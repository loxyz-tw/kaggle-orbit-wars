"""Behavior-cloning trainer for the lean entity-transformer policy (jax_lean_selfplay.LeanEntityPolicy).

Teacher-forced factored supervision on the lean policy:
  focus CE (has_focus, masked to focus_valid) + fire BCE (canfire)
  + target CE (fire=1, masked to alive) + frac CE (fire=1, FB20U bins).

Usage (see train_bc_2p_h640.sh / train_bc_4p_l3pack.sh for the final recipes):
  python bc_train_clone.py --hidden 768 --layers 3 --heads 8 --no-focus --per-step \
      --stream 'bc_chunks/*_c*.npz' --out ckpt/my_bc
"""
from __future__ import annotations
import argparse, pickle, time, glob, os
import numpy as np
import jax, jax.numpy as jnp
import optax
import jax_lean_selfplay as L
from jax_lean_selfplay import LeanEntityPolicy, mask_focus_logits, mask_target_logits

L.OBS_V2 = False; L.RICH_FEATURES = True; L.ORBIT_FEATURES = True

SIZES = {"S": dict(hidden_size=128, num_layers=2, num_heads=4),
         "L": dict(hidden_size=256, num_layers=3, num_heads=8)}


def load(paths):
    parts = [np.load(p) for p in paths]
    keys = set(parts[0].files)
    for p in parts[1:]:
        keys &= set(p.files)                     # intersection: replay npz adds "weight", producer npz lacks it
    out = {k: np.concatenate([p[k] for p in parts], 0) for k in keys}
    if "weight" not in out:                       # producer demos / mixed -> no per-sample outcome weight
        out["weight"] = np.ones(out["feats"].shape[0], np.float32)
    return out


def _rank_frac(sl_ships, sl_me, nseats, xp):
    """My placement among seats by GARRISON ships: (#seats strictly ahead) / (nseats-1).
    0.0 = 1st place, 1.0 = last. `xp` = np (init, host npz) or jnp (on-device batch); the op
    set is shared so one impl serves both. [B,4],[B],[B] -> [B].
    ⚠️ MUST mirror serve-side lean_submission_numpy._rank_frac EXACTLY (strict `>`, garrison-only
    ships, denom max(nseats-1,1)) or the deployed model gets a misaligned col = silent 0%."""
    sh = sl_ships.astype(xp.float32)                               # [B,4] per-seat garrison ships
    me = sl_me.astype(xp.int32); ns = nseats.astype(xp.int32)     # [B]
    ar = xp.arange(sh.shape[0]); seat = xp.arange(4)[None, :]
    my = sh[ar, me]                                                # [B] my ships
    valid = seat < ns[:, None]                                    # [B,4] real seats only
    ahead = xp.where(valid, sh > my[:, None], False).sum(1).astype(xp.float32)
    return ahead / xp.maximum(ns.astype(xp.float32) - 1.0, 1.0)   # [B] in [0,1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=("S", "L"), default=None,    # preset shortcut (S/L); else give --hidden/--layers/--heads
                    help="preset shortcut. For big models give --hidden/--layers/--heads directly "
                         "(states~=params: 11.5M states -> ~h512/L6 ~9M; 28M -> ~h768/L8 ~28M).")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--demos", nargs="+", default=["bc_demos_s0.npz", "bc_demos_s1.npz"])
    ap.add_argument("--stream", default=None, help="glob of chunk npz (gen --chunk-samples). Streams "
                    "one chunk at a time instead of loading all into RAM -> unlocks 28M+ states.")
    ap.add_argument("--val-frac", type=float, default=0.05)   # fraction of CHUNK FILES held out for val (stream mode)
    # defaults = the config of an early strong clone (h512/L6).
    ap.add_argument("--epochs", type=int, default=2)          # sweepable; 2 = proven (supervised acc doesn't predict play -> gate on panel)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=4e-4)
    # optimizer schedule (OFF by default = plain adam constant lr, identical to the proven runs).
    # For SCALE-UP (deeper/wider from-scratch transformer is more lr-sensitive): warmup ramps lr 0->peak
    # over the first steps so early big/noisy grads don't shove a deep net into a bad basin; cosine decay
    # tapers at the end. grad-clip is an early blow-up safety. Gate fresh at scale-up, don't mix into A/Bs.
    ap.add_argument("--lr-schedule", choices=("const", "cosine"), default="const",
                    help="const = plain adam (default, proven). cosine = warmup_cosine_decay (use --warmup).")
    ap.add_argument("--warmup", type=int, default=0, help="warmup steps (cosine schedule only; ramp lr 0->peak).")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="global-norm grad clip (0 = off). ~1.0 for scale-up.")
    ap.add_argument("--per-step", action="store_true",            # big model: fused train_chunk OOMs (XLA
                    help="process each chunk batch-by-batch (train_step) instead of the fused whole-chunk scan. "
                         "The fused scan materializes the WHOLE chunk's trunk output (~chunk*P*hidden) -> OOM at "
                         "h768/L8 on 16GB. per-step bounds memory to one batch; async dispatch hides the overhead.")
    ap.add_argument("--ckpt-every-chunks", type=int, default=0,    # 0 = off (only per-epoch); >0 = mid-epoch curve
                    help="stream mode: ALSO dump a playable ckpt (out.s<step>) every N train chunks, so we get a "
                         "PANEL learning curve over training (acc!=play; CPU-eval them concurrently). 0 = off.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output ckpt path (default ckpt/bc_clone_<size>)")
    ap.add_argument("--resume", action="store_true", help="resume from <out>.resume if it exists (per-epoch checkpoint)")
    ap.add_argument("--init-checkpoint", default="", help="warm-start params from an arbitrary ckpt (e.g. a converged "
                    "ep2) as INIT, then train fresh-opt. Arch MUST match (h/L/heads/frac-bins). For fine-tune-from-"
                    "converged use a LOW const lr (NOT cosine-to-peak). Distinct from --resume (which reloads <out>.resume).")
    ap.add_argument("--no-focus", action="store_true",
                    help="freeze the focus path (focus_proj/bias/head stay at zero-init) and drop the "
                         "focus loss, so per-planet targets are INDEPENDENT (independent "
                         "per-planet heads, no focus). Human turns hit 2.2 targets/turn (59%% multi-target); "
                         "single-focus bottleneck can mangle them. A/B this vs focus-on (37.5 used focus-on).")
    # Key fix: REBALANCE TOWARD DECISIONS. no-op is the overwhelming label -> fire head
    # collapses to always-no-op. Up-weight the positive (fire) class in the BCE so launches don't
    # drown in idle planets. "auto" = (1-rate)/rate from the data's measured fire-rate-on-canfire.
    ap.add_argument("--fire-pos-weight", default="auto",
                    help="fire-head positive-class weight. 'auto' = (1-r)/r from measured fire-rate; "
                         "or pass a float. NEVER 1.0 (=no rebalance=no-op collapse).")
    # Weighting axis = PLAYER RANK ONLY. gen stores the player's ladder score per sample
    # (--leaderboard --min-score selects strong players); we learn ALL their moves -- including the
    # ones from games they LOST (a 1600 beaten by a 1750 is still 1600 play, far above us <1200).
    # The STATE-level outcome down-weighting (old --ow*) was conceptually wrong and is REMOVED.
    ap.add_argument("--is-4p-feature", action="store_true",
                    help="append a broadcast is_4p bit (nseats==4) as an extra feature dim so ONE model "
                         "can condition on player count (2p vs 4p). Uses nseats already in chunks (NO regen). "
                         "feat dim 25->26; inference (lean_submission_numpy) appends it from detected player count.")
    ap.add_argument("--rank-feature", action="store_true",
                    help="append a broadcast rank_frac = my placement by garrison ships among seats, "
                         "(#seats ahead)/(nseats-1), 0=1st..1=last. Computed from sl_ships/sl_me/nseats "
                         "ALREADY in chunks (NO regen). 4p-awareness lever (2p ~= binary, near-redundant w/ adv). "
                         "Appended AFTER is_4p; serve-side recomputes from live obs garrison sums (must match).")
    ap.add_argument("--outcome-weight", action="store_true",
                    help="weight each sample by the demonstrator's LADDER SCORE (player rank). OFF -> "
                         "w=1 (uniform over the selected players). Selection is done at gen time.")
    ap.add_argument("--rank-weight", choices=("linear", "step"), default="linear",
                    help="how --outcome-weight maps ladder score -> sample weight. linear = raw score "
                         "(current default). step = piecewise 1 / 1.5 / 2 / 3 at <1400 / 1400-1500 / "
                         "1500-1600 / >=1600 (sharper rank emphasis; nobody zeroed). No regen needed.")
    ap.add_argument("--win-only", action="store_true",            # (tested = wash; dropping lost games is wrong)
                    help="DEPRECATED/wash: zero loss weight on LOST-game states (sl_win<=0). Drops the "
                         "from-behind coverage the clone needs -> use --win-weight instead (up-weight, keep lost).")
    ap.add_argument("--win-weight", type=float, default=2.0,       # DEFAULT-ON 2.0 (2026-06-19): UP-weight won games x2, KEEP lost (outcome lever)
                    help="multiply loss weight by this on states from games the seat WON (sl_win>0); LOST games "
                         "stay at 1 (NOT dropped). Emphasises aggressive/winning play while keeping the "
                         "from-behind states (lost games) the clone needs for coverage. 1.0=off. Try ~2-3.")
    ap.add_argument("--top-mult", type=float, default=1.0,         # 2026-06-23: up-weight the very top demonstrators (e.g. top-3, sl_ladder>=top-thresh) x this; KEEP all (no filter). 1.0=off.
                    help="multiply loss weight by this on samples whose player's sl_ladder >= --top-thresh "
                         "(lean HARD on the strongest demonstrators without dropping coverage). 1.0=off. Try ~5.")
    ap.add_argument("--top-thresh", type=float, default=1640.0,    # default ~= current ladder #3 (top-3 tier)
                    help="sl_ladder cutoff for the --top-mult up-weight tier (default 1640 ~= top-3).")
    ap.add_argument("--min-score", type=float, default=0.0,        # HARD filter-at-load: keep only sl_ladder>=this
                    help="Drop every sample whose player's ladder score (sl_ladder) < this AT LOAD (dense batches). "
                         "Selectivity lever: clone ONLY the strongest demonstrators (e.g. 1700) so the model learns "
                         "producer-beating play, not the rating-mean. 0=off. Sizes the model: states>=1700 ~2.5M.")
    ap.add_argument("--players", type=int, default=0, choices=(0, 2, 4),  # HARD filter-at-load by game player-count
                    help="Keep only samples from games with this player-count (reads nseats in chunks, NO regen). "
                         "2 = 2p-only (nseats<=2), 4 = 4p-only (nseats>=3), 0 = all (default). Use 2 to train a "
                         "2P-specialist on pure-2p states (the model is dispatched only for 2p). Combines with --min-score.")
    ap.add_argument("--ahead-only", action="store_true",          # distribution shift: learn only from leaders
                    help="per-state dominance filter: zero the loss weight on states where the learning seat is "
                         "NOT currently >= every opponent in BOTH ships AND production (reads sl_ships/sl_prod, "
                         "no regen). Tests 'learn only the moves of players who are out-expanding'. ~43%% survive.")
    ap.add_argument("--capture-weight", type=float, default=1.0,    # distribution shift: rebalance target choice
                    help="up-weight CAPTURE launches (target=enemy/neutral) vs REINFORCE (target=own) in the "
                         "tgt/frac loss. Clone collapses to the reinforce majority (data 71%% reinf/29%% cap "
                         "-> clone 78%%/13%%); this is fire-pos-weight one level deeper (capture-vs-reinforce). "
                         "1.0 = off. Try ~2-4. Target owner read from feats[...,0]=is_mine, no regen.")
    args = ap.parse_args()

    rng = np.random.default_rng(123)
    # ---- data source: --stream (chunk npz, RAM-bounded, unlocks 28M+) OR --demos (full in-RAM load) ----
    if args.stream:
        files = sorted(glob.glob(args.stream))
        if not files:
            raise SystemExit(f"no chunk files match {args.stream}")
        nvf = max(1, int(round(len(files) * args.val_frac)))
        # RANDOM (fixed-seed) hold-out: don't take the first-sorted files -> a chunk order by
        # day/score/seat would silently skew val. Dedicated rng (NOT the main `rng`) keeps the
        # hold-out reproducible across runs AND leaves the train-order rng stream untouched.
        vsel = set(np.random.default_rng(2024).permutation(len(files))[:nvf].tolist())
        val_files = [files[i] for i in range(len(files)) if i in vsel]
        train_files = [files[i] for i in range(len(files)) if i not in vsel] or files[:1]
        if args.outcome_weight:                              # global weight mean across ALL chunks (loss scale)
            s = c = 0.0
            for f in files:
                w = np.load(f)["weight"]; s += float(w.sum()); c += len(w)
            wmean = max(s / c, 1e-6)
            print(f"stream: {len(files)} chunks ({len(train_files)} train/{len(val_files)} val), outcome wmean={wmean:.0f}", flush=True)
        else:
            wmean = None
            print(f"stream: {len(files)} chunks ({len(train_files)} train/{len(val_files)} val)", flush=True)
        _i = np.load(train_files[0]); init_feats = _i["feats"][:2]; init_alive = _i["alive"][:2]
        if args.is_4p_feature:                                # +1 dim so embed in-dim = 26 (is_4p from nseats)
            _i4 = np.broadcast_to((_i["nseats"][:2].astype(np.float32) == 4.0)[:, None, None],
                                  init_feats.shape[:2] + (1,))
            init_feats = np.concatenate([init_feats.astype(np.float32), _i4], axis=-1)
        if args.rank_feature:                                 # +1 dim: rank_frac (appended AFTER is_4p)
            _rf = _rank_frac(_i["sl_ships"][:2], _i["sl_me"][:2], _i["nseats"][:2], np)
            init_feats = np.concatenate([init_feats.astype(np.float32),
                                         np.broadcast_to(_rf[:, None, None], init_feats.shape[:2] + (1,))], axis=-1)
        d = tidx = vidx = None; n_val = 0
    else:
        d = load(args.demos)
        wmean = max(d["weight"].astype(np.float32).mean(), 1e-6) if args.outcome_weight else None
        n = d["feats"].shape[0]
        perm = rng.permutation(n); n_val = max(n // 20, 1)
        vidx, tidx = perm[:n_val], perm[n_val:]
        train_files = val_files = None
        print(f"demos {n} samples ({len(tidx)} train / {n_val} val), feat dim {d['feats'].shape[-1]}"
              + (f", outcome wmean={wmean:.0f}" if wmean else ""), flush=True)
        init_feats = d["feats"][:2]; init_alive = d["alive"][:2]
        if args.is_4p_feature:
            _i4 = np.broadcast_to((d["nseats"][:2].astype(np.float32) == 4.0)[:, None, None],
                                  init_feats.shape[:2] + (1,))
            init_feats = np.concatenate([init_feats.astype(np.float32), _i4], axis=-1)
        if args.rank_feature:                                 # +1 dim: rank_frac (appended AFTER is_4p)
            _rf = _rank_frac(d["sl_ships"][:2], d["sl_me"][:2], d["nseats"][:2], np)
            init_feats = np.concatenate([init_feats.astype(np.float32),
                                         np.broadcast_to(_rf[:, None, None], init_feats.shape[:2] + (1,))], axis=-1)

    # ---- resolve fire-pos-weight (decision rebalance). "auto" = (1-r)/r from measured
    #      fire-rate over canfire planets, so launches don't drown in idle planets. ----
    if str(args.fire_pos_weight) == "auto":
        nfire = ncan = 0.0
        scan = (train_files[:8] if args.stream else None)
        if args.stream:
            for f in scan:
                z = np.load(f); fr = z["fire"].astype(bool); cf = z["canfire"].astype(bool); z.close()
                nfire += float((fr & cf).sum()); ncan += float(cf.sum())
        else:
            fr = d["fire"].astype(bool); cf = d["canfire"].astype(bool)
            nfire += float((fr & cf).sum()); ncan += float(cf.sum())
        rate = max(nfire / max(ncan, 1.0), 1e-4)
        fire_pos_weight = (1.0 - rate) / rate
        print(f"fire-pos-weight auto: fire-rate-on-canfire={rate:.4f} -> pos_weight={fire_pos_weight:.2f}", flush=True)
    else:
        fire_pos_weight = float(args.fire_pos_weight)
        if fire_pos_weight == 1.0:
            print("WARNING: --fire-pos-weight 1.0 = no decision rebalance -> expect no-op collapse", flush=True)

    if args.hidden:                                          # explicit spec (parametrized; ckpt name carries it)
        cfg = dict(hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads)
        spec = f"h{args.hidden}L{args.layers}h{args.heads}"
    elif args.size:
        cfg = SIZES[args.size]; spec = args.size
    else:
        raise SystemExit("give --size S|L OR --hidden/--layers/--heads")
    policy = LeanEntityPolicy(num_frac_bins=20, **cfg)
    key = jax.random.PRNGKey(args.seed)
    f0 = jnp.asarray(init_feats, jnp.float32)
    params = policy.init(key, f0, jnp.asarray(init_alive))["params"]
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"size {spec}: {cfg} -> {n_params/1e3:.0f}K params")

    if args.init_checkpoint:                                  # warm-start params from a converged ckpt (arch must match)
        with open(args.init_checkpoint, "rb") as _f:
            _ck = pickle.load(_f)
        ck_params = _ck["params"] if isinstance(_ck, dict) and "params" in _ck else _ck
        # shape-check: fail loud on any mismatch (wrong arch/frac-bins = silent garbage otherwise)
        s_init = {"/".join(map(str, p)): v.shape for p, v in jax.tree_util.tree_flatten_with_path(params)[0]}
        s_ck = {"/".join(map(str, p)): v.shape for p, v in jax.tree_util.tree_flatten_with_path(ck_params)[0]}
        if s_init != s_ck:
            raise SystemExit(f"--init-checkpoint arch mismatch vs this run's policy:\n  only-in-init={set(s_init)-set(s_ck)}\n"
                             f"  only-in-ckpt={set(s_ck)-set(s_init)}\n  shape-diffs="
                             f"{ {k:(s_init[k],s_ck[k]) for k in s_init if k in s_ck and s_init[k]!=s_ck[k]} }")
        params = jax.tree_util.tree_map(lambda a: jnp.asarray(a), ck_params)
        print(f"warm-start from {args.init_checkpoint} (arch matched, fresh opt)", flush=True)

    if args.lr_schedule == "const" and args.grad_clip <= 0:
        tx = optax.adam(args.lr)                              # default path: unchanged + resume-compatible
    else:
        if args.lr_schedule == "cosine":                     # total optimizer steps (for decay length)
            def _nkeep(f):                                    # count post-filter samples for correct decay length
                if args.min_score > 0 or args.players:        # must mirror _load_chunk's keep mask exactly
                    z = np.load(f); k = np.ones(z["weight"].shape[0], dtype=bool)
                    if args.min_score > 0:
                        k &= (z["sl_ladder"] >= args.min_score)
                    if args.players:
                        k &= (z["nseats"] <= 2) if args.players == 2 else (z["nseats"] >= 3)
                    n = int(k.sum()); z.close(); return n
                return int(np.load(f)["weight"].shape[0])
            per_ep = (sum(_nkeep(f) // args.batch for f in train_files)
                      if args.stream else len(tidx) // args.batch)
            total_steps = max(1, per_ep * args.epochs)
            sched = optax.warmup_cosine_decay_schedule(
                init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup,
                decay_steps=total_steps, end_value=args.lr * 0.1)
            print(f"lr-schedule cosine: warmup {args.warmup} -> peak {args.lr:g} -> end {args.lr*0.1:g} "
                  f"over {total_steps} steps", flush=True)
        else:
            sched = args.lr
        chain = []
        if args.grad_clip > 0:
            chain.append(optax.clip_by_global_norm(args.grad_clip))
            print(f"grad-clip global-norm {args.grad_clip}", flush=True)
        chain.append(optax.adam(sched))
        tx = optax.chain(*chain)                              # NOTE: chain-wrapped opt state -> only resume ckpts trained with the SAME flags
    opt = tx.init(params)

    # --no-focus: grad mask that pins focus_proj/focus_bias/focus_head at their zero-init, turning
    # the net into an independent per-planet supervised model (multi-target representable).
    def _focus_mask(params):
        def f(path, x):
            nm = "/".join(getattr(k, "key", str(k)) for k in path)
            return jnp.zeros_like(x) if any(s in nm for s in ("focus_proj", "focus_bias", "focus_head")) else jnp.ones_like(x)
        return jax.tree_util.tree_map_with_path(f, params)
    focus_mask = _focus_mask(params) if args.no_focus else None

    def loss_fn(params, batch):
        feats, alive = batch["feats"], batch["alive"]
        w = batch["weight"]                                    # [B] per-sample outcome weight (1.0 if --outcome-weight off)
        wp = w[:, None]                                        # broadcast over planet slots
        x, focus_logits, _ = policy.apply({"params": params}, feats, alive, method=policy.trunk)
        focus_ml = mask_focus_logits(focus_logits, batch["focus_valid"])
        lf = optax.softmax_cross_entropy_with_integer_labels(focus_ml, batch["focus"])
        hf = batch["has_focus"] * w
        l_focus = (lf * hf).sum() / jnp.maximum(hf.sum(), 1.0)

        fire_logit, target_logits, frac_logits = policy.apply(
            {"params": params}, x, batch["focus"], alive, method=policy.action_heads)
        # fire BCE with positive-class up-weighting (anti no-op collapse). Denominator uses the
        # SAME per-sample weight so the loss stays a proper weighted mean (no scale blow-up).
        fpw = jnp.where(batch["fire"], fire_pos_weight, 1.0)
        cf = batch["canfire"] * wp * fpw
        lb = optax.sigmoid_binary_cross_entropy(fire_logit, batch["fire"].astype(jnp.float32))
        l_fire = (lb * cf).sum() / jnp.maximum(cf.sum(), 1.0)

        tml = mask_target_logits(target_logits, alive)
        lt = optax.softmax_cross_entropy_with_integer_labels(tml, batch["tgt"].astype(jnp.int32))
        fm = (batch["fire"] & batch["canfire"]) * wp
        if args.capture_weight != 1.0:                        # up-weight CAPTURE (target != own) vs REINFORCE
            P = feats.shape[1]
            tslot = jnp.clip(batch["tgt"].astype(jnp.int32), 0, P - 1)
            tgt_is_mine = jnp.take_along_axis(feats[..., 0], tslot, axis=1)   # is_mine of the launch's target
            fm = fm * jnp.where(tgt_is_mine < 0.5, args.capture_weight, 1.0)
        l_tgt = (lt * fm).sum() / jnp.maximum(fm.sum(), 1.0)
        lfr = optax.softmax_cross_entropy_with_integer_labels(frac_logits, batch["frac"].astype(jnp.int32))
        l_frac = (lfr * fm).sum() / jnp.maximum(fm.sum(), 1.0)
        loss = (0.0 if args.no_focus else l_focus) + l_fire + l_tgt + l_frac
        return loss, (l_focus, l_fire, l_tgt, l_frac)

    @jax.jit
    def train_step(params, opt, batch):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        if focus_mask is not None:                       # pin focus_proj/bias/head at zero-init
            g = jax.tree_util.tree_map(lambda gi, mi: gi * mi, g, focus_mask)
        up, opt = tx.update(g, opt, params)
        return optax.apply_updates(params, up), opt, loss, aux

    @jax.jit
    def val_metrics(params, batch):
        feats, alive = batch["feats"], batch["alive"]
        x, focus_logits, _ = policy.apply({"params": params}, feats, alive, method=policy.trunk)
        focus_ml = mask_focus_logits(focus_logits, batch["focus_valid"])
        focus_acc = ((focus_ml.argmax(-1) == batch["focus"]) * batch["has_focus"]).sum() / jnp.maximum(batch["has_focus"].sum(), 1.0)
        fire_logit, target_logits, frac_logits = policy.apply(
            {"params": params}, x, batch["focus"], alive, method=policy.action_heads)
        cf = batch["canfire"]
        fire_pred = fire_logit > 0
        fire_acc = ((fire_pred == batch["fire"]) * cf).sum() / jnp.maximum(cf.sum(), 1.0)
        fm = batch["fire"] & cf
        pos_rec = ((fire_pred & batch["fire"]) * cf).sum() / jnp.maximum(fm.sum(), 1.0)
        tml = mask_target_logits(target_logits, alive)
        tgt_acc = ((tml.argmax(-1) == batch["tgt"]) * fm).sum() / jnp.maximum(fm.sum(), 1.0)
        frac_acc = ((frac_logits.argmax(-1) == batch["frac"]) * fm).sum() / jnp.maximum(fm.sum(), 1.0)
        # cap_rate = of the model's PREDICTED launches, fraction whose PREDICTED target is NOT own
        # (the capture% knob target; data label cap rate = 29%, ep2 clone ~13% -> want this UP).
        P = feats.shape[1]
        pred_launch = fire_pred & cf
        ptgt_mine = jnp.take_along_axis(feats[..., 0], jnp.clip(tml.argmax(-1), 0, P - 1), axis=1)
        cap_rate = ((ptgt_mine < 0.5) & pred_launch).sum() / jnp.maximum(pred_launch.sum(), 1.0)
        return focus_acc, fire_acc, pos_rec, tgt_acc, frac_acc, cap_rate

    def _mk_batch(src, idx):                                  # src = DEVICE dict (chunk already on GPU); idx = jnp
        # weighting axis = PLAYER RANK ONLY (ladder score). learn every move of the selected players,
        # incl. their lost games. (state-level outcome down-weighting was wrong -> removed.)
        if args.outcome_weight:
            wkey = "sl_ladder" if "sl_ladder" in src else "weight"
            w = src[wkey][idx].astype(jnp.float32)
            if args.rank_weight == "step":                    # piecewise rank emphasis (nobody zeroed)
                w = jnp.where(w >= 1600, 3.0, jnp.where(w >= 1500, 2.0, jnp.where(w >= 1400, 1.5, 1.0)))
        else:
            w = jnp.ones(idx.shape[0], jnp.float32)
        if args.win_only and "sl_win" in src:                 # (wash) drop lost-game states
            w = w * (src["sl_win"][idx] > 0).astype(jnp.float32)
        if args.win_weight != 1.0 and "sl_win" in src:        # UP-weight won games, KEEP lost (coverage)
            w = w * jnp.where(src["sl_win"][idx] > 0, args.win_weight, 1.0)
        if args.top_mult != 1.0 and "sl_ladder" in src:       # UP-weight the top-tier demonstrators (top-3), KEEP all
            w = w * jnp.where(src["sl_ladder"][idx] >= args.top_thresh, args.top_mult, 1.0)
        if args.ahead_only:                                   # per-state dominance: zero non-ahead states
            sh = src["sl_ships"][idx].astype(jnp.float32); pr = src["sl_prod"][idx].astype(jnp.float32)  # [B,4]
            me = src["sl_me"][idx].astype(jnp.int32); ns = src["nseats"][idx].astype(jnp.int32)          # [B]
            ar = jnp.arange(idx.shape[0]); seat = jnp.arange(4)[None, :]
            oppmask = (seat < ns[:, None]) & (seat != me[:, None])     # real opponents only
            opp_sh = jnp.where(oppmask, sh, -1.0).max(1); opp_pr = jnp.where(oppmask, pr, -1.0).max(1)
            ahead = (sh[ar, me] >= opp_sh) & (pr[ar, me] >= opp_pr)    # I lead BOTH ships & prod right now
            w = w * ahead.astype(jnp.float32)
        fb = src["feats"][idx].astype(jnp.float32)
        if args.is_4p_feature:                                # append broadcast is_4p (nseats==4) -> +1 dim
            i4 = jnp.broadcast_to((src["nseats"][idx].astype(jnp.float32) == 4.0)[:, None, None],
                                  fb.shape[:2] + (1,))
            fb = jnp.concatenate([fb, i4], axis=-1)
        if args.rank_feature:                                 # append broadcast rank_frac (AFTER is_4p) -> +1 dim
            rf = _rank_frac(src["sl_ships"][idx], src["sl_me"][idx], src["nseats"][idx], jnp)
            fb = jnp.concatenate([fb, jnp.broadcast_to(rf[:, None, None], fb.shape[:2] + (1,))], axis=-1)
        return {                                              # all gathers happen ON-DEVICE (no per-batch host->device xfer)
            "feats": fb,
            "alive": src["alive"][idx],
            "canfire": src["canfire"][idx],
            "focus_valid": src["focus_valid"][idx],
            "fire": src["fire"][idx],
            "tgt": src["tgt"][idx],
            "frac": src["frac"][idx],
            "focus": src["focus"][idx].astype(jnp.int32),
            "has_focus": src["has_focus"][idx],
            "weight": w,
        }

    def _load_chunk(f):                                       # decompress ONE chunk, push WHOLE chunk to GPU ONCE
        z = np.load(f); gd = {k: jnp.asarray(z[k]) for k in z.files}; z.close()
        keep = None                                           # HARD filter-at-load (dense batches)
        if args.min_score > 0 and "sl_ladder" in gd:          #   selectivity: strong demonstrators only
            keep = gd["sl_ladder"] >= args.min_score
        if args.players and "nseats" in gd:                   #   player-count: 2p-only / 4p-only specialist
            pk = (gd["nseats"] <= 2) if args.players == 2 else (gd["nseats"] >= 3)
            keep = pk if keep is None else (keep & pk)
        if keep is not None:
            gd = {k: (v[keep] if v.shape[0] == keep.shape[0] else v) for k, v in gd.items()}
        return gd

    @jax.jit
    def train_chunk(params, opt, chunk, perm):                # ALL batches of a chunk in ONE jit (no per-step py dispatch)
        B = args.batch; nb = perm.shape[0] // B
        pm = perm[:nb * B].reshape(nb, B)
        def body(carry, bidx):
            p, o = carry
            (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, _mk_batch(chunk, bidx))
            if focus_mask is not None:
                g = jax.tree_util.tree_map(lambda gi, mi: gi * mi, g, focus_mask)
            up, o = tx.update(g, o, p)
            return (optax.apply_updates(p, up), o), loss
        (params, opt), losses = jax.lax.scan(body, (params, opt), pm)
        return params, opt, losses

    out = args.out or f"ckpt/bc_clone_{spec}"
    rpath = out + ".resume"
    start_ep = 0
    resume_pos = 0; resume_perm = None
    if args.resume and os.path.exists(rpath):                # MID-EPOCH checkpoint: survive hang/reboot
        rs = pickle.load(open(rpath, "rb"))
        params = jax.device_put(rs["params"]); opt = jax.device_put(rs["opt"]); start_ep = rs["ep"]
        resume_pos = rs.get("pos", 0); resume_perm = rs.get("perm", None)   # within-epoch chunk position + order
        print(f"RESUMED from {rpath}: epoch {start_ep}/{args.epochs}, chunk pos {resume_pos}", flush=True)

    t0 = time.time()
    gstep = start_ep * 10**9                                  # monotone-ish id for filenames (epoch-prefixed)
    nchunk = 0
    for ep in range(start_ep, args.epochs):
        step = 0
        if args.stream:
            if ep == start_ep and resume_perm is not None:   # resuming MID-epoch: reuse saved order + position
                ep_perm = list(resume_perm); pos0 = resume_pos
            else:
                ep_perm = list(rng.permutation(len(train_files))); pos0 = 0
            for ci in range(pos0, len(ep_perm)):
                fi = ep_perm[ci]
                cd = _load_chunk(train_files[fi])
                N = cd["feats"].shape[0]; cperm = rng.permutation(N); nbp = N // args.batch
                if nbp == 0:                                  # chunk smaller than one batch (rare tail) -> skip
                    for _v in cd.values():
                        _v.delete()
                    del cd; nchunk += 1; continue
                if args.per_step:                            # memory-safe: per-batch (fused scan OOMs the big model)
                    for bi in range(nbp):
                        bidx = jnp.asarray(cperm[bi * args.batch:(bi + 1) * args.batch])
                        params, opt, loss, aux = train_step(params, opt, _mk_batch(cd, bidx))
                    last_loss = float(loss)
                else:                                        # fused whole-chunk scan (fast; small models only)
                    params, opt, losses = train_chunk(params, opt, cd, jnp.asarray(cperm))
                    last_loss = float(losses[-1])
                step += nbp; gstep += nbp; nchunk += 1
                print(f"  ep{ep} step {step} loss={last_loss:.4f} [{time.time()-t0:.0f}s]", flush=True)
                for _v in cd.values():                       # free this chunk's device arrays BEFORE loading the next
                    _v.delete()
                del cd
                if args.ckpt_every_chunks and nchunk % args.ckpt_every_chunks == 0:
                    pg = jax.device_get(params)
                    # MID-EPOCH RESUME point (params+opt+position+order) -> a crash loses <= ckpt-every-chunks chunks
                    pickle.dump({"params": pg, "opt": jax.device_get(opt), "ep": ep,
                                 "pos": ci + 1, "perm": list(ep_perm)}, open(rpath, "wb"))
                    sp = f"{out}.ep{ep}s{step}"               # + PLAYABLE ckpt -> panel learning curve
                    pickle.dump({"params": pg, "config": {**cfg, "num_frac_bins": 20, "obs": "rich+orbit", "bc": True, "is_4p_feature": args.is_4p_feature, "rank_feature": args.rank_feature}},
                                open(sp, "wb"))
                    print(f"  [curve+resume] saved {sp} (pos {ci + 1}/{len(ep_perm)})", flush=True)
        else:
            rng.shuffle(tidx)
            for i in range(len(tidx) // args.batch):
                params, opt, loss, aux = train_step(params, opt, _mk_batch(d, tidx[i * args.batch:(i + 1) * args.batch]))
                if i % 200 == 0:
                    print(f"  ep{ep} step {i} loss={float(loss):.4f} (foc {float(aux[0]):.3f} "
                          f"fire {float(aux[1]):.3f} tgt {float(aux[2]):.3f} frac {float(aux[3]):.3f}) "
                          f"[{time.time()-t0:.0f}s]", flush=True)
        # val
        accs = []
        if args.stream:
            for f in val_files:
                cd = _load_chunk(f); nn = cd["feats"].shape[0]
                for j in range(0, nn - args.batch + 1, args.batch):
                    accs.append([float(a) for a in val_metrics(params, _mk_batch(cd, jnp.arange(j, j + args.batch)))])
        else:
            for j in range(0, n_val, args.batch):
                accs.append([float(a) for a in val_metrics(params, _mk_batch(d, vidx[j:j + args.batch]))])
        m = np.mean(accs, 0)
        print(f"[ep{ep}] VAL focus_acc={m[0]:.3f} fire_acc={m[1]:.3f} fire_recall={m[2]:.3f} "
              f"tgt_acc={m[3]:.3f} frac_acc={m[4]:.3f} CAP_RATE={m[5]:.3f} (data=0.29)", flush=True)
        pg = jax.device_get(params)
        pickle.dump({"params": pg, "opt": jax.device_get(opt), "ep": ep + 1, "pos": 0, "perm": None},
                    open(rpath, "wb"))                       # epoch-boundary resume point (next epoch starts fresh)
        # ALSO save a PLAYABLE per-epoch ckpt (same format as the final `out`). acc doesn't predict
        # play, so we can't pick the best epoch by val -> keep every epoch and panel-eval them.
        ep_ckpt = f"{out}.ep{ep + 1}"
        pickle.dump({"params": pg, "config": {**cfg, "num_frac_bins": 20, "obs": "rich+orbit", "bc": True, "is_4p_feature": args.is_4p_feature, "rank_feature": args.rank_feature}},
                    open(ep_ckpt, "wb"))
        print(f"  [ep{ep}] saved per-epoch ckpt {ep_ckpt}", flush=True)

    pickle.dump({"params": jax.device_get(params),
                 "config": {**cfg, "num_frac_bins": 20, "obs": "rich+orbit", "bc": True, "is_4p_feature": args.is_4p_feature, "rank_feature": args.rank_feature}},
                open(out, "wb"))
    if os.path.exists(rpath):
        os.remove(rpath)                                     # done -> drop the resume checkpoint
    print(f"saved {out} ({n_params/1e3:.0f}K params)")


if __name__ == "__main__":
    main()
