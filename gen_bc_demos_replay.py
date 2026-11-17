"""Build a supervised (behavior-cloning) dataset from official ladder replays.

Selective imitation of strong players: clone moves of strong demonstrators (filtered by public
leaderboard score), weight by outcome, and rebalance toward DECISION states so the no-op
majority doesn't drown out real moves. Learning from real top-ladder replays is a strictly
stronger data source than cloning a fixed heuristic.

Reuses the exact train feature+label machinery so the dataset is drop-in for
bc_train_clone.py and feature drift vs serving is impossible:
  - obs_to_state_np (obs_state)        : Kaggle obs dict -> JaxOrbitState (slot = obs order)
  - planet_features (jax_lean_selfplay): [1,P,25] RICH+ORBIT, demonstrator's perspective
  - labels_from_rows (this file)       : raw launch rows -> factored per-planet labels

Outcome weighting  : winners only (default); per-sample weight = winner's final margin.
Decision rebalance : keep every step with a launch; subsample pure-idle steps (--idle-frac).

Usage: python gen_bc_demos_replay.py --replays-dir replays/ --recursive \
                        --leaderboard leaderboard.csv --min-score 0 \
                        [--out bc_replay_demos.npz] [--stride 1] [--idle-frac 0.2]
                        [--obs-shift 1] [--limit N] [--chunk-samples 200000]
"""
from __future__ import annotations
# Feature recipe: RICH 23-dim base + ORBIT (ang_vel + prod_density) = 25-dim per planet.
# MIN_LAUNCH is irrelevant to feature/label generation but pinned for parity.
import os
import argparse, glob, json, time, functools
import numpy as np
import jax, jax.numpy as jnp
import jax_lean_selfplay as L
import jax_min_env_spike as _env
from jax_min_env_spike import JaxOrbitState
from obs_state import obs_to_state_np, MAXP, MAXG, MAXPATH, COMET_QUADRANTS, COMET_RADIUS


FB20U = np.array([0.05 * i for i in range(1, 21)], np.float32)   # 20 uniform frac-of-source bins


def labels_from_rows(B, P, ships, src, tgt, send, val):
    """Merge [B,M] labeled launch rows into per-planet factored labels."""
    fire = np.zeros((B, P), bool)
    tgt_lab = np.zeros((B, P), np.int8)
    frac_lab = np.zeros((B, P), np.int8)
    acc = np.zeros((B, P), np.float32)
    bb, mm = np.where(val)
    s_r = src[bb, mm]; t_r = tgt[bb, mm]; d_r = send[bb, mm]
    np.add.at(acc, (bb, s_r), d_r)
    order = np.argsort(d_r, kind="stable")               # ascending; last write = max send
    tgt_lab[bb[order], s_r[order]] = t_r[order].astype(np.int8)
    fire = acc >= 1.0
    frac = np.clip(acc / np.maximum(ships.astype(np.float32), 1.0), 0.0, 1.0)
    frac_lab = np.abs(frac[..., None] - FB20U[None, None, :]).argmin(-1).astype(np.int8)
    # focus = slot receiving the largest total send per env
    inc = np.zeros((B, P), np.float32)
    np.add.at(inc, (bb, t_r), d_r)
    focus = inc.argmax(1).astype(np.int16)
    has_focus = inc.max(1) >= 1.0
    return fire, tgt_lab, frac_lab, focus, has_focus





# pin the feature recipe (RICH + ORBIT = 25-dim); 4p is_4p/rank columns are appended at train time
L.OBS_V2 = False; L.RICH_FEATURES = True; L.ORBIT_FEATURES = True
L.GRAVITY_FEATURES = False; L.ROI_FEATURES = False; L.MIN_LAUNCH = 10
_env.NUM_PLAYERS = 2
_FEAT = jax.jit(lambda s, p: L.planet_features(s, p))


def _board_obs(step_row):
    """The full board observation is carried under seat 0 in kaggle replays."""
    o = step_row[0].get("observation")
    return o if o and o.get("planets") else None


def _winner_seat(steps, nseats):
    """Rank-1 seat by final reward; fall back to final planet+ship count if rewards are null."""
    final = steps[-1]
    rews = [final[s].get("reward") if s < len(final) else None for s in range(nseats)]
    if any(r is not None for r in rews):
        rr = [(-1e18 if r is None else float(r)) for r in rews]
        win = int(np.argmax(rr))
        srt = sorted(rr, reverse=True)
        margin = srt[0] - srt[1] if len(srt) > 1 else srt[0]
        return win, float(margin)
    ob = _board_obs(final)                                   # reward-less: count planets+ships
    cnt = np.zeros(nseats)
    if ob:
        for p in ob["planets"]:
            o = int(p[1])
            if 0 <= o < nseats:
                cnt[o] += 1 + int(p[5])
    return int(np.argmax(cnt)), float(np.sort(cnt)[-1] - (np.sort(cnt)[-2] if nseats > 1 else 0))


def _state_outcome_factor(owner_slot, ships_slot, prod_slot, alive_slot, seat, nseats, step, episode_steps, loss_w):
    """STATE-level 'how well is this seat doing AT THIS state' factor in [loss_w, 1.0] (outcome
    weighting — weight toward the STATES that went well). STRENGTH = ships + production x H,
    H = remaining turns (prod is a flow -> future ships; planet COUNT is irrelevant, a prod-5 >> prod-1).
    margin vs the STRONGEST opponent -> [loss_w, 1.0]: dominating 1.0, crushed loss_w, even midpoint.
    Early game prod-lead dominates (decides the long game); ships weigh more late."""
    H = max(episode_steps - step, 1)
    def stren(s):
        m = (owner_slot == s) & alive_slot
        return float(ships_slot[m].sum() + H * prod_slot[m].sum())      # stock + future production
    score = np.array([stren(s) for s in range(nseats)])
    mine = score[seat]
    opp = float(np.delete(score, seat).max()) if nseats > 1 else 0.0    # strongest opponent
    tot = mine + opp
    margin = (mine - opp) / tot if tot > 0 else 0.0                     # [-1,1]: +1 dominate, -1 crushed
    return loss_w + (1.0 - loss_w) * (margin + 1.0) / 2.0


def _fleet_speed(ships):
    return float(np.clip(1.0 + 5.0 * (np.log(max(ships, 1)) / np.log(1000.0)) ** 1.5, None, 6.0))


def _board_orbital(state):
    """Per-board arrays for fleet-trajectory target recovery (batch=1)."""
    x = np.asarray(state.x)[0]; y = np.asarray(state.y)[0]; rad = np.asarray(state.radius)[0]
    alive = np.asarray(state.alive)[0]; comet = np.asarray(state.is_comet)[0]
    ix = np.asarray(state.initial_x)[0]; iy = np.asarray(state.initial_y)[0]
    rfc = np.sqrt((ix - 50.0) ** 2 + (iy - 50.0) ** 2)
    rotating = alive & (~comet) & ((rfc + rad) < 50.0)
    return dict(x=x, y=y, rad=rad, alive=alive, rfc=rfc, a0=np.arctan2(iy - 50.0, ix - 50.0),
                av=float(np.asarray(state.angular_velocity)[0]), sc=int(np.asarray(state.step_count)[0]),
                rotating=rotating, n=len(x))


# ---- LINCHPIN: target recovery via the REAL env (jax_min_env_spike) ----------------------------
# Orbit Wars action = (from_planet, ANGLE, ships) with NO target column. We launch the human's exact
# (src, angle, ships) into the env at the board's true state and let the env's OWN swept-collision
# (phase_f) decide which planet the fleet hits = ground-truth target slot. Zero hand-rolled physics
# (the old straight-line _recover_target diverged: single global av, custom speed, H=90, argmin-bias
# to own cluster -> ~71% own-target).
RECOVER_H = 160                                              # max flight steps (board diag ~140 @ min speed 1)

@functools.partial(jax.jit, static_argnums=(3, 4))          # nseats keys NUM_PLAYERS trace; H static
def _fly_to_target(state, act, slot0, nseats, H):
    s = _env.launch_fleets(state, act, actions_are_angles=True)
    launched = s.fleet_alive[0, slot0]                       # False if invalid (owner/ships/amount)
    def body(s, _):
        s, pnx, pny, pcheck = _env.phase_e_planet_paths(s)   # planets/comets orbit one step
        s, cp, ah = _env.phase_f_fleet_movement(s, pnx, pny, pcheck)  # env swept-collision
        s = s._replace(x=pnx, y=pny, step_count=s.step_count + 1)  # FIX: advance step_count or planets FREEZE (was: orbiting planets never moved -> aimed launches missed -> 14.5% strong-player launches falsely dropped as void)
        return s, (ah[0, slot0], cp[0, slot0])
    _, (hits, cps) = jax.lax.scan(body, s, None, length=H)
    idx = jnp.argmax(hits)                                   # first True step (argmax of bool)
    return jnp.where(launched & hits.any(), cps[idx].astype(jnp.int32), -1)


def _recover_target(state, src_slot, angle, ships, seat, nseats):
    """Return the planet SLOT the human's launch (src_slot, angle, ships) actually reaches in the
    real env, or -1 (flew into space / sun / oob, or invalid launch). Pure env ground truth."""
    src_pid = int(np.asarray(state.planet_id)[0, src_slot])  # id that resolve_planet_slot -> src_slot
    slot0 = int(np.asarray(state.next_fleet)[0])             # fleet slot this launch will occupy
    act = np.zeros((1, nseats, 1, 3), np.float32)
    act[0, seat, 0, 0] = src_pid; act[0, seat, 0, 1] = float(angle); act[0, seat, 0, 2] = float(ships)
    return int(_fly_to_target(state, jnp.asarray(act), slot0, int(nseats), RECOVER_H))


# ---- BATCHED recovery + feats (host-sync-bound -> 1 GPU call per replay, not per launch/cell) -----
@functools.partial(jax.jit, static_argnums=(3, 4))          # nseats, H static; batch B padded to fixed CHUNK
def _fly_to_target_batch(states, acts, slot0s, nseats, H):
    B = slot0s.shape[0]; bidx = jnp.arange(B)
    s = _env.launch_fleets(states, acts, actions_are_angles=True)
    launched = s.fleet_alive[bidx, slot0s]
    def body(s, _):
        s, pnx, pny, pcheck = _env.phase_e_planet_paths(s)
        s, cp, ah = _env.phase_f_fleet_movement(s, pnx, pny, pcheck)
        s = s._replace(x=pnx, y=pny, step_count=s.step_count + 1)  # FIX: advance step_count or planets FREEZE (see _fly_to_target)
        return s, (ah[bidx, slot0s], cp[bidx, slot0s])
    _, (hits, cps) = jax.lax.scan(body, s, None, length=H)   # hits/cps: (H, B)
    first = jnp.argmax(hits, axis=0)                          # (B,) first True step
    return jnp.where(launched & hits.any(axis=0), cps[first, bidx], -1).astype(jnp.int32)


def _stack_states(state_list):                               # concat batch=1 (numpy) JaxOrbitStates on host
    return jax.tree_util.tree_map(lambda *xs: np.concatenate(xs, axis=0), *state_list)   # device xfer once at jit call


_REC_CHUNK = 512


def batch_recover(state_list, act_list, slot0_list, nseats):
    """Fly ALL launches (each a batch=1 state + its action) through the env in CHUNKs; 1 sync/chunk."""
    if not state_list:
        return np.zeros((0,), np.int32)
    out = []
    z_act = np.zeros_like(act_list[0])
    for i in range(0, len(state_list), _REC_CHUNK):
        sl = state_list[i:i + _REC_CHUNK]; al = act_list[i:i + _REC_CHUNK]; s0 = list(slot0_list[i:i + _REC_CHUNK])
        B = len(sl); pad = _REC_CHUNK - B
        if pad:                                              # pad to fixed CHUNK so jit traces once (per nseats)
            sl = sl + [sl[-1]] * pad; al = al + [z_act] * pad; s0 = s0 + [0] * pad
        tg = np.asarray(_fly_to_target_batch(
            _stack_states(sl), jnp.asarray(np.stack(al, 0)),
            jnp.asarray(np.asarray(s0, np.int32)), int(nseats), RECOVER_H))
        out.append(tg[:B])
    return np.concatenate(out, 0)


def batch_feats(state_list, seat_list):
    """planet_features for every (state, seat); grouped by seat (scalar player) -> <=nseats GPU calls."""
    feats = [None] * len(state_list)
    by_seat = {}
    for i, se in enumerate(seat_list):
        by_seat.setdefault(se, []).append(i)
    for se, idxs in by_seat.items():
        fb = np.asarray(_FEAT(_stack_states([state_list[i] for i in idxs]), se), np.float16)
        for j, i in enumerate(idxs):
            feats[i] = fb[j]
    return feats


ANGLE_BINS = 24                                              # option-2 (predict angle): [-pi,pi) -> 24 bins (15 deg)


def _angle_to_bin(a):
    return int(((float(a) + np.pi) % (2 * np.pi)) / (2 * np.pi) * ANGLE_BINS) % ANGLE_BINS


def _rows_for_winner(action, id_to_slot, owner_slot, winner, state, nseats):
    """Winner's launches -> (src_slot, tgt_slot, ships, angle). Target is RECOVERED by flying the
    launch through the REAL env (see _recover_target); angle is also returned raw for option-2 labels."""
    src, tgt, send, ang = [], [], [], []
    for mv in (action or []):
        if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
            continue
        s_id, angle, n = int(mv[0]), float(mv[1]), int(mv[2])
        if n <= 0 or s_id not in id_to_slot:
            continue
        s_slot = id_to_slot[s_id]
        if owner_slot[s_slot] != winner:                     # trust replay: source must be winner's
            continue
        t_slot = _recover_target(state, s_slot, angle, n, winner, nseats)
        if t_slot < 0 or t_slot == s_slot:                   # missed into space / self
            continue
        src.append(s_slot); tgt.append(t_slot); send.append(n); ang.append(angle)
    return src, tgt, send, ang


# dataset fields. sl_* = per-sample raw ingredients for the OUTCOME WEIGHT, so the weighting design
# (strength formula, compare-vs, loss_w, mult/add, state-vs-game) is a TRAIN-TIME function -> sweepable
# without regen. sl_ships/sl_prod = per-seat (padded to 4) ships & production at THIS state; sl_me = the
# learning seat; sl_ladder = player's ladder score; sl_step/sl_eps = step & episode length (for H);
# sl_win = this seat's FINAL reward (for state-vs-game blends).
_OUT_KEYS = ("feats", "alive", "canfire", "focus_valid", "fire", "tgt", "frac", "focus", "has_focus",
             "angle_bin", "weight", "nseats",
             "sl_ships", "sl_prod", "sl_me", "sl_ladder", "sl_step", "sl_eps", "sl_win")


def process_replay(path, stride, idle_frac, obs_shift, rng, winners_only, game_weight=None,
                   strong=None, min_score=0.0, outcome_dim=False, outcome_loss_w=0.4):
    d = json.load(open(path))
    steps = d["steps"]; info = d.get("info", {})
    names = info.get("TeamNames") or [a.get("Name") for a in info.get("Agents", [])]
    episode_steps = int(d.get("configuration", {}).get("episodeSteps", 500))   # H for outcome strength
    nseats = len(steps[0])
    _fin = steps[-1]
    seat_final_rew = np.array([float(_fin[s].get("reward")) if s < len(_fin) and _fin[s].get("reward") is not None
                               else 0.0 for s in range(nseats)], np.float32)   # per-seat final reward (state-vs-game)
    _env.NUM_PLAYERS = nseats                                 # cosmetic (rewards shape); features are count-agnostic
    win_seat, margin = _winner_seat(steps, nseats)
    # outcome weight: the game's manifest rating (trust strong games) if provided,
    # else the reward margin. Trainer normalizes weights to mean 1.
    wval = float(game_weight) if game_weight is not None else float(margin)
    if strong is not None:                                    # selective imitation: learn STRONG players' moves in EVERY
        # game they appear in (WIN OR LOSE) — "learn what good play looks like", not "imitate this replay".
        learn_seats = [s for s in range(nseats) if s < len(names) and names[s] in strong and strong[names[s]] >= min_score]
        seat_ladder = {s: float(strong[names[s]]) for s in learn_seats}   # per-seat ladder strength (outcome factor is STATE-level, applied per-cell below)
    else:
        learn_seats = [win_seat] if winners_only else list(range(nseats))
        seat_ladder = None
    out = {k: [] for k in _OUT_KEYS}
    T = len(steps)
    # ---- PASS 1: collect cells (board arrays + RAW launches) + queue all launches for batch recovery ----
    cells = []                                               # (seat, owner_slot, ships_slot, alive_slot, P, raw)
    cell_states = []                                         # jax state per cell (for batched feats)
    rec_states = []; rec_acts = []; rec_slot0 = []; rec_ref = []   # one entry per launch
    for t in range(0, T, stride):
        board = _board_obs(steps[t])
        if board is None:
            continue
        id_to_slot = {int(p[0]): i for i, p in enumerate(board["planets"])}
        state = obs_to_state_np(board)                       # numpy state (no per-field device xfer)
        owner_slot = np.asarray(state.owner)[0]
        ships_slot = np.asarray(state.ships)[0]
        alive_slot = np.asarray(state.alive)[0]
        prod_slot = np.asarray(state.production)[0]
        step_t = int(np.asarray(state.step_count)[0])
        pid_row = np.asarray(state.planet_id)[0]
        slot0 = int(np.asarray(state.next_fleet)[0])
        P = owner_slot.shape[0]
        at = min(t + obs_shift, T - 1)                       # action aligned to obs (shift if parity demands)
        for seat in learn_seats:
            if seat >= len(steps[at]):
                continue
            owned = (owner_slot == seat) & alive_slot
            if not owned.any():                              # seat eliminated -> nothing to teach
                continue
            raw = []                                         # valid launches (s_slot, angle, ships) PRE-recovery
            for mv in (steps[at][seat].get("action") or []):
                if not (isinstance(mv, (list, tuple)) and len(mv) == 3):
                    continue
                s_id, angle, n = int(mv[0]), float(mv[1]), int(mv[2])
                if n <= 0 or s_id not in id_to_slot:
                    continue
                s_slot = id_to_slot[s_id]
                if owner_slot[s_slot] != seat:               # trust replay: source must be the seat's
                    continue
                raw.append((s_slot, angle, n))
            ci = len(cells)
            cells.append((seat, owner_slot, ships_slot, alive_slot, P, raw, prod_slot, step_t))
            cell_states.append(state)
            for li, (s_slot, angle, n) in enumerate(raw):
                act = np.zeros((nseats, 1, 3), np.float32)
                act[seat, 0, 0] = float(pid_row[s_slot]); act[seat, 0, 1] = angle; act[seat, 0, 2] = float(n)
                rec_states.append(state); rec_acts.append(act); rec_slot0.append(slot0); rec_ref.append((ci, li))
    if not cells:
        return out, 0, nseats, (str(names[win_seat]) if names and win_seat < len(names) else "?"), margin

    # ---- BATCH recover all launches -> target slot per launch ----
    rec_tgt = batch_recover(rec_states, rec_acts, rec_slot0, nseats)
    cell_tgts = [[None] * len(c[5]) for c in cells]
    for (ci, li), tg in zip(rec_ref, rec_tgt):
        cell_tgts[ci][li] = int(tg)

    # ---- PASS 2a: per cell, resolve rows + idle-frac keep decision (rng consumed in cell order = reproducible) ----
    keep = []                                                # (ci, src, tgt, send, angs, is_decision)
    for ci, (seat, owner_slot, ships_slot, alive_slot, P, raw, _prod, _step) in enumerate(cells):
        src = []; tgt = []; send = []; angs = []
        for (s_slot, angle, n), tg in zip(raw, cell_tgts[ci]):
            if tg is None or tg < 0 or tg == s_slot:          # missed into space / self
                continue
            src.append(s_slot); tgt.append(tg); send.append(n); angs.append(angle)
        is_decision = len(src) > 0
        if not is_decision and rng.random() > idle_frac:      # rebalance toward decisions
            continue
        keep.append((ci, src, tgt, send, angs, is_decision))

    # ---- BATCH feats for KEPT cells only (grouped by seat) ----
    feats_list = batch_feats([cell_states[k[0]] for k in keep], [cells[k[0]][0] for k in keep])

    # ---- PASS 2b: build labels (identical recipe to the per-launch version) ----
    for kk, (ci, src, tgt, send, angs, is_decision) in enumerate(keep):
        seat, owner_slot, ships_slot, alive_slot, P, _raw, prod_slot, step_t = cells[ci]
        owned = (owner_slot == seat) & alive_slot
        M = max(len(src), 1)
        srcn = np.zeros((1, M), np.int32); tgtn = np.zeros((1, M), np.int32)
        sendn = np.zeros((1, M), np.float32); valn = np.zeros((1, M), bool)
        if is_decision:
            srcn[0, :len(src)] = src; tgtn[0, :len(tgt)] = tgt
            sendn[0, :len(send)] = send; valn[0, :len(send)] = True
        fire, tgt_lab, frac_lab, _focus, _has = labels_from_rows(
            1, P, ships_slot[None].astype(np.int32), srcn, tgtn, sendn, valn)
        fvalid = alive_slot & ~owned                          # focus over ENEMY/NEUTRAL targets only
        inc = np.zeros(P, np.float32)
        if is_decision:
            np.add.at(inc, tgt, send)
        inc[~fvalid] = 0.0
        focus_slot = int(inc.argmax()); has_f = bool(inc.max() >= 1.0)
        ang_bin = np.zeros(P, np.int16)                       # option-2 label: largest launch's angle bin per source
        if is_decision:
            best = {}
            for s_slot, sn, a in zip(src, send, angs):
                if s_slot not in best or sn > best[s_slot][0]:
                    best[s_slot] = (sn, a)
            for s_slot, (sn, a) in best.items():
                ang_bin[s_slot] = _angle_to_bin(a)
        out["feats"].append(feats_list[kk])
        out["alive"].append(alive_slot)
        out["canfire"].append(owned & (ships_slot >= 1))
        out["focus_valid"].append(fvalid)
        out["fire"].append(fire[0]); out["tgt"].append(tgt_lab[0]); out["frac"].append(frac_lab[0])
        out["focus"].append(np.int16(focus_slot)); out["has_focus"].append(has_f)
        out["angle_bin"].append(ang_bin)
        ladder_v = float(seat_ladder[seat]) if seat_ladder is not None else float(wval)
        out["weight"].append(np.float32(ladder_v))           # base weight = ladder; OUTCOME applied at TRAIN time from sl_*
        out["nseats"].append(np.int8(nseats))                # 2 or 4: split/weight 2p vs 4p downstream
        # ---- per-seat strength ingredients at THIS state (train-time outcome weight, sweepable) ----
        ssh = np.zeros(4, np.float32); spr = np.zeros(4, np.float32)
        for s in range(nseats):
            mm = (owner_slot == s) & alive_slot
            ssh[s] = ships_slot[mm].sum(); spr[s] = prod_slot[mm].sum()
        out["sl_ships"].append(ssh); out["sl_prod"].append(spr)
        out["sl_me"].append(np.int8(seat)); out["sl_ladder"].append(np.float32(ladder_v))
        out["sl_step"].append(np.int16(step_t)); out["sl_eps"].append(np.int16(episode_steps))
        out["sl_win"].append(np.float32(seat_final_rew[seat] if seat < len(seat_final_rew) else 0.0))
    n = len(out["feats"])
    return out, n, nseats, (str(names[win_seat]) if names and win_seat < len(names) else "?"), margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays-dir", required=True)
    ap.add_argument("--out", default="bc_replay_demos.npz")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--idle-frac", type=float, default=0.2)   # fraction of pure-idle steps kept
    ap.add_argument("--obs-shift", type=int, default=1)       # action_(t+1) decided on board_t: audit_recover shows shift=1 -> 100% valid launches vs 2% at shift=0 (env.steps off-by-one)
    ap.add_argument("--winners-only", action="store_true", default=True)
    ap.add_argument("--all-seats", dest="winners_only", action="store_false")
    ap.add_argument("--limit", type=int, default=0)           # cap #replays (quick test on first download)
    ap.add_argument("--recursive", action="store_true")       # glob replays-dir/**/ *.json (date-foldered dataset)
    ap.add_argument("--shard", default=None)                  # "k/n": process takes paths[k::n] (parallel CPU shards)
    ap.add_argument("--days", type=int, default=0)            # keep only N most-recent date-dirs (recursive; meta-recency)
    ap.add_argument("--day-slice", default=None)              # "a:b" slice on sorted date-dirs (disjoint incremental batches: -5: , -10:-5)
    ap.add_argument("--chunk-samples", type=int, default=0)   # >0: flush a chunk npz every M samples (bounded RAM, no merge); trainer streams <out>_c*.npz
    ap.add_argument("--leaderboard", default=None,            # public LB CSV (Rank,TeamName,Score): learn STRONG
                    help="players' moves in EVERY game (win or lose), per-seat weight = ladder Score")
    ap.add_argument("--min-score", type=float, default=0.0)  # only learn seats whose player's LB Score >= this
    ap.add_argument("--outcome-dim", action="store_true")    # weight *= per-seat outcome (lean on games that went well)
    ap.add_argument("--outcome-loss-weight", type=float, default=0.4)  # worst-seat floor for the outcome factor
    ap.add_argument("--manifest", default=None,               # CSV: episode_id,...,avg_score,min_score,...,agent_count
                    help="manifest.csv to filter/weight by rating (learn from strong games)")
    ap.add_argument("--min-score-2p", type=float, default=0)  # keep a 2p game iff its min_score >= this
    ap.add_argument("--min-score-4p", type=float, default=0)  # keep a 4p game iff its min_score >= this
    ap.add_argument("--score-weight", choices=("min", "avg", "none"), default="min",
                    help="per-sample outcome weight source from the manifest (default min_score)")
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()

    patt = os.path.join(args.replays_dir, "**", "*.json") if args.recursive else os.path.join(args.replays_dir, "*.json")
    paths = sorted(glob.glob(patt, recursive=args.recursive))
    if args.days or args.day_slice:                           # select date-dirs (meta-recency / incremental batches)
        import re as _re
        dk = lambda p: (_re.search(r"/(\d{8})/", p) or [None, ""])[1] if _re.search(r"/(\d{8})/", p) else ""
        alldays = sorted({dk(p) for p in paths if dk(p)})
        if args.day_slice:                                    # "a:b" python slice on sorted days (e.g. -5: , -10:-5)
            a, b = (x.strip() for x in args.day_slice.split(":"))
            sel = alldays[(int(a) if a else None):(int(b) if b else None)]
        else:
            sel = alldays[-args.days:]
        keep = set(sel)
        paths = [p for p in paths if dk(p) in keep]
        print(f"days {sorted(keep)} -> {len(paths)} replays")
    if args.shard:                                            # "k/n": this process takes paths[k::n]
        k, nsh = (int(z) for z in args.shard.split("/"))
        paths = paths[k::nsh]
        print(f"shard {k}/{nsh}: {len(paths)} replays")
    # manifest filter: keep games whose min_score clears the per-player-count threshold; the
    # game's rating doubles as the outcome weight (winner's rating >= min_score = "both strong").
    man = {}
    if args.manifest:
        import csv as _csv
        for r in _csv.DictReader(open(args.manifest)):
            man[str(r["episode_id"])] = (float(r["avg_score"]), float(r["min_score"]), int(r["agent_count"]))
        kept = []
        for p in paths:
            eid = os.path.basename(p)[:-5]
            if eid not in man:
                continue
            avg, mn, ac = man[eid]
            thr = args.min_score_2p if ac == 2 else args.min_score_4p
            if mn >= thr:
                kept.append(p)
        n_before = len(paths); paths = kept
        print(f"manifest filter: {n_before} -> {len(paths)} games "
              f"(2p min>={args.min_score_2p}, 4p min>={args.min_score_4p})")
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print(f"no replays in {args.replays_dir}/ (*.json)"); return
    rng = np.random.default_rng(args.seed)
    strong = None
    if args.leaderboard:                                     # TeamName -> ladder Score; learn strong seats win-or-lose
        import csv as _csv2
        strong = {r["TeamName"]: float(r["Score"]) for r in _csv2.DictReader(open(args.leaderboard, encoding="utf-8-sig"))}
        nkeep = sum(1 for v in strong.values() if v >= args.min_score)
        print(f"leaderboard: {len(strong)} teams, {nkeep} with Score>={args.min_score} (strong-player mode, win-or-lose)")
    agg = {k: [] for k in _OUT_KEYS}
    base = args.out[:-4] if args.out.endswith(".npz") else args.out
    chunk = args.chunk_samples; ci = 0; n_total = 0
    def flush():                                              # write agg -> chunk npz, free RAM (bounded gen memory)
        nonlocal ci
        if not agg["feats"]:
            return
        arrs = {k: np.stack(v, 0) for k, v in agg.items()}
        fn = f"{base}_c{ci:04d}.npz"
        np.savez_compressed(fn, **arrs)
        print(f"  flushed {fn}: {arrs['feats'].shape[0]} samples [{time.time()-t0:.0f}s]", flush=True)
        for k in agg:
            agg[k].clear()
        ci += 1
    t0 = time.time(); ng2 = ng4 = skipped = 0
    for i, p in enumerate(paths):
        gw = None
        if man and args.score_weight != "none":
            rec = man.get(os.path.basename(p)[:-5])
            if rec:
                gw = rec[0] if args.score_weight == "avg" else rec[1]
        try:
            out, n, nseats, wname, margin = process_replay(
                p, args.stride, args.idle_frac, args.obs_shift, rng, args.winners_only, game_weight=gw,
                strong=strong, min_score=args.min_score,
                outcome_dim=args.outcome_dim, outcome_loss_w=args.outcome_loss_weight)
        except Exception as e:
            skipped += 1; print(f"  SKIP {os.path.basename(p)}: {type(e).__name__}: {e}", flush=True); continue
        if n == 0:
            skipped += 1; continue
        for k in agg:
            agg[k] += out[k]
        n_total += n; ng2 += (nseats == 2); ng4 += (nseats >= 4)
        if chunk and len(agg["feats"]) >= chunk:              # bounded RAM: flush + free
            flush()
        if i % 25 == 0:
            print(f"  [{i+1}/{len(paths)}] {os.path.basename(p)} {nseats}p win={wname} "
                  f"margin={margin:.1f} +{n} samples [{time.time()-t0:.0f}s]", flush=True)
    if chunk:                                                 # chunked mode: flush remainder, no single file
        flush()
        print(f"\nchunked {ci} files {base}_c*.npz: {n_total} samples from {len(paths)-skipped} replays "
              f"(2p={ng2}, 4p={ng4}, skipped={skipped}) [{time.time()-t0:.0f}s]")
        return
    if not agg["feats"]:
        print("no usable samples extracted"); return
    arrs = {k: np.stack(v, 0) for k, v in agg.items()}
    np.savez_compressed(args.out, **arrs)
    N = arrs["feats"].shape[0]
    fire_rate = arrs["fire"][arrs["canfire"]].mean() if arrs["canfire"].any() else 0.0
    print(f"\nsaved {args.out}: {N} samples from {len(paths)-skipped} replays "
          f"(2p={ng2}, 4p={ng4}, skipped={skipped})")
    print(f"  fire-rate(on canfire)={fire_rate:.3f}  decision-states={int(arrs['has_focus'].sum())} "
          f"({100*arrs['has_focus'].mean():.1f}% of samples)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
