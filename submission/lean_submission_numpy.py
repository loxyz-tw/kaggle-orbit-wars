"""Kaggle Orbit Wars submission — pure-NumPy lean entity-transformer policy.

Single-file, no JAX (JAX init exceeds the 2s agentTimeout). Weights for the
lean self-play checkpoint are embedded as base64(zlib(pickle)) below.

Forward pass parity-verified against jax_lean_selfplay.LeanEntityPolicy
(max abs logit diff ~1e-6). Emits kaggle moves [src_planet_id, angle, ships].
"""
from __future__ import annotations

import base64
import math
import os
import pickle
import zlib

import numpy as np

# Aim/sun-guard knobs (defaults = SHIPPED behavior, resolved by A/B 2026-06-03). Env vars
# override for A/B only. Do NOT switch back to the
# old conservative `ray` guard — it cost ~92 ladder pts (1173->1081) by over-blocking launches.
# aim: lead (intercept prediction; correct) | direct (env-matched arctan2, CATASTROPHIC 1.2%).
# guard: fwdsun (env-exact, suppress ONLY real sun-deaths; plugs segment's overshoot leak,
#   -94% sun-deaths / strength-neutral, 2026-06-22) | fwdsim (also drops never-capture launches)
#   | segmentmargin (source->intercept + overshoot) | segment (leaky, shipping) | ray (over-blocks).
_OW_AIM = os.environ.get("OW_AIM", "lead")
_OW_GUARD = os.environ.get("OW_SUNGUARD", "segment")  # LADDER 2026-06-03 (settled-ish @9h): leaky segment 1169 > segmentmargin 1116 > ray 1082; less guarding = better
_OW_MARGIN_TURNS = float(os.environ.get("OW_MARGIN_TURNS", "4"))
# Feasibility gate (serve-time stop-loss): suppress a launch at a NEUTRAL whose static
# garrison the SOURCE planet can't beat even sending everything (ships < garrison+1) — the
# fleet would bounce and waste the ships; keep them home to accumulate. Default off. The
# observed ladder give-away launches were all this case (single planet too small for the target neutral).
# DEFAULT ON (validated 2026-06-03): 2p +12.5pp / +13.7pp vs two public heuristic baselines (both sig); 4p
# 26.7->30.8% (positive). Set OW_CAPTURE_GATE=0 to disable for A/B.
_OW_CAPTURE_GATE = os.environ.get("OW_CAPTURE_GATE", "1") == "1"

# ---------------------------------------------------------------------------
# embedded weights. _PARAMS_B64 = the 2p (or single) model, filled by
# submission/_build_lean_submission.py. _PARAMS_4P_B64 = optional 4-player model,
# filled by submission/_build_dispatch_submission.py; empty => single-model (no
# dispatch). The agent picks the 4p model in 4-player games, else the 2p model.
_PARAMS_B64 = ""
_PARAMS_4P_B64 = ""
# ---------------------------------------------------------------------------

BOARD_CENTER = 50.0
BOARD_SCALE = 50.0
ROTATION_LIMIT = 50.0
SUN_RADIUS = 10.0
SHIP_LOG_DENOM = math.log(1000.0)
THREAT_ETA_TAU = 20.0
THREAT_AIM_MARGIN = 2.0
CAPTURE_SPEED_REF = 3.0
GRAVITY_R0_SQ = 100.0  # gravity-density softening radius^2 (must match jax_lean_selfplay.GRAVITY_R0_SQ)
ANG_VEL_SCALE = 20.0   # --orbit-features ang_vel scale (must match jax_lean_selfplay.ANG_VEL_SCALE)
NUM_LAYERS = 2
NUM_HEADS = 4
HIDDEN = 128
HEAD_DIM = HIDDEN // NUM_HEADS
MAX_PLANETS = 48


def _load():
    if not _PARAMS_B64:  # clean weightless template (e.g. imported for feature-parity tests)
        return None, None, None, 1, False, False, False, False, False
    def _dec(b64):
        return pickle.loads(zlib.decompress(base64.b64decode(b64)))
    p2 = _dec(_PARAMS_B64)
    params_2p = p2["params"]
    params_4p = _dec(_PARAMS_4P_B64)["params"] if _PARAMS_4P_B64 else params_2p
    return (params_2p, params_4p, np.asarray(p2["frac_bins"], dtype=np.float32),
            int(p2.get("min_launch", 1)), bool(p2.get("rich_features", False)),
            bool(p2.get("orbit_features", False)), bool(p2.get("autoregressive", False)),
            bool(p2.get("is_4p_feature", False)), bool(p2.get("rank_feature", False)))


_PARAMS_2P, _PARAMS_4P, _FRAC_BINS, _MIN_LAUNCH, _RICH, _ORBIT, _AR, _IS4P, _RANK = _load()
_PARAMS = _PARAMS_2P  # default model for direct _forward callers (parity tests)
_NUM_PLAYERS = None   # detected & cached at step 0 each game


# ----- numpy forward (mirrors flax LeanEntityPolicy) -----------------------
def _dense(x, p):
    return x @ p["kernel"] + p["bias"]


def _layernorm(x, p, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * p["scale"] + p["bias"]


def _gelu(x):  # jax.nn.gelu default approximate=True
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _self_attention(h, p, alive):
    q = np.einsum("ph,hnd->pnd", h, p["query"]["kernel"]) + p["query"]["bias"]
    k = np.einsum("ph,hnd->pnd", h, p["key"]["kernel"]) + p["key"]["bias"]
    v = np.einsum("ph,hnd->pnd", h, p["value"]["kernel"]) + p["value"]["bias"]
    q = q / math.sqrt(q.shape[-1])   # head_dim from weights (arch-agnostic; was hardcoded HEAD_DIM)
    scores = np.einsum("qnd,knd->nqk", q, k)
    scores = np.where(alive[None, None, :], scores, np.finfo(np.float32).min)
    w = _softmax(scores, axis=-1)
    ctx = np.einsum("nqk,knd->qnd", w, v)
    return np.einsum("qnd,ndo->qo", ctx, p["out"]["kernel"]) + p["out"]["bias"]


def _fleet_speed(ships):
    fs = np.maximum(ships.astype(np.float32), 1.0)
    return np.maximum(np.minimum(1.0 + 5.0 * (np.log(fs) / SHIP_LOG_DENOM) ** 1.5, 6.0), 1.0)


def _threat_features(owner, alive, px, py, radius, ships, is_comet, av,
                     f_owner, f_x, f_y, f_angle, f_ships, f_alive, player, iters=2):
    P = owner.shape[0]
    if f_x.shape[0] == 0:
        return np.zeros((P, 4), dtype=np.float32)
    spe = _fleet_speed(f_ships)[:, None]
    fxe = f_x[:, None]; fye = f_y[:, None]
    dxe = np.cos(f_angle)[:, None]; dye = np.sin(f_angle)[:, None]
    orb_r = np.sqrt((px - BOARD_CENTER) ** 2 + (py - BOARD_CENTER) ** 2)
    cur_ang = np.arctan2(py - BOARD_CENTER, px - BOARD_CENTER)
    rotating = alive.astype(bool) & (~is_comet.astype(bool)) & ((orb_r + radius) < ROTATION_LIMIT)
    tpx = px[None, :].astype(np.float32); tpy = py[None, :].astype(np.float32)
    eta = np.zeros((f_x.shape[0], P), dtype=np.float32)
    for _ in range(iters):
        rx = tpx - fxe; ry = tpy - fye
        proj = rx * dxe + ry * dye
        eta = np.maximum(proj, 0.0) / spe
        new_ang = cur_ang[None, :] + np.where(rotating[None, :], av * eta, 0.0)
        tpx = BOARD_CENTER + orb_r[None, :] * np.cos(new_ang)
        tpy = BOARD_CENTER + orb_r[None, :] * np.sin(new_ang)
    rx = tpx - fxe; ry = tpy - fye
    proj = rx * dxe + ry * dye
    perp = np.abs(rx * dye - ry * dxe)
    eta = np.maximum(proj, 0.0) / spe
    aimed = (proj > 0.0) & (perp <= radius[None, :] + THREAT_AIM_MARGIN) \
        & f_alive[:, None].astype(bool) & alive[None, :].astype(bool)
    proj_for_min = np.where(aimed, proj, np.inf)
    has_aim = aimed.any(axis=1)
    tgt = np.argmin(proj_for_min, axis=1)
    onehot = np.zeros((f_x.shape[0], P), dtype=np.float32)
    onehot[np.arange(f_x.shape[0]), tgt] = has_aim.astype(np.float32)
    hostile = (((f_owner != player) & (f_owner >= 0)))[:, None].astype(np.float32)
    friendly = (f_owner == player)[:, None].astype(np.float32)
    ships_f = np.maximum(f_ships.astype(np.float32), 0.0)[:, None]
    inc_host = (onehot * hostile * ships_f).sum(axis=0)
    inc_frnd = (onehot * friendly * ships_f).sum(axis=0)
    host_cell = (onehot > 0.0) & (((f_owner != player) & (f_owner >= 0)))[:, None]
    soonest = np.where(host_cell, eta, np.inf).min(axis=0)
    imminence = np.where(np.isfinite(soonest), np.exp(-soonest / THREAT_ETA_TAU), 0.0)
    sp = np.maximum(ships.astype(np.float32), 0.0)
    return np.stack([np.log1p(inc_host) / SHIP_LOG_DENOM, np.log1p(inc_frnd) / SHIP_LOG_DENOM,
                     imminence, np.clip((sp + inc_frnd - inc_host) / 100.0, -3.0, 3.0)], axis=-1).astype(np.float32)


def _nearest_dist(owner, alive, x, y, mask_kind, player):
    """mask_kind in {'enemy','neutral','own'}; returns (P,) normalized /100 (2.0 sentinel)."""
    P = owner.shape[0]
    alive = alive.astype(bool)
    rx = x[:, None] - x[None, :]; ry = y[:, None] - y[None, :]
    pdist = np.sqrt(rx * rx + ry * ry)
    if mask_kind == 'enemy':
        m = ((((owner != player) & (owner != -1))) & alive)[None, :]
    elif mask_kind == 'neutral':
        m = ((owner == -1) & alive)[None, :]
    else:
        m = ((owner == player) & alive)[None, :] & ~np.eye(P, dtype=bool)
    nd = np.min(np.where(m, pdist, np.inf), axis=1)
    return np.where(np.isfinite(nd), nd / 100.0, 2.0).astype(np.float32)


def _time_distance_features(owner, alive, x, y, step, player):
    P = owner.shape[0]
    step_feat = np.full((P,), float(step) / 500.0, dtype=np.float32)
    return np.stack([step_feat, _nearest_dist(owner, alive, x, y, 'enemy', player),
                     _nearest_dist(owner, alive, x, y, 'neutral', player),
                     _nearest_dist(owner, alive, x, y, 'own', player)], axis=-1).astype(np.float32)


def _capture_cost(owner, alive, x, y, ships, production, player):
    near_ow = _nearest_dist(owner, alive, x, y, 'own', player)
    eta_cap = (near_ow * 100.0) / CAPTURE_SPEED_REF
    is_enemy = (((owner != player) & (owner != -1))) & alive.astype(bool)
    defenders = np.maximum(ships.astype(np.float32), 0.0) \
        + np.where(is_enemy, production.astype(np.float32) * eta_cap, 0.0)
    return np.clip(defenders / 100.0, 0.0, 3.0).astype(np.float32)


def _gravity_features(owner, alive, x, y, ships, player):
    """Local enemy/own ship-mass density (gravity ~ Σ ships/(dist^2+R0^2)) -> (P,2).
    Numpy parity of jax_lean_selfplay.planet_features GRAVITY block (raw board-unit
    distance, softening GRAVITY_R0_SQ, log1p). en_j: any opponent (no self-exclude,
    matches near_en); ow_j: own planets excluding self (matches near_own)."""
    P = owner.shape[0]
    al = alive.astype(bool)
    rx = x[:, None] - x[None, :]
    ry = y[:, None] - y[None, :]
    d2 = (rx * rx + ry * ry) + GRAVITY_R0_SQ                                  # (P,P)
    sj = np.maximum(ships.astype(np.float32), 0.0)                            # (P,)
    en_j = (((owner != player) & (owner != -1)) & al)[None, :]               # (1,P)
    ow_j = (((owner == player) & al)[None, :]) & (~np.eye(P, dtype=bool))    # (P,P)
    en_mass = np.log1p((np.where(en_j, sj[None, :], 0.0) / d2).sum(axis=1))  # (P,)
    ow_mass = np.log1p((np.where(ow_j, sj[None, :], 0.0) / d2).sum(axis=1))  # (P,)
    return np.stack([en_mass, ow_mass], axis=-1).astype(np.float32)          # (P,2)


def _orbit_features(owner, alive, x, y, radius, prod, is_comet, av, player):
    """Numpy parity of jax_lean_selfplay.planet_features ORBIT block -> (P,2):
    ang_vel = rot * angular_velocity(scalar) * ANG_VEL_SCALE ;
    prod_density = log1p(Sum_{j alive, j!=self} prod_j / (dist^2 + GRAVITY_R0_SQ))."""
    P = owner.shape[0]
    al = alive.astype(bool)
    orb_r = np.sqrt((x - BOARD_CENTER) ** 2 + (y - BOARD_CENTER) ** 2)
    rot = (al & (~is_comet.astype(bool)) & ((orb_r + radius) < ROTATION_LIMIT)).astype(np.float32)  # (P,)
    ang_vel = rot * float(av) * ANG_VEL_SCALE                                 # (P,)
    rx = x[:, None] - x[None, :]
    ry = y[:, None] - y[None, :]
    d2 = (rx * rx + ry * ry) + GRAVITY_R0_SQ                                  # (P,P)
    pr = prod.astype(np.float32)
    alive_j = al[None, :] & (~np.eye(P, dtype=bool))                         # (P,P) exclude self
    prod_density = np.log1p((np.where(alive_j, pr[None, :], 0.0) / d2).sum(axis=1))  # (P,)
    return np.stack([ang_vel, prod_density], axis=-1).astype(np.float32)      # (P,2)


def _roi_features(owner, alive, x, y, ships, production, player):
    """Capture payoff-time (defenders/prod, log1p) -> (P,). Numpy parity of the JAX
    ROI block; `defenders` mirrors _capture_cost (raw, pre-clip)."""
    near_ow = _nearest_dist(owner, alive, x, y, 'own', player)
    eta_cap = (near_ow * 100.0) / CAPTURE_SPEED_REF
    is_enemy = (((owner != player) & (owner != -1))) & alive.astype(bool)
    defenders = np.maximum(ships.astype(np.float32), 0.0) \
        + np.where(is_enemy, production.astype(np.float32) * eta_cap, 0.0)
    return np.log1p(defenders / np.maximum(production.astype(np.float32), 1.0)).astype(np.float32)  # (P,)


def _global_rotating(owner, alive, x, y, radius, ships, production, is_comet, f_owner, f_ships, f_alive, player):
    P = owner.shape[0]
    alive = alive.astype(bool)
    sp = np.maximum(ships.astype(np.float32), 0.0)
    ff = np.maximum(f_ships.astype(np.float32), 0.0); fal = f_alive.astype(bool)
    my = (sp * ((owner == player) & alive)).sum() + (ff * ((f_owner == player) & fal)).sum()
    en = (sp * ((((owner != player) & (owner != -1))) & alive)).sum() + (ff * ((((f_owner != player) & (f_owner >= 0))) & fal)).sum()
    tot = my + en
    adv = float((my - en) / max(tot, 1e-6)) if tot > 0.0 else 0.0
    # D1: global production advantage (own vs enemy planet production)
    pp = production.astype(np.float32)
    myp = (pp * ((owner == player) & alive)).sum()
    enp = (pp * ((((owner != player) & (owner != -1))) & alive)).sum()
    ptot = myp + enp
    padv = float((myp - enp) / max(ptot, 1e-6)) if ptot > 0.0 else 0.0
    orb_r = np.sqrt((x - BOARD_CENTER) ** 2 + (y - BOARD_CENTER) ** 2)
    rot = (alive & (~is_comet.astype(bool)) & ((orb_r + radius) < ROTATION_LIMIT)).astype(np.float32)
    return np.stack([np.full((P,), adv, np.float32), np.full((P,), padv, np.float32), rot],
                    axis=-1).astype(np.float32)


def _planet_features(owner, alive, x, y, ships, production, is_comet, player):
    alive = alive.astype(bool)
    is_mine = ((owner == player) & alive).astype(np.float32)
    is_enemy = ((((owner != player) & (owner != -1))) & alive).astype(np.float32)
    is_neutral = ((owner == -1) & alive).astype(np.float32)
    alive_f = alive.astype(np.float32)
    sh = np.maximum(ships.astype(np.float32), 0.0)
    ships_lin = np.clip(sh / 100.0, 0.0, 3.0)
    ships_log = np.log1p(sh) / SHIP_LOG_DENOM
    prod = production.astype(np.float32) / 5.0
    xn = (x - BOARD_CENTER) / BOARD_SCALE
    yn = (y - BOARD_CENTER) / BOARD_SCALE
    dist_sun = np.sqrt(xn * xn + yn * yn)
    comet = is_comet.astype(np.float32)
    return np.stack(
        [is_mine, is_enemy, is_neutral, alive_f, ships_lin, ships_log, prod, xn, yn, dist_sun, comet],
        axis=-1,
    ).astype(np.float32)


def _forward(feats, alive, focus_valid, params=None):
    P = params if params is not None else _PARAMS
    alive = alive.astype(bool)
    n_layers = sum(1 for kk in P if kk.startswith("ln_attn_"))   # arch-agnostic (was hardcoded NUM_LAYERS)
    x = _dense(feats, P["embed"])
    for i in range(n_layers):
        h = _layernorm(x, P["ln_attn_%d" % i])
        x = x + _self_attention(h, P["attn_%d" % i], alive)
        h = _layernorm(x, P["ln_mlp_%d" % i])
        h = _dense(h, P["mlp_in_%d" % i])
        h = _gelu(h)
        h = _dense(h, P["mlp_out_%d" % i])
        x = x + h
    x = _layernorm(x, P["ln_out"])
    # C1 commander (only if this checkpoint carries the focus heads; pre-C1 skips it).
    has_c1 = "focus_head" in P
    x_cond = x
    focus_idx = -1
    if has_c1:
        focus_logits = _dense(x, P["focus_head"])[..., 0]
        fv = focus_valid.astype(bool)
        fmask = np.where(fv, focus_logits, np.finfo(focus_logits.dtype).min) if fv.any() \
            else np.zeros_like(focus_logits)
        focus_idx = int(fmask.argmax())
        x_cond = x + _dense(x[focus_idx], P["focus_proj"])
    fire_logit = _dense(x_cond, P["fire_head"])[..., 0]
    frac_logits = _dense(x_cond, P["frac_head"])
    q = _dense(x_cond, P["tgt_q"])
    k = _dense(x_cond, P["tgt_k"])
    target_logits = (q @ k.T) / math.sqrt(P["embed"]["kernel"].shape[1])   # hidden from weights (was HIDDEN)
    if has_c1:
        target_logits[:, focus_idx] += float(np.asarray(P["focus_bias"]))
    return fire_logit, target_logits, frac_logits


def _autoregressive_targets(target_logits, fire, frac_val, ships, owner, alive, player, ch):
    """C2 greedy autoregressive target choice (numpy mirror of jax autoregressive_target).
    Owned planets in ships-DESC order (stable, matches jnp.argsort); each picks argmax target under
    target_logits + commit_head bias from running committed mass, then adds its outgoing ships to
    committed so later planets avoid already-covered targets. commit_head absent -> not called."""
    P = target_logits.shape[0]
    shp = ships.astype(np.float32)
    defenders = shp
    can_fire = (owner == player) & alive & (ships >= _MIN_LAUNCH)
    fire_eff = fire & can_fire
    order = np.argsort(-ships, kind="stable")            # stable to match jnp.argsort tie-order
    ker = np.asarray(ch["kernel"], np.float32)           # (2,1)
    b0 = np.asarray(ch["bias"], np.float32)              # (1,)
    committed = np.zeros(P, np.float32)
    tgt = target_logits.argmax(axis=-1).astype(np.int64)  # default for slots the scan also overwrites
    for i in order:
        cf = np.stack([committed / 100.0, committed / (defenders + 1.0)], axis=-1)  # (P,2)
        bias = (cf @ ker)[:, 0] + b0[0]                  # (P,)
        t = int((target_logits[i] + bias).argmax())
        tgt[i] = t
        if fire_eff[i]:
            committed[t] += frac_val[i] * shp[i]         # unfloored, matches jax committed update
    return tgt


# ----- aiming / sun-guard (numpy mirror of lean_kaggle_agent) --------------
def _lead_angle(sx, sy, tx, ty, radius_t, is_comet_t, amount, av, radius_src=0.0, iters=6):
    speed = 1.0 + 5.0 * (math.log(max(amount, 1)) / SHIP_LOG_DENOM) ** 1.5
    speed = min(max(speed, 1.0), 6.0)
    orb_r = math.hypot(tx - BOARD_CENTER, ty - BOARD_CENTER)
    rotating = (not is_comet_t) and (orb_r + radius_t < ROTATION_LIMIT)
    if not rotating:
        return math.atan2(ty - sy, tx - sx), tx, ty
    cur_ang = math.atan2(ty - BOARD_CENTER, tx - BOARD_CENTER)
    # ★ FLIGHT-TIME FIX (2026-06-20): the env spawns the fleet at planet_center +
    # (radius_src + 0.1) ALONG the launch direction (launch_fleets L739-740), then it flies at
    # `speed`. The aim angle (atan2 from planet center) is unchanged because the fleet travels
    # along that same direction; but the FLIGHT DISTANCE is shorter by (radius_src + 0.1), so the
    # intercept must subtract it — otherwise t is over-estimated → over-leads rotating targets →
    # fleet arrives early → near-miss → straight overshoot into the sun. (case-study/shoot-sun.json)
    spawn_off = radius_src + 0.1
    fx, fy = tx, ty
    for _ in range(iters):
        t = max(0.0, math.hypot(fx - sx, fy - sy) - spawn_off) / speed
        ang = cur_ang + av * t
        fx = BOARD_CENTER + orb_r * math.cos(ang)
        fy = BOARD_CENTER + orb_r * math.sin(ang)
    return math.atan2(fy - sy, fx - sx), fx, fy


def _crosses_sun(sx, sy, ex, ey):
    vx, vy = ex - sx, ey - sy
    wx, wy = BOARD_CENTER - sx, BOARD_CENTER - sy
    c2 = vx * vx + vy * vy
    if c2 < 1e-9:
        return math.hypot(wx, wy) < ROTATION_LIMIT
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / c2))
    px, py = sx + t * vx, sy + t * vy
    return math.hypot(px - BOARD_CENTER, py - BOARD_CENTER) < SUN_RADIUS


def _ray_hits_sun(sx, sy, angle, margin=1.0):
    """True if the forward launch ray comes within the sun radius -> suppress.
    Conservative (ignores planet blockers): a planet might be a moving/orbiting
    "blocker" that drifts away so the fleet misses and flies on into the sun. The
    old _crosses_sun only checked up to the predicted intercept and leaked badly on
    misses. Measured: this conservative form = ZERO sun-deaths with NO loss of launch
    volume (suppressed ships are retained and re-launched at a safe target)."""
    dx, dy = math.cos(angle), math.sin(angle)
    proj_s = (BOARD_CENTER - sx) * dx + (BOARD_CENTER - sy) * dy
    if proj_s <= 0.0:
        return False  # sun behind launch direction
    perp_s = abs((BOARD_CENTER - sx) * dy - (BOARD_CENTER - sy) * dx)
    return perp_s < SUN_RADIUS + margin


def _fwd_sim_sun(sx, sy, sr, angle, amount, px, py, pr, palive, pcomet, av, sun_only=False):
    """ENV-EXACT forward-sim of THIS launch (replicates phase_f swept-circle capture vs all
    planets + sun; validated 99% vs the real env, sun-death detector matched against
    case-study/shoot-sun-0621). True -> suppress.

    `fwdsun` (sun_only=True, RECOMMENDED): SURGICAL — suppress ONLY launches that actually
      die to the sun (overshoot a rotated-away target straight into the center). Plugs the
      leak in the `segment` guard (which only checks source->intercept, so it misses the
      overshoot) WITHOUT changing anything else. Measured (real env, N=30 vs producer):
      sun-deaths 16->1 (-94%), launch volume -3% (only the sun launches), h2h vs segment
      46.7% = strength-neutral.
    `fwdsim` (sun_only=False): ALSO suppresses off-board / never-capture "wasteful" launches
      (launch volume -13%); strength-neutral too (h2h 48.3%) but broader than the sun bug,
      and ladder history warns 'less guarding = better' -> prefer fwdsun for a minimal fix."""
    v = min(max(1.0 + 5.0 * (math.log(max(amount, 1)) / SHIP_LOG_DENOM) ** 1.5, 1.0), 6.0)
    dx, dy = math.cos(angle), math.sin(angle)
    fx, fy = sx + dx * (sr + 0.1), sy + dy * (sr + 0.1)
    orb = np.hypot(px - BOARD_CENTER, py - BOARD_CENTER)
    a0 = np.arctan2(py - BOARD_CENTER, px - BOARD_CENTER)
    rot = palive & (~pcomet) & ((orb + pr) < ROTATION_LIMIT)
    board = 2.0 * BOARD_CENTER
    for _k in range(1, 56):
        nfx, nfy = fx + dx * v, fy + dy * v
        # ★ OFF-BY-ONE FIX (2026-06-22, case-study/shoot-sun-0621-2): the env's phase_e rotates
        # planets to rotate(initial, av*step_count) and the obs we get is ALREADY at the decision
        # step, so during the fleet's k-th move the env sweeps the planet from rotate(av*(k-2)) to
        # rotate(av*(k-1)) — NOT (k-1)->k. Rotating one step too far drifts grazing planets ~1.3u,
        # flipping a real MISS into a phantom CAPTURE -> the guard then thinks the fleet captured and
        # FAILS to suppress a launch that actually flies straight into the sun. Verified: with this
        # shift the sim reproduces the replay's exact death (step 23, pos 41.2,42.4).
        ang_o = a0 + np.where(rot, av * max(_k - 2, 0), 0.0)
        ang_n = a0 + np.where(rot, av * (_k - 1), 0.0)
        pox, poy = BOARD_CENTER + orb * np.cos(ang_o), BOARD_CENTER + orb * np.sin(ang_o)
        pnx, pny = BOARD_CENTER + orb * np.cos(ang_n), BOARD_CENTER + orb * np.sin(ang_n)
        d0x, d0y = fx - pox, fy - poy
        dvx, dvy = (nfx - fx) - (pnx - pox), (nfy - fy) - (pny - poy)
        a = dvx * dvx + dvy * dvy
        b = 2.0 * (d0x * dvx + d0y * dvy)
        c = d0x * d0x + d0y * d0y - pr * pr
        disc = b * b - 4.0 * a * c
        small = a < 1e-12
        sq = np.sqrt(np.maximum(disc, 0.0))
        den = np.maximum(2.0 * a, 1e-30)
        t1 = (-b - sq) / den
        t2 = (-b + sq) / den
        hit = (((disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)) | (small & (c <= 0.0))) & palive
        sa = (nfx - fx) ** 2 + (nfy - fy) ** 2
        sb = 2.0 * ((fx - BOARD_CENTER) * (nfx - fx) + (fy - BOARD_CENTER) * (nfy - fy))
        sc = (fx - BOARD_CENTER) ** 2 + (fy - BOARD_CENTER) ** 2 - SUN_RADIUS ** 2
        sdisc = sb * sb - 4.0 * sa * sc
        sun_t = None
        if sdisc >= 0.0:
            ssq = math.sqrt(sdisc); sden = max(2.0 * sa, 1e-30)
            st1, st2 = (-sb - ssq) / sden, (-sb + ssq) / sden
            if st2 >= 0.0 and st1 <= 1.0:
                sun_t = max(0.0, st1)
        if hit.any():
            tp = float(np.where(hit, t1, np.inf).min())
            return sun_t is not None and sun_t < tp   # captured -> keep, unless sun comes first
        if sun_t is not None:
            return True   # sun death
        if nfx < 0.0 or nfx > board or nfy < 0.0 or nfy > board:
            return False if sun_only else True   # off-board, captured nothing (not a sun death)
        fx, fy = nfx, nfy
    return False if sun_only else True            # never captured within horizon (not a sun death)


def _sun_block(sx, sy, angle, lx, ly, amount):
    """Dispatch the sun-guard by _OW_GUARD. `ray` = old conservative infinite-ray;
    `segment` = leaky source->intercept; `segmentmargin` = principled: the actual
    flight segment plus OW_MARGIN_TURNS of travel past the intercept (a miss is only
    fatal if the fleet overshoots into the sun, so guard exactly that overshoot zone
    without over-blocking legitimate hits that merely pass near the sun afterwards)."""
    if _OW_GUARD == "ray":
        return _ray_hits_sun(sx, sy, angle)
    ex, ey = lx, ly
    if _OW_GUARD == "segmentmargin":
        speed = min(max(1.0 + 5.0 * (math.log(max(amount, 1)) / SHIP_LOG_DENOM) ** 1.5, 1.0), 6.0)
        ex = lx + math.cos(angle) * speed * _OW_MARGIN_TURNS
        ey = ly + math.sin(angle) * speed * _OW_MARGIN_TURNS
    return _crosses_sun(sx, sy, ex, ey)


def _parse_obs(obs):
    planets = obs.get("planets") or []
    n = min(len(planets), MAX_PLANETS)
    comet_ids = {int(v) for v in (obs.get("comet_planet_ids") or [])}
    pid = np.full((MAX_PLANETS,), -1, np.int32)
    owner = np.full((MAX_PLANETS,), -2, np.int32)
    x = np.zeros((MAX_PLANETS,), np.float32)
    y = np.zeros((MAX_PLANETS,), np.float32)
    radius = np.zeros((MAX_PLANETS,), np.float32)
    ships = np.zeros((MAX_PLANETS,), np.int32)
    prod = np.zeros((MAX_PLANETS,), np.int32)
    is_comet = np.zeros((MAX_PLANETS,), bool)
    alive = np.zeros((MAX_PLANETS,), bool)
    if n > 0:
        p = np.asarray(planets[:n], dtype=np.float32)
        pid[:n] = p[:, 0].astype(np.int32)
        owner[:n] = p[:, 1].astype(np.int32)
        x[:n] = p[:, 2]
        y[:n] = p[:, 3]
        radius[:n] = p[:, 4]
        ships[:n] = p[:, 5].astype(np.int32)
        prod[:n] = p[:, 6].astype(np.int32)
        alive[:n] = True
        for i in range(n):
            if int(pid[i]) in comet_ids:
                is_comet[i] = True
    # fleets: [id, owner, x, y, angle, from_planet_id, ships]
    fleets = obs.get("fleets") or []
    if fleets:
        fa = np.asarray(fleets, dtype=np.float32)
        f_owner = fa[:, 1].astype(np.int32)
        f_x = fa[:, 2]; f_y = fa[:, 3]; f_angle = fa[:, 4]
        f_ships = fa[:, 6].astype(np.float32)
    else:
        f_owner = np.zeros((0,), np.int32)
        f_x = f_y = f_angle = f_ships = np.zeros((0,), np.float32)
    f_alive = np.ones(f_owner.shape, bool)
    return (pid, owner, alive, x, y, radius, ships, prod, is_comet,
            f_owner, f_x, f_y, f_angle, f_ships, f_alive)


def agent(obs, config=None):
    planets = obs.get("planets") if obs else None
    if not planets:
        return []
    player = int(obs.get("player", 0))
    av = float(obs.get("angular_velocity", 0.0) or 0.0)
    step = int(obs.get("step", 0))
    (pid, owner, alive, x, y, radius, ships, prod, is_comet,
     f_owner, f_x, f_y, f_angle, f_ships, f_alive) = _parse_obs(obs)

    # dispatch: detect player count at game start (all home planets present) and cache
    # for the rest of the game. 4p if our id >= 2 or >2 distinct non-neutral owners.
    global _NUM_PLAYERS
    if step == 0 or _NUM_PLAYERS is None:
        n_owners = len({int(o) for o in owner if int(o) >= 0})
        _NUM_PLAYERS = 4 if (player >= 2 or n_owners > 2) else 2
    model = _PARAMS_4P if _NUM_PLAYERS == 4 else _PARAMS_2P

    # N-player: features use RAW owner + RAW player. With opponent-collapse features
    # (is_enemy = any opponent) this is correct for 2p AND 4p, and is bit-identical to
    # the old player-0 remap for 2p. No remap needed.
    base = _planet_features(owner, alive, x, y, ships, prod, is_comet, player)            # (P,11)
    if _RICH:
        threat = _threat_features(owner, alive, x, y, radius, ships, is_comet, av,
                                  f_owner, f_x, f_y, f_angle, f_ships, f_alive, player)    # (P,4)
        td = _time_distance_features(owner, alive, x, y, step, player)                     # (P,4)
        cap = _capture_cost(owner, alive, x, y, ships, prod, player)[:, None]              # (P,1)
        gr = _global_rotating(owner, alive, x, y, radius, ships, prod, is_comet,
                              f_owner, f_ships, f_alive, player)                           # (P,3) adv+prod_adv+rot
        feats = np.concatenate([base, threat, td, cap, gr], axis=-1)                      # (P,23)
        if _ORBIT:
            # --rich-features --orbit-features = 23 base + [ang_vel, prod_density] (25-dim).
            # NOTE: 25-dim is ambiguous (rich+gravity vs rich+orbit) so this is flag-gated, not nd-gated.
            feats = np.concatenate([feats, _orbit_features(owner, alive, x, y, radius, prod, is_comet, av, player)], axis=-1)
        else:
            # legacy single-delta models auto-detected from embed in-dim (24=roi, 25=gravity, 26=both).
            nd = model["embed"]["kernel"].shape[0] if model is not None else feats.shape[-1]
            if nd in (24, 26):  # roi payoff-time
                feats = np.concatenate([feats, _roi_features(owner, alive, x, y, ships, prod, player)[:, None]], axis=-1)
            if nd in (25, 26):  # gravity density
                feats = np.concatenate([feats, _gravity_features(owner, alive, x, y, ships, player)], axis=-1)
    else:
        feats = base                                                                     # (P,11)
    if _IS4P:                                  # append broadcast is_4p bit (matches bc_train --is-4p-feature)
        is4 = np.full((feats.shape[0], 1), 1.0 if _NUM_PLAYERS == 4 else 0.0, np.float32)
        feats = np.concatenate([feats, is4], axis=-1)
    if _RANK:                                  # append broadcast rank_frac AFTER is_4p (matches bc_train --rank-feature)
        # my placement by GARRISON ships among seats: (#seats strictly ahead)/(nseats-1), 0=1st..1=last.
        # MUST match bc_train_clone._rank_frac (garrison-only sum, strict >, denom max(nseats-1,1)).
        shp = ships.astype(np.float32)
        seat_ships = np.array([shp[(owner == s) & alive].sum() for s in range(_NUM_PLAYERS)], np.float32)
        my_ships = shp[(owner == player) & alive].sum()
        rank_frac = float((seat_ships > my_ships).sum()) / max(_NUM_PLAYERS - 1, 1)
        feats = np.concatenate([feats, np.full((feats.shape[0], 1), rank_frac, np.float32)], axis=-1)
    focus_valid = alive & (owner != player)  # valid mass targets: neutral + any opponent
    fire_logit, target_logits, frac_logits = _forward(feats, alive, focus_valid, model)

    # mask dead / self targets
    neg = np.finfo(np.float32).min
    tgt_valid = alive[None, :] & (~np.eye(MAX_PLANETS, dtype=bool))
    target_logits = np.where(tgt_valid, target_logits, neg)
    fire = fire_logit > 0.0  # sigmoid(.)>0.5
    frac_idx = frac_logits.argmax(axis=-1)
    frac_val = _FRAC_BINS[frac_idx]
    # C2: autoregressive target choice only if the model was TRAINED autoregressive (_AR flag);
    # commit_head may exist as a dead arch default (e.g. B10M trained non-AR) -> parallel argmax.
    if _AR and model is not None and "commit_head" in model:
        target_idx = _autoregressive_targets(target_logits, fire, frac_val, ships, owner, alive, player,
                                             model["commit_head"])
    else:
        target_idx = target_logits.argmax(axis=-1)

    moves = []
    for i in range(MAX_PLANETS):
        if not alive[i] or owner[i] != player or not fire[i]:
            continue
        if int(ships[i]) < _MIN_LAUNCH:  # match training can_fire mask
            continue
        amt = int(math.floor(float(frac_val[i]) * int(ships[i])))
        amt = max(1, min(amt, int(ships[i])))
        if amt <= 0:
            continue
        j = int(target_idx[i])
        if j == i or not alive[j]:
            continue
        if _OW_CAPTURE_GATE and int(owner[j]) == -1 and int(ships[i]) < int(ships[j]) + 1:
            continue  # can't capture this neutral even sending all -> keep ships home to accumulate
        sxi, syi = float(x[i]), float(y[i])
        if _OW_AIM == "direct":  # env-matched: aim at target's CURRENT position, no lead
            txj, tyj = float(x[j]), float(y[j])
            angle, lx, ly = math.atan2(tyj - syi, txj - sxi), txj, tyj
        else:
            angle, lx, ly = _lead_angle(sxi, syi, float(x[j]), float(y[j]),
                                        float(radius[j]), bool(is_comet[j]), amt, av,
                                        radius_src=float(radius[i]))
        if _OW_GUARD in ("fwdsim", "fwdsun"):
            blocked = _fwd_sim_sun(sxi, syi, float(radius[i]), angle, amt,
                                   x, y, radius, alive, is_comet, av,
                                   sun_only=(_OW_GUARD == "fwdsun"))
        else:
            blocked = _sun_block(sxi, syi, angle, lx, ly, amt)
        if blocked:
            continue
        moves.append([int(pid[i]), float(angle), amt])
    return moves
