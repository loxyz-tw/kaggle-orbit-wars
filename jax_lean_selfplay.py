"""Lean self-play PPO for Orbit Wars (JAX), plus the shared entity-transformer policy.

Replaces the 71MB/sample per-candidate codec with factored action heads:
per planet -> {fire, target softmax over planets, fraction discrete bins}.
The env (jax_min_env_spike.step) already validates legality (owner/ships/alive),
so the policy just emits (src_pid, tgt_pid, amount); illegal moves become no-ops.

Action layout fed to step(): (B, 2 players, P moves, 3) = (src_pid, tgt_pid, amount).
Pure self-play: both players share one parameter set.

Reward: sparse +1/-1 win/loss at done (enough for 2p; value head does credit
assignment). Watch explained_variance / KL / clip_frac (key health metrics).

Phase 0 throughput probe: --probe (40K SPS on 5080).
Phase 1 signs-of-life: default train loop; gate = EV climbs past 0.8 within ~100 iters.
"""

from __future__ import annotations

import argparse
import pickle
import time
from functools import partial
from pathlib import Path

if __name__ == "__main__":
    from jax_train_runtime import configure_jax_training_runtime

    configure_jax_training_runtime()

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

import jax_min_env_spike as _env
from jax_min_env_spike import JaxOrbitState, reset, reset_from_bank, reset_from_bank_4p, reset_from_map, step


# ---------------------------------------------------------------------------
# Optional heuristic opponent ("producer") for opponent-mix / scaffold training.
# NOT included in this repository: pass --producer-module <file.py | module.name> pointing to
# your own JAX heuristic exposing:
#   CONFIG_2P, CONFIG_4P                         (NamedTuple configs supporting ._replace(...))
#   producer_moves(state, player, cfg, num_players) -> (B, M, 3) moves (src_pid, tgt_pid, amount)
#   producer_plan_per_planet(state, player, cfg, num_players)   (only for --scaffold)
# Pure self-play (the default) never loads it.
PRODUCER_MODULE = ""
_PRODUCER = None


def _load_producer():
    global _PRODUCER
    if _PRODUCER is None:
        if not PRODUCER_MODULE:
            raise SystemExit("this mode needs a heuristic opponent: pass --producer-module <file.py|module>")
        import importlib
        import importlib.util
        if PRODUCER_MODULE.endswith(".py"):
            spec = importlib.util.spec_from_file_location("user_producer", PRODUCER_MODULE)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        else:
            mod = importlib.import_module(PRODUCER_MODULE)
        _PRODUCER = mod
    return _PRODUCER

BOARD_CENTER = 50.0
BOARD_SCALE = 50.0
ROTATION_LIMIT = 50.0  # planet rotates if (orbit_r + radius) < this (env line 562)
SHIP_LOG_DENOM = jnp.log(1000.0)
FEATURE_DIM = 23  # 11 planet + 4 threat(A1) + 1 time(A2) + 3 dist(A3) + 1 capture_cost + 1 ship_adv + 1 prod_adv(D1) + 1 is_rotating
SUN_RADIUS = 10.0
SUN_R2 = SUN_RADIUS * SUN_RADIUS
THREAT_ETA_TAU = 20.0  # imminence decay: exp(-eta/tau)
THREAT_AIM_MARGIN = 2.0  # perp slack (units) when deciding a fleet is aimed at a planet
MIN_LAUNCH = 1  # a planet must hold >= this many ships to launch (set from args in main); 1 = no-op
RICH_FEATURES = False  # set from args: False=11-dim base, True=23-dim (A1 threat + time + dist + capture + ship_adv + prod_adv + rotating)
GRAVITY_FEATURES = False  # set from args: appends 2 dims (local enemy/own ship-mass density ~ Σships/dist^2) -> 25-dim. Default OFF keeps 23-dim bit-identical for existing ckpts.
ROI_FEATURES = False  # set from args: appends 1 dim (capture payoff-time = defenders/prod, log1p) -> 24-dim. Default OFF keeps 23-dim bit-identical.
GRAVITY_R0_SQ = 100.0  # softening radius^2 (R0=10 board units, ~planet influence scale) for the gravity-density features
OBS_V2 = False  # set from --obs-v2: redesigned 20-dim obs. Cut is_neutral/alive/ships_lin/thr_net/is_comet; add angular_velocity + prod_density. For the from-scratch robustness base.
ROTATION_AUG = False  # set from --rotation-aug: rotate the (x,y) obs cols 90/180/270 deg per state at update time (4x data, rotational invariance). Only x,y are orientation-dependent.
ORBIT_FEATURES = False  # set from --orbit-features: appends 2 dims to RICH (ang_vel + prod_density, obs-v2's probe-gated adds) -> 25-dim. Default OFF keeps 23-dim bit-identical.
IS4P_FEATURE = False  # set from --is-4p-feature: appends 1 dim (broadcast is_4p = N_PLAYERS==4) AFTER orbit, to match BC 4p-pack actor (27-dim). Default OFF = bit-identical.
RANK_FEATURE = False  # set from --rank-feature: appends 1 dim (my garrison-ship rank frac among seats) AFTER is_4p, to match BC 4p-pack. MUST mirror bc_train_clone._rank_frac (garrison-only, strict >, /(N-1)).
N_PLAYERS = 2         # set from --num-players: needed by is_4p/rank features (seat count + denom).
ANG_VEL_SCALE = 20.0   # angular_velocity is ~0.025-0.05 rad/step; *20 -> ~0.5-1.0 O(1) feature scale


def fleet_speed(ships):
    """Mirror env phase_h fleet speed: clip(1 + 5*(log(ships)/log1000)^1.5, max=6)."""
    fs = jnp.maximum(ships.astype(jnp.float32), 1.0)
    return jnp.maximum(jnp.minimum(1.0 + 5.0 * (jnp.log(fs) / SHIP_LOG_DENOM) ** 1.5, 6.0), 1.0)


def fleet_threat_features(state: "JaxOrbitState", player: int, iters: int = 2) -> jax.Array:
    """Per-planet in-transit-fleet threat features -> (B, P, 4).

    For each live fleet (owner/x/y/angle/ships only — same info kaggle obs exposes),
    project its straight-line ray and find the planet it is aimed at, accounting for
    the target planet's orbital motion over the flight (a few fixed-point iters, the
    inverse of `lead_angle`). Aggregate per planet:
      f1 = log1p(incoming hostile ships)/log1000
      f2 = log1p(incoming friendly ships)/log1000
      f3 = hostile imminence exp(-soonest_eta/tau)  (0 if none incoming)
      f4 = clip((planet ships + incoming friendly - incoming hostile)/100, -3, 3)
    Approximation (consistent train<->kaggle inference): straight fleet ray, orbital
    target projection, single closest-aimed planet per fleet. Mirrors main.py's
    get_planets_under_attack intent without a per-tick loop.
    """
    B, P = state.owner.shape
    spe = fleet_speed(state.fleet_ships)[:, :, None]          # (B,F,1)
    fxe = state.fleet_x[:, :, None]                           # (B,F,1)
    fye = state.fleet_y[:, :, None]
    dxe = jnp.cos(state.fleet_angle)[:, :, None]
    dye = jnp.sin(state.fleet_angle)[:, :, None]

    px = state.x; py = state.y                               # (B,P) current
    orb_r = jnp.sqrt((px - BOARD_CENTER) ** 2 + (py - BOARD_CENTER) ** 2)
    cur_ang = jnp.arctan2(py - BOARD_CENTER, px - BOARD_CENTER)
    rotating = state.alive & (~state.is_comet) & ((orb_r + state.radius) < ROTATION_LIMIT)  # (B,P)
    av = state.angular_velocity[:, None, None]               # (B,1,1)

    tpx = px[:, None, :]; tpy = py[:, None, :]               # (B,1,P) -> broadcasts to (B,F,P)
    eta = jnp.zeros((B, state.fleet_x.shape[1], P), dtype=jnp.float32)
    for _ in range(iters):
        rx = tpx - fxe; ry = tpy - fye
        proj = rx * dxe + ry * dye                           # (B,F,P) distance ahead along ray
        eta = jnp.maximum(proj, 0.0) / spe
        new_ang = cur_ang[:, None, :] + jnp.where(rotating[:, None, :], av * eta, 0.0)
        tpx = BOARD_CENTER + orb_r[:, None, :] * jnp.cos(new_ang)
        tpy = BOARD_CENTER + orb_r[:, None, :] * jnp.sin(new_ang)

    rx = tpx - fxe; ry = tpy - fye
    proj = rx * dxe + ry * dye
    perp = jnp.abs(rx * dye - ry * dxe)
    eta = jnp.maximum(proj, 0.0) / spe
    aimed = (proj > 0.0) & (perp <= state.radius[:, None, :] + THREAT_AIM_MARGIN) \
        & state.fleet_alive[:, :, None] & state.alive[:, None, :]
    proj_for_min = jnp.where(aimed, proj, jnp.inf)
    has_aim = aimed.any(axis=2)                              # (B,F)
    tgt = jnp.argmin(proj_for_min, axis=2)                   # (B,F)
    onehot = jax.nn.one_hot(tgt, P, dtype=jnp.float32) * has_aim[:, :, None].astype(jnp.float32)  # (B,F,P)

    hostile = ((state.fleet_owner != player) & (state.fleet_owner >= 0))[:, :, None].astype(jnp.float32)  # (B,F,1)
    friendly = (state.fleet_owner == player)[:, :, None].astype(jnp.float32)
    ships_f = jnp.maximum(state.fleet_ships.astype(jnp.float32), 0.0)[:, :, None]
    inc_host = (onehot * hostile * ships_f).sum(axis=1)      # (B,P)
    inc_frnd = (onehot * friendly * ships_f).sum(axis=1)

    host_cell = (onehot > 0.0) & ((state.fleet_owner != player) & (state.fleet_owner >= 0))[:, :, None]  # (B,F,P)
    soonest = jnp.min(jnp.where(host_cell, eta, jnp.inf), axis=1)  # (B,P)
    imminence = jnp.where(jnp.isfinite(soonest), jnp.exp(-soonest / THREAT_ETA_TAU), 0.0)

    ships_planet = jnp.maximum(state.ships.astype(jnp.float32), 0.0)
    f1 = jnp.log1p(inc_host) / SHIP_LOG_DENOM
    f2 = jnp.log1p(inc_frnd) / SHIP_LOG_DENOM
    f3 = imminence
    f4 = jnp.clip((ships_planet + inc_frnd - inc_host) / 100.0, -3.0, 3.0)
    return jnp.stack([f1, f2, f3, f4], axis=-1)


def _seg_sun_dist2(sx, sy, ex, ey):
    """Squared min distance from sun center (50,50) to segment (s)->(e)."""
    vx, vy = ex - sx, ey - sy
    wx, wy = BOARD_CENTER - sx, BOARD_CENTER - sy
    c2 = vx * vx + vy * vy
    t = jnp.clip((wx * vx + wy * vy) / jnp.maximum(c2, 1e-9), 0.0, 1.0)
    px, py = sx + t * vx, sy + t * vy
    return (px - BOARD_CENTER) ** 2 + (py - BOARD_CENTER) ** 2


def _ray_hits_sun(sx, sy, angle, margin=1.0):
    """(B,P) bool: forward launch ray comes within the sun radius -> suppress.
    Conservative (ignores blockers): a moving/orbiting "blocker" can drift away so
    the fleet misses and flies into the sun. The old segment-to-intercept check
    leaked badly on misses; this conservative form gives ZERO sun-deaths with no loss
    of launch volume (suppressed ships are retained for a safe target)."""
    dx, dy = jnp.cos(angle), jnp.sin(angle)                      # (B,P)
    cx = BOARD_CENTER - sx; cy = BOARD_CENTER - sy               # (B,P) source->sun
    proj_s = cx * dx + cy * dy                                   # (B,P) forward projection
    perp_s = jnp.abs(cx * dy - cy * dx)                          # (B,P) perpendicular dist
    return (proj_s > 0.0) & (perp_s < SUN_RADIUS + margin)       # (B,P)


def planet_features(state: JaxOrbitState, player: int) -> jax.Array:
    """Per-planet, player-relative features -> (B, P, F). Pure JAX (on-device)."""
    if OBS_V2:
        return planet_features_v2(state, player)
    owner = state.owner
    alive = state.alive
    is_mine = ((owner == player) & alive).astype(jnp.float32)
    is_enemy = ((owner != player) & (owner != -1) & alive).astype(jnp.float32)
    is_neutral = ((owner == -1) & alive).astype(jnp.float32)
    alive_f = alive.astype(jnp.float32)
    ships = jnp.maximum(state.ships.astype(jnp.float32), 0.0)
    ships_lin = jnp.clip(ships / 100.0, 0.0, 3.0)
    ships_log = jnp.log1p(ships) / SHIP_LOG_DENOM
    prod = state.production.astype(jnp.float32) / 5.0
    x = (state.x - BOARD_CENTER) / BOARD_SCALE
    y = (state.y - BOARD_CENTER) / BOARD_SCALE
    dist_sun = jnp.sqrt(x * x + y * y)
    is_comet = state.is_comet.astype(jnp.float32)
    base = jnp.stack(
        [is_mine, is_enemy, is_neutral, alive_f, ships_lin, ships_log, prod, x, y, dist_sun, is_comet],
        axis=-1,
    )  # (B, P, 11)
    if not RICH_FEATURES:
        return base  # clean 11-dim baseline
    threat = fleet_threat_features(state, player)            # (B, P, 4)  A1: in-transit fleets
    B, P = owner.shape
    # A2: time — normalized step, broadcast to every planet
    step_feat = jnp.broadcast_to((state.step_count.astype(jnp.float32) / 500.0)[:, None], (B, P))[..., None]
    # A3: distance to nearest enemy / neutral / own planet (raw board units / 100; 2.0 = none)
    rx = state.x[:, :, None] - state.x[:, None, :]
    ry = state.y[:, :, None] - state.y[:, None, :]
    pdist = jnp.sqrt(rx * rx + ry * ry)                     # (B, P, P)
    eye = jnp.eye(P, dtype=bool)[None]
    en_j = ((owner != player) & (owner != -1) & alive)[:, None, :]
    nt_j = ((owner == -1) & alive)[:, None, :]
    ow_j = ((owner == player) & alive)[:, None, :] & ~eye

    def _nearest(mask_j):
        nd = jnp.min(jnp.where(mask_j, pdist, jnp.inf), axis=2)
        return jnp.where(jnp.isfinite(nd), nd / 100.0, 2.0)

    near_en = _nearest(en_j)
    near_nt = _nearest(nt_j)
    near_ow = _nearest(ow_j)
    dist_feats = jnp.stack([near_en, near_nt, near_ow], axis=-1)  # (B,P,3)

    # capture cost: defenders this planet will have when a fleet arrives from my nearest
    # own planet, accounting for production accrued during flight (enemy planets only,
    # mirroring predict_total_ships). Source-agnostic per-target estimate.
    CAPTURE_SPEED_REF = 3.0
    eta_cap = (near_ow * 100.0) / CAPTURE_SPEED_REF  # near_ow is /100 board units (2.0 if none)
    is_enemy_p = (owner != player) & (owner != -1) & alive
    defenders = jnp.maximum(state.ships.astype(jnp.float32), 0.0) \
        + jnp.where(is_enemy_p, state.production.astype(jnp.float32) * eta_cap, 0.0)
    capture_cost = jnp.clip(defenders / 100.0, 0.0, 3.0)[..., None]

    # global ship advantage (incl in-transit fleets), broadcast to every planet (main.py get_ship_advantage)
    ships_pos = jnp.maximum(state.ships.astype(jnp.float32), 0.0)
    f_pos = jnp.maximum(state.fleet_ships.astype(jnp.float32), 0.0)
    fa = state.fleet_alive
    my_total = (ships_pos * ((owner == player) & alive)).sum(1) \
        + (f_pos * ((state.fleet_owner == player) & fa)).sum(1)
    en_total = (ships_pos * ((owner != player) & (owner != -1) & alive)).sum(1) \
        + (f_pos * (((state.fleet_owner != player) & (state.fleet_owner >= 0)) & fa)).sum(1)
    tot = my_total + en_total
    adv = jnp.where(tot > 0.0, (my_total - en_total) / jnp.maximum(tot, 1e-6), 0.0)  # (B,)
    adv_feat = jnp.broadcast_to(adv[:, None], (B, P))[..., None]
    # D1: global PRODUCTION advantage (own vs enemy planet production), broadcast.
    # strong heuristics weight production ~8x ships; ship_adv alone is blind to it.
    prod_all = state.production.astype(jnp.float32)
    my_prod = (prod_all * ((owner == player) & alive)).sum(1)
    en_prod = (prod_all * ((owner != player) & (owner != -1) & alive)).sum(1)
    ptot = my_prod + en_prod
    padv = jnp.where(ptot > 0.0, (my_prod - en_prod) / jnp.maximum(ptot, 1e-6), 0.0)  # (B,)
    padv_feat = jnp.broadcast_to(padv[:, None], (B, P))[..., None]
    # is_rotating: planet orbits the sun (needs lead-angle to hit)
    orb_r2 = jnp.sqrt((state.x - BOARD_CENTER) ** 2 + (state.y - BOARD_CENTER) ** 2)
    rot_feat = (alive & (~state.is_comet) & ((orb_r2 + state.radius) < ROTATION_LIMIT)).astype(jnp.float32)[..., None]

    feats = [base, threat, step_feat, dist_feats, capture_cost, adv_feat, padv_feat, rot_feat]
    if ROI_FEATURES:
        # Payoff time: turns of production to recoup the capture cost (defenders/prod, log1p).
        # HIGH = a defended-but-low-production TRAP (prod-1 with heavy garrison); LOW = a cheap
        # high-production prize. Encodes "prod-5 >> 3x prod-1" cost-adjusted, at target granularity.
        # `defenders` is the raw (pre-clip) capture_cost numerator from above. ALIVE planets
        # always have prod 1-5; prod==0 only on dead/padding slots (alive=False, attn-masked),
        # so max(prod,1) guards div-by-0 WITHOUT distorting the 1-5 payoff-time on real planets.
        roi = jnp.log1p(defenders / jnp.maximum(state.production.astype(jnp.float32), 1.0))[..., None]  # (B,P,1)
        feats.append(roi)
    if GRAVITY_FEATURES:
        # Local ship-mass density (gravity ~ Σ ships/dist^2): "meat-grinder frontline" vs
        # "safe backyard" — distinct from near_* (shortest distance, mass-blind) and adv (global).
        # Reuses pdist/en_j/ow_j/eye from the dist block above; no new distance compute.
        # RAW board-unit distance (~0-100) with a softening core GRAVITY_R0_SQ (planet
        # influence radius^2): board is ~100 units, so normalizing dist/100 before ^2 would
        # collapse the near field (every d^2 << 1) and flatten frontline vs backyard.
        d2 = pdist ** 2 + GRAVITY_R0_SQ                                   # (B,P,P) softened, no div-by-0
        sj = jnp.maximum(state.ships.astype(jnp.float32), 0.0)           # (B,P) raw ship masses
        en_mass = jnp.log1p((jnp.where(en_j, sj[:, None, :], 0.0) / d2).sum(2))[..., None]  # (B,P,1) ~0-3.5
        ow_mass = jnp.log1p((jnp.where(ow_j, sj[:, None, :], 0.0) / d2).sum(2))[..., None]  # (B,P,1); ow_j excludes self (~eye)
        feats += [en_mass, ow_mass]
    if ORBIT_FEATURES:
        # obs-v2's two probe-gated ADDs, ported onto RICH (replay evidence 2026-06-11):
        # ang_vel = orbital rate for rotating planets (lead-aim signal; RAW R^2 0.43 gap),
        # prod_density = local production density (expansion-direction signal). Same math
        # as planet_features_v2.
        av_b = jnp.broadcast_to(state.angular_velocity[:, None].astype(jnp.float32), (B, P))
        ang_vel = (rot_feat[..., 0] * av_b * ANG_VEL_SCALE)[..., None]               # (B,P,1)
        d2o = pdist ** 2 + GRAVITY_R0_SQ
        alive_j = alive[:, None, :] & ~eye
        pr_all = state.production.astype(jnp.float32)
        prod_density = jnp.log1p((jnp.where(alive_j, pr_all[:, None, :], 0.0) / d2o).sum(2))[..., None]
        feats += [ang_vel, prod_density]
    if IS4P_FEATURE:  # broadcast is_4p (matches BC 4p-pack); appended AFTER orbit
        feats.append(jnp.full((B, P, 1), 1.0 if N_PLAYERS == 4 else 0.0, jnp.float32))
    if RANK_FEATURE:  # my garrison-ship rank-frac among seats (mirror bc_train_clone._rank_frac); AFTER is_4p
        sp = jnp.maximum(state.ships.astype(jnp.float32), 0.0)                                   # (B,P)
        seat_ships = jnp.stack([(sp * ((owner == s) & alive)).sum(1) for s in range(N_PLAYERS)], axis=1)  # (B,N)
        my = seat_ships[:, player]                                                               # (B,)
        ahead = (seat_ships > my[:, None]).sum(1).astype(jnp.float32)                            # (B,) strict >
        rank = ahead / max(N_PLAYERS - 1, 1)
        feats.append(jnp.broadcast_to(rank[:, None, None], (B, P, 1)))
    return jnp.concatenate(feats, axis=-1)  # 23 base; +1 roi (24); +2 gravity; +2 --orbit-features; +is_4p +rank (4p-pack 27)


class LeanEntityPolicy(nn.Module):
    """Entity transformer over planets + factored action heads (lean PPO recipe).

    C1 commander/intent: a global focus-target latent makes the per-planet action
    heads conditionally coupled. trunk() emits the planet embeddings x, the
    focus-head logits, and value; a focus is sampled OUTSIDE (in sample_player);
    action_heads() then conditions fire/target/frac on the sampled focus. Joint
    policy = P(focus|s) * prod_p P(a_p|s,focus). focus_proj/focus_bias init to ZERO
    so a warm-started model initially behaves identically to the pre-C1 policy.
    """

    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    num_frac_bins: int = 3
    # 20-bin thaw plan-1 (2026-06-11): physically isolate the trunk from frac-head
    # gradients (lax.stop_gradient on the frac input). Forward values are IDENTICAL
    # (eval unaffected); only the backward path is cut, so frac_head still learns but
    # cannot corrupt the 400M trunk (the measured soft-8 damage channel).
    frac_stop_gradient: bool = False
    # FRESH-training fix (2026-06-14): focus_head/focus_proj are zero-init BY DEFAULT for
    # warm-start step-0 identity (C1). But for a FROM-SCRATCH run that zero-init traps the
    # focus/commander head at MAX entropy (measured: focus_head L2 stuck ~0.1 vs tgt_q ~14
    # at 150M, entFoc pinned at ln(P)) -> the coordination head never learns, model plays on
    # target-attention only. When True, focus_head/focus_proj use lecun_normal so the focus
    # head can bootstrap fresh. ONLY for fresh runs (NOT warm-start, which needs the zeros).
    fresh_focus_init: bool = False
    # 2026-06-19 DUAL-TRUNK critic for PPO handover: actor trunk (h768/L4) stays warm/frozen from
    # the L4 BC ckpt; critic grows its OWN smaller trunk (h384/L4/h4 ~4M) so PPO
    # value gradients NEVER touch the actor's BC algo-cut memory. Depth aligned (same hop count for
    # 56-planet attack topology), width halved (value=scalar, doesn't need 768). dual_trunk=False ->
    # classic shared value_head (BC eval / old RL paths unaffected).
    dual_trunk: bool = False
    critic_hidden: int = 384
    critic_layers: int = 4
    critic_heads: int = 4

    def setup(self):
        self.embed = nn.Dense(self.hidden_size, name="embed")
        self.ln_attn = [nn.LayerNorm(name=f"ln_attn_{i}") for i in range(self.num_layers)]
        self.attn = [
            nn.SelfAttention(
                num_heads=self.num_heads, qkv_features=self.hidden_size,
                out_features=self.hidden_size, dropout_rate=0.0, deterministic=True,
                name=f"attn_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.ln_mlp = [nn.LayerNorm(name=f"ln_mlp_{i}") for i in range(self.num_layers)]
        self.mlp_in = [nn.Dense(self.hidden_size, name=f"mlp_in_{i}") for i in range(self.num_layers)]
        self.mlp_out = [nn.Dense(self.hidden_size, name=f"mlp_out_{i}") for i in range(self.num_layers)]
        self.ln_out = nn.LayerNorm(name="ln_out")
        self.fire_head = nn.Dense(1, name="fire_head")
        self.frac_head = nn.Dense(self.num_frac_bins, name="frac_head")
        self.tgt_q = nn.Dense(self.hidden_size, name="tgt_q")
        self.tgt_k = nn.Dense(self.hidden_size, name="tgt_k")
        self.value_head = nn.Dense(1, name="value_head")
        # C1 commander head: zero-init so a warm-started model == pre-C1 at step 0.
        _focus_init = nn.initializers.lecun_normal() if self.fresh_focus_init else nn.initializers.zeros
        self.focus_head = nn.Dense(1, name="focus_head", kernel_init=_focus_init)
        self.focus_proj = nn.Dense(self.hidden_size, name="focus_proj", kernel_init=_focus_init)
        self.focus_bias = self.param("focus_bias", nn.initializers.zeros, ())
        # C2 autoregressive collision-avoidance: per-target bias on the target logits
        # from how much friendly mass is already COMMITTED to that target (running, in
        # ships-desc planet order). zero-init -> step-0 == pre-C2.
        self.commit_head = nn.Dense(1, name="commit_head",
                                    kernel_init=nn.initializers.zeros,
                                    bias_init=nn.initializers.zeros)
        # DUAL-TRUNK critic: own (smaller) attention trunk + value head. Only built when dual_trunk
        # so non-dual ckpts (BC eval, old RL) keep identical params. Depth = actor's (hop count),
        # width = critic_hidden (384), heads = critic_heads (4).
        if self.dual_trunk:
            ch, cl, chd = self.critic_hidden, self.critic_layers, self.critic_heads
            self.c_embed = nn.Dense(ch, name="c_embed")
            self.c_ln_attn = [nn.LayerNorm(name=f"c_ln_attn_{i}") for i in range(cl)]
            self.c_attn = [nn.SelfAttention(num_heads=chd, qkv_features=ch, out_features=ch,
                                            dropout_rate=0.0, deterministic=True, name=f"c_attn_{i}")
                           for i in range(cl)]
            self.c_ln_mlp = [nn.LayerNorm(name=f"c_ln_mlp_{i}") for i in range(cl)]
            self.c_mlp_in = [nn.Dense(ch, name=f"c_mlp_in_{i}") for i in range(cl)]
            self.c_mlp_out = [nn.Dense(ch, name=f"c_mlp_out_{i}") for i in range(cl)]
            self.c_ln_out = nn.LayerNorm(name="c_ln_out")
            self.c_value_head = nn.Dense(1, name="c_value_head")

    def critic_value(self, feats: jax.Array, alive: jax.Array):
        """DUAL-TRUNK value: independent critic trunk -> pooled -> scalar value (B,).
        Actor params untouched (no shared gradient path). Only valid when dual_trunk=True."""
        x = self.c_embed(feats)
        key_mask = alive[:, None, None, :]
        for i in range(self.critic_layers):
            x = x + self.c_attn[i](self.c_ln_attn[i](x), mask=key_mask)
            h = self.c_mlp_out[i](nn.gelu(self.c_mlp_in[i](self.c_ln_mlp[i](x))))
            x = x + h
        x = self.c_ln_out(x)
        mask_f = alive.astype(x.dtype)[..., None]
        pooled = (x * mask_f).sum(axis=1) / jnp.maximum(mask_f.sum(axis=1), 1.0)
        return self.c_value_head(pooled).squeeze(-1)

    def trunk(self, feats: jax.Array, alive: jax.Array):
        """Planet embeddings + focus-head logits + value. (B,P,H), (B,P), (B,)."""
        x = self.embed(feats)
        key_mask = alive[:, None, None, :]
        for i in range(self.num_layers):
            x = x + self.attn[i](self.ln_attn[i](x), mask=key_mask)
            h = self.mlp_out[i](nn.gelu(self.mlp_in[i](self.ln_mlp[i](x))))
            x = x + h
        x = self.ln_out(x)
        focus_logits = self.focus_head(x).squeeze(-1)  # (B,P) raw; masked to valid targets outside
        mask_f = alive.astype(x.dtype)[..., None]
        pooled = (x * mask_f).sum(axis=1) / jnp.maximum(mask_f.sum(axis=1), 1.0)
        value = self.value_head(pooled).squeeze(-1)  # (B,)
        return x, focus_logits, value

    def action_heads(self, x: jax.Array, focus_idx: jax.Array, alive: jax.Array):
        """Per-planet fire/target/frac logits, conditioned on the sampled focus."""
        x_focus = jnp.take_along_axis(x, focus_idx[:, None, None], axis=1)  # (B,1,H)
        x_cond = x + self.focus_proj(x_focus)                               # (B,P,H), broadcast
        fire_logit = self.fire_head(x_cond).squeeze(-1)                     # (B,P)
        frac_in = jax.lax.stop_gradient(x_cond) if self.frac_stop_gradient else x_cond
        frac_logits = self.frac_head(frac_in)                              # (B,P,K)
        q = self.tgt_q(x_cond)
        k = self.tgt_k(x_cond)
        target_logits = jnp.einsum("bih,bjh->bij", q, k) / jnp.sqrt(jnp.float32(self.hidden_size))
        # direct bias on the focus column: mechanically tilts targeting toward the
        # team focus (anti-collapse insurance independent of focus_proj).
        focus_oh = jax.nn.one_hot(focus_idx, alive.shape[-1], dtype=target_logits.dtype)  # (B,P)
        target_logits = target_logits + (self.focus_bias * focus_oh)[:, None, :]
        return fire_logit, target_logits, frac_logits

    def commit_bias(self, committed_feat: jax.Array) -> jax.Array:
        """C2: per-candidate-target bias on target logits from committed friendly mass.
        committed_feat: (B,P,2) = [committed/100, committed/(defenders+1)] -> (B,P) bias.
        zero-init kernel+bias -> returns 0 at step 0 (autoregressive head inert)."""
        return self.commit_head(committed_feat).squeeze(-1)

    def __call__(self, feats: jax.Array, alive: jax.Array):
        """Greedy full forward (used for param init; value via [3] for bootstrap)."""
        x, focus_logits, value = self.trunk(feats, alive)
        focus_idx = jnp.argmax(jnp.where(alive, focus_logits, -jnp.inf), axis=-1)
        fire_logit, target_logits, frac_logits = self.action_heads(x, focus_idx, alive)
        # Exercise commit_head during init so its (zero-init) params are registered even
        # when --autoregressive is off (warm-start merge treats it as a fresh-init new key).
        _ = self.commit_bias(jnp.zeros(alive.shape + (2,), x.dtype))
        # Exercise dual critic trunk during init so its params register (value here still from the
        # shared trunk for back-compat; PPO loop calls critic_value explicitly when dual_trunk).
        if self.dual_trunk:
            _ = self.critic_value(feats, alive)
        return fire_logit, target_logits, frac_logits, value, focus_logits


def planet_features_v2(state: JaxOrbitState, player: int) -> jax.Array:
    """Redesigned 20-dim obs (2026-06-07). Order (cols):
    0 is_mine 1 is_enemy 2 ships_log 3 prod 4 dist_sun 5 x 6 y 7 is_rotating 8 angular_velocity
    9 thr_host 10 thr_frnd 11 thr_imm 12 near_en 13 near_nt 14 near_ow 15 capture_cost
    16 prod_density 17 time 18 ship_adv 19 prod_adv.  (x,y at cols 5,6 -> rotation-aug rotates those.)
    Cut vs old 23-dim: is_neutral (=1-mine-enemy), alive (=attn mask), ships_lin (dup of ships_log),
    thr_net (linear combo), is_comet (dead bit: env has no comets). Added: angular_velocity, prod_density."""
    owner = state.owner
    alive = state.alive
    B, P = owner.shape
    is_mine = ((owner == player) & alive).astype(jnp.float32)
    is_enemy = ((owner != player) & (owner != -1) & alive).astype(jnp.float32)
    ships = jnp.maximum(state.ships.astype(jnp.float32), 0.0)
    ships_log = jnp.log1p(ships) / SHIP_LOG_DENOM
    prod = state.production.astype(jnp.float32) / 5.0
    xx = (state.x - BOARD_CENTER) / BOARD_SCALE
    yy = (state.y - BOARD_CENTER) / BOARD_SCALE
    dist_sun = jnp.sqrt(xx * xx + yy * yy)
    # orbital motion
    orb_r = jnp.sqrt((state.x - BOARD_CENTER) ** 2 + (state.y - BOARD_CENTER) ** 2)
    is_rotating = (alive & (~state.is_comet) & ((orb_r + state.radius) < ROTATION_LIMIT)).astype(jnp.float32)
    av = jnp.broadcast_to(state.angular_velocity[:, None].astype(jnp.float32), (B, P))
    ang_vel = is_rotating * av * ANG_VEL_SCALE                       # only rotating planets carry the rate
    # in-transit threat (host/frnd/imminence; drop the net combo f4)
    threat = fleet_threat_features(state, player)                    # (B,P,4)
    # nearest enemy/neutral/own distances
    rx = state.x[:, :, None] - state.x[:, None, :]
    ry = state.y[:, :, None] - state.y[:, None, :]
    pdist = jnp.sqrt(rx * rx + ry * ry)
    eye = jnp.eye(P, dtype=bool)[None]
    en_j = ((owner != player) & (owner != -1) & alive)[:, None, :]
    nt_j = ((owner == -1) & alive)[:, None, :]
    ow_j = ((owner == player) & alive)[:, None, :] & ~eye

    def _nearest(mask_j):
        nd = jnp.min(jnp.where(mask_j, pdist, jnp.inf), axis=2)
        return jnp.where(jnp.isfinite(nd), nd / 100.0, 2.0)

    near_en = _nearest(en_j)
    near_nt = _nearest(nt_j)
    near_ow = _nearest(ow_j)
    # capture cost: defenders at arrival from nearest own source (enemy production accrues in flight)
    eta_cap = (near_ow * 100.0) / 3.0
    is_enemy_p = (owner != player) & (owner != -1) & alive
    defenders = ships + jnp.where(is_enemy_p, state.production.astype(jnp.float32) * eta_cap, 0.0)
    capture_cost = jnp.clip(defenders / 100.0, 0.0, 3.0)
    # prod_density (NEW): local PRODUCTION density Σ_j prod_j/(d^2+R0), alive neighbours, log1p
    d2 = pdist ** 2 + GRAVITY_R0_SQ
    alive_j = alive[:, None, :] & ~eye
    pr = state.production.astype(jnp.float32)
    prod_density = jnp.log1p((jnp.where(alive_j, pr[:, None, :], 0.0) / d2).sum(2))
    # globals (broadcast)
    time_feat = jnp.broadcast_to((state.step_count.astype(jnp.float32) / 500.0)[:, None], (B, P))
    f_pos = jnp.maximum(state.fleet_ships.astype(jnp.float32), 0.0)
    fa = state.fleet_alive
    my_tot = (ships * ((owner == player) & alive)).sum(1) + (f_pos * ((state.fleet_owner == player) & fa)).sum(1)
    en_tot = (ships * ((owner != player) & (owner != -1) & alive)).sum(1) \
        + (f_pos * (((state.fleet_owner != player) & (state.fleet_owner >= 0)) & fa)).sum(1)
    tot = my_tot + en_tot
    adv = jnp.broadcast_to(jnp.where(tot > 0.0, (my_tot - en_tot) / jnp.maximum(tot, 1e-6), 0.0)[:, None], (B, P))
    my_p = (pr * ((owner == player) & alive)).sum(1)
    en_p = (pr * ((owner != player) & (owner != -1) & alive)).sum(1)
    pt = my_p + en_p
    padv = jnp.broadcast_to(jnp.where(pt > 0.0, (my_p - en_p) / jnp.maximum(pt, 1e-6), 0.0)[:, None], (B, P))
    cols = [
        is_mine, is_enemy, ships_log, prod, dist_sun, xx, yy, is_rotating, ang_vel,
        threat[..., 0], threat[..., 1], threat[..., 2], near_en, near_nt, near_ow,
        capture_cost, prod_density, time_feat, adv, padv]  # (B,P,20)
    # OPTIONAL appended features (cols 20+), gated by the same flags as old obs. APPENDED LAST so a
    # 20-dim ckpt warm-starts EXACT via zero-init embed-extend (extend_ckpt.py pads /embed/kernel).
    if ROI_FEATURES:  # +1: capture payoff-time (defenders/prod, log1p)
        cols.append(jnp.log1p(defenders / jnp.maximum(state.production.astype(jnp.float32), 1.0)))
    if GRAVITY_FEATURES:  # +2: local enemy/own ship-mass density Σ ships/(d^2+R0)
        cols.append(jnp.log1p((jnp.where(en_j, ships[:, None, :], 0.0) / d2).sum(2)))  # en_mass
        cols.append(jnp.log1p((jnp.where(ow_j, ships[:, None, :], 0.0) / d2).sum(2)))  # ow_mass
    return jnp.stack(cols, axis=-1)  # 20 (+1 roi)(+2 gravity)


def rotate_xy(feats: jax.Array, key, xcol: int = 5, ycol: int = 6) -> jax.Array:
    """Rotational data augmentation: rotate the (x,y) obs cols of each state by a random k*90 deg.
    Only x,y are orientation-dependent (rest are rotation-invariant); the dynamics are 90-deg-rotation
    equivariant, and the action labels (target=planet index, fire, frac) are invariant -> valid 4x aug."""
    M = feats.shape[0]
    k = jax.random.randint(key, (M,), 0, 4)
    cos = jnp.array([1.0, 0.0, -1.0, 0.0])[k][:, None]   # (M,1) broadcast over planets
    sin = jnp.array([0.0, 1.0, 0.0, -1.0])[k][:, None]
    x = feats[:, :, xcol]
    y = feats[:, :, ycol]
    xr = cos * x - sin * y
    yr = sin * x + cos * y
    return feats.at[:, :, xcol].set(xr).at[:, :, ycol].set(yr)


def mask_target_logits(target_logits: jax.Array, alive: jax.Array) -> jax.Array:
    """Disallow dead targets and self-target; keep a safe fallback row."""
    P = alive.shape[-1]
    eye = jnp.eye(P, dtype=bool)[None, :, :]
    tgt_valid = alive[:, None, :] & (~eye)
    neg_inf = jnp.finfo(target_logits.dtype).min
    out = jnp.where(tgt_valid, target_logits, neg_inf)
    no_valid = ~tgt_valid.any(axis=-1)
    return jnp.where(no_valid[..., None] & eye, 0.0, out)


def _categorical_logp(logits: jax.Array, idx: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, idx[..., None], axis=-1).squeeze(-1)


def factored_logp_entropy(fire_logit, target_logits, frac_logits, owned, fire, target_idx, frac_idx):
    """Joint logp (B,) and entropy (B,) for the per-planet factored action."""
    fire_p = jnp.clip(jax.nn.sigmoid(fire_logit), 1e-6, 1.0 - 1e-6)
    logp_fire = jnp.where(fire, jnp.log(fire_p), jnp.log(1.0 - fire_p))
    logp_target = _categorical_logp(target_logits, target_idx)
    logp_frac = _categorical_logp(frac_logits, frac_idx)
    logp_planet = jnp.where(owned, logp_fire + jnp.where(fire, logp_target + logp_frac, 0.0), 0.0)
    logp = logp_planet.sum(axis=1)

    ent_fire = -(fire_p * jnp.log(fire_p) + (1.0 - fire_p) * jnp.log(1.0 - fire_p))
    ent_tgt = -(jax.nn.softmax(target_logits, -1) * jax.nn.log_softmax(target_logits, -1)).sum(-1)
    ent_frac = -(jax.nn.softmax(frac_logits, -1) * jax.nn.log_softmax(frac_logits, -1)).sum(-1)
    ent_planet = jnp.where(owned, ent_fire + fire_p * (ent_tgt + ent_frac), 0.0)
    # Mean over owned planets (not sum): keeps entropy O(1) and ent_coef
    # invariant to planet count P and per-map owned count (kaggle maps vary
    # 20-40 planets). Summed entropy exploded as P grew. logp stays summed.
    denom = jnp.maximum(owned.sum(axis=1).astype(jnp.float32), 1.0)
    entropy = ent_planet.sum(axis=1) / denom
    # Fire-head entropy alone (mean over owned, max=ln2~=0.693). Key
    # collapse signal: protect it with its own coef so the policy keeps firing.
    ent_fire_mean = jnp.where(owned, ent_fire, 0.0).sum(axis=1) / denom
    return logp, entropy, ent_fire_mean


def _commit_feat(committed, defenders):
    """Per-candidate-target committed-mass features for the C2 commit head. (B,P)->(B,P,2).
    [committed/100, committed/(defenders+1)]: 2nd >= 1 means "a prior launch this turn
    already plans to cover this target's garrison" -> don't pile on / don't send a 2nd."""
    return jnp.stack([committed / 100.0, committed / (defenders + 1.0)], axis=-1)


def autoregressive_target(bias_apply, base_tgt, fire_eff, frac_val, ships, defenders, order,
                          *, key=None, greedy=False, recorded_target=None):
    """C2 autoregressive target choice over owned planets in `order` (ships-desc).

    Each planet i (in order) chooses its target under `base_tgt[:,i,:] + commit_bias(committed)`,
    then ADDS its outgoing mass to committed[target] so later (smaller) planets see what is
    already covered -> emergent collision-avoidance / no-redundant-send.

    SHARED by sampling (key/greedy given) and PPO replay (recorded_target given). Because replay
    teacher-forces the recorded target, the committed trajectory is bit-identical to sampling, so
    the autoregressive logp is consistent. zero-init commit_head -> bias==0 -> EXACT match to the
    parallel target head (step-0 identity).

    Returns per-planet (B,P): target_idx, logp_target, ent_target (scattered back from rank order).
    """
    B, P = ships.shape
    bidx = jnp.arange(B)
    sample = recorded_target is None
    keys = jax.random.split(key, P) if (sample and not greedy) else jnp.zeros((P, 2), jnp.uint32)

    def body(committed, xs):
        j, kj = xs
        i = order[:, j]                                                       # (B,) planet at rank j
        base_row = jnp.take_along_axis(base_tgt, i[:, None, None], 1).squeeze(1)   # (B,P) masked logits
        logits = base_row + bias_apply(_commit_feat(committed, defenders))   # (B,P)
        if not sample:
            tgt = jnp.take_along_axis(recorded_target, i[:, None], 1).squeeze(1).astype(jnp.int32)
        elif greedy:
            tgt = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        else:
            tgt = jax.random.categorical(kj, logits, axis=-1).astype(jnp.int32)
        logp = _categorical_logp(logits, tgt)
        ent = -(jax.nn.softmax(logits, -1) * jax.nn.log_softmax(logits, -1)).sum(-1)
        fe = jnp.take_along_axis(fire_eff, i[:, None], 1).squeeze(1)          # (B,) gated fire
        fv = jnp.take_along_axis(frac_val, i[:, None], 1).squeeze(1)
        sh = jnp.take_along_axis(ships, i[:, None], 1).squeeze(1).astype(jnp.float32)
        sent = jnp.where(fe, fv * sh, 0.0)                                    # ships this planet commits
        committed = committed.at[bidx, tgt].add(sent)
        return committed, (tgt, logp, ent)

    init = jnp.zeros((B, P), jnp.float32)
    _, (tgt_seq, logp_seq, ent_seq) = jax.lax.scan(body, init, (jnp.arange(P), keys))
    # scatter rank-ordered (P,B) outputs back to per-planet (B,P) via order
    tgt_full = jnp.zeros((B, P), jnp.int32).at[bidx[:, None], order].set(tgt_seq.T)
    logp_full = jnp.zeros((B, P), jnp.float32).at[bidx[:, None], order].set(logp_seq.T)
    ent_full = jnp.zeros((B, P), jnp.float32).at[bidx[:, None], order].set(ent_seq.T)
    return tgt_full, logp_full, ent_full


def factored_logp_entropy_ar(fire_logit, frac_logits, logp_target, ent_target, owned, fire, frac_idx):
    """Joint logp/entropy with an AUTOREGRESSIVE target component (precomputed per-planet
    logp_target/ent_target from autoregressive_target). Mirrors factored_logp_entropy exactly
    for the fire/frac heads; at zero-commit (logp_target==parallel) it is EXACT-equal to it."""
    fire_p = jnp.clip(jax.nn.sigmoid(fire_logit), 1e-6, 1.0 - 1e-6)
    logp_fire = jnp.where(fire, jnp.log(fire_p), jnp.log(1.0 - fire_p))
    logp_frac = _categorical_logp(frac_logits, frac_idx)
    logp_planet = jnp.where(owned, logp_fire + jnp.where(fire, logp_target + logp_frac, 0.0), 0.0)
    logp = logp_planet.sum(axis=1)

    ent_fire = -(fire_p * jnp.log(fire_p) + (1.0 - fire_p) * jnp.log(1.0 - fire_p))
    ent_frac = -(jax.nn.softmax(frac_logits, -1) * jax.nn.log_softmax(frac_logits, -1)).sum(-1)
    ent_planet = jnp.where(owned, ent_fire + fire_p * (ent_target + ent_frac), 0.0)
    denom = jnp.maximum(owned.sum(axis=1).astype(jnp.float32), 1.0)
    entropy = ent_planet.sum(axis=1) / denom
    ent_fire_mean = jnp.where(owned, ent_fire, 0.0).sum(axis=1) / denom
    return logp, entropy, ent_fire_mean


def mask_focus_logits(focus_logits, focus_valid):
    """Restrict the commander's focus to alive enemy/neutral targets (valid mass
    targets). Rows with no valid target fall back to a uniform no-op."""
    neg_inf = jnp.finfo(focus_logits.dtype).min
    out = jnp.where(focus_valid, focus_logits, neg_inf)
    no_valid = ~focus_valid.any(axis=-1, keepdims=True)
    return jnp.where(no_valid, 0.0, out)


def focus_logp_entropy(focus_logits, focus_valid, focus_idx):
    """logp (B,) and entropy (B,) of the global focus choice. Gated to 0 when the
    env has no valid focus target (focus is then an inert no-op)."""
    fl = mask_focus_logits(focus_logits, focus_valid)
    any_valid = focus_valid.any(axis=-1)
    logp = jnp.where(any_valid, _categorical_logp(fl, focus_idx), 0.0)
    ent = -(jax.nn.softmax(fl, -1) * jax.nn.log_softmax(fl, -1)).sum(-1)
    ent = jnp.where(any_valid, ent, 0.0)
    return logp, ent


def lead_angle(state, target_idx, amount, iters: int = 6):
    """Interception aim: source planet -> chosen target's *future* position.

    kaggle env consumes a raw launch angle and planets orbit the sun, so aiming
    at a target's current position misses moving planets. Fixed-point solve for
    travel time t (= distance / fleet_speed) against the target's circular orbit,
    matching env physics (phase_e + fleet speed formula). Static planets and
    comets fall back to current position (rotation term is masked off).
    """
    def gx(a):
        return jnp.take_along_axis(a, target_idx, axis=1)

    sx, sy = state.x, state.y  # source = each planet slot (B,P)
    # ★ FLIGHT-TIME FIX (2026-06-20): env spawns the fleet at source_center +
    # (radius_src + 0.1) ALONG the launch dir (launch_fleets), so flight DISTANCE is
    # shorter by (radius_src + 0.1); subtract it (else over-leads rotating targets -> sun).
    spawn_off = state.radius + 0.1  # (B,P), per source slot
    t_ix, t_iy = gx(state.initial_x), gx(state.initial_y)
    t_rad = gx(state.radius)
    t_x, t_y = gx(state.x), gx(state.y)
    t_alive = gx(state.alive)
    t_comet = gx(state.is_comet)

    dx, dy = t_ix - BOARD_CENTER, t_iy - BOARD_CENTER
    orb_r = jnp.sqrt(dx * dx + dy * dy)
    rotating = t_alive & (~t_comet) & ((orb_r + t_rad) < BOARD_CENTER)
    init_ang = jnp.arctan2(dy, dx)
    av = state.angular_velocity[:, None]
    step0 = state.step_count[:, None].astype(jnp.float32)

    speed = 1.0 + 5.0 * (jnp.log(jnp.maximum(amount.astype(jnp.float32), 1.0)) / SHIP_LOG_DENOM) ** 1.5
    speed = jnp.clip(speed, 1.0, 6.0)

    def pos_at(s):
        ang = init_ang + av * s
        rx = BOARD_CENTER + orb_r * jnp.cos(ang)
        ry = BOARD_CENTER + orb_r * jnp.sin(ang)
        return jnp.where(rotating, rx, t_x), jnp.where(rotating, ry, t_y)

    fx, fy = t_x, t_y
    for _ in range(iters):
        dist = jnp.sqrt((fx - sx) ** 2 + (fy - sy) ** 2)
        t = jnp.maximum(dist - spawn_off, 0.0) / speed
        fx, fy = pos_at(step0 + t)
    return jnp.arctan2(fy - sy, fx - sx), fx, fy


def sample_player(params, policy, state, player, key, frac_bins, *, greedy=False, autoregressive=False):
    """Sample a player's factored action. Returns dict of moves + components + logp + value + feats."""
    feats = planet_features(state, player)
    alive = state.alive
    owned = (state.owner == player) & alive          # true ownership (clean)
    ships = state.ships.astype(jnp.int32)
    can_fire = owned & (ships >= MIN_LAUNCH)          # fire-eligibility: must hold >= MIN_LAUNCH ships
    focus_valid = alive & (state.owner != player)    # valid mass targets: alive enemy/neutral

    k_focus, k_fire, k_tgt, k_frac = jax.random.split(key, 4)

    # C1 stage 1: trunk + commander. Sample a global focus target, masked to valid.
    x, focus_logits, value = policy.apply({"params": params}, feats, alive, method=policy.trunk)
    if policy.dual_trunk:  # value from independent critic trunk (actor x/focus untouched)
        value = policy.apply({"params": params}, feats, alive, method=policy.critic_value)
    focus_masked = mask_focus_logits(focus_logits, focus_valid)
    if greedy:
        focus_idx = jnp.argmax(focus_masked, axis=-1).astype(jnp.int32)
    else:
        focus_idx = jax.random.categorical(k_focus, focus_masked, axis=-1).astype(jnp.int32)

    # C1 stage 2: per-planet heads, conditioned on the sampled focus.
    fire_logit, target_logits, frac_logits = policy.apply(
        {"params": params}, x, focus_idx, alive, method=policy.action_heads)
    target_logits = mask_target_logits(target_logits, alive)

    # fire/frac sampled in parallel (NOT conditioned on committed, per plan decision 2).
    fire_p = jax.nn.sigmoid(fire_logit)
    if greedy:
        fire = fire_p > 0.5
        frac_idx = jnp.argmax(frac_logits, axis=-1).astype(jnp.int32)
    else:
        fire = jax.random.bernoulli(k_fire, fire_p)
        frac_idx = jax.random.categorical(k_frac, frac_logits, axis=-1).astype(jnp.int32)
    fire = fire & can_fire

    # gate logp/entropy on can_fire so PPO is consistent: sub-threshold planets are
    # treated as un-actionable (forced fire=False, zero logp/entropy contribution).
    if autoregressive:
        # C2: target chosen autoregressively over owned planets (ships-desc), coupling on
        # committed friendly mass so later (smaller) planets avoid already-covered targets.
        frac_val = frac_bins[frac_idx]
        order = jnp.argsort(-ships, axis=1)
        bias_apply = lambda cf: policy.apply({"params": params}, cf, method=policy.commit_bias)
        target_idx, logp_target, ent_target = autoregressive_target(
            bias_apply, target_logits, fire, frac_val, ships, ships.astype(jnp.float32), order,
            key=k_tgt, greedy=greedy)
        logp_act, _, _ = factored_logp_entropy_ar(
            fire_logit, frac_logits, logp_target, ent_target, can_fire, fire, frac_idx)
    else:
        if greedy:
            target_idx = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
        else:
            target_idx = jax.random.categorical(k_tgt, target_logits, axis=-1).astype(jnp.int32)
        logp_act, _, _ = factored_logp_entropy(fire_logit, target_logits, frac_logits, can_fire, fire, target_idx, frac_idx)
    # Total joint logp = focus logp + per-planet action logp.
    logp_focus, _ = focus_logp_entropy(focus_logits, focus_valid, focus_idx)
    logp = logp_act + logp_focus

    src_pid = state.planet_id
    frac_val = frac_bins[frac_idx]
    amount = jnp.floor(frac_val * ships.astype(jnp.float32)).astype(jnp.int32)
    amount = jnp.where(fire, jnp.maximum(amount, 1), 0)
    amount = jnp.minimum(amount, ships)
    angle, lx, ly = lead_angle(state, target_idx, amount)
    # Sun guard: a straight fleet whose path crosses the sun (<r=10 of center)
    # is destroyed by the env. Suppress such launches (ships stay home) instead
    # of feeding them into the sun. Mirrors main.py's reject behaviour.
    crosses_sun = _ray_hits_sun(state.x, state.y, angle)
    amount = jnp.where(crosses_sun, 0, amount)
    moves = jnp.stack(
        [src_pid.astype(jnp.float32), angle, amount.astype(jnp.float32)], axis=-1
    )  # (B,P,3) = (src_pid, lead_angle, amount); fed with actions_are_angles=True
    return {
        "moves": moves,
        "feats": feats,
        "alive": alive,
        "owned": can_fire,  # stored mask = fire-eligibility (what train_step's logp/fire_rate gate on)
        "fire": fire,
        "target_idx": target_idx,
        "frac_idx": frac_idx,
        "focus_idx": focus_idx,
        "focus_valid": focus_valid,
        "ships": ships,  # raw per-planet ships (B,P): C2 replay needs order + committed mass
        "logp": logp,
        "value": value,
    }


def sample_player_scaffold(params, policy, state, player, key, frac_bins, scaffold_cfg,
                           *, greedy=False):
    """Producer-as-scaffold action (2026-06-12): the frozen jax producer proposes its full
    plan; the policy's FIRE head decides per proposing planet {follow, veto}. Executed
    moves are the producer's exact rows with vetoed sources zeroed. logp = Bernoulli(fire)
    over proposing planets ONLY (focus greedy, no logp; target/frac heads unused).
    Returns the same dict shape as sample_player ("owned" := proposal mask)."""
    _JP = _load_producer()
    feats = planet_features(state, player)
    alive = state.alive
    ships = state.ships.astype(jnp.int32)

    moves_rows, src_rows, val_rows, prop_fire = _JP.producer_plan_per_planet(
        state, player, scaffold_cfg, 2)

    k_fire, _ = jax.random.split(key)
    x, focus_logits, value = policy.apply({"params": params}, feats, alive, method=policy.trunk)
    focus_valid = alive & (state.owner != player)
    focus_masked = mask_focus_logits(focus_logits, focus_valid)
    focus_idx = jnp.argmax(focus_masked, axis=-1).astype(jnp.int32)   # greedy, no logp

    fire_logit, target_logits, frac_logits = policy.apply(
        {"params": params}, x, focus_idx, alive, method=policy.action_heads)
    fire_p = jax.nn.sigmoid(fire_logit)
    if greedy:
        fire = fire_p > 0.5
    else:
        fire = jax.random.bernoulli(k_fire, fire_p)
    fire = fire & prop_fire                                            # decisions only where proposed

    # execute: zero the ships of vetoed source rows
    B = src_rows.shape[0]
    follow_rows = fire[jnp.arange(B)[:, None], jnp.clip(src_rows, 0, fire.shape[1] - 1)] & val_rows
    moves_exec = moves_rows.at[:, :, 2].set(jnp.where(follow_rows, moves_rows[:, :, 2], 0.0))
    P = state.owner.shape[1]
    M = moves_exec.shape[1]
    moves_exec = jnp.pad(moves_exec, ((0, 0), (0, max(P - M, 0)), (0, 0)))[:, :P, :]

    logp_f = -jax.nn.softplus(-fire_logit) * fire - jax.nn.softplus(fire_logit) * (~fire)
    logp = jnp.where(prop_fire, logp_f, 0.0).sum(axis=1)

    return {
        "moves": moves_exec,
        "feats": feats,
        "alive": alive,
        "owned": prop_fire,            # loss gate = proposal mask
        "fire": fire,
        "target_idx": jnp.zeros(prop_fire.shape, jnp.int32),
        "frac_idx": jnp.zeros(prop_fire.shape, jnp.int32),
        "focus_idx": focus_idx,
        "focus_valid": focus_valid,
        "ships": ships,
        "logp": logp,
        "value": value,
    }


def score_by_player(state, player, mode="ships"):
    """Terminal score for the Copeland reward = total ships (garrison on alive planets + ships in
    alive fleets). Matches the RULES win condition EXACTLY ("Final score = total ships on owned
    planets + total ships in owned fleets. Highest wins.").

    `mode` kept for signature compat but only "ships" is valid. The old "planets" / "blend" modes
    were DELETED 2026-06-07: they ranked on PLANET COUNT, which is NOT the win condition (you can win
    planets while losing ships) -> a misaligned objective (tested = win-rate wash; b0 cross-play
    ship-scored had the planets-trained seed LOSE to the ships-trained one). The 4p-turtle motivation
    for planet-reward is better served by PBRS expansion-shaping (policy-invariant, keeps the terminal
    ship-Copeland intact), not by changing the terminal metric.
    """
    owned = (state.owner == player) & state.alive
    planet_ships = (owned * state.ships).sum(axis=1)
    fleet_ships = (((state.fleet_owner == player) & state.fleet_alive) * state.fleet_ships).sum(axis=1)
    return planet_ships + fleet_ships


def reset_done_envs(state, reset_state, done):
    def sel(cur, init):
        mask = done.reshape((done.shape[0],) + (1,) * (cur.ndim - 1))
        return jnp.where(mask, init, cur)

    return jax.tree_util.tree_map(sel, state, reset_state)


def make_rollout(policy, frac_bins, episode_steps, rollout_steps, gamma=0.99, num_players=2,
                 score_mode="ships", reward_mode="copeland", autoregressive=False,
                 expansion_shaping=False, expansion_weight=0.0,
                 net_asset_shaping=False, nas_prod_weight=0.0, nas_combat_weight=0.0, producer_envs=0,
                 producer_cfg_overrides=None, producer_levels=None, scaffold_cfg=None,
                 frozen_opponents=None):
    """Self-play rollout via lax.scan over N players (all share one parameter set).
    Stores per-step per-player tensors for PPO.

    Pure sparse terminal reward (lean PPO recipe). For N players the terminal reward
    is the mean over opponents of sign(my_score - their_score) (a normalized Copeland
    score in [-1,1]); for 2p this reduces EXACTLY to sign(score0 - score1).
    score_mode is "ships" only (planets/blend deleted — misaligned with the ship win-condition).

    producer_envs > 0 (D4 pool training): in the FIRST `producer_envs` envs, player 1's
    actions are replaced by the frozen heuristic opponent (--producer-module).
    The stored p1 tensors in those envs describe UNSAMPLED actions — the main loop drops
    those lanes from the PPO data (KEEP_IDX); p0's reward there = beat-producer signal.
    """
    if producer_envs > 0:
        _JP = _load_producer()
        assert num_players == 2, "--opponent-producer-frac is 2p-only"
        base_cfg = _JP.CONFIG_2P
        if producer_cfg_overrides:
            base_cfg = base_cfg._replace(**producer_cfg_overrides)
            print(f"[lean] curriculum producer cfg overrides: {producer_cfg_overrides}", flush=True)
        if producer_levels:
            # mixed-level pool: split producer envs evenly among horizons (env-index order)
            n_lv = len(producer_levels)
            per = producer_envs // n_lv
            _level_ranges = []
            for i, hz in enumerate(producer_levels):
                lo = i * per
                hi = (i + 1) * per if i < n_lv - 1 else producer_envs
                _level_ranges.append((lo, hi, base_cfg._replace(horizon=int(hz))))
            print(f"[lean] mixed-level pool: {[(lo, hi, c.horizon) for lo, hi, c in _level_ranges]}", flush=True)
        else:
            _level_ranges = [(0, producer_envs, base_cfg)]

    @jax.jit
    def rollout(params, state, reset_state, key):
        n_frozen = len(frozen_opponents) if frozen_opponents else 0
        def body(carry, _):
            st, k = carry
            keys = jax.random.split(k, num_players + 1 + n_frozen)
            k = keys[0]
            if scaffold_cfg is not None:
                outs = [sample_player_scaffold(params, policy, st, 0, keys[1], frac_bins, scaffold_cfg)]
                outs += [sample_player(params, policy, st, p, keys[p + 1], frac_bins,
                                       autoregressive=autoregressive)
                         for p in range(1, num_players)]
            else:
                outs = [sample_player(params, policy, st, p, keys[p + 1], frac_bins, autoregressive=autoregressive)
                        for p in range(num_players)]
            actions = jnp.stack([o["moves"] for o in outs], axis=1)  # (B,N,P,3)
            if producer_envs > 0:
                P_rows = actions.shape[2]
                p1 = actions[:, 1]
                env_ar = jnp.arange(actions.shape[0])
                for _lo, _hi, _cfg in _level_ranges:
                    pm = _JP.producer_moves(st, 1, _cfg, num_players)         # (B,M,3) M=12 <= P
                    pm_pad = jnp.zeros((pm.shape[0], P_rows, 3), actions.dtype)
                    pm_pad = pm_pad.at[:, : pm.shape[1], :].set(pm.astype(actions.dtype))
                    env_mask = ((env_ar >= _lo) & (env_ar < _hi))[:, None, None]
                    p1 = jnp.where(env_mask, pm_pad, p1)
                actions = actions.at[:, 1].set(p1)
            if frozen_opponents:                                          # PAST-SELF LEAGUE: p1 = frozen ckpt
                p1 = actions[:, 1]
                env_ar = jnp.arange(actions.shape[0])
                for fi, (_fp, _lo, _hi) in enumerate(frozen_opponents):
                    fo = sample_player(_fp, policy, st, 1, keys[num_players + 1 + fi], frac_bins,
                                       autoregressive=autoregressive)
                    env_mask = ((env_ar >= _lo) & (env_ar < _hi))[:, None, None]
                    p1 = jnp.where(env_mask, fo["moves"], p1)
                actions = actions.at[:, 1].set(p1)
            nst = step(st, actions, episode_steps, actions_are_angles=True)
            done = nst.done
            scores = jnp.stack([score_by_player(nst, p, score_mode) for p in range(num_players)], axis=1).astype(jnp.float32)  # (B,N)
            if reward_mode == "winner":
                is_max = scores >= scores.max(axis=1, keepdims=True)          # (B,N); ties -> all +1
                rew = jnp.where(is_max, 1.0, -1.0)                            # winner-take-all (ladder-aligned)
            else:
                diff = scores[:, :, None] - scores[:, None, :]                # (B,N,N)
                rew = jnp.sign(diff).sum(axis=2) / max(num_players - 1, 1)    # (B,N) normalized Copeland
            rew = jnp.where(done[:, None], rew, 0.0)

            # C3 expansion shaping (4p turtle root-cause): potential-based reward shaping
            # F = gamma*Phi(s') - Phi(s), Phi_p = w * sum(production of planets p controls).
            # Policy-INVARIANT (Ng 1999): doesn't change the optimum, only densifies the gradient
            # toward expansion to escape the self-play turtle basin (real-ladder replay: we stall
            # at ~5 planets mid-game while winners snowball to ~19). Terminal Copeland unchanged.
            # Episodic exact-invariance: Phi(terminal)=0.
            if expansion_shaping:
                def _potential(s):
                    pr = s.production.astype(jnp.float32)                                    # (B,P)
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]  # (B,P,N)
                    return expansion_weight * (pr[:, :, None] * oh).sum(axis=1)              # (B,N)
                phi_post = jnp.where(done[:, None], 0.0, _potential(nst))
                rew = rew + (gamma * phi_post - _potential(st))
            if net_asset_shaping:
                # 2026-06-23 "Net Asset Capture" (4p melee, arm B): dense per-step
                # reward = w1*dProduction - w2*combat_losses. dProduction>0 only when you CAPTURE
                # (forces active expansion); combat_losses = drop in total ships (garrison+fleet;
                # launching keeps the ship counted -> a drop = real death) -> activates the BC's
                # 20-bin "is this trade worth it" calc. Terminal reward kept (use --reward-mode winner).
                def _prodsum(s):
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]
                    return (s.production.astype(jnp.float32)[:, :, None] * oh).sum(axis=1)        # (B,N)
                def _shipsum(s):
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]
                    pl = (s.ships.astype(jnp.float32)[:, :, None] * oh).sum(axis=1)
                    fo = (s.fleet_owner[:, :, None] == jnp.arange(num_players)) & s.fleet_alive[:, :, None]
                    fl = (s.fleet_ships.astype(jnp.float32)[:, :, None] * fo).sum(axis=1)
                    return pl + fl                                                                # (B,N)
                dprod = jnp.where(done[:, None], 0.0, _prodsum(nst) - _prodsum(st))
                combat_loss = jnp.where(done[:, None], 0.0, jnp.maximum(0.0, _shipsum(st) - _shipsum(nst)))
                rew = rew + nas_prod_weight * dprod - nas_combat_weight * combat_loss

            def stkN(key_name):
                return jnp.stack([o[key_name] for o in outs], axis=1)  # (B, N, ...)

            out = {
                "feats": stkN("feats"), "alive": stkN("alive"), "owned": stkN("owned"),
                "fire": stkN("fire"), "target_idx": stkN("target_idx"), "frac_idx": stkN("frac_idx"),
                "focus_idx": stkN("focus_idx"), "focus_valid": stkN("focus_valid"),
                "ships": stkN("ships"),
                "logp": stkN("logp"), "value": stkN("value"),
                "reward": rew, "done": done,
            }
            nst = reset_done_envs(nst, reset_state, done)
            return (nst, k), out

        (final_state, final_key), traj = jax.lax.scan(body, (state, key), None, length=rollout_steps)
        if policy.dual_trunk:
            nvs = [policy.apply({"params": params}, planet_features(final_state, p), final_state.alive,
                                method=policy.critic_value) for p in range(num_players)]
        else:
            nvs = [policy.apply({"params": params}, planet_features(final_state, p), final_state.alive,
                                method=policy.trunk)[2] for p in range(num_players)]
        next_value = jnp.stack(nvs, axis=1)  # (B,N)
        return final_state, final_key, traj, next_value

    return rollout


def make_rollout_4p(policy, frac_bins, episode_steps, rollout_steps, gamma=0.99, num_players=4,
                    score_mode="ships", reward_mode="copeland", autoregressive=False,
                    expansion_shaping=False, expansion_weight=0.0,
                 net_asset_shaping=False, nas_prod_weight=0.0, nas_combat_weight=0.0, producer_envs=0,
                    producer_cfg_overrides=None, producer_levels=None, scaffold_cfg=None,
                    frozen_opponents=None):
    """4p-DEDICATED rollout (2026-06-20). IDENTICAL to make_rollout
    EXCEPT the producer-mix lane assigns the frozen producer to ALL opponent seats (1..N-1),
    not just seat 1: a producer-env = 1 learner (seat 0) vs (N-1) producers, matching
    eval_4p_full's 1-vs-3. base_cfg = CONFIG_4P. Non-producer envs = full N-way self-play.
    The 2p make_rollout is left byte-identical (b10m line untouched). reward/expansion/scan
    are already N-player. KEEP_IDX in the main loop must drop seats 1..N-1 in producer envs.
    """
    if producer_envs > 0:
        _JP = _load_producer()
        assert num_players == 4, "make_rollout_4p is for 4p; 2p uses make_rollout"
        base_cfg = _JP.CONFIG_4P
        if producer_cfg_overrides:
            base_cfg = base_cfg._replace(**producer_cfg_overrides)
            print(f"[lean] 4p producer cfg overrides: {producer_cfg_overrides}", flush=True)
        if producer_levels:
            # mixed-level pool: split producer envs evenly among horizons (env-index order)
            n_lv = len(producer_levels)
            per = producer_envs // n_lv
            _level_ranges = []
            for i, hz in enumerate(producer_levels):
                lo = i * per
                hi = (i + 1) * per if i < n_lv - 1 else producer_envs
                _level_ranges.append((lo, hi, base_cfg._replace(horizon=int(hz))))
            print(f"[lean] 4p mixed-level pool: {[(lo, hi, c.horizon) for lo, hi, c in _level_ranges]}", flush=True)
        else:
            _level_ranges = [(0, producer_envs, base_cfg)]

    @jax.jit
    def rollout(params, state, reset_state, key):
        n_frozen = len(frozen_opponents) if frozen_opponents else 0
        def body(carry, _):
            st, k = carry
            keys = jax.random.split(k, num_players + 1 + n_frozen)
            k = keys[0]
            if scaffold_cfg is not None:
                outs = [sample_player_scaffold(params, policy, st, 0, keys[1], frac_bins, scaffold_cfg)]
                outs += [sample_player(params, policy, st, p, keys[p + 1], frac_bins,
                                       autoregressive=autoregressive)
                         for p in range(1, num_players)]
            else:
                outs = [sample_player(params, policy, st, p, keys[p + 1], frac_bins, autoregressive=autoregressive)
                        for p in range(num_players)]
            actions = jnp.stack([o["moves"] for o in outs], axis=1)  # (B,N,P,3)
            if producer_envs > 0:
                P_rows = actions.shape[2]
                env_ar = jnp.arange(actions.shape[0])
                # producer-env = seat 0 learner vs producer in EVERY opponent seat (1..N-1)
                for opp in range(1, num_players):
                    seat = actions[:, opp]
                    for _lo, _hi, _cfg in _level_ranges:
                        pm = _JP.producer_moves(st, opp, _cfg, num_players)       # (B,M,3) M<=P
                        pm_pad = jnp.zeros((pm.shape[0], P_rows, 3), actions.dtype)
                        pm_pad = pm_pad.at[:, : pm.shape[1], :].set(pm.astype(actions.dtype))
                        env_mask = ((env_ar >= _lo) & (env_ar < _hi))[:, None, None]
                        seat = jnp.where(env_mask, pm_pad, seat)
                    actions = actions.at[:, opp].set(seat)
            if frozen_opponents:                                          # (4p league unused now; seat1-only, harmless)
                p1 = actions[:, 1]
                env_ar = jnp.arange(actions.shape[0])
                for fi, (_fp, _lo, _hi) in enumerate(frozen_opponents):
                    fo = sample_player(_fp, policy, st, 1, keys[num_players + 1 + fi], frac_bins,
                                       autoregressive=autoregressive)
                    env_mask = ((env_ar >= _lo) & (env_ar < _hi))[:, None, None]
                    p1 = jnp.where(env_mask, fo["moves"], p1)
                actions = actions.at[:, 1].set(p1)
            nst = step(st, actions, episode_steps, actions_are_angles=True)
            done = nst.done
            scores = jnp.stack([score_by_player(nst, p, score_mode) for p in range(num_players)], axis=1).astype(jnp.float32)  # (B,N)
            if reward_mode == "winner":
                is_max = scores >= scores.max(axis=1, keepdims=True)
                rew = jnp.where(is_max, 1.0, -1.0)
            else:
                diff = scores[:, :, None] - scores[:, None, :]                # (B,N,N)
                rew = jnp.sign(diff).sum(axis=2) / max(num_players - 1, 1)    # (B,N) normalized Copeland
            rew = jnp.where(done[:, None], rew, 0.0)
            if expansion_shaping:
                def _potential(s):
                    pr = s.production.astype(jnp.float32)
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]
                    return expansion_weight * (pr[:, :, None] * oh).sum(axis=1)
                phi_post = jnp.where(done[:, None], 0.0, _potential(nst))
                rew = rew + (gamma * phi_post - _potential(st))
            if net_asset_shaping:
                # 2026-06-23 "Net Asset Capture" (4p melee, arm B): dense per-step
                # reward = w1*dProduction - w2*combat_losses. dProduction>0 only when you CAPTURE
                # (forces active expansion); combat_losses = drop in total ships (garrison+fleet;
                # launching keeps the ship counted -> a drop = real death) -> activates the BC's
                # 20-bin "is this trade worth it" calc. Terminal reward kept (use --reward-mode winner).
                def _prodsum(s):
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]
                    return (s.production.astype(jnp.float32)[:, :, None] * oh).sum(axis=1)        # (B,N)
                def _shipsum(s):
                    oh = (s.owner[:, :, None] == jnp.arange(num_players)) & s.alive[:, :, None]
                    pl = (s.ships.astype(jnp.float32)[:, :, None] * oh).sum(axis=1)
                    fo = (s.fleet_owner[:, :, None] == jnp.arange(num_players)) & s.fleet_alive[:, :, None]
                    fl = (s.fleet_ships.astype(jnp.float32)[:, :, None] * fo).sum(axis=1)
                    return pl + fl                                                                # (B,N)
                dprod = jnp.where(done[:, None], 0.0, _prodsum(nst) - _prodsum(st))
                combat_loss = jnp.where(done[:, None], 0.0, jnp.maximum(0.0, _shipsum(st) - _shipsum(nst)))
                rew = rew + nas_prod_weight * dprod - nas_combat_weight * combat_loss

            def stkN(key_name):
                return jnp.stack([o[key_name] for o in outs], axis=1)

            out = {
                "feats": stkN("feats"), "alive": stkN("alive"), "owned": stkN("owned"),
                "fire": stkN("fire"), "target_idx": stkN("target_idx"), "frac_idx": stkN("frac_idx"),
                "focus_idx": stkN("focus_idx"), "focus_valid": stkN("focus_valid"),
                "ships": stkN("ships"),
                "logp": stkN("logp"), "value": stkN("value"),
                "reward": rew, "done": done,
            }
            nst = reset_done_envs(nst, reset_state, done)
            return (nst, k), out

        (final_state, final_key), traj = jax.lax.scan(body, (state, key), None, length=rollout_steps)
        nvs = [policy.apply({"params": params}, planet_features(final_state, p), final_state.alive,
                            method=policy.trunk)[2] for p in range(num_players)]
        next_value = jnp.stack(nvs, axis=1)
        return final_state, final_key, traj, next_value

    return rollout


def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """rewards/values/dones: (T, M). next_value: (M,). dones broadcast already to (T,M)."""

    def body(carry, inp):
        r, v, d = inp
        gae_acc, next_v = carry
        nonterminal = 1.0 - d
        delta = r + gamma * next_v * nonterminal - v
        gae_acc = delta + gamma * lam * nonterminal * gae_acc
        return (gae_acc, v), gae_acc

    init = (jnp.zeros_like(next_value), next_value)
    _, adv = jax.lax.scan(body, init, (rewards, values, dones), reverse=True)
    return adv, adv + values


def make_train_step(policy, clip_coef, vf_coef, ent_coef, optimizer, ent_fire_coef=0.0, ent_focus_coef=0.0,
                    autoregressive=False, frac_bins=None, ent_frac_coef=0.0, scaffold=False,
                    bc_ref_params=None, kl_coef=0.0):
    @jax.jit
    def train_step(params, opt_state, feats, alive, owned, fire, target_idx, frac_idx,
                   focus_idx, focus_valid, ships, old_logp, adv, ret):
        def loss_fn(p):
            # C1 two-stage recompute: trunk -> focus logp; action_heads(focus) -> action logp.
            x, focus_logits, value = policy.apply({"params": p}, feats, alive, method=policy.trunk)
            if policy.dual_trunk:  # value from independent critic trunk
                value = policy.apply({"params": p}, feats, alive, method=policy.critic_value)
            fl, tl, frl = policy.apply({"params": p}, x, focus_idx, alive, method=policy.action_heads)
            tl = mask_target_logits(tl, alive)
            if scaffold:
                # scaffold mode: action = per-planet {follow producer, veto}; the only sampled
                # head is FIRE on proposing planets ("owned" = proposal mask). Focus was greedy
                # (no logp); target/frac heads unused.
                logp_f = -jax.nn.softplus(-fl) * fire - jax.nn.softplus(fl) * (~fire)
                logp_act = jnp.where(owned, logp_f, 0.0).sum(axis=1)
                pf = jax.nn.sigmoid(fl)
                ent_pl = -(pf * jnp.log(pf + 1e-8) + (1 - pf) * jnp.log(1 - pf + 1e-8))
                denom_f = jnp.maximum(owned.sum(axis=1).astype(jnp.float32), 1.0)
                entropy = jnp.where(owned, ent_pl, 0.0).sum(axis=1) / denom_f
                ent_fire = entropy
                logp_focus = jnp.zeros_like(logp_act)
                ent_focus = jnp.zeros_like(entropy)
            elif autoregressive:
                # C2 teacher-forcing: rebuild committed from RECORDED targets (identical to the
                # sampling-time trajectory) and score target logp under the current params'
                # commit-biased logits. Same scan as sampling -> logp consistent for PPO.
                frac_val = frac_bins[frac_idx]
                order = jnp.argsort(-ships, axis=1)
                bias_apply = lambda cf: policy.apply({"params": p}, cf, method=policy.commit_bias)
                _, logp_target, ent_target = autoregressive_target(
                    bias_apply, tl, fire, frac_val, ships, ships.astype(jnp.float32), order,
                    recorded_target=target_idx)
                logp_act, entropy, ent_fire = factored_logp_entropy_ar(
                    fl, frl, logp_target, ent_target, owned, fire, frac_idx)
            else:
                logp_act, entropy, ent_fire = factored_logp_entropy(fl, tl, frl, owned, fire, target_idx, frac_idx)
            if not scaffold:
                logp_focus, ent_focus = focus_logp_entropy(focus_logits, focus_valid, focus_idx)
            logp = logp_act + logp_focus
            logratio = logp - old_logp
            ratio = jnp.exp(logratio)
            pg1 = -adv * ratio
            pg2 = -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = jnp.mean(jnp.maximum(pg1, pg2))
            value_loss = 0.5 * jnp.mean((value - ret) ** 2)
            ent = jnp.mean(entropy)
            entf = jnp.mean(ent_fire)  # fire-head entropy (watch for collapse)
            entfoc = jnp.mean(ent_focus)  # commander entropy (C1: watch for collapse)
            # Targeted frac-head entropy (20-bin thaw experiment 2026-06-11): global
            # ent_coef's frac term is gated by fire_p and dominated by the target head,
            # so frozen new bins (bias -8) never melt. Direct unweighted frac entropy
            # over owned planets gives the melting force a dedicated dial (same per-head
            # pattern as ent_fire_coef). Default 0.0 = exact old loss.
            ent_frac_pl = -(jax.nn.softmax(frl, -1) * jax.nn.log_softmax(frl, -1)).sum(-1)
            denom_o = jnp.maximum(owned.sum(axis=1).astype(jnp.float32), 1.0)
            entfr = jnp.mean(jnp.where(owned, ent_frac_pl, 0.0).sum(axis=1) / denom_o)
            # Per-head fire entropy bonus. Global ent_coef's pressure is dominated by
            # the target head (softmax over ~28 planets, ent up to ln28~3.3) and cannot
            # reach the binary fire head (ent <= ln2~0.69), so the fire head collapses to
            # fire-everything. A per-head ent_coef is the standard fix for
            # entropy_fire collapse. Default 0.0 = pure global.
            # ent_focus_coef does the same for the commander head (C1).
            # Stage B: KL anchor to frozen BC ref. Anchors the actor's action distribution to L4-BC
            # on VISITED states (small actor LR alone won't stop covariate-shift drift off the algo
            # cut). Mirrors the entropy mask/mean (owned planets, focus over valid). 0 when kl_coef==0.
            kl_anchor = 0.0
            if kl_coef > 0.0 and bc_ref_params is not None:
                xr, focus_r, _ = policy.apply({"params": bc_ref_params}, feats, alive, method=policy.trunk)
                flr, tlr, frlr = policy.apply({"params": bc_ref_params}, xr, focus_idx, alive, method=policy.action_heads)
                tlr = mask_target_logits(tlr, alive)
                def _ckl(cl, rl):  # KL(softmax(cl)||softmax(rl)) over last axis
                    clp = jax.nn.log_softmax(cl, -1); rlp = jax.nn.log_softmax(rl, -1)
                    return (jnp.exp(clp) * (clp - rlp)).sum(-1)
                pc = jnp.clip(jax.nn.sigmoid(fl), 1e-6, 1 - 1e-6); pr = jnp.clip(jax.nn.sigmoid(flr), 1e-6, 1 - 1e-6)
                kl_fire = pc * jnp.log(pc / pr) + (1 - pc) * jnp.log((1 - pc) / (1 - pr))  # (B,P)
                kl_pl = jnp.where(owned, kl_fire + _ckl(tl, tlr) + _ckl(frl, frlr), 0.0).sum(axis=1) / denom_o
                fc_m = mask_focus_logits(focus_logits, focus_valid); fr_m = mask_focus_logits(focus_r, focus_valid)
                kl_anchor = jnp.mean(kl_pl + _ckl(fc_m, fr_m))
            loss = (policy_loss + vf_coef * value_loss
                    - ent_coef * ent - ent_fire_coef * entf - ent_focus_coef * entfoc
                    - ent_frac_coef * entfr + kl_coef * kl_anchor)
            approx_kl = jnp.mean((ratio - 1.0) - logratio)
            clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))
            # explained variance
            var_ret = jnp.var(ret)
            ev = 1.0 - jnp.var(ret - value) / jnp.maximum(var_ret, 1e-8)
            return loss, (policy_loss, value_loss, ent, approx_kl, clip_frac, ev, entf, entfoc, entfr, kl_anchor)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        gnorm = optax.global_norm(grads)
        return params, opt_state, loss, aux, gnorm

    return train_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true", help="Measure rollout throughput only.")
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--rollout-steps", type=int, default=32)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--epochs", type=int, default=4,
                   help="PPO epochs per rollout (reuse the batch). Was 1 (sample-inefficient, "
                        "underused PPO's clip-enabled reuse; clip-frac was ~0.05-0.12). Now 4 (PPO "
                        "standard 3-10) to extract more learning per sample — matters since models "
                        "only train 50-150M. ⚠️ A/B candidate later: epochs 1 vs 4 (1 favours on-policy "
                        "stability, 4 favours sample efficiency; the right value is empirical).")
    p.add_argument("--minibatch-size", type=int, default=2048)
    p.add_argument("--planets", type=int, default=20)
    p.add_argument("--fleets", type=int, default=2048)
    p.add_argument("--episode-steps", type=int, default=500)
    p.add_argument("--map-json", default="")
    p.add_argument("--map-bank", default="", help="npz map bank; each env samples a kaggle-distribution map.")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--frac-bins", default="0.1,0.25,0.5,0.75,1.0",
                   help="Discrete launch sizes = fraction of source ships. The old 3-bin set "
                        "{0.52,0.87,1.0} had NO small option -> forced over-sending, blocked the "
                        "efficient small captures producer-style expansion needs. Now 5 bins incl "
                        "0.1/0.25. Changes frac_head dim -> NOT warm-start-compatible with 3-bin ckpts.")
    # PPO hyperparams
    p.add_argument("--lr", type=float, default=3e-5)
    # DUAL-TRUNK PPO handover (2026-06-19): actor warm from L4 BC + own LR; critic fresh smaller
    # trunk + bigger LR + warmup. value goes through critic_value when --dual-trunk.
    p.add_argument("--dual-trunk", action="store_true", help="independent critic trunk; value via critic_value")
    p.add_argument("--critic-hidden", type=int, default=384)
    p.add_argument("--critic-layers", type=int, default=4)
    p.add_argument("--critic-heads", type=int, default=4)
    p.add_argument("--actor-lr", type=float, default=1e-5, help="dual: actor LR (BC 1/10~1/20); 0 during warmup")
    p.add_argument("--critic-lr", type=float, default=3e-4, help="dual: critic LR (small fresh net, fast MSE)")
    p.add_argument("--warmup-critic-updates", type=int, default=0, help="dual: # updates with actor LR=0 (train critic only)")
    # Stage B: KL anchor to a FROZEN BC reference, so the unfrozen actor cannot drift off the BC
    # algo-cut policy (small LR alone is not enough; covariate-shift snowball). Penalty =
    # kl_coef * mean KL(current_actor || bc_ref) over visited states (fire+target+frac+focus heads).
    p.add_argument("--kl-coef", type=float, default=0.0, help="KL(actor||bc_ref) anchor coef on visited states (0=off)")
    p.add_argument("--bc-ref-ckpt", default="", help="frozen BC ref ckpt for KL anchor (default: reuse --init-checkpoint)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--ent-fire-coef", type=float, default=0.0,
                   help="Per-head fire entropy bonus (fix for entropy_fire collapse). 0=off.")
    p.add_argument("--ent-focus-coef", type=float, default=0.0,
                   help="C1 commander-head entropy bonus (prevents focus posterior collapse). 0=off.")
    p.add_argument("--ent-frac-coef", type=float, default=0.0,
                   help="Per-head frac entropy bonus (20-bin thaw: melt frozen new bins without "
                        "heating target/focus heads via global ent). 0=off (exact old loss).")
    p.add_argument("--frac-stop-gradient", action="store_true",
                   help="Cut frac-head gradients to the trunk (lax.stop_gradient on its input); "
                        "forward identical, trunk physically immune to frac exploration noise.")
    p.add_argument("--fresh-focus-init", action="store_true",
                   help="FRESH-only: lecun-init focus_head/focus_proj instead of zeros so the "
                        "focus/commander head bootstraps from scratch (zero-init traps it at max "
                        "entropy in from-scratch runs). Do NOT use with --init-checkpoint warm-start.")
    p.add_argument("--min-launch", type=int, default=1,
                   help="A planet must hold >= this many ships to launch (forces accumulation; 1=off).")
    p.add_argument("--num-players", type=int, default=2, choices=(2, 4),
                   help="Self-play player count (2 or 4). Sets the env NUM_PLAYERS; rollout samples N players.")
    p.add_argument("--score-mode", default="ships", choices=("ships",),  # planets/blend DELETED: misaligned w/ ship win-condition
                   help="Terminal Copeland score basis. ONLY 'ships' (= the RULES win condition). "
                        "planets/blend DELETED 2026-06-07 (ranked on planet count = misaligned with the "
                        "ship win-condition; tested wash). 4p expansion -> use --expansion-shaping (PBRS) instead.")
    p.add_argument("--reward-mode", default="copeland", choices=("copeland", "winner"),
                   help="Terminal reward shape: copeland (default, graded sign-Copeland/(N-1); 2p "
                        "bit-identical) or winner (winner-take-all +1 to 1st else -1, aligns 4p "
                        "training with the LADDER's scoring + removes the turtle middle-rank net).")
    p.add_argument("--league-ckpts", type=str, default=None,
                   help="PAST-SELF LEAGUE: comma-sep frozen self-ckpt paths used as PLAYER 1 in a slice "
                        "of envs (fictitious self-play; anti rock-paper-scissors). Mutually exclusive "
                        "with --opponent-producer-frac (both occupy the first envs).")
    p.add_argument("--league-fracs", type=str, default=None,
                   help="comma-sep env fraction per --league-ckpts entry (rest = current-policy self-play). "
                        "e.g. 0.25,0.25 = 25%% vs ckpt1, 25%% vs ckpt2, 50%% self.")
    p.add_argument("--producer-module", default="",
                   help="path to a .py file (or importable module) providing YOUR heuristic opponent; "
                        "required only for --opponent-producer-frac > 0 or --scaffold (not shipped here).")
    p.add_argument("--opponent-producer-frac", type=float, default=0.0,
                   help="D4 pool training: fraction of envs where PLAYER 1 is the FROZEN heuristic "
                        "opponent from --producer-module. Those (env,p1) lanes are excluded from "
                        "the PPO loss; p0's terminal Copeland vs producer is the signal. First "
                        "round(frac*num_envs) envs, static set. 2p only.")
    p.add_argument("--producer-roi", type=float, default=None,
                   help="Curriculum: override the in-loop producer's roi_threshold (weakened-producer ladder).")
    p.add_argument("--producer-waves", type=int, default=None,
                   help="Curriculum: override the in-loop producer's max_waves.")
    p.add_argument("--producer-horizon", type=int, default=None,
                   help="Curriculum: override the in-loop producer's horizon.")
    p.add_argument("--producer-levels", type=str, default=None,
                   help="Mixed-level pool: comma horizons (e.g. '12,14,16'). The producer envs "
                        "(opponent-producer-frac) are split EVENLY among these levels (env-index "
                        "order). Overrides --producer-horizon.")
    p.add_argument("--producer-min-ships", type=float, default=None,
                   help="Override the in-loop producer's min_ships (e.g. 6 = the shipped 62.5% config).")
    p.add_argument("--scaffold", action="store_true",
                   help="Producer-as-scaffold: p0 = frozen producer plan (min6 CONFIG_2P) with the "
                        "policy's fire head deciding per-planet {follow, veto}. Fire-only PPO loss.")
    p.add_argument("--scaffold-fire-bias", type=float, default=2.0,
                   help="One-time fire_head bias boost at init in --scaffold mode (sigmoid(2)~0.88 "
                        "follow rate => step-0 ~= the wrapped producer).")
    p.add_argument("--rich-features", action="store_true",
                   help="Use 23-dim features (A1 threat + time + dist + capture + ship_adv + prod_adv + rotating). Default=11-dim base.")
    p.add_argument("--gravity-features", action="store_true",
                   help="Append 2 dims: local enemy/own ship-mass density (Σ ships/dist^2). 23->25-dim. Requires a 25-dim ckpt (zero-init embed extend from a 23-dim model).")
    p.add_argument("--obs-v2", action="store_true",
                   help="Redesigned 20-dim obs: cut is_neutral/alive/ships_lin/thr_net/is_comet, "
                        "add angular_velocity + prod_density. Overrides --rich-features. For the from-scratch base.")
    p.add_argument("--rotation-aug", action="store_true",
                   help="Rotate the (x,y) obs cols 90/180/270 deg per state at update time (4x data, rotational "
                        "invariance). Use with --obs-v2 (x,y at cols 5,6).")
    p.add_argument("--expansion-shaping", action="store_true",
                   help="C3: potential-based reward shaping F=gamma*Phi'-Phi, Phi=w*controlled-production, "
                        "every step (terminal Copeland unchanged). Policy-invariant; densifies the expansion "
                        "gradient to escape the 4p turtle basin.")
    p.add_argument("--expansion-weight", type=float, default=0.03,
                   help="C3 potential weight w (Phi = w * sum controlled production). Sweep ~0.01-0.1.")
    p.add_argument("--net-asset-shaping", action="store_true",
                   help="2026-06-23 'Net Asset Capture' dense reward (4p melee arm B): "
                        "w1*dProduction - w2*combat_losses per step. NOT potential-based (distorts "
                        "toward expansion/opportunism). Pair with --reward-mode winner.")
    p.add_argument("--nas-prod-weight", type=float, default=0.03,
                   help="w1 for net-asset shaping: reward per unit of production GAINED (capture).")
    p.add_argument("--nas-combat-weight", type=float, default=0.01,
                   help="w2 for net-asset shaping: penalty per ship LOST (total-ship-count drop).")
    p.add_argument("--autoregressive", action="store_true",
                   help="C2: sample target autoregressively over owned planets (ships-desc), "
                        "coupling on committed friendly mass (collision-avoidance). zero-init "
                        "commit_head -> step-0 identical to non-AR.")
    p.add_argument("--orbit-features", action="store_true",
                   help="Append ang_vel + prod_density (obs-v2's probe-gated adds) to RICH -> 25-dim.")
    p.add_argument("--is-4p-feature", action="store_true",
                   help="Append broadcast is_4p (N_PLAYERS==4) AFTER orbit -> matches BC 4p-pack actor (27-dim).")
    p.add_argument("--rank-feature", action="store_true",
                   help="Append my garrison-ship rank-frac AFTER is_4p -> matches BC 4p-pack. Mirrors bc_train_clone._rank_frac.")
    p.add_argument("--roi-features", action="store_true",
                   help="Append 1 dim: capture payoff-time (defenders/prod, log1p). 23->24-dim. Requires a 24-dim ckpt (zero-init embed extend from a 23-dim model).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--probe-iters", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--eval-checkpoint", default="", help="Eval this checkpoint (greedy p0) vs opponent.")
    p.add_argument("--eval-opponent", choices=("random", "self"), default="random")
    p.add_argument("--eval-games", type=int, default=512)
    p.add_argument("--init-checkpoint", default="",
                   help="Warm-start training params from this checkpoint (cumulative training; Adam state resets).")
    p.add_argument("--resume-opt-state", action="store_true",
                   help="Restore the optimizer (Adam moments + step count) from --init-checkpoint for a "
                        "SEAMLESS continuation (no re-warmup transient). Default OFF = reset Adam each leg, "
                        "the validated scale-chain methodology. Only applies if the ckpt has opt_state and "
                        "its tree matches (no fresh-init new params); otherwise warns and resets.")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--checkpoint-interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260531)
    # consumed by jax_train_runtime before argparse; declared here so argparse accepts them.
    p.add_argument("--allow-cpu-training", action="store_true",
                   help="Allow CPU JAX training (smoke tests only).")
    p.add_argument("--cpu-threads", type=int, default=6, help="CPU BLAS/OMP thread cap.")
    return p.parse_args()


def main():
    args = parse_args()
    global PRODUCER_MODULE
    PRODUCER_MODULE = args.producer_module
    global MIN_LAUNCH, RICH_FEATURES, GRAVITY_FEATURES, ROI_FEATURES, OBS_V2, ROTATION_AUG, ORBIT_FEATURES
    global IS4P_FEATURE, RANK_FEATURE, N_PLAYERS
    MIN_LAUNCH = int(args.min_launch)
    RICH_FEATURES = bool(args.rich_features)
    GRAVITY_FEATURES = bool(args.gravity_features)
    ROI_FEATURES = bool(args.roi_features)
    OBS_V2 = bool(args.obs_v2)
    ROTATION_AUG = bool(args.rotation_aug)
    ORBIT_FEATURES = bool(args.orbit_features)
    IS4P_FEATURE = bool(args.is_4p_feature)
    RANK_FEATURE = bool(args.rank_feature)
    N_PLAYERS = int(args.num_players)
    _env.NUM_PLAYERS = int(args.num_players)  # env combat/scoring/launch use this at trace time
    print(f"[lean] devices={jax.devices()} num_players={args.num_players}", flush=True)
    frac_bins = jnp.array([float(v) for v in args.frac_bins.split(",")], dtype=jnp.float32)
    policy = LeanEntityPolicy(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_frac_bins=int(frac_bins.shape[0]),
        frac_stop_gradient=args.frac_stop_gradient,
        fresh_focus_init=args.fresh_focus_init,
        dual_trunk=args.dual_trunk,
        critic_hidden=args.critic_hidden,
        critic_layers=args.critic_layers,
        critic_heads=args.critic_heads,
    )

    if args.map_bank:
        # 4p uses the full-permutation seat-fairness reset; 2p keeps the byte-identical coin-flip path
        _reset_bank = reset_from_bank_4p if args.num_players == 4 else reset_from_bank
        state = _reset_bank(args.map_bank, args.num_envs, args.fleets, args.seed)
    elif args.map_json:
        state = reset_from_map(args.map_json, args.num_envs, args.fleets)
    else:
        state = reset(args.num_envs, args.planets, args.fleets)
    reset_state = state

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = policy.init(init_key, planet_features(state, 0), state.alive)["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[lean] params={n_params/1e3:.1f}K hidden={args.hidden_size} layers={args.num_layers} P={state.owner.shape[1]}", flush=True)
    base_steps = 0
    resume_opt_state = None  # populated below if --resume-opt-state and the ckpt carries a matching opt_state
    if args.init_checkpoint:
        with open(args.init_checkpoint, "rb") as f:
            ick = pickle.load(f)
        # Merge (not replace): overlay checkpoint params onto freshly-init params,
        # keeping init values for keys the checkpoint lacks (e.g. C1 focus_* heads
        # when warm-starting from a pre-C1 checkpoint). Identical to a full replace
        # when the architectures match.
        def merge(init_p, ck_p):
            out = dict(init_p)
            for kk, vv in init_p.items():
                if kk in ck_p:
                    out[kk] = merge(vv, ck_p[kk]) if isinstance(vv, dict) else jnp.asarray(ck_p[kk])
            return out
        new_keys = [k for k in params if k not in ick["params"]]
        params = merge(params, ick["params"])
        base_steps = int(ick.get("update", 0)) * args.num_envs * args.rollout_steps
        if args.resume_opt_state:
            if new_keys:
                print(f"[lean] --resume-opt-state IGNORED: ckpt lacks fresh-init params {new_keys}; "
                      f"opt_state tree would mismatch. Resetting Adam.", flush=True)
            elif ick.get("opt_state") is None:
                print("[lean] --resume-opt-state IGNORED: ckpt has no opt_state (pre-this-feature). "
                      "Resetting Adam.", flush=True)
            else:
                resume_opt_state = ick["opt_state"]
        adam_note = "Adam state RESUMED (seamless)" if resume_opt_state is not None else "Adam state reset"
        print(f"[lean] warm-start from {args.init_checkpoint} @u{ick.get('update','?')} "
              f"(~{base_steps/1e6:.0f}M prior samples; {adam_note})"
              + (f"; fresh-init new params: {new_keys}" if new_keys else ""), flush=True)
        if args.scaffold and args.scaffold_fire_bias:
            fh = dict(params["fire_head"])
            fh["bias"] = fh["bias"] + args.scaffold_fire_bias
            params = {**params, "fire_head": fh}
            resume_opt_state = None  # bias surgery invalidates the Adam tree state semantics
            print(f"[lean] scaffold: fire_head bias +{args.scaffold_fire_bias} "
                  f"(step-0 follow-rate boost); Adam reset.", flush=True)

    n_prod_envs = int(round(args.opponent_producer_frac * args.num_envs))
    if n_prod_envs > 0:
        _opp_desc = "players 1.." + str(args.num_players - 1) if args.num_players == 4 else "player 1"
        print(f"[lean] D4 pool: {_opp_desc} = FROZEN heuristic opponent in envs [0, {n_prod_envs}) "
              f"of {args.num_envs} ({100*args.opponent_producer_frac:.0f}%); those opponent lanes "
              "excluded from the PPO loss.", flush=True)

    # PAST-SELF LEAGUE: load frozen self-ckpts as PLAYER 1 in the first envs (mutually excl. w/ producer pool)
    frozen_opponents = None
    n_league_envs = 0
    if args.league_ckpts:
        assert n_prod_envs == 0, "--league-ckpts and --opponent-producer-frac both use the first envs; pick one"
        _paths = [s for s in args.league_ckpts.split(",") if s]
        _fracs = [float(x) for x in args.league_fracs.split(",")]
        assert len(_paths) == len(_fracs), "--league-ckpts and --league-fracs length mismatch"
        frozen_opponents = []
        _lo = 0
        for _p, _fr in zip(_paths, _fracs):
            _n = int(round(_fr * args.num_envs))
            with open(_p, "rb") as _f:
                _fp = jax.tree_util.tree_map(jnp.asarray, pickle.load(_f)["params"])
            frozen_opponents.append((_fp, _lo, _lo + _n)); _lo += _n
        n_league_envs = _lo
        print(f"[lean] PAST-SELF LEAGUE: p1 = {list(zip(_paths, [(l, h) for _, l, h in frozen_opponents]))} "
              f"of {args.num_envs} envs; rest = current self-play; league p1 lanes excluded from PPO.", flush=True)
    scaffold_cfg = None
    if args.scaffold:
        _JPm = _load_producer()
        scaffold_cfg = _JPm.CONFIG_2P._replace(min_ships=6.0)
        print(f"[lean] SCAFFOLD: p0 = frozen producer plan (min6 h{scaffold_cfg.horizon}) + "
              f"policy fire-head veto; fire-only PPO loss; init fire bias +{args.scaffold_fire_bias}",
              flush=True)
    _mk_rollout = make_rollout_4p if args.num_players == 4 else make_rollout   # 4p: producer in all opp seats
    rollout = _mk_rollout(
        policy, frac_bins, args.episode_steps, args.rollout_steps, gamma=args.gamma,
        num_players=args.num_players, score_mode=args.score_mode, reward_mode=args.reward_mode,
        autoregressive=args.autoregressive,
        expansion_shaping=args.expansion_shaping, expansion_weight=args.expansion_weight,
        net_asset_shaping=args.net_asset_shaping, nas_prod_weight=args.nas_prod_weight,
        nas_combat_weight=args.nas_combat_weight,
        producer_envs=n_prod_envs,
        producer_cfg_overrides={k: v for k, v in (
            ("roi_threshold", args.producer_roi), ("max_waves", args.producer_waves),
            ("horizon", args.producer_horizon), ("min_ships", args.producer_min_ships)) if v is not None} or None,
        producer_levels=[int(x) for x in args.producer_levels.split(",")] if args.producer_levels else None,
        scaffold_cfg=scaffold_cfg,
        frozen_opponents=frozen_opponents,
    )

    if args.eval_checkpoint:
        with open(args.eval_checkpoint, "rb") as f:
            ck = pickle.load(f)
        p0_params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        if args.eval_opponent == "self":
            p1_params = p0_params
            opp_desc = "self(greedy)"
        else:
            key, ok = jax.random.split(key)
            p1_params = policy.init(ok, planet_features(state, 1), state.alive)["params"]
            opp_desc = "random-init"

        @jax.jit
        def play_game(pa, pb, st, key):
            def body(carry, _):
                s, k = carry
                k, k0, k1 = jax.random.split(k, 3)
                m0 = sample_player(pa, policy, s, 0, k0, frac_bins, greedy=True, autoregressive=args.autoregressive)["moves"]
                m1 = sample_player(pb, policy, s, 1, k1, frac_bins, greedy=True, autoregressive=args.autoregressive)["moves"]
                acts = jnp.stack([m0, m1], axis=1)
                ns = step(s, acts, args.episode_steps, actions_are_angles=True)
                return (ns, k), None

            (final, _), _ = jax.lax.scan(body, (st, key), None, length=args.episode_steps)
            return final

        key, gk = jax.random.split(key)
        final = play_game(p0_params, p1_params, state, gk)
        jax.block_until_ready(final.ships)
        s0 = score_by_player(final, 0)
        s1 = score_by_player(final, 1)
        wins = int(jnp.sum(s0 > s1))
        losses = int(jnp.sum(s1 > s0))
        draws = int(jnp.sum(s0 == s1))
        n = args.num_envs
        print(f"[eval] ckpt(greedy p0) vs {opp_desc}: {n} games, {args.episode_steps} steps", flush=True)
        print(f"[eval] W={wins} ({100*wins/n:.1f}%) L={losses} ({100*losses/n:.1f}%) D={draws}", flush=True)
        print(f"[eval] mean ship-diff (p0-p1) = {float(jnp.mean((s0-s1).astype(jnp.float32))):+.1f}", flush=True)
        return

    if args.probe:
        t0 = time.perf_counter()
        print("[probe] compiling + first rollout...", flush=True)
        st, k, traj, _ = rollout(params, state, reset_state, key)
        jax.block_until_ready(traj)
        print(f"[probe] first rollout (incl compile) {time.perf_counter()-t0:.1f}s", flush=True)
        t1 = time.perf_counter()
        for _ in range(args.probe_iters):
            st, k, traj, _ = rollout(params, st, reset_state, k)
        jax.block_until_ready(traj)
        elapsed = time.perf_counter() - t1
        env_steps = args.probe_iters * args.rollout_steps * args.num_envs
        sps = env_steps / elapsed
        print(f"[probe] STEADY: {sps:,.0f} SPS ({env_steps:,} env-steps in {elapsed:.2f}s)", flush=True)
        print(f"[probe] GATE >= 5000 SPS: {'PASS' if sps >= 5000 else 'FAIL'} (target 10000)", flush=True)
        return

    # --- PPO training ---
    if args.dual_trunk:
        # actor/critic dual LR + actor warmup: actor LR=0 for the first --warmup-critic-updates
        # optimizer steps so the fresh critic learns to read the board BEFORE its (initially garbage)
        # advantage gradients reach the actor. Then actor LR opens at args.actor_lr (BC 1/10~1/20).
        import flax
        actor_sched = optax.join_schedules(
            [optax.constant_schedule(0.0), optax.constant_schedule(args.actor_lr)],
            [max(args.warmup_critic_updates, 1)])
        def _label(params):
            flat = flax.traverse_util.flatten_dict(params)
            return flax.traverse_util.unflatten_dict(
                {k: ("critic" if k[0].startswith("c_") else "actor") for k in flat})
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.multi_transform(
                {"actor": optax.adamw(actor_sched, weight_decay=args.weight_decay),
                 "critic": optax.adamw(args.critic_lr, weight_decay=args.weight_decay)},
                _label),
        )
        print(f"[lean] DUAL-TRUNK opt: actor_lr={args.actor_lr} (warmup {args.warmup_critic_updates} opt-steps @0) "
              f"critic_lr={args.critic_lr}", flush=True)
    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(args.lr, weight_decay=args.weight_decay),
        )
    opt_state = optimizer.init(params)
    if resume_opt_state is not None:
        fresh_struct = jax.tree_util.tree_structure(opt_state)
        loaded = jax.tree_util.tree_map(jnp.asarray, resume_opt_state)
        if jax.tree_util.tree_structure(loaded) == fresh_struct:
            opt_state = loaded
            print("[lean] optimizer state restored from checkpoint (continuation)", flush=True)
        else:
            print("[lean] --resume-opt-state IGNORED: opt_state tree mismatch vs current optimizer; "
                  "using fresh Adam.", flush=True)
    # Stage B: load frozen BC ref for KL anchor (default reuse --init-checkpoint = the L4 BC we warm from)
    bc_ref = None
    if args.kl_coef > 0.0:
        _ref_path = args.bc_ref_ckpt or args.init_checkpoint
        assert _ref_path, "--kl-coef>0 needs --bc-ref-ckpt or --init-checkpoint"
        with open(_ref_path, "rb") as _f:
            bc_ref = jax.tree_util.tree_map(jnp.asarray, pickle.load(_f)["params"])
        print(f"[lean] KL anchor ON: kl_coef={args.kl_coef} ref={_ref_path}", flush=True)
    train_step = make_train_step(policy, args.clip_coef, args.value_coef, args.entropy_coef, optimizer,
                                 ent_fire_coef=args.ent_fire_coef, ent_focus_coef=args.ent_focus_coef,
                                 autoregressive=args.autoregressive, frac_bins=frac_bins,
                                 ent_frac_coef=args.ent_frac_coef, scaffold=args.scaffold,
                                 bc_ref_params=bc_ref, kl_coef=args.kl_coef)

    T, B = args.rollout_steps, args.num_envs
    print(
        f"[train] updates={args.updates} num_envs={B} rollout={T} epochs={args.epochs} "
        f"minibatch={args.minibatch_size} lr={args.lr} ent={args.entropy_coef} entFire={args.ent_fire_coef} gamma={args.gamma}",
        flush=True,
    )
    print(
        "[train] update | EV | KL | clipfrac | ent | v_loss | p_loss | mean_R | gnorm | sps | s",
        flush=True,
    )

    # D4 producer lanes: static keep-index dropping (env < n_prod_envs, player==1)
    # samples from the flattened PPO data (their stored logp/value describe actions
    # the producer overrode — garbage for the loss; GAE is per-lane so other lanes
    # are unaffected, and p0's lanes in those envs carry the beat-producer reward).
    KEEP_IDX = None
    lane_keep = None
    if n_prod_envs > 0 or n_league_envs > 0:
        _keep = np.ones((B, args.num_players), bool)
        if n_prod_envs > 0:
            if args.num_players == 4:
                _keep[:n_prod_envs, 1:] = False     # 4p: drop ALL opponent seats (1v3 producer envs)
            else:
                _keep[:n_prod_envs, 1] = False       # 2p: unchanged

        if n_league_envs > 0:                                             # league: p1 is frozen, drop those lanes
            _keep[:n_league_envs, 1] = False
        lane_keep = jnp.asarray(_keep)                                     # (B,NP)
        _keep_flat = np.tile(_keep[None], (T, 1, 1)).reshape(-1)
        KEEP_IDX = jnp.asarray(np.nonzero(_keep_flat)[0])

    run_start = time.perf_counter()
    for update in range(1, args.updates + 1):
        u0 = time.perf_counter()
        state, key, traj, next_value = rollout(params, state, reset_state, key)

        # flatten (T,B,NP,...) -> trajectories per (env,player); GAE on (T, M) with M=B*NP
        NP = args.num_players
        rewards = traj["reward"].reshape(T, B * NP)
        values = traj["value"].reshape(T, B * NP)
        dones = jnp.broadcast_to(traj["done"][:, :, None], (T, B, NP)).reshape(T, B * NP)
        nv = next_value.reshape(B * NP)
        adv_TM, ret_TM = compute_gae(rewards, values, dones, nv, args.gamma, args.gae_lambda)

        # flatten everything to N = T*B*NP samples
        def flat(x):
            return x.reshape((T * B * NP,) + x.shape[3:])

        feats = flat(traj["feats"])
        if ROTATION_AUG:
            key, _rk = jax.random.split(key)
            feats = rotate_xy(feats, _rk)  # random k*90deg on (x,y) cols per state (obs-v2: cols 5,6)
        alive = flat(traj["alive"])
        owned = flat(traj["owned"])
        fire = flat(traj["fire"])
        target_idx = flat(traj["target_idx"])
        frac_idx = flat(traj["frac_idx"])
        focus_idx = flat(traj["focus_idx"])
        focus_valid = flat(traj["focus_valid"])
        ships_flat = flat(traj["ships"])
        old_logp = traj["logp"].reshape(T * B * NP)
        adv = adv_TM.reshape(T * B * NP)
        ret = ret_TM.reshape(T * B * NP)
        if KEEP_IDX is not None:
            feats = feats[KEEP_IDX]; alive = alive[KEEP_IDX]; owned = owned[KEEP_IDX]
            fire = fire[KEEP_IDX]; target_idx = target_idx[KEEP_IDX]; frac_idx = frac_idx[KEEP_IDX]
            focus_idx = focus_idx[KEEP_IDX]; focus_valid = focus_valid[KEEP_IDX]
            ships_flat = ships_flat[KEEP_IDX]; old_logp = old_logp[KEEP_IDX]
            adv = adv[KEEP_IDX]; ret = ret[KEEP_IDX]
        if args.norm_adv:
            adv = (adv - jnp.mean(adv)) / (jnp.std(adv) + 1e-8)

        N = int(adv.shape[0])
        mb = min(args.minibatch_size, N)
        n_mb = N // mb
        acc = {kk: 0.0 for kk in ("ev", "kl", "clip", "ent", "vloss", "ploss", "gnorm", "entf", "entfoc", "entfr", "klanc")}
        steps = 0
        for _ in range(args.epochs):
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, N)
            for i in range(n_mb):
                idx = perm[i * mb : (i + 1) * mb]
                params, opt_state, loss, aux, gnorm = train_step(
                    params, opt_state,
                    feats[idx], alive[idx], owned[idx], fire[idx], target_idx[idx], frac_idx[idx],
                    focus_idx[idx], focus_valid[idx], ships_flat[idx], old_logp[idx], adv[idx], ret[idx],
                )
                ploss, vloss, ent, kl, clipf, ev, entf, entfoc, entfr, kl_anchor = aux
                acc["ev"] += float(ev); acc["kl"] += float(kl); acc["clip"] += float(clipf)
                acc["ent"] += float(ent); acc["vloss"] += float(vloss); acc["ploss"] += float(ploss)
                acc["gnorm"] += float(gnorm); acc["entf"] += float(entf); acc["entfoc"] += float(entfoc)
                acc["entfr"] += float(entfr); acc["klanc"] += float(kl_anchor)
                steps += 1

        for kk in acc:
            acc[kk] /= max(steps, 1)
        done_rate = float(jnp.mean(traj["done"]))
        abs_r = float(jnp.mean(jnp.abs(traj["reward"])))
        if lane_keep is not None:  # exclude producer lanes (their sampled fire stats are unused)
            lk = lane_keep[None, :, :, None]                               # (1,B,NP,1)
            fire_rate = float(jnp.sum(traj["fire"] * lk) / jnp.maximum(jnp.sum(traj["owned"] * lk), 1))
        else:
            fire_rate = float(jnp.sum(traj["fire"]) / jnp.maximum(jnp.sum(traj["owned"]), 1))
        dt = time.perf_counter() - u0
        sps = (T * B) / dt
        if update % args.log_interval == 0 or update == 1 or update == args.updates:
            print(
                f"[u{update:04d}] EV={acc['ev']:+.3f} KL={acc['kl']:.4f} clip={acc['clip']:.3f} "
                f"ent={acc['ent']:.3f} entF={acc['entf']:.3f}/0.69 entFoc={acc['entfoc']:.3f} entFr={acc['entfr']:.3f} KLanc={acc['klanc']:.4f} vL={acc['vloss']:.4f} pL={acc['ploss']:+.4f} "
                f"fire={fire_rate:.3f} done={done_rate:.3f} absR={abs_r:.4f} gn={acc['gnorm']:.2f} sps={sps:,.0f} {dt:.2f}s",
                flush=True,
            )

        if args.checkpoint and args.checkpoint_interval and update % args.checkpoint_interval == 0:
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            with open(args.checkpoint, "wb") as f:
                pickle.dump({"params": jax.device_get(params), "opt_state": jax.device_get(opt_state),
                             "config": vars(args), "update": update}, f)
            print(f"[ckpt] saved {args.checkpoint} @ u{update}", flush=True)

    print(f"[train] done {args.updates} updates in {time.perf_counter()-run_start:.1f}s", flush=True)
    if args.checkpoint:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        with open(args.checkpoint, "wb") as f:
            pickle.dump({"params": jax.device_get(params), "opt_state": jax.device_get(opt_state),
                         "config": vars(args), "update": args.updates}, f)
        print(f"[ckpt] final saved {args.checkpoint}", flush=True)


if __name__ == "__main__":
    main()
