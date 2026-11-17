import argparse
import json
import time
from typing import NamedTuple
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BOARD_CENTER = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
COMET_RADIUS = 1.0
COMET_PRODUCTION = 1
COMET_QUADRANTS = 4
# Number of players in the game (2 or 4). Set before jit-tracing step/rollout,
# same pattern as MIN_LAUNCH/RICH_FEATURES. Combat (phase_i_combat) and scoring are
# generic over this; the per-game count is fixed. 2p stays bit-identical when =2.
NUM_PLAYERS = 2


FIXTURE_PLANETS = jnp.array(
    [
        [0, 0, 20.0, 50.0, 3.0, 50, 2],
        [1, -1, 35.0, 50.0, 3.0, 10, 3],
        [2, -1, 50.0, 50.0, 3.0, 10, 1],
        [3, 1, 80.0, 50.0, 3.0, 50, 2],
    ],
    dtype=jnp.float32,
)
ROTATION_FIXTURE_PLANETS = jnp.array(
    [
        [0, 0, 70.0, 50.0, 3.0, 50, 2],
        [1, -1, 50.0, 70.0, 3.0, 10, 3],
        [2, -1, 50.0, 50.0, 3.0, 10, 1],
        [3, 1, 95.0, 50.0, 3.0, 50, 2],
    ],
    dtype=jnp.float32,
)
REINFORCE_FIXTURE_PLANETS = jnp.array(
    [
        [0, 0, 20.0, 50.0, 3.0, 50, 2],
        [1, 0, 35.0, 50.0, 3.0, 10, 3],
        [2, -1, 50.0, 50.0, 3.0, 10, 1],
        [3, 1, 80.0, 50.0, 3.0, 50, 2],
    ],
    dtype=jnp.float32,
)
MULTI_HIT_FIXTURE_PLANETS = jnp.array(
    [
        [0, 0, 38.0, 50.0, 1.0, 50, 2],
        [1, -1, 43.0, 50.0, 2.0, 10, 1],
        [2, -1, 45.0, 50.0, 2.0, 10, 1],
        [3, 1, 80.0, 50.0, 3.0, 50, 2],
    ],
    dtype=jnp.float32,
)


class JaxOrbitState(NamedTuple):
    planet_id: jax.Array
    owner: jax.Array
    x: jax.Array
    y: jax.Array
    radius: jax.Array
    ships: jax.Array
    production: jax.Array
    initial_x: jax.Array
    initial_y: jax.Array
    angular_velocity: jax.Array
    alive: jax.Array
    is_comet: jax.Array
    next_planet_id: jax.Array
    next_planet_slot: jax.Array
    comet_paths: jax.Array
    comet_path_len: jax.Array
    comet_spawn_step: jax.Array
    comet_group_ships: jax.Array
    comet_group_defined: jax.Array
    comet_group_spawned: jax.Array
    comet_group_alive: jax.Array
    comet_group_path_idx: jax.Array
    comet_group_planet_slot: jax.Array
    fleet_owner: jax.Array
    fleet_x: jax.Array
    fleet_y: jax.Array
    fleet_angle: jax.Array
    fleet_ships: jax.Array
    fleet_alive: jax.Array
    next_fleet: jax.Array
    step_count: jax.Array
    done: jax.Array
    rewards: jax.Array


def reset(batch_size: int, planets: int, fleets: int) -> JaxOrbitState:
    idx = jnp.arange(planets)
    angle = 2.0 * jnp.pi * idx / planets
    ring = 32.0
    x_row = 50.0 + ring * jnp.cos(angle)
    y_row = 50.0 + ring * jnp.sin(angle)
    radius_row = jnp.full((planets,), 3.0, dtype=jnp.float32)
    production_row = (idx % 5 + 1).astype(jnp.int32)
    owner_row = jnp.full((planets,), -1, dtype=jnp.int32)
    owner_row = owner_row.at[0].set(0)
    owner_row = owner_row.at[planets // 2].set(1)
    ships_row = jnp.full((planets,), 20, dtype=jnp.int32)
    ships_row = ships_row.at[0].set(50)
    ships_row = ships_row.at[planets // 2].set(50)
    alive_row = jnp.ones((planets,), dtype=bool)
    groups = 0
    max_path = 1
    return JaxOrbitState(
        planet_id=jnp.tile(idx.astype(jnp.int32)[None, :], (batch_size, 1)),
        owner=jnp.tile(owner_row[None, :], (batch_size, 1)),
        x=jnp.tile(x_row[None, :], (batch_size, 1)),
        y=jnp.tile(y_row[None, :], (batch_size, 1)),
        radius=jnp.tile(radius_row[None, :], (batch_size, 1)),
        ships=jnp.tile(ships_row[None, :], (batch_size, 1)),
        production=jnp.tile(production_row[None, :], (batch_size, 1)),
        initial_x=jnp.tile(x_row[None, :], (batch_size, 1)),
        initial_y=jnp.tile(y_row[None, :], (batch_size, 1)),
        angular_velocity=jnp.full((batch_size,), 0.0, dtype=jnp.float32),
        alive=jnp.tile(alive_row[None, :], (batch_size, 1)),
        is_comet=jnp.zeros((batch_size, planets), dtype=bool),
        next_planet_id=jnp.full((batch_size,), planets, dtype=jnp.int32),
        next_planet_slot=jnp.full((batch_size,), planets, dtype=jnp.int32),
        comet_paths=jnp.zeros((batch_size, groups, COMET_QUADRANTS, max_path, 2), dtype=jnp.float32),
        comet_path_len=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_spawn_step=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_ships=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_defined=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_spawned=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_alive=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_path_idx=jnp.full((batch_size, groups), -1, dtype=jnp.int32),
        comet_group_planet_slot=jnp.full((batch_size, groups, COMET_QUADRANTS), -1, dtype=jnp.int32),
        fleet_owner=jnp.full((batch_size, fleets), -2, dtype=jnp.int32),
        fleet_x=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_y=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_angle=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_ships=jnp.zeros((batch_size, fleets), dtype=jnp.int32),
        fleet_alive=jnp.zeros((batch_size, fleets), dtype=bool),
        next_fleet=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        rewards=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32),
    )


def reset_fixture(batch_size: int, fleets: int, fixture_kind: str = "base", angular_velocity: float = 0.0) -> JaxOrbitState:
    fixture = FIXTURE_PLANETS
    if fixture_kind == "rotation":
        fixture = ROTATION_FIXTURE_PLANETS
    elif fixture_kind == "reinforce":
        fixture = REINFORCE_FIXTURE_PLANETS
    elif fixture_kind == "multi_hit":
        fixture = MULTI_HIT_FIXTURE_PLANETS
    planets = int(fixture.shape[0])
    groups = 0
    max_path = 1
    return JaxOrbitState(
        planet_id=jnp.tile(fixture[:, 0].astype(jnp.int32)[None, :], (batch_size, 1)),
        owner=jnp.tile(fixture[:, 1].astype(jnp.int32)[None, :], (batch_size, 1)),
        x=jnp.tile(fixture[:, 2][None, :], (batch_size, 1)),
        y=jnp.tile(fixture[:, 3][None, :], (batch_size, 1)),
        radius=jnp.tile(fixture[:, 4][None, :], (batch_size, 1)),
        ships=jnp.tile(fixture[:, 5].astype(jnp.int32)[None, :], (batch_size, 1)),
        production=jnp.tile(fixture[:, 6].astype(jnp.int32)[None, :], (batch_size, 1)),
        initial_x=jnp.tile(fixture[:, 2][None, :], (batch_size, 1)),
        initial_y=jnp.tile(fixture[:, 3][None, :], (batch_size, 1)),
        angular_velocity=jnp.full((batch_size,), float(angular_velocity), dtype=jnp.float32),
        alive=jnp.ones((batch_size, planets), dtype=bool),
        is_comet=jnp.zeros((batch_size, planets), dtype=bool),
        next_planet_id=jnp.full((batch_size,), planets, dtype=jnp.int32),
        next_planet_slot=jnp.full((batch_size,), planets, dtype=jnp.int32),
        comet_paths=jnp.zeros((batch_size, groups, COMET_QUADRANTS, max_path, 2), dtype=jnp.float32),
        comet_path_len=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_spawn_step=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_ships=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_defined=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_spawned=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_alive=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_path_idx=jnp.full((batch_size, groups), -1, dtype=jnp.int32),
        comet_group_planet_slot=jnp.full((batch_size, groups, COMET_QUADRANTS), -1, dtype=jnp.int32),
        fleet_owner=jnp.full((batch_size, fleets), -2, dtype=jnp.int32),
        fleet_x=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_y=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_angle=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_ships=jnp.zeros((batch_size, fleets), dtype=jnp.int32),
        fleet_alive=jnp.zeros((batch_size, fleets), dtype=bool),
        next_fleet=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        rewards=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32),
    )


def reset_from_map(path: str, batch_size: int, fleets: int) -> JaxOrbitState:
    payload = json.loads(Path(path).read_text())
    base_rows = payload["planets"]
    comet_groups = payload.get("comet_groups", [])
    groups = len(comet_groups)
    max_path = max([1] + [int(group.get("path_len", 0)) for group in comet_groups])
    base_planets = len(base_rows)
    planets = base_planets + groups * COMET_QUADRANTS

    planet_id_row = [-1 for _ in range(planets)]
    owner_row = [-1 for _ in range(planets)]
    x_row = [-99.0 for _ in range(planets)]
    y_row = [-99.0 for _ in range(planets)]
    radius_row = [COMET_RADIUS for _ in range(planets)]
    ships_row = [0 for _ in range(planets)]
    production_row = [0 for _ in range(planets)]
    alive_row = [False for _ in range(planets)]
    is_comet_row = [False for _ in range(planets)]

    for idx, row in enumerate(base_rows):
        planet_id_row[idx] = int(row[0])
        owner_row[idx] = int(row[1])
        x_row[idx] = float(row[2])
        y_row[idx] = float(row[3])
        radius_row[idx] = float(row[4])
        ships_row[idx] = int(row[5])
        production_row[idx] = int(row[6])
        alive_row[idx] = True

    comet_paths = jnp.zeros((groups, COMET_QUADRANTS, max_path, 2), dtype=jnp.float32)
    comet_path_len = jnp.zeros((groups,), dtype=jnp.int32)
    comet_spawn_step = jnp.zeros((groups,), dtype=jnp.int32)
    comet_group_ships = jnp.zeros((groups,), dtype=jnp.int32)
    comet_group_defined = jnp.zeros((groups,), dtype=bool)
    for group_idx, group in enumerate(comet_groups):
        group_paths = jnp.array(group["paths"], dtype=jnp.float32)
        path_len = int(group.get("path_len", group_paths.shape[1]))
        comet_paths = comet_paths.at[group_idx, :, :path_len, :].set(group_paths[:, :path_len, :])
        comet_path_len = comet_path_len.at[group_idx].set(path_len)
        comet_spawn_step = comet_spawn_step.at[group_idx].set(int(group["spawn_step"]))
        comet_group_ships = comet_group_ships.at[group_idx].set(int(group["comet_ships"]))
        comet_group_defined = comet_group_defined.at[group_idx].set(True)

    next_planet_id = max(int(row[0]) for row in base_rows) + 1 if base_rows else 0
    return JaxOrbitState(
        planet_id=jnp.tile(jnp.array(planet_id_row, dtype=jnp.int32)[None, :], (batch_size, 1)),
        owner=jnp.tile(jnp.array(owner_row, dtype=jnp.int32)[None, :], (batch_size, 1)),
        x=jnp.tile(jnp.array(x_row, dtype=jnp.float32)[None, :], (batch_size, 1)),
        y=jnp.tile(jnp.array(y_row, dtype=jnp.float32)[None, :], (batch_size, 1)),
        radius=jnp.tile(jnp.array(radius_row, dtype=jnp.float32)[None, :], (batch_size, 1)),
        ships=jnp.tile(jnp.array(ships_row, dtype=jnp.int32)[None, :], (batch_size, 1)),
        production=jnp.tile(jnp.array(production_row, dtype=jnp.int32)[None, :], (batch_size, 1)),
        initial_x=jnp.tile(jnp.array(x_row, dtype=jnp.float32)[None, :], (batch_size, 1)),
        initial_y=jnp.tile(jnp.array(y_row, dtype=jnp.float32)[None, :], (batch_size, 1)),
        angular_velocity=jnp.full((batch_size,), float(payload["angular_velocity"]), dtype=jnp.float32),
        alive=jnp.tile(jnp.array(alive_row, dtype=bool)[None, :], (batch_size, 1)),
        is_comet=jnp.tile(jnp.array(is_comet_row, dtype=bool)[None, :], (batch_size, 1)),
        next_planet_id=jnp.full((batch_size,), next_planet_id, dtype=jnp.int32),
        next_planet_slot=jnp.full((batch_size,), base_planets, dtype=jnp.int32),
        comet_paths=jnp.tile(comet_paths[None, :, :, :, :], (batch_size, 1, 1, 1, 1)),
        comet_path_len=jnp.tile(comet_path_len[None, :], (batch_size, 1)),
        comet_spawn_step=jnp.tile(comet_spawn_step[None, :], (batch_size, 1)),
        comet_group_ships=jnp.tile(comet_group_ships[None, :], (batch_size, 1)),
        comet_group_defined=jnp.tile(comet_group_defined[None, :], (batch_size, 1)),
        comet_group_spawned=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_alive=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_path_idx=jnp.full((batch_size, groups), -1, dtype=jnp.int32),
        comet_group_planet_slot=jnp.full((batch_size, groups, COMET_QUADRANTS), -1, dtype=jnp.int32),
        fleet_owner=jnp.full((batch_size, fleets), -2, dtype=jnp.int32),
        fleet_x=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_y=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_angle=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_ships=jnp.zeros((batch_size, fleets), dtype=jnp.int32),
        fleet_alive=jnp.zeros((batch_size, fleets), dtype=bool),
        next_fleet=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        rewards=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32),
    )


def reset_from_bank_4p(bank_path: str, batch_size: int, fleets: int, seed: int = 0) -> JaxOrbitState:
    """4p-DEDICATED bank reset (2026-06-20). IDENTICAL to reset_from_bank EXCEPT seat fairness:
    relabel owners 0..NUM_PLAYERS-1 by a RANDOM PER-ENV PERMUTATION (not the 2p 0<->1 coin-flip),
    so every seat index sees every starting position equally in 4p training (the 2p version left
    4p slots 2/3 untouched = seat bias). Home positions unchanged; only owner LABEL relabeled;
    rewards are per-index symmetric so this is fair. ⚠️ Use ONLY for 4p; the 2p path keeps the
    byte-identical reset_from_bank.
    """
    data = np.load(bank_path)
    planets, alive_b, av_b, counts_b = data["planets"], data["alive"], data["av"], data["counts"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, planets.shape[0], size=batch_size)
    sel, sal, sav, scnt = planets[idx], alive_b[idx], av_b[idx], counts_b[idx]
    P = planets.shape[1]

    pid = sel[..., 0].astype(np.int32)
    owner = np.where(sal, sel[..., 1].astype(np.int32), -1)
    # full N-player permutation (vs reset_from_bank's 2p coin-flip)
    perms = np.argsort(rng.random((batch_size, NUM_PLAYERS)), axis=1).astype(np.int32)   # (B,N) per-env perm
    new_owner = np.take_along_axis(perms, np.clip(owner, 0, NUM_PLAYERS - 1), axis=1)     # relabel via perm
    owner = np.where(owner >= 0, new_owner, owner)                                        # keep neutral/dead (-1)
    x = sel[..., 2].astype(np.float32)
    y = sel[..., 3].astype(np.float32)
    radius = sel[..., 4].astype(np.float32)
    ships = sel[..., 5].astype(np.int32)
    prod = sel[..., 6].astype(np.int32)
    groups, max_path = 0, 1
    return JaxOrbitState(
        planet_id=jnp.asarray(pid),
        owner=jnp.asarray(owner),
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        radius=jnp.asarray(radius),
        ships=jnp.asarray(ships),
        production=jnp.asarray(prod),
        initial_x=jnp.asarray(x),
        initial_y=jnp.asarray(y),
        angular_velocity=jnp.asarray(sav.astype(np.float32)),
        alive=jnp.asarray(sal),
        is_comet=jnp.zeros((batch_size, P), dtype=bool),
        next_planet_id=jnp.asarray(scnt.astype(np.int32)),
        next_planet_slot=jnp.asarray(scnt.astype(np.int32)),
        comet_paths=jnp.zeros((batch_size, groups, COMET_QUADRANTS, max_path, 2), dtype=jnp.float32),
        comet_path_len=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_spawn_step=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_ships=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_defined=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_spawned=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_alive=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_path_idx=jnp.full((batch_size, groups), -1, dtype=jnp.int32),
        comet_group_planet_slot=jnp.full((batch_size, groups, COMET_QUADRANTS), -1, dtype=jnp.int32),
        fleet_owner=jnp.full((batch_size, fleets), -2, dtype=jnp.int32),
        fleet_x=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_y=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_angle=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_ships=jnp.zeros((batch_size, fleets), dtype=jnp.int32),
        fleet_alive=jnp.zeros((batch_size, fleets), dtype=bool),
        next_fleet=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        rewards=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32),
    )


def reset_from_bank(bank_path: str, batch_size: int, fleets: int, seed: int = 0) -> JaxOrbitState:
    """Sample `batch_size` maps (with replacement) from a kaggle map bank npz.

    Each env gets a (possibly) different kaggle-distribution map padded to the
    bank's fixed Pmax. Dead padding slots: owner=-1, alive=False (masked
    everywhere). No comets (groups=0); comet support is a later delta.
    """
    data = np.load(bank_path)
    planets, alive_b, av_b, counts_b = data["planets"], data["alive"], data["av"], data["counts"]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, planets.shape[0], size=batch_size)
    sel, sal, sav, scnt = planets[idx], alive_b[idx], av_b[idx], counts_b[idx]
    P = planets.shape[1]

    pid = sel[..., 0].astype(np.int32)
    owner = np.where(sal, sel[..., 1].astype(np.int32), -1)
    # Seat fairness (2026-06-12): the bank's slot-1 spawn is systematically better
    # (model/producer mirrors: P1 wins 64% of decided games; owner-swap flips it to 36%,
    # kaggle itself is balanced). Coin-flip the 0<->1 labels per env so neither player
    # index inherits the positional edge. Deterministic given `seed`. 2p labels only;
    # owners >= 2 (4p maps) are left untouched.
    swap = (rng.random(batch_size) < 0.5)[:, None]
    owner = np.where(swap & (owner == 0), 1, np.where(swap & (owner == 1), 0, owner))
    x = sel[..., 2].astype(np.float32)
    y = sel[..., 3].astype(np.float32)
    radius = sel[..., 4].astype(np.float32)
    ships = sel[..., 5].astype(np.int32)
    prod = sel[..., 6].astype(np.int32)
    groups, max_path = 0, 1
    return JaxOrbitState(
        planet_id=jnp.asarray(pid),
        owner=jnp.asarray(owner),
        x=jnp.asarray(x),
        y=jnp.asarray(y),
        radius=jnp.asarray(radius),
        ships=jnp.asarray(ships),
        production=jnp.asarray(prod),
        initial_x=jnp.asarray(x),
        initial_y=jnp.asarray(y),
        angular_velocity=jnp.asarray(sav.astype(np.float32)),
        alive=jnp.asarray(sal),
        is_comet=jnp.zeros((batch_size, P), dtype=bool),
        next_planet_id=jnp.asarray(scnt.astype(np.int32)),
        next_planet_slot=jnp.asarray(scnt.astype(np.int32)),
        comet_paths=jnp.zeros((batch_size, groups, COMET_QUADRANTS, max_path, 2), dtype=jnp.float32),
        comet_path_len=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_spawn_step=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_ships=jnp.zeros((batch_size, groups), dtype=jnp.int32),
        comet_group_defined=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_spawned=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_alive=jnp.zeros((batch_size, groups), dtype=bool),
        comet_group_path_idx=jnp.full((batch_size, groups), -1, dtype=jnp.int32),
        comet_group_planet_slot=jnp.full((batch_size, groups, COMET_QUADRANTS), -1, dtype=jnp.int32),
        fleet_owner=jnp.full((batch_size, fleets), -2, dtype=jnp.int32),
        fleet_x=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_y=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_angle=jnp.zeros((batch_size, fleets), dtype=jnp.float32),
        fleet_ships=jnp.zeros((batch_size, fleets), dtype=jnp.int32),
        fleet_alive=jnp.zeros((batch_size, fleets), dtype=bool),
        next_fleet=jnp.zeros((batch_size,), dtype=jnp.int32),
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        done=jnp.zeros((batch_size,), dtype=bool),
        rewards=jnp.zeros((batch_size, NUM_PLAYERS), dtype=jnp.int32),
    )


def first_planet_id_by_owner(state: JaxOrbitState, owner: int) -> jax.Array:
    batch = state.owner.shape[0]
    slots = jnp.arange(state.owner.shape[1])[None, :]
    found = (state.owner == owner).any(axis=1)
    first_slot = jnp.where(state.owner == owner, slots, state.owner.shape[1]).argmin(axis=1).astype(jnp.int32)
    env_idx = jnp.arange(batch)
    fallback = state.planet_id[env_idx, 0]
    return jnp.where(found, state.planet_id[env_idx, first_slot], fallback)


def first_planet_id_matching(state: JaxOrbitState, mask: jax.Array, fallback: jax.Array) -> jax.Array:
    batch = state.owner.shape[0]
    slots = jnp.arange(state.owner.shape[1])[None, :]
    found = mask.any(axis=1)
    first_slot = jnp.where(mask, slots, state.owner.shape[1]).argmin(axis=1).astype(jnp.int32)
    env_idx = jnp.arange(batch)
    picked = state.planet_id[env_idx, first_slot]
    return jnp.where(found, picked, fallback)


def make_actions(state: JaxOrbitState, mode: str, launch_ships: int, max_moves_per_player: int = 1) -> jax.Array:
    batch = state.owner.shape[0]
    if mode == "noop":
        return jnp.zeros((batch, 2, max_moves_per_player, 3), dtype=jnp.int32)
    if mode == "comet_attack":
        launch_cap = jnp.full((batch,), launch_ships, dtype=jnp.int32)
        out = []
        env_idx = jnp.arange(batch)
        for player in range(NUM_PLAYERS):
            valid_src = state.alive & (state.owner == player) & (state.ships >= launch_ships)
            slot_range = jnp.arange(state.owner.shape[1])[None, :]
            src_rank = jnp.where(valid_src, slot_range, state.owner.shape[1])
            src_sorted = jnp.argsort(src_rank, axis=1).astype(jnp.int32)
            player_rows = []
            move_limit = min(max_moves_per_player, state.owner.shape[1])
            for move_idx in range(move_limit):
                src_slot = src_sorted[:, move_idx]
                has_src = valid_src[env_idx, src_slot]
                src_x = state.x[env_idx, src_slot]
                src_y = state.y[env_idx, src_slot]
                enemy_mask = state.alive & (state.owner != player)
                comet_enemy_mask = enemy_mask & state.is_comet
                has_comet_enemy = comet_enemy_mask.any(axis=1)
                target_mask = jnp.where(has_comet_enemy[:, None], comet_enemy_mask, enemy_mask)
                has_target = target_mask.any(axis=1)
                dx = state.x - src_x[:, None]
                dy = state.y - src_y[:, None]
                dist2 = dx * dx + dy * dy
                dist2 = jnp.where(target_mask, dist2, jnp.inf)
                tgt_slot = jnp.argmin(dist2, axis=1).astype(jnp.int32)
                launch_ok = has_src & has_target
                src_id = state.planet_id[env_idx, src_slot]
                tgt_id = state.planet_id[env_idx, tgt_slot]
                ships = jnp.where(launch_ok, launch_cap, 0)
                player_rows.append(jnp.stack([src_id, tgt_id, ships], axis=1))
            while len(player_rows) < max_moves_per_player:
                player_rows.append(jnp.stack([jnp.full((batch,), -1, dtype=jnp.int32), jnp.full((batch,), -1, dtype=jnp.int32), jnp.zeros((batch,), dtype=jnp.int32)], axis=1))
            out.append(jnp.stack(player_rows, axis=1))
        return jnp.stack(out, axis=1).transpose((0, 1, 2, 3))
    slot_range = jnp.arange(state.owner.shape[1])[None, :]
    env_idx = jnp.arange(batch)
    launch_cap = jnp.full((batch,), launch_ships, dtype=jnp.int32)
    out = []
    for player in range(NUM_PLAYERS):
        valid_src = state.alive & (state.owner == player) & (state.ships >= launch_ships)
        src_rank = jnp.where(valid_src, slot_range, state.owner.shape[1])
        src_sorted = jnp.argsort(src_rank, axis=1).astype(jnp.int32)
        player_rows = []
        move_limit = min(max_moves_per_player, state.owner.shape[1])
        for move_idx in range(move_limit):
            src_slot = src_sorted[:, move_idx]
            has_src = valid_src[env_idx, src_slot]
            src_x = state.x[env_idx, src_slot]
            src_y = state.y[env_idx, src_slot]
            enemy_mask = state.alive & (state.owner != player)
            dx = state.x - src_x[:, None]
            dy = state.y - src_y[:, None]
            dist2 = dx * dx + dy * dy
            dist2 = jnp.where(enemy_mask, dist2, jnp.inf)
            tgt_slot = jnp.argmin(dist2, axis=1).astype(jnp.int32)
            has_target = enemy_mask.any(axis=1)
            launch_ok = has_src & has_target
            src_id = state.planet_id[env_idx, src_slot]
            tgt_id = state.planet_id[env_idx, tgt_slot]
            ships = jnp.where(launch_ok, launch_cap, 0)
            if player == 1 and mode in ("capture", "terminal"):
                ships = jnp.zeros((batch,), dtype=jnp.int32)
            player_rows.append(jnp.stack([src_id, tgt_id, ships], axis=1))
        while len(player_rows) < max_moves_per_player:
            player_rows.append(jnp.stack([jnp.full((batch,), -1, dtype=jnp.int32), jnp.full((batch,), -1, dtype=jnp.int32), jnp.zeros((batch,), dtype=jnp.int32)], axis=1))
        out.append(jnp.stack(player_rows, axis=1))
    return jnp.stack(out, axis=1).transpose((0, 1, 2, 3))


def make_fixture_actions(state: JaxOrbitState, mode: str, launch_ships: int, max_moves_per_player: int = 1) -> jax.Array:
    batch = state.owner.shape[0]
    if mode == "noop":
        return jnp.zeros((batch, 2, max_moves_per_player, 3), dtype=jnp.int32)
    home0 = first_planet_id_by_owner(state, 0)
    home1 = first_planet_id_by_owner(state, 1)
    target0 = first_planet_id_matching(state, state.owner != 0, home0)
    target1 = first_planet_id_matching(state, state.owner != 1, home1)
    if mode in ("enemy_capture", "sun_collision", "multi_planet_intersection", "rotating_target_collision"):
        target0 = home1
    if mode == "friendly_reinforce":
        target0 = first_planet_id_matching(state, (state.owner == 0) & (state.planet_id != home0[:, None]), home0)
    ships = jnp.full((batch,), launch_ships, dtype=jnp.int32)
    p1_enabled = mode in ("fixed_launch", "launch")
    p1_ships = ships if p1_enabled else jnp.zeros((batch,), dtype=jnp.int32)
    base = jnp.stack(
        [
            jnp.stack([home0, target0, ships], axis=1),
            jnp.stack([home1, target1, p1_ships], axis=1),
        ],
        axis=1,
    )[:, :, None, :]
    pad_moves = max(max_moves_per_player - 1, 0)
    pad = jnp.zeros((batch, 2, pad_moves, 3), dtype=jnp.int32)
    return jnp.concatenate([base, pad], axis=2)


def step(
    state: JaxOrbitState,
    actions: jax.Array,
    episode_steps: int = 500,
    actions_are_angles: bool = False,
) -> JaxOrbitState:
    state = phase_a_expire_comets(state)
    state = phase_b_spawn_comets(state)
    active = ~state.done
    state = launch_fleets(state, actions, actions_are_angles=actions_are_angles)
    owned = state.alive & (state.owner >= 0) & active[:, None]
    ships = state.ships + owned.astype(jnp.int32) * state.production
    state = state._replace(ships=ships)
    state, planet_new_x, planet_new_y, planet_check = phase_e_planet_paths(state)
    state, combat_planet, combat_fleet = phase_f_fleet_movement(state, planet_new_x, planet_new_y, planet_check)
    state = state._replace(x=planet_new_x, y=planet_new_y)
    state = phase_h_expire_comets(state)
    state = phase_i_combat(state, combat_planet, combat_fleet)
    return terminate_and_score(state, episode_steps)


def phase_a_expire_comets(state: JaxOrbitState) -> JaxOrbitState:
    expired = state.comet_group_alive & (state.comet_group_path_idx >= state.comet_path_len)
    return kill_comet_groups(state, expired)


def phase_b_spawn_comets(state: JaxOrbitState) -> JaxOrbitState:
    groups = state.comet_group_defined.shape[1]
    if groups == 0:
        return state
    spawn_step = state.step_count[:, None] + 1
    should_spawn = (
        (~state.done)[:, None]
        & state.comet_group_defined
        & (~state.comet_group_spawned)
        & (state.comet_spawn_step == spawn_step)
    )
    has_spawn = should_spawn.any(axis=1)
    group_slots = jnp.where(
        should_spawn,
        jnp.arange(groups, dtype=jnp.int32)[None, :],
        groups,
    ).argmin(axis=1).astype(jnp.int32)
    env_idx = jnp.arange(state.owner.shape[0])
    next_pid = state.next_planet_id
    next_slot = state.next_planet_slot
    spawn_ships = state.comet_group_ships[env_idx, group_slots]

    planet_id = state.planet_id
    owner = state.owner
    x = state.x
    y = state.y
    radius = state.radius
    ships = state.ships
    production = state.production
    initial_x = state.initial_x
    initial_y = state.initial_y
    alive = state.alive
    is_comet = state.is_comet
    group_planet_slot = state.comet_group_planet_slot

    for q in range(COMET_QUADRANTS):
        slot_q = next_slot + q
        pid_q = next_pid + q
        do_spawn = has_spawn & (slot_q < state.owner.shape[1])
        planet_id = planet_id.at[env_idx, slot_q].set(jnp.where(do_spawn, pid_q, planet_id[env_idx, slot_q]))
        owner = owner.at[env_idx, slot_q].set(jnp.where(do_spawn, -1, owner[env_idx, slot_q]))
        x = x.at[env_idx, slot_q].set(jnp.where(do_spawn, -99.0, x[env_idx, slot_q]))
        y = y.at[env_idx, slot_q].set(jnp.where(do_spawn, -99.0, y[env_idx, slot_q]))
        radius = radius.at[env_idx, slot_q].set(jnp.where(do_spawn, COMET_RADIUS, radius[env_idx, slot_q]))
        ships = ships.at[env_idx, slot_q].set(jnp.where(do_spawn, spawn_ships, ships[env_idx, slot_q]))
        production = production.at[env_idx, slot_q].set(jnp.where(do_spawn, COMET_PRODUCTION, production[env_idx, slot_q]))
        alive = alive.at[env_idx, slot_q].set(do_spawn | alive[env_idx, slot_q])
        is_comet = is_comet.at[env_idx, slot_q].set(do_spawn | is_comet[env_idx, slot_q])
        initial_x = initial_x.at[env_idx, slot_q].set(jnp.where(do_spawn, -99.0, initial_x[env_idx, slot_q]))
        initial_y = initial_y.at[env_idx, slot_q].set(jnp.where(do_spawn, -99.0, initial_y[env_idx, slot_q]))
        group_planet_slot = group_planet_slot.at[env_idx, group_slots, q].set(jnp.where(do_spawn, slot_q, group_planet_slot[env_idx, group_slots, q]))

    comet_group_spawned = state.comet_group_spawned.at[env_idx, group_slots].set(has_spawn | state.comet_group_spawned[env_idx, group_slots])
    comet_group_alive = state.comet_group_alive.at[env_idx, group_slots].set(has_spawn | state.comet_group_alive[env_idx, group_slots])
    comet_group_path_idx = state.comet_group_path_idx.at[env_idx, group_slots].set(jnp.where(has_spawn, -1, state.comet_group_path_idx[env_idx, group_slots]))
    return state._replace(
        planet_id=planet_id,
        owner=owner,
        x=x,
        y=y,
        radius=radius,
        ships=ships,
        production=production,
        initial_x=initial_x,
        initial_y=initial_y,
        alive=alive,
        is_comet=is_comet,
        next_planet_id=jnp.where(has_spawn, next_pid + COMET_QUADRANTS, next_pid),
        next_planet_slot=jnp.where(has_spawn, next_slot + COMET_QUADRANTS, next_slot),
        comet_group_spawned=comet_group_spawned,
        comet_group_alive=comet_group_alive,
        comet_group_path_idx=comet_group_path_idx,
        comet_group_planet_slot=group_planet_slot,
    )


def phase_e_planet_paths(state: JaxOrbitState) -> tuple[JaxOrbitState, jax.Array, jax.Array, jax.Array]:
    dx = state.initial_x - BOARD_CENTER
    dy = state.initial_y - BOARD_CENTER
    radius_from_center = jnp.sqrt(dx * dx + dy * dy)
    rotating = state.alive & (~state.is_comet) & ((radius_from_center + state.radius) < BOARD_CENTER)
    init_angle = jnp.arctan2(dy, dx)
    angle = init_angle + state.angular_velocity[:, None] * state.step_count[:, None].astype(jnp.float32)
    rot_x = BOARD_CENTER + radius_from_center * jnp.cos(angle)
    rot_y = BOARD_CENTER + radius_from_center * jnp.sin(angle)
    new_x = jnp.where(rotating, rot_x, state.x)
    new_y = jnp.where(rotating, rot_y, state.y)
    groups = state.comet_group_alive.shape[1]
    path_idx = state.comet_group_path_idx
    if groups > 0:
        path_idx = state.comet_group_path_idx + state.comet_group_alive.astype(jnp.int32)
        group_alive = state.comet_group_alive & (path_idx >= 0) & (path_idx < state.comet_path_len)
        planet_check = state.alive
        env_idx = jnp.arange(state.owner.shape[0])[:, None]
        group_idx = jnp.arange(groups, dtype=jnp.int32)[None, :]
        for q in range(COMET_QUADRANTS):
            slots = state.comet_group_planet_slot[:, :, q]
            valid = group_alive & (slots >= 0)
            safe_slots = jnp.where(valid, slots, 0)
            safe_idx = jnp.clip(path_idx, 0, jnp.maximum(state.comet_path_len - 1, 0))
            comet_pos = jnp.take_along_axis(
                state.comet_paths[:, :, q, :, :],
                safe_idx[:, :, None, None],
                axis=2,
            ).squeeze(axis=2)
            cx = comet_pos[..., 0]
            cy = comet_pos[..., 1]
            prev_x = state.x[env_idx, safe_slots]
            first_placement = prev_x < 0.0
            cur_x = new_x[env_idx, safe_slots]
            cur_y = new_y[env_idx, safe_slots]
            write_x = jnp.where(valid, cx, cur_x)
            write_y = jnp.where(valid, cy, cur_y)
            new_x = new_x.at[env_idx, safe_slots].set(write_x)
            new_y = new_y.at[env_idx, safe_slots].set(write_y)
            cur_check = planet_check[env_idx, safe_slots]
            write_check = jnp.where(valid & first_placement, False, cur_check)
            planet_check = planet_check.at[env_idx, safe_slots].set(write_check)
        return state._replace(comet_group_path_idx=path_idx), new_x, new_y, planet_check
    return state, new_x, new_y, state.alive


def phase_h_expire_comets(state: JaxOrbitState) -> JaxOrbitState:
    expired = state.comet_group_alive & (state.comet_group_path_idx >= state.comet_path_len)
    return kill_comet_groups(state, expired)


def kill_comet_groups(state: JaxOrbitState, expired: jax.Array) -> JaxOrbitState:
    groups = state.comet_group_alive.shape[1]
    if groups == 0:
        return state
    alive = state.alive
    is_comet = state.is_comet
    env_idx = jnp.arange(state.owner.shape[0])[:, None]
    for q in range(COMET_QUADRANTS):
        slots = state.comet_group_planet_slot[:, :, q]
        kill = expired & (slots >= 0)
        safe_slots = jnp.where(kill, slots, 0)
        alive = alive.at[env_idx, safe_slots].set(jnp.where(kill, False, alive[env_idx, safe_slots]))
        is_comet = is_comet.at[env_idx, safe_slots].set(jnp.where(kill, False, is_comet[env_idx, safe_slots]))
    return state._replace(
        alive=alive,
        is_comet=is_comet,
        comet_group_alive=state.comet_group_alive & ~expired,
    )


def launch_fleets(state: JaxOrbitState, actions: jax.Array, actions_are_angles: bool = False) -> JaxOrbitState:
    batch = state.owner.shape[0]
    env_idx = jnp.arange(batch)
    if actions.ndim == 3:
        actions = actions[:, :, None, :]
    ships = state.ships
    next_fleet = state.next_fleet
    fleet_owner = state.fleet_owner
    fleet_x = state.fleet_x
    fleet_y = state.fleet_y
    fleet_angle = state.fleet_angle
    fleet_ships = state.fleet_ships
    fleet_alive = state.fleet_alive

    move_count = actions.shape[2]
    for player in range(NUM_PLAYERS):
        for move_idx in range(move_count):
            src_pid = actions[:, player, move_idx, 0].astype(jnp.int32)
            src = resolve_planet_slot(state, src_pid)
            amount = actions[:, player, move_idx, 2].astype(jnp.int32)
            src_owner = state.owner[env_idx, src]
            src_alive = state.alive[env_idx, src]
            src_ships = ships[env_idx, src]
            valid = (~state.done) & src_alive & (src_owner == player) & (amount > 0) & (src_ships >= amount)
            slot = jnp.minimum(next_fleet, state.fleet_alive.shape[1] - 1)
            sx = state.x[env_idx, src]
            sy = state.y[env_idx, src]
            if actions_are_angles:
                angle = actions[:, player, move_idx, 1].astype(jnp.float32)
            else:
                tgt_pid = actions[:, player, move_idx, 1].astype(jnp.int32)
                tgt = resolve_planet_slot(state, tgt_pid)
                tx = state.x[env_idx, tgt]
                ty = state.y[env_idx, tgt]
                angle = jnp.arctan2(ty - sy, tx - sx)
            start_x = sx + jnp.cos(angle) * (state.radius[env_idx, src] + 0.1)
            start_y = sy + jnp.sin(angle) * (state.radius[env_idx, src] + 0.1)
            ships = ships.at[env_idx, src].add(jnp.where(valid, -amount, 0))
            fleet_owner = fleet_owner.at[env_idx, slot].set(jnp.where(valid, player, fleet_owner[env_idx, slot]))
            fleet_x = fleet_x.at[env_idx, slot].set(jnp.where(valid, start_x, fleet_x[env_idx, slot]))
            fleet_y = fleet_y.at[env_idx, slot].set(jnp.where(valid, start_y, fleet_y[env_idx, slot]))
            fleet_angle = fleet_angle.at[env_idx, slot].set(jnp.where(valid, angle, fleet_angle[env_idx, slot]))
            fleet_ships = fleet_ships.at[env_idx, slot].set(jnp.where(valid, amount, fleet_ships[env_idx, slot]))
            fleet_alive = fleet_alive.at[env_idx, slot].set(valid | fleet_alive[env_idx, slot])
            next_fleet = jnp.where(valid, jnp.minimum(next_fleet + 1, state.fleet_alive.shape[1] - 1), next_fleet)

    return state._replace(
        ships=ships,
        fleet_owner=fleet_owner,
        fleet_x=fleet_x,
        fleet_y=fleet_y,
        fleet_angle=fleet_angle,
        fleet_ships=fleet_ships,
        fleet_alive=fleet_alive,
        next_fleet=next_fleet,
    )


def resolve_planet_slot(state: JaxOrbitState, planet_ids: jax.Array) -> jax.Array:
    matches = state.planet_id == planet_ids[:, None]
    return matches.argmax(axis=1).astype(jnp.int32)


def phase_f_fleet_movement(
    state: JaxOrbitState,
    planet_new_x: jax.Array,
    planet_new_y: jax.Array,
    planet_check: jax.Array,
) -> tuple[JaxOrbitState, jax.Array, jax.Array]:
    speed = 1.0 + 5.0 * (jnp.log(jnp.maximum(state.fleet_ships, 1)) / jnp.log(1000.0)) ** 1.5
    speed = jnp.clip(speed, max=6.0)
    old_x = state.fleet_x
    old_y = state.fleet_y
    new_x = state.fleet_x + jnp.cos(state.fleet_angle) * speed
    new_y = state.fleet_y + jnp.sin(state.fleet_angle) * speed

    ax_old = old_x[:, :, None]
    ay_old = old_y[:, :, None]
    ax_new = new_x[:, :, None]
    ay_new = new_y[:, :, None]
    px_old = state.x[:, None, :]
    py_old = state.y[:, None, :]
    px_new = planet_new_x[:, None, :]
    py_new = planet_new_y[:, None, :]
    pr = state.radius[:, None, :]
    d0x = ax_old - px_old
    d0y = ay_old - py_old
    dvx = (ax_new - ax_old) - (px_new - px_old)
    dvy = (ay_new - ay_old) - (py_new - py_old)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - pr * pr
    disc = b * b - 4.0 * a * c
    small_a = a < 1e-12
    sq = jnp.sqrt(jnp.maximum(disc, 0.0))
    denom = jnp.maximum(2.0 * a, 1e-30)
    t1 = (-b - sq) / denom
    t2 = (-b + sq) / denom
    hit = ((disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)) | (small_a & (c <= 0.0))
    planet_valid = (state.alive & planet_check)[:, None, :]
    fleet_valid = state.fleet_alive[:, :, None]
    hit = hit & planet_valid & fleet_valid
    big = jnp.full_like(t1, jnp.inf)
    t1_for_argmin = jnp.where(hit, t1, big)
    any_hit = hit.any(axis=2)
    hit_slot = jnp.argmin(t1_for_argmin, axis=2).astype(jnp.int32)
    oob = (new_x < 0.0) | (new_x > BOARD_SIZE) | (new_y < 0.0) | (new_y > BOARD_SIZE)
    seg_dx = new_x - old_x
    seg_dy = new_y - old_y
    seg_len2 = seg_dx * seg_dx + seg_dy * seg_dy
    t_proj = jnp.where(
        seg_len2 > 0.0,
        ((BOARD_CENTER - old_x) * seg_dx + (BOARD_CENTER - old_y) * seg_dy) / jnp.maximum(seg_len2, 1e-30),
        jnp.zeros_like(seg_len2),
    )
    t_proj = jnp.clip(t_proj, 0.0, 1.0)
    proj_x = old_x + t_proj * seg_dx
    proj_y = old_y + t_proj * seg_dy
    sun_hit = jnp.sqrt((proj_x - BOARD_CENTER) ** 2 + (proj_y - BOARD_CENTER) ** 2) < SUN_RADIUS
    any_hit = any_hit & state.fleet_alive
    oob_die = (oob | sun_hit) & state.fleet_alive & ~any_hit
    survives = state.fleet_alive & ~any_hit & ~oob_die
    combat_planet = jnp.where(any_hit, hit_slot, -1).astype(jnp.int32)
    state = state._replace(
        fleet_x=jnp.where(survives, new_x, state.fleet_x),
        fleet_y=jnp.where(survives, new_y, state.fleet_y),
        fleet_alive=survives,
    )
    return state, combat_planet, any_hit


def phase_i_combat(state: JaxOrbitState, combat_planet: jax.Array, combat_fleet: jax.Array) -> JaxOrbitState:
    batch, planets = state.owner.shape
    owners = NUM_PLAYERS
    env_idx = jnp.arange(batch)[:, None]
    owner_idx = jnp.clip(state.fleet_owner, 0, owners - 1)
    valid = combat_fleet & (state.fleet_owner >= 0) & (state.fleet_owner < owners)
    arrivals = jnp.zeros((batch, planets, owners), dtype=jnp.int32)
    planet_slot = jnp.clip(combat_planet, 0, planets - 1)
    flat = env_idx * (planets * owners) + planet_slot * owners + owner_idx
    flat = jnp.where(valid, flat, 0)
    contrib = jnp.where(valid, state.fleet_ships, 0)
    arrivals = arrivals.reshape(-1).at[flat.reshape(-1)].add(contrib.reshape(-1)).reshape(batch, planets, owners)

    topk_idx = jnp.argsort(arrivals, axis=2)[:, :, ::-1]
    top1_owner = topk_idx[:, :, 0]
    top1_ships = jnp.take_along_axis(arrivals, top1_owner[:, :, None], axis=2).squeeze(2)
    top2_owner = topk_idx[:, :, 1]
    top2_ships = jnp.take_along_axis(arrivals, top2_owner[:, :, None], axis=2).squeeze(2)
    contested = top1_ships > 0
    survivor = top1_ships - top2_ships
    tied = top1_ships == top2_ships
    survivor = jnp.where(tied, 0, survivor)
    survivor_owner = jnp.where(survivor > 0, top1_owner, -1)

    cur_owner = state.owner
    cur_ships = state.ships
    same_owner = (cur_owner == survivor_owner) & contested & (survivor > 0)
    diff_owner = (cur_owner != survivor_owner) & contested & (survivor > 0)
    ships_after_friendly = cur_ships + survivor
    ships_after_hostile = cur_ships - survivor
    flips = diff_owner & (ships_after_hostile < 0)
    new_ships = cur_ships
    new_owner = cur_owner
    new_ships = jnp.where(same_owner, ships_after_friendly, new_ships)
    defender_survives = diff_owner & (ships_after_hostile >= 0)
    new_ships = jnp.where(defender_survives, ships_after_hostile, new_ships)
    new_ships = jnp.where(flips, -ships_after_hostile, new_ships)
    new_owner = jnp.where(flips, survivor_owner, new_owner)
    active_planet = state.alive & contested
    final_ships = jnp.where(active_planet, new_ships, cur_ships)
    final_owner = jnp.where(active_planet, new_owner, cur_owner)
    return state._replace(ships=final_ships.astype(jnp.int32), owner=final_owner.astype(jnp.int32))


def terminate_and_score(state: JaxOrbitState, episode_steps: int) -> JaxOrbitState:
    # N-player generic (reduces bit-identically to the old 2p code when NUM_PLAYERS==2).
    pidx = jnp.arange(NUM_PLAYERS)                                          # (P,)
    own_planet = (state.owner[:, None, :] == pidx[None, :, None]) & state.alive[:, None, :]        # (B,P,planets)
    own_fleet = (state.fleet_owner[:, None, :] == pidx[None, :, None]) & state.fleet_alive[:, None, :]  # (B,P,fleets)
    score = (own_planet * state.ships[:, None, :]).sum(2) + (own_fleet * state.fleet_ships[:, None, :]).sum(2)  # (B,P)
    alive_p = own_planet.any(2) | own_fleet.any(2)                          # (B,P) still in the game
    num_alive = alive_p.sum(axis=1)                                         # (B,)
    new_step = state.step_count + (~state.done).astype(jnp.int32)
    newly_done = (~state.done) & ((new_step >= episode_steps - 2) | (num_alive <= 1))
    # reward = +1 if your score is the max (ties → all leaders win), else -1.
    # 2p: score_p >= other ⟺ score_p == max → matches old rew_p = score>=other ? 1 : -1.
    is_max = score >= score.max(axis=1, keepdims=True)
    rew = jnp.where(is_max, 1, -1).astype(state.rewards.dtype)              # (B,P)
    rewards = jnp.where(newly_done[:, None], rew, state.rewards)
    return state._replace(step_count=new_step, done=state.done | newly_done, rewards=rewards)


def main() -> None:
    args = parse_args()
    action_fn = make_fixture_actions if args.fixture else make_actions
    replay_actions = load_replay_actions(args.actions_json, args.steps)

    @jax.jit
    def run_steps(state):
        def body(carry, _):
            actions = action_fn(carry, args.action_mode, args.launch_ships, args.max_moves_per_player)
            return step(carry, actions, args.episode_steps), None

        return jax.lax.scan(body, state, None, length=args.steps)[0]

    @jax.jit
    def run_steps_with_history(state):
        def body(carry, _):
            actions = action_fn(carry, args.action_mode, args.launch_ships, args.max_moves_per_player)
            nxt = step(carry, actions, args.episode_steps)
            return nxt, snapshot(nxt)

        return jax.lax.scan(body, state, None, length=args.steps)

    initial_state = initial_env_state(args)
    summaries = [summary(initial_state, "initial")] if args.fixture or args.map_json else []
    wants_summary = args.fixture or bool(args.map_json)
    if replay_actions is None:
        if wants_summary and args.step_history:
            compiled_state, compiled_history = run_steps_with_history(initial_state)
            jax.block_until_ready((compiled_state, compiled_history))
        else:
            compiled_state = run_steps(initial_state)
            jax.block_until_ready(compiled_state)
    state = initial_env_state(args)
    start = time.perf_counter()
    history = None
    if replay_actions is not None:
        replay_arr = jnp.array(replay_actions, dtype=jnp.float32)
        hist_rows: list[dict[str, jax.Array]] = []
        for idx in range(args.steps):
            act = replay_arr[idx][None, :, :, :]
            state = step(state, act, args.episode_steps, actions_are_angles=True)
            if wants_summary and args.step_history:
                hist_rows.append(snapshot(state))
        if wants_summary and args.step_history:
            history = {
                key: jnp.stack([row[key] for row in hist_rows], axis=0)
                for key in hist_rows[0].keys()
            } if hist_rows else None
        jax.block_until_ready(state)
    elif wants_summary and args.step_history:
        state, history = run_steps_with_history(state)
        jax.block_until_ready((state, history))
    else:
        state = run_steps(state)
        jax.block_until_ready(state)
    elapsed = time.perf_counter() - start
    if wants_summary:
        if history is not None:
            summaries.extend(summaries_from_history(history))
        else:
            summaries.append(summary(state, f"step_{args.steps}"))
    report = {
        "config": vars(args),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "elapsed_sec": elapsed,
        "env_steps": args.num_envs * args.steps,
        "env_sps": args.num_envs * args.steps / max(elapsed, 1e-9),
        "done_count": int(state.done.sum()),
        "reward_sum": [int(x) for x in state.rewards.sum(axis=0).tolist()],
        "ship_sum": int(state.ships.sum()),
        "fleet_alive": int(state.fleet_alive.sum()),
    }
    if wants_summary:
        report["summaries"] = summaries
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal no-comets JAX Orbit Wars env spike.")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--planets", type=int, default=8)
    parser.add_argument("--fleets", type=int, default=512)
    parser.add_argument("--episode-steps", type=int, default=500)
    parser.add_argument(
        "--action-mode",
        choices=(
            "noop",
            "fixed_launch",
            "launch",
            "capture",
            "terminal",
            "enemy_capture",
            "friendly_reinforce",
            "sun_collision",
            "multi_planet_intersection",
            "rotating_target_collision",
            "comet_attack",
            "replay",
        ),
        default="fixed_launch",
    )
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-kind", choices=("base", "rotation", "reinforce", "multi_hit"), default="base")
    parser.add_argument("--angular-velocity", type=float, default=0.0)
    parser.add_argument("--map-json", default="")
    parser.add_argument("--step-history", action="store_true")
    parser.add_argument("--launch-ships", type=int, default=5)
    parser.add_argument("--max-moves-per-player", type=int, default=1)
    parser.add_argument("--actions-json", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def load_replay_actions(path: str, steps: int) -> list[list[list[list[float]]]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    rows = payload.get("steps", [])
    out: list[list[list[list[float]]]] = []
    global_max_moves = 1
    normalized_rows: list[list[list[list[float]]]] = []
    for idx in range(steps):
        if idx < len(rows):
            step_row = rows[idx]
        else:
            step_row = [[], []]
        p0_raw = step_row[0] if len(step_row) > 0 else []
        p1_raw = step_row[1] if len(step_row) > 1 else []
        if len(p0_raw) == 3 and not isinstance(p0_raw[0], list):
            p0_raw = [p0_raw]
        if len(p1_raw) == 3 and not isinstance(p1_raw[0], list):
            p1_raw = [p1_raw]
        p0 = [[float(move[0]), float(move[1]), float(move[2])] for move in p0_raw]
        p1 = [[float(move[0]), float(move[1]), float(move[2])] for move in p1_raw]
        max_moves = max(len(p0), len(p1), 1)
        while len(p0) < max_moves:
            p0.append([-1.0, 0.0, 0.0])
        while len(p1) < max_moves:
            p1.append([-1.0, 0.0, 0.0])
        row = [p0, p1]
        normalized_rows.append(row)
        global_max_moves = max(global_max_moves, len(p0), len(p1))
    for row in normalized_rows:
        p0 = row[0]
        p1 = row[1]
        while len(p0) < global_max_moves:
            p0.append([-1.0, 0.0, 0.0])
        while len(p1) < global_max_moves:
            p1.append([-1.0, 0.0, 0.0])
        out.append([p0, p1])
    return out


def initial_env_state(args: argparse.Namespace) -> JaxOrbitState:
    if args.map_json:
        return reset_from_map(args.map_json, args.num_envs, args.fleets)
    if args.fixture:
        return reset_fixture(args.num_envs, args.fleets, args.fixture_kind, args.angular_velocity)
    return reset(args.num_envs, args.planets, args.fleets)


def snapshot(state: JaxOrbitState) -> dict[str, jax.Array]:
    owner = state.owner[0]
    ships = state.ships[0]
    alive = state.alive[0]
    fleet_alive = state.fleet_alive[0]
    return {
        "step_count": state.step_count[0],
        "done": state.done[0],
        "rewards": state.rewards[0],
        "planet_owner": owner,
        "planet_id": state.planet_id[0],
        "planet_radius": state.radius[0],
        "planet_production": state.production[0],
        "planet_ships": ships,
        "planet_x": state.x[0],
        "planet_y": state.y[0],
        "alive": alive,
        "fleet_alive": fleet_alive.sum(),
        "fleet_ships": (state.fleet_ships[0] * fleet_alive.astype(jnp.int32)).sum(),
        "comet_group_alive": state.comet_group_alive[0],
        "comet_group_path_idx": state.comet_group_path_idx[0],
        "comet_group_path_len": state.comet_path_len[0],
        "score0": (((owner == 0) & alive).astype(jnp.int32) * ships).sum(),
        "score1": (((owner == 1) & alive).astype(jnp.int32) * ships).sum(),
    }


def summaries_from_history(history: dict[str, jax.Array]) -> list[dict]:
    out = []
    for idx in range(int(history["step_count"].shape[0])):
        out.append(
            {
                "label": f"step_{idx + 1}",
                "step_count": int(history["step_count"][idx]),
                "done": bool(history["done"][idx]),
                "rewards": [int(x) for x in history["rewards"][idx].tolist()],
                "planet_owner": [int(x) for x in history["planet_owner"][idx].tolist()],
                "planet_id": [int(x) for x in history["planet_id"][idx].tolist()],
                "planet_radius": [round(float(x), 4) for x in history["planet_radius"][idx].tolist()],
                "planet_production": [int(x) for x in history["planet_production"][idx].tolist()],
                "planet_ships": [int(x) for x in history["planet_ships"][idx].tolist()],
                "planet_x": [round(float(x), 4) for x in history["planet_x"][idx].tolist()],
                "planet_y": [round(float(x), 4) for x in history["planet_y"][idx].tolist()],
                "alive": [bool(x) for x in history["alive"][idx].tolist()],
                "fleet_alive": int(history["fleet_alive"][idx]),
                "fleet_ships": int(history["fleet_ships"][idx]),
                "comet_group_alive": [bool(x) for x in history["comet_group_alive"][idx].tolist()],
                "comet_group_path_idx": [int(x) for x in history["comet_group_path_idx"][idx].tolist()],
                "comet_group_path_len": [int(x) for x in history["comet_group_path_len"][idx].tolist()],
                "score0": int(history["score0"][idx]),
                "score1": int(history["score1"][idx]),
            }
        )
    return out


def summary(state: JaxOrbitState, label: str) -> dict:
    owner = state.owner[0]
    ships = state.ships[0]
    alive = state.alive[0]
    fleet_alive = state.fleet_alive[0]
    score0 = (((owner == 0) & alive).astype(jnp.int32) * ships).sum()
    score1 = (((owner == 1) & alive).astype(jnp.int32) * ships).sum()
    return {
        "label": label,
        "step_count": int(state.step_count[0]),
        "done": bool(state.done[0]),
        "rewards": [int(x) for x in state.rewards[0].tolist()],
        "planet_owner": [int(x) for x in owner.tolist()],
        "planet_id": [int(x) for x in state.planet_id[0].tolist()],
        "planet_radius": [round(float(x), 4) for x in state.radius[0].tolist()],
        "planet_production": [int(x) for x in state.production[0].tolist()],
        "planet_ships": [int(x) for x in ships.tolist()],
        "planet_x": [round(float(x), 4) for x in state.x[0].tolist()],
        "planet_y": [round(float(x), 4) for x in state.y[0].tolist()],
        "alive": [bool(x) for x in alive.tolist()],
        "fleet_alive": int(fleet_alive.sum()),
        "fleet_ships": int((state.fleet_ships[0] * fleet_alive.astype(jnp.int32)).sum()),
        "comet_group_alive": [bool(x) for x in state.comet_group_alive[0].tolist()],
        "comet_group_path_idx": [int(x) for x in state.comet_group_path_idx[0].tolist()],
        "comet_group_path_len": [int(x) for x in state.comet_path_len[0].tolist()],
        "score0": int(score0),
        "score1": int(score1),
    }


if __name__ == "__main__":
    main()
