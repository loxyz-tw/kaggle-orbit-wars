"""Kaggle observation -> JaxOrbitState conversion shared by the dataset generator.

Constants match the JAX environment's fixed shapes (jax_min_env_spike).
"""
from __future__ import annotations

import numpy as np

import jax_min_env_spike as _env
from jax_min_env_spike import JaxOrbitState

COMET_QUADRANTS = getattr(_env, "COMET_QUADRANTS", 4)
COMET_RADIUS = getattr(_env, "COMET_RADIUS", 1.0)
MAXP = 56; MAXG = 2; MAXPATH = 80


def obs_to_state_np(obs, fleet_cap=256):
    """Kaggle observation dict -> single-env JaxOrbitState with NUMPY fields (slot = obs order).
    No per-field jnp.asarray device transfer: callers stack many states and transfer once
    (per-field transfers were ~57% of dataset-generation time when profiled)."""
    planets = obs["planets"]
    init_by_id = {int(p[0]): p for p in obs.get("initial_planets", [])}
    comets = obs.get("comets") or []
    step = int(obs["step"]); av = float(obs["angular_velocity"]); n = len(planets)
    pid = np.full(MAXP, -1, np.int32); owner = np.full(MAXP, -1, np.int32)
    x = np.zeros(MAXP, np.float32); y = np.zeros(MAXP, np.float32)
    radius = np.full(MAXP, COMET_RADIUS, np.float32); ships = np.zeros(MAXP, np.int32)
    prod = np.zeros(MAXP, np.int32); alive = np.zeros(MAXP, bool)
    init_x = np.zeros(MAXP, np.float32); init_y = np.zeros(MAXP, np.float32)
    for i, p in enumerate(planets):
        pid[i] = int(p[0]); owner[i] = int(p[1]); x[i] = float(p[2]); y[i] = float(p[3])
        radius[i] = float(p[4]); ships[i] = int(p[5]); prod[i] = int(p[6]); alive[i] = True
        ip = init_by_id.get(int(p[0]))
        init_x[i] = float(ip[2]) if ip else float(p[2]); init_y[i] = float(ip[3]) if ip else float(p[3])
    id_to_slot = {int(pid[i]): i for i in range(n)}
    comet_id_set = set()
    for c in comets:
        comet_id_set.update(int(i) for i in c["planet_ids"])
    is_comet = np.array([int(pid[i]) in comet_id_set for i in range(MAXP)], bool)
    cpaths = np.zeros((MAXG, COMET_QUADRANTS, MAXPATH, 2), np.float32)
    cpath_len = np.zeros(MAXG, np.int32); cidx = np.full(MAXG, -1, np.int32); cships = np.zeros(MAXG, np.int32)
    cdef = np.zeros(MAXG, bool); cspawned = np.zeros(MAXG, bool); calive = np.zeros(MAXG, bool)
    cslot = np.full((MAXG, COMET_QUADRANTS), -1, np.int32)
    for g, c in enumerate(comets[:MAXG]):
        pth = np.array(c["paths"], np.float32); Lp = min(pth.shape[1], MAXPATH)
        cpaths[g, :, :Lp, :] = pth[:, :Lp, :]; cpath_len[g] = Lp; cidx[g] = int(c["path_index"])
        cdef[g] = cspawned[g] = calive[g] = True
        ids = [int(i) for i in c["planet_ids"]]
        cships[g] = int(ships[id_to_slot[ids[0]]]) if ids[0] in id_to_slot else 0
        for q, pq in enumerate(ids[:COMET_QUADRANTS]):
            if pq in id_to_slot: cslot[g, q] = id_to_slot[pq]
    fl = obs.get("fleets") or []; F = fleet_cap
    fo = np.full(F, -2, np.int32); fx = np.zeros(F, np.float32); fy = np.zeros(F, np.float32)
    fa = np.zeros(F, np.float32); fs = np.zeros(F, np.int32); falive = np.zeros(F, bool)
    if len(fl) > F:
        fl = sorted(fl, key=lambda f: -int(f[6]))[:F]
    for k, f in enumerate(fl):
        fo[k] = int(f[1]); fx[k] = float(f[2]); fy[k] = float(f[3]); fa[k] = float(f[4]); fs[k] = int(f[6]); falive[k] = True
    b = lambda a: np.asarray(a)[None]                 # NUMPY (not jnp) -> no device transfer here
    return JaxOrbitState(
        planet_id=b(pid), owner=b(owner), x=b(x), y=b(y), radius=b(radius), ships=b(ships),
        production=b(prod), initial_x=b(init_x), initial_y=b(init_y),
        angular_velocity=np.array([av], np.float32), alive=b(alive), is_comet=b(is_comet),
        next_planet_id=np.array([int(pid.max()) + 1 if n else 0], np.int32),
        next_planet_slot=np.array([n], np.int32),
        comet_paths=b(cpaths), comet_path_len=b(cpath_len), comet_spawn_step=b(np.zeros(MAXG, np.int32)),
        comet_group_ships=b(cships), comet_group_defined=b(cdef), comet_group_spawned=b(cspawned),
        comet_group_alive=b(calive), comet_group_path_idx=b(cidx), comet_group_planet_slot=b(cslot),
        fleet_owner=b(fo), fleet_x=b(fx), fleet_y=b(fy), fleet_angle=b(fa), fleet_ships=b(fs),
        fleet_alive=b(falive), next_fleet=np.array([len(fl)], np.int32),
        step_count=np.array([step], np.int32), done=np.array([False]),
        rewards=np.zeros((1, _env.NUM_PLAYERS), np.float32))
