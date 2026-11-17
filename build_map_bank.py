"""Build a kaggle-distribution map bank for multi-map RL training.

Reuses the *exact* kaggle env generator (`generate_planets`) and replicates the
interpreter's init RNG order (angular_velocity -> generate_planets -> home group)
so each seed reproduces the same map the live env would create. No reverse
engineering: this is the env's own code.

Output: npz with padded arrays
  planets (N, Pmax, 7) float32  [id, owner, x, y, radius, ships, production]
  alive   (N, Pmax)    bool
  av      (N,)         float32
"""
import argparse
import random

import numpy as np

from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets


def build_one(seed: int, num_agents: int = 2):
    rng = random.Random(seed)
    # Mirror interpreter() init block ordering exactly.
    av = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    num_groups = len(planets) // 4
    if num_groups > 0:
        home = rng.randint(0, num_groups - 1)
        base = home * 4
        if num_agents == 2:
            planets[base][1] = 0
            planets[base][5] = 10
            planets[base + 3][1] = 1
            planets[base + 3][5] = 10
        elif num_agents == 4:
            for j in range(4):
                planets[base + j][1] = j
                planets[base + j][5] = 10
    return av, planets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-maps", type=int, default=512)
    p.add_argument("--seed-base", type=int, default=1_000_000)
    p.add_argument("--num-agents", type=int, default=2, choices=(2, 4),
                   help="Home-planet assignment: 2p (diagonal owners 0,3) or 4p (all 4 owners).")
    p.add_argument("--out", default="")
    args = p.parse_args()
    out = args.out or (f"kaggle_map_bank_{args.num_agents}p.npz" if args.num_agents != 2 else "kaggle_map_bank.npz")

    maps = [build_one(args.seed_base + i, args.num_agents) for i in range(args.num_maps)]
    pmax = max(len(pl) for _, pl in maps)
    n = len(maps)

    planets = np.zeros((n, pmax, 7), dtype=np.float32)
    alive = np.zeros((n, pmax), dtype=bool)
    av = np.zeros((n,), dtype=np.float32)
    counts = np.zeros((n,), dtype=np.int32)
    for i, (a, pl) in enumerate(maps):
        av[i] = a
        counts[i] = len(pl)
        for j, row in enumerate(pl):
            planets[i, j] = np.asarray(row, dtype=np.float32)
            alive[i, j] = True

    np.savez(out, planets=planets, alive=alive, av=av, counts=counts)
    print(f"[map-bank] {n} maps ({args.num_agents}p home assignment) -> {out}")
    print(f"[map-bank] planet count: min={counts.min()} max={counts.max()} mean={counts.mean():.1f} (Pmax={pmax})")
    print(f"[map-bank] angular_velocity: min={av.min():.4f} max={av.max():.4f}")
    orbiting = (
        (np.sqrt((planets[..., 2] - 50.0) ** 2 + (planets[..., 3] - 50.0) ** 2) + planets[..., 4] < 50.0)
        & alive
    ).sum(axis=1)
    print(f"[map-bank] orbiting planets/map: min={orbiting.min()} max={orbiting.max()} mean={orbiting.mean():.1f}")


if __name__ == "__main__":
    main()
