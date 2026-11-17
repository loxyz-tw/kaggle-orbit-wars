"""Measure OUR fleets that die to the SUN, directly in the REAL kaggle env (ground truth,
no replay reconstruction). Runs N games AGENT vs OPP (rotating seats), tracks every fleet
we launch by id across env.steps, and classifies its disappearance:
  - CAPTURE: last pos within (planet_radius + 1) of some alive planet surface  -> productive
  - SUN: last pos within SUN_RADIUS+margin of board center                      -> sun-death
  - OOB/OTHER: vanished far from any planet and not at the sun                  -> wasteful
Reports per-guard sun-death rate AND win-rate so we can see if a guard removes
sun-deaths WITHOUT costing win-rate.

Also serves as a 4p win-rate gauge (OW_NP=4: AGENT vs 3x OPP).

Usage: OW_AGENT=<our submission .py> OW_SUNGUARD=segment OW_OPP=<opponent .py, default self-play> \
       [OW_NP=4] OMP_NUM_THREADS=1 python measure_sun_deaths.py [N]
"""
import os, sys, math, multiprocessing as mp

def _require(var):
    v = os.environ.get(var, "")
    if not v:
        raise SystemExit(f"set {var}=<path to agent .py>")
    return v

AGENT = _require("OW_AGENT")
OPP = os.environ.get("OW_OPP", AGENT)  # default = self-play
NP = int(os.environ.get("OW_NP", "2"))  # number of players (2 or 4)
C = 50.0
SUN = 10.0
SUN_MARGIN = float(os.environ.get("SUN_MARGIN", "3.0"))  # last pos before crossing -> within SUN+margin


def _resolve(path):
    # our own submission subs load fine via importlib; opponents (some heuristics)
    # break under custom-name importlib -> hand kaggle the raw path (ladder-identical).
    if "submission/sub_" not in path and not path.startswith("submission/"):
        return path
    import importlib.util as iu
    name = "aw_" + os.path.basename(path).replace(".py", "").replace("-", "_")
    spec = iu.spec_from_file_location(name, path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.agent


def run_one(args):
    i, seed = args
    import logging; logging.disable(logging.INFO)
    from kaggle_environments import make
    seat = i % NP
    agents = [_resolve(OPP) for _ in range(NP)]; agents[seat] = _resolve(AGENT)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    steps = env.steps
    final = steps[-1]
    # score-based ranking (env reward tie-inflates in 4p; mirror games -> all reward=1)
    fobs = final[0]["observation"]
    sc = [0.0] * NP
    for p in (fobs.get("planets") or []):
        o = int(p[1])
        if 0 <= o < NP: sc[o] += float(p[5])
    for fl in (fobs.get("fleets") or []):
        o = int(fl[1])
        if 0 <= o < NP: sc[o] += float(fl[6])
    mx = max(sc); leaders = [s for s in range(NP) if sc[s] == mx]
    win = 1 if (seat in leaders and len(leaders) == 1) else 0
    tie = 1 if (seat in leaders and len(leaders) > 1) else 0

    # track fleets by id from seat-0 observation (full board, absolute owners)
    last = {}   # fid -> (x, y, planets_snapshot)
    prev_ids = set()
    n_capture = n_sun = n_oob = n_launch = 0
    for st in steps:
        o = st[0].get("observation") or {}
        if "fleets" not in o:
            continue
        planets = o.get("planets") or []
        cur = {}
        for f in o["fleets"]:
            fid, own = int(f[0]), int(f[1])
            if own != seat:
                continue
            cur[fid] = (float(f[2]), float(f[3]))
            if fid not in last:
                n_launch += 1
            # store pos, heading angle, ships, planet snapshot
            last[fid] = (float(f[2]), float(f[3]), float(f[4]), int(f[6]), planets)
        # fleets that disappeared this step
        for fid in prev_ids - set(cur.keys()):
            x, y, ang, sh, pl = last[fid]
            # speed (env formula) -> how far it travels per step
            v = min(max(1.0 + 5.0 * (math.log(max(sh, 1)) / math.log(1000.0)) ** 1.5, 1.0), 6.0)
            # nearest alive planet surface distance from last pos
            nd = 1e9
            for p in pl:
                pd = math.hypot(float(p[2]) - x, float(p[3]) - y) - float(p[4])
                nd = min(nd, pd)
            # SUN test: forward segment from last pos along heading (up to 1 step) within SUN
            ex, ey = x + math.cos(ang) * v, y + math.sin(ang) * v
            vx, vy = ex - x, ey - y
            seg2 = vx * vx + vy * vy
            tproj = max(0.0, min(1.0, ((C - x) * vx + (C - y) * vy) / max(seg2, 1e-9)))
            px, py = x + tproj * vx, y + tproj * vy
            sun_cross = math.hypot(px - C, py - C) < SUN
            if nd <= v + 0.5:        # could have captured a planet this step
                n_capture += 1
            elif sun_cross or math.hypot(x - C, y - C) <= SUN + SUN_MARGIN:
                n_sun += 1
            else:
                n_oob += 1
        prev_ids = set(cur.keys())
    return (win, tie, n_launch, n_capture, n_sun, n_oob)


def wilson(w, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = w / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return ((c-m)/d, (c+m)/d)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(os.environ.get("OW_WORKERS", "8"))
    guard = os.environ.get("OW_SUNGUARD", "segment")
    tasks = [(i, 20260000 + i) for i in range(N)]
    print(f"[sun] AGENT={AGENT} guard={guard} vs OPP={OPP}  N={N}", flush=True)
    res = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(run_one, tasks):
            res.append(r)
            if len(res) % 20 == 0:
                print(f"[sun] {len(res)}/{N} done", flush=True)
    n = len(res)
    wins = sum(r[0] for r in res); ties = sum(r[1] for r in res)
    launch = sum(r[2] for r in res); cap = sum(r[3] for r in res)
    sun = sum(r[4] for r in res); oob = sum(r[5] for r in res)
    games_with_sun = sum(1 for r in res if r[4] > 0)
    lo, hi = wilson(wins, n)
    print(f"\n[sun] DONE guard={guard} (N={n})")
    print(f"  win = {wins}/{n} = {100*wins/n:.1f}%  (Wilson95 {100*lo:.1f}-{100*hi:.1f}%)  tie={100*ties/n:.1f}%")
    print(f"  our fleets launched = {launch}")
    print(f"  resolved: capture={cap}  SUN-death={sun}  oob/other={oob}")
    print(f"  *** SUN-deaths = {sun} ({100*sun/max(launch,1):.2f}% of launches), in {games_with_sun}/{n} games ***")


if __name__ == "__main__":
    main()
