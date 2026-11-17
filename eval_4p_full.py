"""4p eval with MECHANISM metrics — win% PLUS the turtle-diagnosis metrics in one pass.
AGENT vs 3x OPP, rotating seats, same seeds as eval_4p_kaggle (20260000+i).

Reports, for AGENT: WIN% (1st-place rate, fair=25%) + launches/alive-step, peak planets
(us vs best opp), elimination rate, avg elimination step. These are what diagnosed the 4p
turtle on the ladder (launches/alive-step 0.51 vs 2p 1.60) — so they directly show whether
planet-reward made the model expand more aggressively.

Usage: OW_AGENT=<our submission .py> OW_OPP=<your opponent agent .py> OW_WORKERS=8 python eval_4p_full.py [N]
"""
import os, sys, math, multiprocessing as mp

def _require(var):
    v = os.environ.get(var, "")
    if not v:
        raise SystemExit(f"set {var}=<path to agent .py>")
    return v

AGENT = _require("OW_AGENT")
OPP = _require("OW_OPP")


def run_one(args):
    i, seed = args
    import logging; logging.disable(logging.INFO)
    from kaggle_environments import make
    seat = i % 4
    agents = [OPP, OPP, OPP, OPP]; agents[seat] = AGENT
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    steps = env.steps; T = len(steps)
    final = steps[-1]
    # SCORE-based ranking (the env reward tie-inflates: near-mirror games give all
    # seats reward=1 -> 'win>=max' was always true -> 100% artifact, 2026-06-10).
    fobs = final[0]["observation"]
    sc = [0.0, 0.0, 0.0, 0.0]
    for p in (fobs.get("planets") or []):
        o = int(p[1])
        if 0 <= o < 4: sc[o] += float(p[5])
    for fl in (fobs.get("fleets") or []):
        o = int(fl[1])
        if 0 <= o < 4: sc[o] += float(fl[6])
    mx = max(sc)
    leaders = [i for i in range(4) if sc[i] == mx]
    win = (1.0 / len(leaders)) if seat in leaders else 0.0   # tie-split share
    # mechanism: peaks, elimination, launches while alive (planet owner = absolute seat)
    mypk = 0; opppk = 0; elim = None
    for j, st in enumerate(steps):
        pl = st[0]["observation"].get("planets") or []
        cnt = {0: 0, 1: 0, 2: 0, 3: 0}
        for p in pl:
            o = int(p[1])
            if o >= 0: cnt[o] = cnt.get(o, 0) + 1
        mypk = max(mypk, cnt[seat])
        opppk = max(opppk, max(cnt[s] for s in range(4) if s != seat))
        if cnt[seat] == 0 and mypk > 0 and elim is None: elim = j
    aT = elim if elim is not None else T
    L = sum(len(steps[j][seat].get("action") or []) for j in range(aT))
    lpas = L / max(1, aT)
    return (win, lpas, mypk, opppk, 1 if elim is not None else 0, elim if elim is not None else T)


def wilson(w, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = w / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return ((c-m)/d, (c+m)/d)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    workers = int(os.environ.get("OW_WORKERS", "8"))
    tasks = [(i, 20260000 + i) for i in range(N)]
    print(f"[4p-full] {AGENT} vs 3x {OPP}  N={N}", flush=True)
    res = []
    # NO maxtasksperchild=1: process restart cost 5.8s/game (import 1.8 + re-jit 4.0) was 82% of eval time
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(run_one, tasks):
            res.append(r)
            if len(res) % 20 == 0:
                print(f"[4p-full] {len(res)}/{N} done", flush=True)
    n = len(res)
    wins = sum(r[0] for r in res); lo, hi = wilson(wins, n)
    lpas = sum(r[1] for r in res)/n
    mypk = sum(r[2] for r in res)/n; opppk = sum(r[3] for r in res)/n
    elim = sum(r[4] for r in res); elim_step = sum(r[5] for r in res)/n
    print(f"\n[4p-full] DONE: {AGENT}  (N={n})")
    print(f"  WIN(1st)        = {wins}/{n} = {100*wins/n:.1f}%  (Wilson95 {100*lo:.1f}-{100*hi:.1f}%)  [fair=25%]")
    print(f"  launches/alive  = {lpas:.2f}   <-- turtle metric (ladder 4p was 0.51, 2p 1.60)")
    print(f"  peak planets    = US {mypk:.1f}  vs best-opp {opppk:.1f}")
    print(f"  eliminated      = {elim}/{n} = {100*elim/n:.0f}%   (avg elim/end step {elim_step:.0f})")


if __name__ == "__main__":
    main()
