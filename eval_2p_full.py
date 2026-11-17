"""2p head-to-head via the official kaggle env: AGENT vs OPP, rotating seats, fixed seeds.

Usage: OW_AGENT=<our submission .py> OW_OPP=<your opponent agent .py> OW_WORKERS=8 \
       OMP_NUM_THREADS=1 python eval_2p_full.py [N]
Reports AGENT win% with a Wilson 95% interval (and OPP win% = 100 - win% - tie%).
"""
import os, sys, math, multiprocessing as mp

def _require(var):
    v = os.environ.get(var, "")
    if not v:
        raise SystemExit(f"set {var}=<path to agent .py>")
    return v

AGENT = _require("OW_AGENT")
OPP = _require("OW_OPP")


# Agents are handed to kaggle as file paths (ladder-identical fresh-exec semantics each game).
def _resolve(path):
    return path


def run_one(args):
    i, seed = args
    import logging; logging.disable(logging.INFO)
    from kaggle_environments import make
    seat = i % 2
    agents = [_resolve(OPP), _resolve(OPP)]; agents[seat] = _resolve(AGENT)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    final = env.steps[-1]
    rew = [float(final[s]["reward"]) if final[s]["reward"] is not None else -1e9 for s in range(2)]
    me, opp = rew[seat], rew[1 - seat]
    win = 1 if me > opp else 0
    tie = 1 if me == opp else 0
    return (win, tie)


def wilson(w, n):
    if n == 0: return (0.0, 0.0)
    z = 1.96; p = w / n; d = 1 + z*z/n; c = p + z*z/(2*n)
    m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return ((c-m)/d, (c+m)/d)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    workers = int(os.environ.get("OW_WORKERS", "8"))
    tasks = [(i, 20260000 + i) for i in range(N)]
    print(f"[2p] {AGENT} vs {OPP}  N={N}", flush=True)
    res = []
    # NO maxtasksperchild=1: process restart cost 5.8s/game (import 1.8 + re-jit 4.0) was 82% of eval time
    with mp.get_context("spawn").Pool(workers) as pool:
        for r in pool.imap_unordered(run_one, tasks):
            res.append(r)
            if len(res) % 20 == 0:
                print(f"[2p] {len(res)}/{N} done", flush=True)
    n = len(res)
    wins = sum(r[0] for r in res); ties = sum(r[1] for r in res)
    lo, hi = wilson(wins, n)
    print(f"\n[2p] DONE: {AGENT}  (N={n})")
    print(f"  AGENT win = {wins}/{n} = {100*wins/n:.1f}%  (Wilson95 {100*lo:.1f}-{100*hi:.1f}%)  tie={100*ties/n:.1f}%")
    print(f"  OPP wins = {100*(n-wins-ties)/n:.1f}%")


if __name__ == "__main__":
    main()
