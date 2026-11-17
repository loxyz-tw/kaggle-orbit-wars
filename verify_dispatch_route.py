import sys, os, importlib.util as iu
from kaggle_environments import make
# Route verification: the dispatch submission's moves must equal the standalone single-model
# submission it routes to, move for move (expect 0 mismatches). Run 2p and 4p in SEPARATE processes
# (the dispatch caches the detected player count per game).
# Note: replay observations lack obs['step']; we inject it so detection runs once at step 0 as on the ladder.
#   Usage: OW_DISP=<dispatch.py> OW_STD2P=<2p single-model .py> python verify_dispatch_route.py 2
#          OW_DISP=<dispatch.py> OW_STD4P=<4p single-model .py> python verify_dispatch_route.py 4
def load(p,n):
    s=iu.spec_from_file_location(n,p); m=iu.module_from_spec(s); s.loader.exec_module(m); return m
def need(v):
    x=os.environ.get(v,"")
    if not x: raise SystemExit(f"set {v}=<path>")
    return x
if len(sys.argv)<2 or sys.argv[1] not in ("2","4"): raise SystemExit("usage: verify_dispatch_route.py 2|4")
disp=load(need("OW_DISP"),"disp")
std=load(need("OW_STD2P") if sys.argv[1]=="2" else need("OW_STD4P"),"std")
npl=int(sys.argv[1])
def aeq(a,b,tol=1e-5):
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(aeq(x,y,tol) for x,y in zip(a,b))
    try: return abs(float(a)-float(b))<=tol
    except: return a==b
mm=tot=0
seeds=[20260000,20260001] if npl==2 else [20260002,20260003]
for sd in seeds:
    env=make("orbit_wars",configuration={"seed":sd},debug=False)
    env.run([disp.agent]*npl)
    for t,st in enumerate(env.steps):
        for seat in range(npl):
            ob=st[seat].get("observation")
            if not ob or not ob.get("planets"): continue
            cfg=st[seat].get("configuration")
            ob=dict(ob); ob["step"]=t
            if not aeq(disp.agent(ob,cfg), std.agent(ob,cfg)): mm+=1
            tot+=1
print(f"{npl}P: {mm}/{tot} mismatches (expect 0)")
