"""Zero-init embed-extend a ckpt for FEATURE SURGERY warm-start: pad the input embedding
(/embed/kernel, shape (F,H)) by n_extra ZERO rows -> (F+n_extra, H). The new feature dims contribute
NOTHING at step 0, so behavior == the source ckpt until training adapts the new weights (EXACT
warm-start). Use as --init-checkpoint of a run with --gravity-features/--roi-features (which append
n_extra cols to obs-v2). Drops opt_state (shape changed -> must reset Adam).
Usage: python extend_ckpt.py <in_ckpt> <out_ckpt> <n_extra>
"""
import sys, pickle
import numpy as np


def find_embed(params):
    """Return (parent_dict, key) of the input embedding kernel (a 2D leaf whose first dim is the
    feature count). Prefer an exact 'embed'/'kernel' path; else the 2D leaf with the smallest dim0."""
    if "embed" in params and isinstance(params["embed"], dict) and "kernel" in params["embed"]:
        return params["embed"], "kernel"
    best = None
    def walk(d):
        nonlocal best
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v)
            else:
                a = np.asarray(v)
                if a.ndim == 2 and (best is None or a.shape[0] < np.asarray(best[0][best[1]]).shape[0]):
                    best = (d, k)
    walk(params)
    if best is None:
        raise SystemExit("no 2D embedding kernel found")
    return best


def main():
    inp, out, n_extra = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ck = pickle.load(open(inp, "rb"))
    parent, key = find_embed(ck["params"])
    emb = np.asarray(parent[key])
    F, H = emb.shape
    ext = np.zeros((F + n_extra, H), emb.dtype)
    ext[:F] = emb
    parent[key] = ext
    ck["opt_state"] = None  # shapes changed -> force Adam reset (no --resume-opt-state)
    pickle.dump(ck, open(out, "wb"))
    print(f"extended embed {F}->{F + n_extra} (rows {F}..{F+n_extra-1} zero) -> {out}", flush=True)


if __name__ == "__main__":
    main()
