"""Build a self-contained dual-expert dispatch submission.

Embeds TWO checkpoints into the clean template: a 2-player model (_PARAMS_B64) and a
4-player model (_PARAMS_4P_B64). The agent detects the player count at game start and
routes to the matching model (see lean_submission_numpy.agent).

Run from repo root:
    python submission/_build_dispatch_submission.py <2p_ckpt> <4p_ckpt> [out_path]
Default out_path: submission/sub_dispatch_<YYYYMMDD>.py
"""
from __future__ import annotations

import base64
import datetime
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

if len(sys.argv) < 3:
    raise SystemExit("usage: python submission/_build_dispatch_submission.py <2p_ckpt> <4p_ckpt> [out_path]")
CKPT_2P = sys.argv[1]
CKPT_4P = sys.argv[2]
TEMPLATE = Path("submission/lean_submission_numpy.py")
if len(sys.argv) > 3:
    OUT = Path(sys.argv[3])
else:
    OUT = Path(f"submission/sub_dispatch_{datetime.date.today():%Y%m%d}.py")


def to_numpy_tree(d):
    if hasattr(d, "items"):
        return {str(k): to_numpy_tree(v) for k, v in d.items()}
    return np.asarray(d, dtype=np.float32)


def embed(ckpt_path):
    payload = pickle.loads(Path(ckpt_path).read_bytes())
    params = to_numpy_tree(payload["params"])
    for k in [k for k in list(params) if k.startswith("c_")]:  # drop dual-trunk critic
        del params[k]
    cfg = payload.get("config", {})
    nfb = int(params["frac_head"]["kernel"].shape[-1])    # bin count from weights
    fb = cfg.get("frac_bins")
    if fb and len(fb.split(",")) == nfb:
        frac_bins = np.asarray([float(v) for v in fb.split(",")], dtype=np.float32)
    elif nfb == 20:
        frac_bins = np.asarray([0.05 * i for i in range(1, 21)], dtype=np.float32)
    else:                                                 # NO silent 3-bin default; fail loud
        raise SystemExit(f"can't determine frac_bins for nfb={nfb}; cfg frac_bins={fb!r}")
    blob = {
        "params": params, "frac_bins": frac_bins,
        "min_launch": int(cfg.get("min_launch", 1)),
        "rich_features": bool(cfg.get("rich_features", False)) or ("rich" in str(cfg.get("obs", ""))),
        "orbit_features": bool(cfg.get("orbit_features", False)) or ("orbit" in str(cfg.get("obs", ""))),
        "autoregressive": bool(cfg.get("autoregressive", False)),
        "is_4p_feature": bool(cfg.get("is_4p_feature", False)),
        "rank_feature": bool(cfg.get("rank_feature", False)),
    }
    raw = pickle.dumps(blob, protocol=4)
    b64 = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    print(f"  {ckpt_path}: min_launch={blob['min_launch']} rich={blob['rich_features']} "
          f"raw={len(raw)}B b64={len(b64)} chars")
    return b64


def main():
    print("embedding 2p model:")
    b64_2p = embed(CKPT_2P)
    print("embedding 4p model:")
    b64_4p = embed(CKPT_4P)

    text = TEMPLATE.read_text()
    if '_PARAMS_B64 = ""' not in text or '_PARAMS_4P_B64 = ""' not in text:
        raise SystemExit(f"{TEMPLATE} is not a clean dual template (empty _PARAMS_B64 / _PARAMS_4P_B64 markers not found).")
    if OUT.resolve() == TEMPLATE.resolve():
        raise SystemExit("refusing to overwrite the template; pass a distinct out_path.")
    text = text.replace('_PARAMS_B64 = ""', f'_PARAMS_B64 = "{b64_2p}"', 1)
    text = text.replace('_PARAMS_4P_B64 = ""', f'_PARAMS_4P_B64 = "{b64_4p}"', 1)
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes) — 2p={CKPT_2P}  4p={CKPT_4P}")


if __name__ == "__main__":
    main()
