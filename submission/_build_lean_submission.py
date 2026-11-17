"""Embed a lean checkpoint into a self-contained Kaggle submission file.

Reads the CLEAN template submission/lean_submission_numpy.py (must keep an empty
`_PARAMS_B64 = ""` marker) and writes the filled agent to a NEW dated/named file
so an already-uploaded submission is never overwritten.

Run from repo root:
    python submission/_build_lean_submission.py <ckpt_path> [out_path]

Default out_path: submission/sub_<ckpt_basename>_<YYYYMMDD>.py
"""
from __future__ import annotations

import base64
import datetime
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np

if len(sys.argv) < 2:
    raise SystemExit("usage: python submission/_build_lean_submission.py <ckpt_path> [out_path]")
CKPT = sys.argv[1]
TEMPLATE = Path("submission/lean_submission_numpy.py")
if len(sys.argv) > 2:
    OUT = Path(sys.argv[2])
else:
    stamp = datetime.date.today().strftime("%Y%m%d")
    OUT = Path(f"submission/sub_{Path(CKPT).name}_{stamp}.py")


def to_numpy_tree(d):
    if hasattr(d, "items"):
        return {str(k): to_numpy_tree(v) for k, v in d.items()}
    return np.asarray(d, dtype=np.float32)


def main():
    payload = pickle.loads(Path(CKPT).read_bytes())
    params = to_numpy_tree(payload["params"])
    # dual-trunk ckpts carry an independent critic under c_* keys -> dead weight at inference
    # (the actor forward never reads them); drop to keep the submission small.
    dropped = [k for k in list(params) if k.startswith("c_")]
    for k in dropped:
        del params[k]
    if dropped:
        print(f"dropped {len(dropped)} critic (c_*) keys from dual-trunk ckpt")
    cfg = payload.get("config", {})
    nfb = int(params["frac_head"]["kernel"].shape[-1])   # authoritative bin count from weights
    fb = cfg.get("frac_bins")
    if fb and len(fb.split(",")) == nfb:
        frac_bins = np.asarray([float(v) for v in fb.split(",")], dtype=np.float32)
    elif nfb == 20:                                       # FB20U (BC ckpts store num_frac_bins, not the string)
        frac_bins = np.asarray([0.05 * i for i in range(1, 21)], dtype=np.float32)
    else:                                                 # NO silent 3-bin default (caused 0% builds); fail loud
        raise SystemExit(f"can't determine frac_bins for nfb={nfb}; cfg frac_bins={fb!r}. "
                         "Provide a matching frac_bins string in the ckpt config.")
    print(f"frac_bins: {len(frac_bins)} bins (frac_head out={nfb})")
    min_launch = int(cfg.get("min_launch", 1))
    import os as _os
    if _os.environ.get("OW_BUILD_MINLAUNCH"):   # sweep override (min_launch is baked into the submission)
        min_launch = int(_os.environ["OW_BUILD_MINLAUNCH"])
    rich = bool(cfg.get("rich_features", False)) or ("rich" in str(cfg.get("obs", "")))
    orbit = bool(cfg.get("orbit_features", False)) or ("orbit" in str(cfg.get("obs", "")))
    autoregressive = bool(cfg.get("autoregressive", False))
    is_4p = bool(cfg.get("is_4p_feature", False))
    rank = bool(cfg.get("rank_feature", False))
    print(f"min_launch={min_launch} rich_features={rich} orbit_features={orbit} autoregressive={autoregressive} is_4p_feature={is_4p} rank_feature={rank}")

    raw = pickle.dumps({"params": params, "frac_bins": frac_bins,
                        "min_launch": min_launch, "rich_features": rich,
                        "orbit_features": orbit, "autoregressive": autoregressive,
                        "is_4p_feature": is_4p, "rank_feature": rank}, protocol=4)
    b64 = base64.b64encode(zlib.compress(raw, 9)).decode("ascii")
    print(f"ckpt={CKPT} raw={len(raw)} bytes, compressed b64={len(b64)} chars")

    text = TEMPLATE.read_text()
    marker = '_PARAMS_B64 = ""'
    if marker not in text:
        raise SystemExit(f"{TEMPLATE} is not a clean template (empty _PARAMS_B64 marker not found). "
                         "Keep the template weightless; builds go to a separate dated file.")
    if OUT.resolve() == TEMPLATE.resolve():
        raise SystemExit("refusing to overwrite the template; pass a distinct out_path.")
    text = text.replace(marker, f'_PARAMS_B64 = "{b64}"', 1)
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
