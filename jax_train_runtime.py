"""JAX BC/PPO training runtime guard — import before `import jax`."""

from __future__ import annotations

import os
import sys


def _argv_has_flag(*names: str) -> bool:
    return any(n in sys.argv for n in names)


def _argv_int(flag: str, *, default: int) -> int:
    if flag not in sys.argv:
        return default
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return default
    try:
        return max(1, int(sys.argv[i + 1]))
    except ValueError:
        return default


def configure_cpu_thread_budget(max_threads: int) -> None:
    """Cap BLAS / OpenMP threads so host-side board-feature loops do not grab all cores."""
    n = str(max(1, int(max_threads)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = n


def configure_jax_training_runtime() -> None:
    """Set platform allocator; refuse accidental CPU-only training (CLI entrypoints only)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    configure_cpu_thread_budget(_argv_int("--cpu-threads", default=6))
    if _argv_has_flag("--allow-cpu-training", "--help", "-h"):
        return
    cpu_platform = os.environ.get("JAX_PLATFORMS", "").strip().lower() == "cpu"
    cuda_hidden = os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
    if cpu_platform or cuda_hidden:
        reasons = []
        if cpu_platform:
            reasons.append("JAX_PLATFORMS=cpu")
        if cuda_hidden:
            reasons.append("CUDA_VISIBLE_DEVICES=''")
        raise SystemExit(
            "refuse to run JAX training on CPU: "
            + ", ".join(reasons)
            + ". Unset those vars or pass --allow-cpu-training."
        )
