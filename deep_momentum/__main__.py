"""Command-line entry point for the deep-momentum and gating experiments."""

from __future__ import annotations

import os


# Apply the single-thread-per-process policy before NumPy/SciPy or PyTorch is
# imported. This keeps direct CLI invocations equivalent to the Make targets
# and prevents nested BLAS pools from oversubscribing multi-core hosts.
os.environ["KERAS_BACKEND"] = "torch"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from .cli import main


if __name__ == "__main__":
    main()
