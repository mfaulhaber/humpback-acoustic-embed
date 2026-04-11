"""Device selection for PyTorch workloads.

Order of preference: MPS (macOS) → CUDA (Linux) → CPU. The
``HUMPBACK_FORCE_CPU=1`` env var overrides the auto-selection so CI and
determinism-sensitive tests can pin to CPU.
"""

from __future__ import annotations

import os

import torch


def select_device() -> torch.device:
    """Return the preferred torch device for training/inference.

    Honors ``HUMPBACK_FORCE_CPU=1`` as a hard override. Otherwise prefers
    MPS (Apple Silicon) when available, then CUDA, falling back to CPU.
    """
    if os.environ.get("HUMPBACK_FORCE_CPU") == "1":
        return torch.device("cpu")

    mps_ok = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if mps_ok:
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")
