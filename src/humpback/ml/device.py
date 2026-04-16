"""Device selection for PyTorch workloads.

Order of preference: MPS (macOS) → CUDA (Linux) → CPU. The
``HUMPBACK_FORCE_CPU=1`` env var overrides the auto-selection so CI and
determinism-sensitive tests can pin to CPU.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Default tolerances for ``select_and_validate_device``. These compare the
# CPU and target-device output of one forward pass on a deterministic
# sample input. They are intentionally loose enough to accept normal
# fp32 GPU rounding but tight enough to catch the kind of silent
# divergence MPS BiGRU layers have historically produced. Tunable here
# if false fallbacks appear in practice.
DEFAULT_VALIDATION_RTOL = 1e-4
DEFAULT_VALIDATION_ATOL = 1e-5


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


def select_and_validate_device(
    model: nn.Module,
    sample_input: torch.Tensor,
    *,
    rtol: float = DEFAULT_VALIDATION_RTOL,
    atol: float = DEFAULT_VALIDATION_ATOL,
) -> tuple[torch.device, str | None]:
    """Pick a device for ``model`` and validate the choice with one forward pass.

    Returns ``(device, fallback_reason)``:

    - If ``select_device()`` returns CPU (no GPU available, or
      ``HUMPBACK_FORCE_CPU=1``), short-circuits to ``(cpu, None)``
      without running any forward call. The model is left untouched.
    - Otherwise runs one forward on CPU and one on the target device
      with the same ``sample_input``, and compares with
      ``torch.allclose(rtol=rtol, atol=atol)``. The model is mutated in
      place — moved to whichever device is ultimately selected.

    Fallback reasons (only set when a non-CPU target was attempted and
    rejected): ``"<backend>_load_error"`` if the target-device forward
    raised, ``"<backend>_output_mismatch"`` if outputs diverged.
    """
    target_device = select_device()
    if target_device.type == "cpu":
        return target_device, None

    backend = target_device.type

    try:
        model.to("cpu")
        with torch.no_grad():
            cpu_output = model(sample_input.to("cpu"))
    except Exception:
        logger.warning(
            "CPU validation forward failed for %s target device; staying on CPU",
            backend,
            exc_info=True,
        )
        model.to("cpu")
        return torch.device("cpu"), f"{backend}_load_error"

    try:
        model.to(target_device)
        with torch.no_grad():
            target_output = model(sample_input.to(target_device))
    except Exception:
        logger.warning(
            "Target-device forward failed on %s; falling back to CPU",
            backend,
            exc_info=True,
        )
        model.to("cpu")
        return torch.device("cpu"), f"{backend}_load_error"

    target_output_cpu = target_output.detach().to("cpu")
    if not torch.allclose(cpu_output, target_output_cpu, rtol=rtol, atol=atol):
        logger.warning(
            "Output mismatch between CPU and %s beyond tolerance (rtol=%s, atol=%s); "
            "falling back to CPU",
            backend,
            rtol,
            atol,
        )
        model.to("cpu")
        return torch.device("cpu"), f"{backend}_output_mismatch"

    return target_device, None
