"""Tests for ``humpback.ml.device.select_device``."""

from __future__ import annotations

from unittest.mock import patch

import torch

from humpback.ml.device import select_device


def test_force_cpu_env_overrides_everything() -> None:
    with patch.dict("os.environ", {"HUMPBACK_FORCE_CPU": "1"}):
        assert select_device() == torch.device("cpu")


def test_selects_mps_when_available() -> None:
    import os

    os.environ.pop("HUMPBACK_FORCE_CPU", None)
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.backends.mps.is_built", return_value=True),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert select_device() == torch.device("mps")


def test_selects_cuda_when_mps_unavailable() -> None:
    import os

    os.environ.pop("HUMPBACK_FORCE_CPU", None)
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.backends.mps.is_built", return_value=False),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert select_device() == torch.device("cuda")


def test_falls_back_to_cpu_when_nothing_available() -> None:
    import os

    os.environ.pop("HUMPBACK_FORCE_CPU", None)
    with (
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.backends.mps.is_built", return_value=False),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert select_device() == torch.device("cpu")
