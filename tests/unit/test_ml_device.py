"""Tests for ``humpback.ml.device``."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
from torch import nn

from humpback.ml.device import select_and_validate_device, select_device


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


class _FakeDeviceModel(nn.Module):
    """A test-only nn.Module that fakes device moves.

    Stores the most recent device passed to ``.to(...)`` in
    ``_fake_device`` and adapts ``forward`` based on that. Lets us
    simulate target-device load errors and output divergence without
    needing real GPU hardware.
    """

    def __init__(
        self,
        *,
        raise_on_target: bool = False,
        diverge_on_target: bool = False,
    ) -> None:
        super().__init__()
        # A real parameter so .to(...) forwards the move through nn.Module
        # mechanics, even though we override the recorded device below.
        self.dummy = nn.Parameter(torch.zeros(1))
        self._fake_device = torch.device("cpu")
        self._raise_on_target = raise_on_target
        self._diverge_on_target = diverge_on_target
        self.forward_calls = 0
        self.move_calls: list[torch.device] = []

    def to(self, *args, **kwargs):  # type: ignore[override]
        device: torch.device | None = None
        if args:
            arg0 = args[0]
            if isinstance(arg0, torch.device):
                device = arg0
            elif isinstance(arg0, str):
                device = torch.device(arg0)
        if device is None and "device" in kwargs:
            raw = kwargs["device"]
            device = raw if isinstance(raw, torch.device) else torch.device(raw)
        if device is not None:
            self._fake_device = device
            self.move_calls.append(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        if self._fake_device.type != "cpu":
            if self._raise_on_target:
                raise RuntimeError(
                    f"simulated {self._fake_device.type} forward failure"
                )
            if self._diverge_on_target:
                return torch.full((1, 2), 100.0)
        return torch.zeros(1, 2)


def test_validate_force_cpu_short_circuits() -> None:
    model = _FakeDeviceModel()
    sample = torch.zeros(1, 4)
    with patch.dict("os.environ", {"HUMPBACK_FORCE_CPU": "1"}):
        device, reason = select_and_validate_device(model, sample)
    assert device == torch.device("cpu")
    assert reason is None
    assert model.forward_calls == 0
    assert model.move_calls == []


def test_validate_no_gpu_short_circuits() -> None:
    model = _FakeDeviceModel()
    sample = torch.zeros(1, 4)
    with patch("humpback.ml.device.select_device", return_value=torch.device("cpu")):
        device, reason = select_and_validate_device(model, sample)
    assert device == torch.device("cpu")
    assert reason is None
    assert model.forward_calls == 0
    assert model.move_calls == []


class _FakeInput:
    """A stand-in for ``sample_input`` that no-ops ``.to(...)``.

    Lets us drive the validation path with target devices that may not
    be physically present (e.g. CUDA on macOS), since
    ``torch.Tensor.to("cuda")`` would otherwise raise during the test.
    """

    def to(self, _device):  # noqa: ANN001 - test helper
        return self


def test_validate_target_load_error_falls_back() -> None:
    fake_target = torch.device("mps")
    model = _FakeDeviceModel(raise_on_target=True)
    with patch("humpback.ml.device.select_device", return_value=fake_target):
        device, reason = select_and_validate_device(model, _FakeInput())  # type: ignore[arg-type]
    assert device == torch.device("cpu")
    assert reason == "mps_load_error"
    assert model._fake_device == torch.device("cpu")


def test_validate_output_mismatch_falls_back() -> None:
    fake_target = torch.device("cuda")
    model = _FakeDeviceModel(diverge_on_target=True)
    with patch("humpback.ml.device.select_device", return_value=fake_target):
        device, reason = select_and_validate_device(model, _FakeInput())  # type: ignore[arg-type]
    assert device == torch.device("cpu")
    assert reason == "cuda_output_mismatch"
    assert model._fake_device == torch.device("cpu")


@pytest.mark.skipif(
    not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ),
    reason="MPS not available on this host",
)
def test_validate_happy_path_mps() -> None:
    import os

    os.environ.pop("HUMPBACK_FORCE_CPU", None)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    sample = torch.randn(2, 8)
    device, reason = select_and_validate_device(model, sample)
    assert device == torch.device("mps")
    assert reason is None
    assert next(model.parameters()).device.type == "mps"
