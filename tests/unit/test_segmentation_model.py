"""Tests for the Pass 2 SegmentationCRNN model module."""

from __future__ import annotations

import torch

from humpback.call_parsing.segmentation.model import SegmentationCRNN


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_forward_shape_matches_input_time_dim() -> None:
    model = SegmentationCRNN()
    model.eval()
    batch, frames = 2, 128
    x = torch.zeros(batch, 1, model.n_mels, frames)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch, frames)


def test_parameter_count_in_target_range() -> None:
    model = SegmentationCRNN()
    count = _count_parameters(model)
    assert 240_000 <= count <= 360_000, f"param count {count} outside target range"


def test_forward_is_deterministic_under_fixed_seed() -> None:
    torch.manual_seed(42)
    model_a = SegmentationCRNN()
    torch.manual_seed(42)
    model_b = SegmentationCRNN()

    model_a.eval()
    model_b.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 1, model_a.n_mels, 64)
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b)


def test_variable_length_inputs_produce_variable_length_outputs() -> None:
    model = SegmentationCRNN()
    model.eval()
    x_short = torch.zeros(1, 1, model.n_mels, 64)
    x_long = torch.zeros(1, 1, model.n_mels, 96)
    with torch.no_grad():
        out_short = model(x_short)
        out_long = model(x_long)
    assert out_short.shape[-1] == 64
    assert out_long.shape[-1] == 96
    assert out_short.shape[-1] != out_long.shape[-1]


def test_gradients_flow_through_all_leaf_parameters() -> None:
    model = SegmentationCRNN()
    model.train()
    x = torch.randn(2, 1, model.n_mels, 64)
    out = model(x)
    loss = out.mean()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
