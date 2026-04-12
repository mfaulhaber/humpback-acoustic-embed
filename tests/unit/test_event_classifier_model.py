"""Tests for EventClassifierCNN (Pass 3 model architecture)."""

from __future__ import annotations

import pytest
import torch

from humpback.call_parsing.event_classifier.model import EventClassifierCNN


class TestEventClassifierCNN:
    def test_output_shape_various_time_lengths(self) -> None:
        model = EventClassifierCNN(n_types=5)
        model.eval()
        for t in (6, 15, 50, 156):
            x = torch.randn(2, 1, 64, t)
            out = model(x)
            assert out.shape == (2, 5), f"Failed for T={t}: got {out.shape}"

    def test_single_sample_batch(self) -> None:
        model = EventClassifierCNN(n_types=3)
        model.eval()
        x = torch.randn(1, 1, 64, 20)
        out = model(x)
        assert out.shape == (1, 3)

    def test_minimum_time_dimension(self) -> None:
        model = EventClassifierCNN(n_types=2)
        model.eval()
        x = torch.randn(1, 1, 64, 1)
        out = model(x)
        assert out.shape == (1, 2)

    def test_frequency_only_pooling(self) -> None:
        model = EventClassifierCNN(n_types=3)
        model.eval()
        x = torch.randn(1, 1, 64, 30)
        feat = model.conv(x)
        assert feat.shape[2] == 64 // (2**4), "Frequency should be halved 4 times"
        assert feat.shape[3] == 30, "Time dimension should be preserved"

    def test_param_count_in_expected_range(self) -> None:
        model = EventClassifierCNN(n_types=10)
        n_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < n_params < 1_000_000, f"Param count {n_params} out of range"

    def test_deterministic_output(self) -> None:
        torch.manual_seed(42)
        model = EventClassifierCNN(n_types=3)
        model.eval()
        x = torch.randn(1, 1, 64, 20)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        assert torch.equal(out1, out2)

    def test_gradients_flow(self) -> None:
        model = EventClassifierCNN(n_types=3)
        model.train()
        x = torch.randn(2, 1, 64, 20)
        out = model(x)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_outputs_raw_logits(self) -> None:
        model = EventClassifierCNN(n_types=3)
        model.eval()
        x = torch.randn(4, 1, 64, 20)
        with torch.no_grad():
            out = model(x)
        assert out.min() < 0 or out.max() > 1, (
            "Outputs should be raw logits, not bounded"
        )

    def test_custom_channels(self) -> None:
        model = EventClassifierCNN(n_types=2, conv_channels=(16, 32))
        model.eval()
        x = torch.randn(1, 1, 64, 10)
        out = model(x)
        assert out.shape == (1, 2)

    def test_invalid_n_types(self) -> None:
        with pytest.raises(ValueError, match="n_types must be >= 1"):
            EventClassifierCNN(n_types=0)

    def test_empty_conv_channels(self) -> None:
        with pytest.raises(
            ValueError, match="conv_channels must contain at least one entry"
        ):
            EventClassifierCNN(n_types=2, conv_channels=())

    def test_invalid_input_ndim(self) -> None:
        model = EventClassifierCNN(n_types=2)
        with pytest.raises(ValueError, match="expects 4-D input"):
            model(torch.randn(64, 20))
