"""SegmentationCRNN — Pass 2 framewise humpback-presence model.

A convolutional-recurrent model that takes a log-mel tile and returns a
per-frame logit vector. The architecture is four ``Conv2d → BatchNorm2d
→ ReLU`` blocks (only the last strides in time ×2), an
``AdaptiveAvgPool2d`` that collapses the mel/frequency axis, a two-layer
bidirectional GRU, a frame-head ``Linear``, and a nearest-neighbor
upsample that restores the original time resolution. See
``docs/specs/2026-04-11-call-parsing-pass2-segmentation-design.md`` for
the rationale on the ~300k-parameter target.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_N_MELS = 64
DEFAULT_CONV_CHANNELS: tuple[int, ...] = (32, 64, 96, 128)
DEFAULT_GRU_HIDDEN = 64
DEFAULT_GRU_LAYERS = 2


class SegmentationCRNN(nn.Module):
    def __init__(
        self,
        n_mels: int = DEFAULT_N_MELS,
        conv_channels: Sequence[int] = DEFAULT_CONV_CHANNELS,
        gru_hidden: int = DEFAULT_GRU_HIDDEN,
        gru_layers: int = DEFAULT_GRU_LAYERS,
    ) -> None:
        super().__init__()
        channels = tuple(conv_channels)
        if not channels:
            raise ValueError("conv_channels must contain at least one entry")
        if gru_layers < 1:
            raise ValueError("gru_layers must be >= 1")

        self.n_mels = n_mels
        self.conv_channels = channels
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers

        blocks: list[nn.Module] = []
        prev_channels = 1
        last_idx = len(channels) - 1
        for idx, out_channels in enumerate(channels):
            stride: tuple[int, int] = (1, 2) if idx == last_idx else (1, 1)
            blocks.append(
                nn.Conv2d(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                )
            )
            blocks.append(nn.BatchNorm2d(out_channels))
            blocks.append(nn.ReLU(inplace=True))
            prev_channels = out_channels
        self.conv = nn.Sequential(*blocks)

        # Collapse the mel/frequency axis before the recurrent head so the
        # BiGRU sees one summary vector per time step. The conv stack still
        # learns per-bin features; pooling just stops the GRU input from
        # ballooning to ``conv_channels[-1] * n_mels`` and keeps the model
        # near the ~300k parameter target documented in the spec.
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.gru = nn.GRU(
            input_size=prev_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.frame_head = nn.Linear(2 * gru_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SegmentationCRNN expects a 4-D input (B, 1, n_mels, T), got "
                f"{tuple(x.shape)}"
            )
        input_t = int(x.shape[-1])

        feat = self.conv(x)
        feat = self.freq_pool(feat)
        feat = feat.squeeze(2).transpose(1, 2)
        rnn_out, _ = self.gru(feat)
        logits = self.frame_head(rnn_out).squeeze(-1)
        logits = logits.unsqueeze(1)
        # ``size=input_t`` restores the original frame count even when the
        # conv stride on an odd-length input would otherwise leave us one
        # frame short of a clean 2× upsample.
        logits = F.interpolate(logits, size=input_t, mode="nearest")
        return logits.squeeze(1)
