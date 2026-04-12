"""EventClassifierCNN — Pass 3 multi-label event-type classifier.

A small convolutional network that takes a variable-length log-mel
spectrogram of an event crop and returns per-type logits.  Four
``Conv2d -> BatchNorm2d -> ReLU`` blocks with ``MaxPool2d((2, 1))``
after each — pooling only in the frequency axis so the time dimension
is preserved for arbitrarily short events (down to ~0.2 s).
``AdaptiveAvgPool2d((1, 1))`` collapses both axes into a fixed-size
vector before the multi-label classification head.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

DEFAULT_N_MELS = 64
DEFAULT_CONV_CHANNELS: tuple[int, ...] = (32, 64, 128, 256)


class EventClassifierCNN(nn.Module):
    def __init__(
        self,
        n_types: int,
        n_mels: int = DEFAULT_N_MELS,
        conv_channels: Sequence[int] = DEFAULT_CONV_CHANNELS,
    ) -> None:
        super().__init__()
        if n_types < 1:
            raise ValueError("n_types must be >= 1")
        channels = tuple(conv_channels)
        if not channels:
            raise ValueError("conv_channels must contain at least one entry")

        self.n_types = n_types
        self.n_mels = n_mels
        self.conv_channels = channels

        blocks: list[nn.Module] = []
        prev_ch = 1
        for out_ch in channels:
            blocks.extend(
                [
                    nn.Conv2d(prev_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d((2, 1)),
                ]
            )
            prev_ch = out_ch
        self.conv = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(channels[-1], n_types)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "EventClassifierCNN expects 4-D input (B, 1, n_mels, T), "
                f"got shape {tuple(x.shape)}"
            )
        feat = self.conv(x)
        feat = self.pool(feat)
        feat = feat.flatten(1)
        return self.head(feat)
