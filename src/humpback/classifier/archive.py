"""Archive provider protocol for detection pipeline audio sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class StreamSegment:
    """Timeline segment metadata for a single audio object."""

    key: str
    start_ts: float
    duration_sec: float

    @property
    def end_ts(self) -> float:
        return self.start_ts + self.duration_sec


@runtime_checkable
class ArchiveProvider(Protocol):
    """Protocol for audio archive sources used by the detection pipeline.

    Each provider encapsulates discovery, fetching, and decoding for a
    specific archive format (HLS .ts, NOAA .aif, local .wav, etc.).
    """

    @property
    def name(self) -> str:
        """Display name (e.g. 'Orcasound Lab')."""
        ...

    @property
    def source_id(self) -> str:
        """Machine identifier (e.g. 'rpi_orcasound_lab')."""
        ...

    def build_timeline(self, start_ts: float, end_ts: float) -> list[StreamSegment]:
        """Return ordered segments covering [start_ts, end_ts)."""
        ...

    def count_segments(self, start_ts: float, end_ts: float) -> int:
        """Estimate total segments in range (for progress bars)."""
        ...

    def fetch_segment(self, key: str) -> bytes:
        """Download/read a single segment by key."""
        ...

    def decode_segment(self, raw_bytes: bytes, target_sr: int) -> np.ndarray:
        """Decode raw segment bytes to float32 audio array."""
        ...

    def invalidate_cached_segment(self, key: str) -> bool:
        """Invalidate a cached segment. Returns False if no cache."""
        ...
