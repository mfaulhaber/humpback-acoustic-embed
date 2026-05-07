"""Shared audio fixture writers for tests."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def write_sine_wav(
    path: Path,
    duration_sec: float,
    sample_rate: int = 16000,
    freq_hz: float = 440.0,
    amplitude: float = 0.7,
) -> None:
    """Write a mono 16-bit PCM sine-wave WAV fixture."""
    n_samples = int(sample_rate * duration_sec)
    peak = max(-1.0, min(1.0, amplitude))
    samples = [
        int(32767 * peak * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n_samples)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
