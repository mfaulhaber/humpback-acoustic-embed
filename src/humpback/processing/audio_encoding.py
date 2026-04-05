"""Audio encoding utilities for WAV and MP3 output."""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to 16-bit PCM WAV bytes with peak normalization."""
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm = (audio_clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to MP3 via ffmpeg subprocess."""
    wav_bytes = encode_wav(audio, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
        wav_f.write(wav_bytes)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
        mp3_path = mp3_f.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-ac",
                "1",
                mp3_path,
            ],
            capture_output=True,
            check=True,
        )
        return Path(mp3_path).read_bytes()
    finally:
        Path(wav_path).unlink(missing_ok=True)
        Path(mp3_path).unlink(missing_ok=True)
