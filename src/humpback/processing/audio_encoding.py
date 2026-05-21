"""Audio encoding utilities for WAV, MP3, and FLAC output."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

_SILENCE_RMS_FLOOR = 1e-8


def normalize_for_playback(
    audio: np.ndarray,
    target_rms_dbfs: float = -20.0,
    ceiling: float = 0.95,
) -> np.ndarray:
    """Scale audio to a target RMS level and soft-clip for listener comfort.

    This is used for timeline viewer playback so that short playback chunks
    that span an abrupt mic-gain change do not force the listener to ride
    the volume knob. The ``tanh`` soft-clip prevents harsh clipping on the
    transients that survive the RMS scaling.

    Args:
        audio: Input float audio chunk.
        target_rms_dbfs: Desired output RMS in dBFS (e.g. ``-20.0``).
        ceiling: Soft-clip ceiling in the ``[0.0, 1.0)`` range.

    Returns:
        A new float32 array; the input is not modified.
    """
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    scaled = audio.astype(np.float32, copy=True)
    rms = float(np.sqrt(np.mean(scaled**2)))
    if rms > _SILENCE_RMS_FLOOR:
        target_rms = 10.0 ** (target_rms_dbfs / 20.0)
        scaled *= target_rms / rms

    return (ceiling * np.tanh(scaled / ceiling)).astype(np.float32)


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float audio to 16-bit PCM WAV bytes.

    Callers are expected to have already scaled the audio to the range
    ``[-1.0, 1.0]`` (e.g. via :func:`normalize_for_playback`); this
    function only clips and quantizes.
    """
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
    """Encode float audio to MP3 via ffmpeg subprocess."""
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


def write_flac_clip(samples: np.ndarray, sr: int, path: Path) -> None:
    """Write float32 mono samples to a 16-bit PCM FLAC file.

    Unlike :func:`normalize_for_playback` and the classifier's training-set
    FLAC writer, this helper does NOT scale or normalize the input. The
    resulting clip preserves the loudness the piano-roll player rendered so
    DAW alignment carries through to absolute sample levels.

    The write is atomic: samples are written to ``<path>.tmp`` first and
    only renamed to ``path`` on success.

    Args:
        samples: 1-D float32 audio buffer. Must be finite.
        sr: Sample rate in Hz (must be positive).
        path: Destination ``.flac`` path.

    Raises:
        ValueError: ``samples`` is not 1-D, is not finite, or ``sr <= 0``.
    """
    if samples.ndim != 1:
        raise ValueError(
            f"write_flac_clip requires 1-D samples, got ndim={samples.ndim}"
        )
    if sr <= 0:
        raise ValueError(f"write_flac_clip requires positive sample rate, got sr={sr}")
    if samples.size > 0 and not np.all(np.isfinite(samples)):
        raise ValueError("write_flac_clip received non-finite samples")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        sf.write(
            str(tmp),
            samples.astype(np.float32, copy=False),
            sr,
            format="FLAC",
            subtype="PCM_16",
        )
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
