import wave
from pathlib import Path

import numpy as np


def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode a WAV file to float32 numpy array and sample rate.

    For MP3 support, requires librosa (optional dependency).
    """
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return _decode_wav(path)
    elif suffix == ".mp3":
        return _decode_mp3(path)
    elif suffix == ".flac":
        return _decode_flac(path)
    else:
        raise ValueError(f"Unsupported audio format: {suffix}")


def _decode_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "r") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        raw = wf.readframes(n_frames)
        sampwidth = wf.getsampwidth()

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    # Normalize to [-1, 1]
    audio /= np.iinfo(dtype).max
    return audio, sr


def _decode_mp3(path: Path) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for MP3 decoding. Install with: pip install librosa")
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return audio, sr


def _decode_flac(path: Path) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required for FLAC decoding. Install with: uv add librosa")
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    return audio, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback: simple linear interpolation
        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
