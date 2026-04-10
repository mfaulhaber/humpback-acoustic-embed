"""PCEN-based spectrogram normalization for timeline tile rendering.

Wraps ``librosa.pcen`` to replace heuristic gain-step detection and per-job
reference-dB computation with a single stateless per-tile normalization.
PCEN is the field-standard bioacoustic AGC described in Lostanlen et al.,
"Per-Channel Energy Normalization: Why and How".
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class PcenParams:
    """PCEN tuning parameters with defaults chosen for hydrophone audio.

    Defaults use a slightly longer ``time_constant`` than librosa's 0.4 s
    preset so a 1–2 s whale vocalization is less likely to AGC itself out
    mid-call, combined with stronger root compression (``power=0.25``,
    ``bias=10``) for clearer contrast on timeline tiles.
    """

    time_constant: float = 0.5
    gain: float = 0.98
    bias: float = 10.0
    power: float = 0.25
    eps: float = 1e-6


def render_tile_pcen(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    warmup_samples: int = 0,
    params: PcenParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a PCEN-normalized spectrogram for a timeline tile.

    The caller is expected to fetch ``warmup_samples`` of audio immediately
    preceding the tile so PCEN's per-frequency low-pass filter has room to
    settle before the first rendered frame. Those leading frames are
    trimmed off the left edge of the output before returning.

    Returns:
        ``(frequencies, pcen_power)`` — a 1-D array of STFT frequency bins
        and a 2-D ``(n_freqs, n_frames)`` PCEN output array. The output is
        in PCEN's bounded range; callers should render it with fixed
        ``vmin``/``vmax`` rather than deriving a dynamic range from the
        data.

    Raises:
        ValueError: If the input audio or the PCEN output contains
            non-finite values.
    """
    if params is None:
        params = PcenParams()

    freqs_ref = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    if len(audio) == 0:
        return freqs_ref, np.zeros((len(freqs_ref), 0), dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Input audio contains non-finite values")

    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    noverlap = n_fft - hop_length
    # boundary=None / padded=False disable scipy's default zero-padding
    # at the STFT edges. Boundary frames dominated by zeros would corrupt
    # PCEN's per-frequency low-pass filter state at the start of every
    # tile; skipping them is cheaper and more accurate than trying to
    # compensate downstream.
    freqs, _times, Zxx = stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary=None,  # type: ignore[arg-type]
        padded=False,
    )
    # librosa.pcen operates on a magnitude spectrogram; feeding |Zxx|^2
    # produces runaway values because PCEN's internal low-pass filter
    # is tuned for magnitude envelopes.
    magnitude = np.abs(Zxx).astype(np.float32)

    pcen = librosa.pcen(
        magnitude,
        sr=sample_rate,
        hop_length=hop_length,
        time_constant=params.time_constant,
        gain=params.gain,
        bias=params.bias,
        power=params.power,
        eps=params.eps,
    )

    if not np.all(np.isfinite(pcen)):
        raise ValueError("PCEN output contains non-finite values")

    if warmup_samples > 0 and pcen.shape[1] > 0:
        warmup_frames = min(warmup_samples // hop_length, pcen.shape[1])
        pcen = pcen[:, warmup_frames:]

    return freqs, pcen
