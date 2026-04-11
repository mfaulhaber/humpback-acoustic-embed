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
import scipy.signal
from scipy.signal import stft


@dataclass(frozen=True)
class PcenParams:
    """PCEN tuning parameters with defaults tuned for hydrophone audio.

    Uses librosa's default ``bias``/``power`` (2.0, 0.5) for well-behaved
    dynamic range, ``gain=0.98`` for aggressive per-bin AGC (empirically
    lowering gain produces severe darkness at ``vmax=1.0`` because the
    per-bin steady-state output scales roughly as ``M^(1-gain)``), and a
    longer ``time_constant=2.0`` so a 1–3 s whale vocalization is not
    fully tracked out mid-call at coarse zoom levels.
    """

    time_constant: float = 2.0
    gain: float = 0.98
    bias: float = 2.0
    power: float = 0.5
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

    # librosa.pcen's default initial filter state is lfilter_zi scaled
    # for a unit-amplitude step input, which leaves the per-bin low-pass
    # starting around 1.0. Hydrophone STFT magnitudes are typically
    # orders of magnitude smaller, so the default zi causes the first
    # many frames to have near-zero PCEN output (the denominator is
    # still decaying from ~1 toward the actual signal level). That
    # shows up as a dark strip at the left edge of every tile.
    #
    # Pre-scaling zi by the first frame's magnitude places each
    # frequency bin's low-pass filter at its own settled level from
    # the first frame on, eliminating the cold-start transient.
    t_frames = params.time_constant * sample_rate / float(hop_length)
    b_coef = (np.sqrt(1.0 + 4.0 * t_frames**2) - 1.0) / (2.0 * t_frames**2)
    zi_base = scipy.signal.lfilter_zi([b_coef], [1.0, b_coef - 1.0])
    zi = magnitude[:, 0:1] * zi_base

    pcen = librosa.pcen(
        magnitude,
        sr=sample_rate,
        hop_length=hop_length,
        time_constant=params.time_constant,
        gain=params.gain,
        bias=params.bias,
        power=params.power,
        eps=params.eps,
        b=b_coef,
        zi=zi,
    )

    if not np.all(np.isfinite(pcen)):
        raise ValueError("PCEN output contains non-finite values")

    if warmup_samples > 0 and pcen.shape[1] > 0:
        warmup_frames = min(warmup_samples // hop_length, pcen.shape[1])
        pcen = pcen[:, warmup_frames:]

    return freqs, pcen
