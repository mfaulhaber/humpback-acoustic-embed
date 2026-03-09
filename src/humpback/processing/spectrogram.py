"""Generate STFT spectrogram PNG images for detection clips."""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import stft  # noqa: E402


def generate_spectrogram_png(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 128,
    n_fft: int = 2048,
    dynamic_range_db: float = 80.0,
    width_px: int = 640,
    height_px: int = 320,
) -> bytes:
    """Render an STFT spectrogram of *audio* and return PNG bytes.

    Parameters
    ----------
    audio : 1-D float array
    sample_rate : sample rate in Hz
    hop_length : STFT hop in samples
    n_fft : FFT window size
    dynamic_range_db : dB range below peak to display
    width_px, height_px : output image dimensions in pixels

    Returns
    -------
    bytes – PNG image data
    """
    # Handle very short audio: pad to at least n_fft samples
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    noverlap = n_fft - hop_length
    f, t, Zxx = stft(audio, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap)

    power = np.abs(Zxx) ** 2
    # Avoid log10(0)
    power = np.maximum(power, 1e-12)
    power_db = 10.0 * np.log10(power)

    vmax = float(power_db.max())
    vmin = vmax - dynamic_range_db

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    # Crop to 0–3 kHz
    freq_mask = f <= 3000
    f = f[freq_mask]
    power_db = power_db[freq_mask, :]

    ax.pcolormesh(t, f, power_db, vmin=vmin, vmax=vmax, cmap="inferno", shading="gouraud")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
