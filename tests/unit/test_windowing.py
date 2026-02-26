import numpy as np

from humpback.processing.windowing import count_windows, slice_windows


def test_exact_division():
    audio = np.ones(32000, dtype=np.float32)  # 1 second at 32kHz
    windows = list(slice_windows(audio, 32000, 0.5))
    assert len(windows) == 2
    assert all(len(w) == 16000 for w in windows)


def test_zero_padding():
    audio = np.ones(48000, dtype=np.float32)  # 1.5 seconds at 32kHz
    windows = list(slice_windows(audio, 32000, 1.0))
    assert len(windows) == 2
    # First window is all ones
    assert windows[0].sum() == 32000
    # Second window has 16000 ones and 16000 zeros
    assert windows[1][:16000].sum() == 16000
    assert windows[1][16000:].sum() == 0


def test_count_windows():
    assert count_windows(32000, 32000, 1.0) == 1
    assert count_windows(48000, 32000, 1.0) == 2
    assert count_windows(160000, 16000, 5.0) == 2


def test_single_short_window():
    audio = np.ones(100, dtype=np.float32)
    windows = list(slice_windows(audio, 16000, 5.0))
    assert len(windows) == 1
    assert len(windows[0]) == 80000
