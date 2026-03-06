import numpy as np

from humpback.processing.windowing import count_windows, slice_windows, slice_windows_with_metadata


def test_exact_division():
    audio = np.ones(32000, dtype=np.float32)  # 1 second at 32kHz
    windows = list(slice_windows(audio, 32000, 0.5))
    assert len(windows) == 2
    assert all(len(w) == 16000 for w in windows)


def test_overlap_last_window():
    """When the last chunk is shorter than a full window, it overlaps backward."""
    audio = np.arange(48000, dtype=np.float32)  # 1.5 seconds at 32kHz
    windows = list(slice_windows(audio, 32000, 1.0))
    assert len(windows) == 2
    # First window: samples 0..31999
    assert windows[0][0] == 0
    assert len(windows[0]) == 32000
    # Second window: shifted back to end at sample 48000 → starts at 16000
    assert windows[1][0] == 16000
    assert windows[1][-1] == 47999
    assert len(windows[1]) == 32000


def test_count_windows():
    assert count_windows(32000, 32000, 1.0) == 1
    assert count_windows(48000, 32000, 1.0) == 2
    assert count_windows(160000, 16000, 5.0) == 2


def test_short_audio_skipped():
    """Audio shorter than one window produces no windows."""
    audio = np.ones(100, dtype=np.float32)
    windows = list(slice_windows(audio, 16000, 5.0))
    assert len(windows) == 0


def test_count_windows_short_audio():
    """Audio shorter than one window → 0 windows."""
    assert count_windows(100, 16000, 5.0) == 0
    assert count_windows(79999, 16000, 5.0) == 0


def test_overlap_12s_audio():
    """12s audio, 5s window → 3 windows, last one overlapped starting at 7s."""
    sr = 16000
    audio = np.ones(sr * 12, dtype=np.float32)
    windows = list(slice_windows(audio, sr, 5.0))
    assert len(windows) == 3


# ---- slice_windows_with_metadata tests ----


def test_metadata_exact_division():
    audio = np.ones(32000, dtype=np.float32)
    results = list(slice_windows_with_metadata(audio, 32000, 0.5))
    assert len(results) == 2
    for window, meta in results:
        assert len(window) == 16000
        assert not meta.is_overlapped
        assert meta.original_samples == 16000
    assert results[0][1].window_index == 0
    assert results[0][1].offset_sec == 0.0
    assert results[1][1].window_index == 1
    assert results[1][1].offset_sec == 0.5


def test_metadata_overlapped_window():
    audio = np.arange(48000, dtype=np.float32)
    results = list(slice_windows_with_metadata(audio, 32000, 1.0))
    assert len(results) == 2
    # First window: not overlapped
    assert not results[0][1].is_overlapped
    assert results[0][1].original_samples == 32000
    # Second window: overlapped, starts at sample 16000
    assert results[1][1].is_overlapped
    assert results[1][1].offset_sec == 16000 / 32000  # 0.5s
    assert results[1][1].original_samples == 32000


def test_metadata_short_audio_skipped():
    """Audio shorter than one window produces no results."""
    audio = np.ones(100, dtype=np.float32)
    results = list(slice_windows_with_metadata(audio, 16000, 5.0))
    assert len(results) == 0


def test_metadata_matches_slice_windows():
    """slice_windows_with_metadata should produce identical audio to slice_windows."""
    audio = np.random.randn(50000).astype(np.float32)
    plain = list(slice_windows(audio, 16000, 1.0))
    meta_results = list(slice_windows_with_metadata(audio, 16000, 1.0))
    assert len(plain) == len(meta_results)
    for pw, (mw, _meta) in zip(plain, meta_results):
        np.testing.assert_array_equal(pw, mw)
