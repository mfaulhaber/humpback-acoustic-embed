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


# ---- hop_seconds tests ----


def test_hop_half_overlap():
    """Half-overlap hop produces correct window count and offsets."""
    sr = 16000
    audio = np.ones(sr * 10, dtype=np.float32)  # 10s
    # 5s window, 2.5s hop → windows at 0, 2.5, 5.0 (7.5 would need overlap-back at 5.0 = duplicate → skip)
    windows = list(slice_windows(audio, sr, 5.0, hop_seconds=2.5))
    assert len(windows) == 3
    assert all(len(w) == sr * 5 for w in windows)


def test_hop_equals_window():
    """hop == window produces identical output to no-hop."""
    audio = np.random.randn(48000).astype(np.float32)
    no_hop = list(slice_windows(audio, 16000, 1.0))
    with_hop = list(slice_windows(audio, 16000, 1.0, hop_seconds=1.0))
    assert len(no_hop) == len(with_hop)
    for a, b in zip(no_hop, with_hop):
        np.testing.assert_array_equal(a, b)


def test_hop_metadata_offsets():
    """Verify offset_sec values with hop."""
    sr = 16000
    audio = np.ones(sr * 12, dtype=np.float32)  # 12s
    results = list(slice_windows_with_metadata(audio, sr, 5.0, hop_seconds=1.0))
    offsets = [meta.offset_sec for _, meta in results]
    # 12s audio, 5s window, 1s hop → windows at 0,1,2,...,7 (8 total)
    # Last window at 7.0 ends at exactly 12.0 — not overlap-back
    assert offsets[0] == 0.0
    assert offsets[1] == 1.0
    assert offsets[-1] == 7.0
    assert len(offsets) == 8


def test_hop_metadata_offsets_with_overlap_back():
    """Verify overlap-back window is marked when audio doesn't fit exactly."""
    sr = 16000
    audio = np.ones(int(sr * 12.5), dtype=np.float32)  # 12.5s
    results = list(slice_windows_with_metadata(audio, sr, 5.0, hop_seconds=1.0))
    offsets = [meta.offset_sec for _, meta in results]
    # Last hop-aligned window at 7.0, then overlap-back at 7.5
    assert offsets[-1] == 7.5
    assert results[-1][1].is_overlapped


def test_hop_validation():
    """hop > window raises ValueError."""
    import pytest

    audio = np.ones(32000, dtype=np.float32)
    with pytest.raises(ValueError, match="hop_seconds must be <= window_seconds"):
        list(slice_windows(audio, 16000, 1.0, hop_seconds=2.0))


def test_hop_validation_negative():
    """hop <= 0 raises ValueError."""
    import pytest

    audio = np.ones(32000, dtype=np.float32)
    with pytest.raises(ValueError, match="hop_seconds must be positive"):
        list(slice_windows(audio, 16000, 1.0, hop_seconds=0))


def test_count_windows_with_hop():
    """Verify count formula with hop."""
    sr = 16000
    # 12s audio, 5s window, 1s hop
    # Windows at: 0,1,2,3,4,5,6,7 = 8 (hop-aligned: 0..7, no remainder)
    assert count_windows(sr * 12, sr, 5.0, hop_seconds=1.0) == 8
    # 12.5s audio, 5s window, 1s hop → 8 + 1 overlap-back = 9
    assert count_windows(int(sr * 12.5), sr, 5.0, hop_seconds=1.0) == 9
    # 10s audio, 5s window, 2.5s hop → k=(80000-80000)/(40000)=0 → wait, let me compute
    # n=160000, w=80000, h=40000: k=(160000-80000)/40000=2, count=3, remainder=0
    assert count_windows(sr * 10, sr, 5.0, hop_seconds=2.5) == 3


def test_hop_exact_fit_no_overlap_back():
    """When audio fits exactly into hop-aligned windows, no overlap-back."""
    sr = 16000
    # 10s audio, 5s window, 5s hop → 2 windows, no overlap-back
    results = list(slice_windows_with_metadata(np.ones(sr * 10, dtype=np.float32), sr, 5.0, hop_seconds=5.0))
    assert len(results) == 2
    assert not any(meta.is_overlapped for _, meta in results)
