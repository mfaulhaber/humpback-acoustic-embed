"""Golden-output tests for ``iter_inference_windows``.

Covers boundary cases that the Pass 2 inference loop relies on:
exact-multiple, partial last window (with pull-back), single-window
regions, and short regions that fit in one window.
"""

from __future__ import annotations

import numpy as np

from humpback.call_parsing.segmentation.window_iter import iter_inference_windows


def _materialize(audio, sample_rate, frame_hop_samples, window_seconds, hop_seconds):
    """Run the iterator and return the list of ``(slice_length, frame_offset)``."""
    return [
        (int(chunk.shape[0]), int(frame_off))
        for chunk, frame_off in iter_inference_windows(
            audio=audio,
            sample_rate=sample_rate,
            frame_hop_samples=frame_hop_samples,
            window_seconds=window_seconds,
            hop_seconds=hop_seconds,
        )
    ]


def test_region_shorter_than_one_window_yields_full_audio():
    sr = 16000
    frame_hop_samples = 500  # 32 fps
    audio = np.zeros(sr * 10, dtype=np.float32)  # 10 s, window=30 s

    out = _materialize(audio, sr, frame_hop_samples, 30.0, 15.0)

    assert out == [(sr * 10, 0)]


def test_region_exactly_one_window_yields_full_audio():
    sr = 16000
    frame_hop_samples = 500
    audio = np.zeros(sr * 30, dtype=np.float32)  # exactly the window

    out = _materialize(audio, sr, frame_hop_samples, 30.0, 15.0)

    assert out == [(sr * 30, 0)]


def test_region_exactly_n_times_hop():
    """45 s region with 30 s window and 15 s hop: 30 s @0, 30 s @15 s pulled to end."""
    sr = 16000
    frame_hop_samples = 500
    window_samples = 30 * sr
    audio = np.zeros(45 * sr, dtype=np.float32)

    out = _materialize(audio, sr, frame_hop_samples, 30.0, 15.0)

    # First window starts at sample 0, second starts at sample 15s.
    expected_frame_offset_15s = int(np.round((15 * sr) / frame_hop_samples))
    assert out == [
        (window_samples, 0),
        (window_samples, expected_frame_offset_15s),
    ]


def test_non_integer_window_count_clamps_final_window():
    """A 50 s region: windows at 0/15/30 s. The last window has a 20 s
    tail (>= window/2 = 15 s) so the pull-back guard does NOT trigger
    and the final window is simply clamped to the audio end (20 s long).
    """
    sr = 16000
    frame_hop_samples = 500
    window_samples = 30 * sr
    audio = np.zeros(50 * sr, dtype=np.float32)

    out = _materialize(audio, sr, frame_hop_samples, 30.0, 15.0)

    expected = [
        (window_samples, 0),
        (window_samples, int(np.round((15 * sr) / frame_hop_samples))),
        (20 * sr, int(np.round((30 * sr) / frame_hop_samples))),
    ]
    assert out == expected


def test_pullback_triggers_when_tail_below_half_window():
    """With ``hop > window/2`` a tail < ``window // 2`` can occur and the
    final window is anchored at the audio end. Use ``window=10 s``,
    ``hop=8 s`` and 20 s of audio:

      iter 1: offset=0,  end=10 s, advance to 8 s.
      iter 2: offset=8 s, end=18 s, tail=12 s (no pull-back), advance to 16 s.
      iter 3: offset=16 s, tail=4 s < 5 s → pull-back to offset=10 s, end=20 s.
    """
    sr = 16000
    frame_hop_samples = 500
    window_samples = 10 * sr
    audio = np.zeros(20 * sr, dtype=np.float32)

    out = _materialize(audio, sr, frame_hop_samples, 10.0, 8.0)

    expected = [
        (window_samples, 0),
        (window_samples, int(np.round((8 * sr) / frame_hop_samples))),
        (window_samples, int(np.round((10 * sr) / frame_hop_samples))),
    ]
    assert out == expected


def test_long_region_yields_four_windows_no_pullback():
    """75 s region: windows at 0/15/30/45 s; last window's tail is 15 s
    which is NOT strictly less than ``window // 2`` so the pull-back
    guard does not trigger.
    """
    sr = 16000
    frame_hop_samples = 500
    window_samples = 30 * sr
    audio = np.zeros(75 * sr, dtype=np.float32)

    out = _materialize(audio, sr, frame_hop_samples, 30.0, 15.0)

    expected = [
        (window_samples, 0),
        (window_samples, int(np.round((15 * sr) / frame_hop_samples))),
        (window_samples, int(np.round((30 * sr) / frame_hop_samples))),
        (window_samples, int(np.round((45 * sr) / frame_hop_samples))),
    ]
    assert out == expected


def test_empty_audio_yields_nothing():
    sr = 16000
    out = _materialize(np.zeros(0, dtype=np.float32), sr, 500, 30.0, 15.0)
    assert out == []
