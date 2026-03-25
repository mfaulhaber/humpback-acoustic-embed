"""Tests for timeline audio resolution from HLS cache."""

import numpy as np
from unittest.mock import patch


def test_resolve_returns_correct_duration():
    """Resolved audio length should match requested duration at target sample rate."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 10.0
    fake_audio = np.zeros(int(sr * duration), dtype=np.float32)

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = fake_audio
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)


def test_resolve_silence_for_gap():
    """When HLS cache has no segments for a range, return silence."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 5.0

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = np.zeros(int(sr * duration), dtype=np.float32)
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)
    assert result.dtype == np.float32


def test_resolve_clamps_to_job_bounds():
    """Requested range beyond job bounds should be clamped."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    job_start = 1000000.0
    job_end = 1086400.0

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = np.zeros(sr * 5, dtype=np.float32)
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=job_start,
            job_end_timestamp=job_end,
            start_sec=job_start - 100.0,
            duration_sec=5.0,
            target_sr=sr,
        )
    assert len(result) == sr * 5


def test_resolve_pads_short_audio():
    """When resolved audio is shorter than requested, pad with zeros."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 10.0
    # Return only 5 seconds of audio
    short_audio = np.ones(int(sr * 5), dtype=np.float32) * 0.5

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = short_audio
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)
    assert result.dtype == np.float32
    # First 5 seconds should have signal, rest should be zero-padded
    assert np.all(result[: int(sr * 5)] == 0.5)
    assert np.all(result[int(sr * 5) :] == 0.0)


def test_resolve_trims_long_audio():
    """When resolved audio is longer than requested, trim to exact length."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 5.0
    # Return 10 seconds of audio
    long_audio = np.ones(int(sr * 10), dtype=np.float32) * 0.3

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = long_audio
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)


def test_resolve_entirely_outside_job_bounds_returns_silence():
    """When start_sec + duration is entirely before job_start, return silence."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    job_start = 1000000.0
    job_end = 1086400.0

    result = resolve_timeline_audio(
        hydrophone_id="rpi_north_sjc",
        local_cache_path="/fake/cache",
        job_start_timestamp=job_start,
        job_end_timestamp=job_end,
        start_sec=job_end + 100.0,
        duration_sec=5.0,
        target_sr=sr,
    )
    assert len(result) == sr * 5
    assert np.all(result == 0.0)
    assert result.dtype == np.float32


def test_resolve_calls_hls_cache_with_clamped_params():
    """Verify _resolve_audio_from_hls_cache is called with clamped start/duration."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    job_start = 1000000.0
    job_end = 1086400.0

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = np.zeros(sr * 5, dtype=np.float32)
        resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=job_start,
            job_end_timestamp=job_end,
            start_sec=job_start - 100.0,
            duration_sec=200.0,
            target_sr=sr,
        )
    mock_resolve.assert_called_once()
    call_kwargs = mock_resolve.call_args[1]
    # effective_start should be clamped to job_start
    assert call_kwargs["start_sec"] == job_start
    # effective_duration should be clamped to min(start+duration, job_end) - job_start
    expected_duration = min(job_start - 100.0 + 200.0, job_end) - job_start
    assert abs(call_kwargs["duration_sec"] - expected_duration) < 1e-6


def test_resolve_dispatches_to_provider_for_noaa():
    """NOAA source IDs should dispatch to _resolve_audio_from_provider."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 5.0
    fake_audio = np.ones(int(sr * duration), dtype=np.float32) * 0.7

    with (
        patch(
            "humpback.processing.timeline_audio._resolve_audio_from_provider"
        ) as mock_provider,
        patch("humpback.config.get_archive_source") as mock_source,
    ):
        mock_source.return_value = {"provider_kind": "noaa_gcs"}
        mock_provider.return_value = fake_audio
        result = resolve_timeline_audio(
            hydrophone_id="sanctsound_oc",
            local_cache_path="",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
            noaa_cache_path="/fake/noaa-cache",
        )
    mock_provider.assert_called_once()
    call_kwargs = mock_provider.call_args[1]
    assert call_kwargs["hydrophone_id"] == "sanctsound_oc"
    assert call_kwargs["noaa_cache_path"] == "/fake/noaa-cache"
    assert len(result) == int(sr * duration)
    assert np.allclose(result, 0.7)


def test_build_hls_timeline_for_range_integration(tmp_path):
    """build_hls_timeline_for_range discovers local cache segments."""
    from humpback.classifier.s3_stream import build_hls_timeline_for_range
    from humpback.config import ORCASOUND_S3_BUCKET

    hydrophone_id = "rpi_north_sjc"
    folder_ts = "1700000000"
    cache_root = tmp_path / ORCASOUND_S3_BUCKET
    hls_dir = cache_root / hydrophone_id / "hls" / folder_ts
    hls_dir.mkdir(parents=True)

    # Create fake .ts files
    for i in range(5):
        (hls_dir / f"live00{i}.ts").write_bytes(b"\x00" * 100)

    # Create a playlist
    playlist = "#EXTM3U\n"
    for i in range(5):
        playlist += f"#EXTINF:10.0,\nlive00{i}.ts\n"
    (hls_dir / "live.m3u8").write_text(playlist)

    result = build_hls_timeline_for_range(
        hydrophone_id=hydrophone_id,
        local_cache_path=str(tmp_path),
        start_epoch=1700000000.0,
        end_epoch=1700000050.0,
    )

    assert len(result) == 5
    # Check segment ordering and timestamps
    for i, (seg_path, seg_start, seg_duration) in enumerate(result):
        assert seg_path.endswith(f"live00{i}.ts")
        assert seg_duration == 10.0
        assert seg_start == 1700000000.0 + i * 10.0


def test_decode_segments_to_audio_fills_gaps():
    """decode_segments_to_audio fills gaps with silence in the output array."""
    from humpback.classifier.s3_stream import decode_segments_to_audio

    sr = 32000
    start_epoch = 1000.0
    end_epoch = 1030.0

    # Two segments with a gap: [1000-1010] and [1020-1030]
    # (middle 10 seconds is a gap)
    timeline = [
        ("/fake/seg1.ts", 1000.0, 10.0),
        ("/fake/seg2.ts", 1020.0, 10.0),
    ]

    # Mock decode_ts_bytes to return known audio
    with patch("humpback.classifier.s3_stream._decode_local_ts_file") as mock_decode:
        seg1_audio = np.ones(sr * 10, dtype=np.float32) * 0.5
        seg2_audio = np.ones(sr * 10, dtype=np.float32) * 0.8
        mock_decode.side_effect = [seg1_audio, seg2_audio]

        result = decode_segments_to_audio(
            timeline=timeline,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            target_sr=sr,
        )

    expected_len = int((end_epoch - start_epoch) * sr)
    assert len(result) == expected_len
    # First 10 seconds should have signal from seg1
    assert np.allclose(result[: sr * 10], 0.5)
    # Middle 10 seconds (gap) should be silence
    assert np.all(result[sr * 10 : sr * 20] == 0.0)
    # Last 10 seconds should have signal from seg2
    assert np.allclose(result[sr * 20 : sr * 30], 0.8)
