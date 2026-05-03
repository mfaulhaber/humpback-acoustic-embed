"""Tests for shared timeline tile repository paths and metadata."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _source_ref(start: float = 1000.0, end: float = 1100.0):
    from humpback.processing.timeline_repository import TimelineSourceRef

    return TimelineSourceRef(
        hydrophone_id="rpi_orcasound_lab",
        source_identity="/tmp/cache",
        job_start_timestamp=start,
        job_end_timestamp=end,
    )


def _request(freq_min: int = 0, freq_max: int = 3000):
    from humpback.processing.timeline_repository import TimelineTileRequest

    return TimelineTileRequest(
        zoom_level="1m",
        tile_index=7,
        freq_min=freq_min,
        freq_max=freq_max,
        width_px=512,
        height_px=256,
    )


def test_source_refs_for_same_span_have_same_key():
    from humpback.processing.timeline_repository import TimelineSourceRef

    settings = SimpleNamespace(
        s3_cache_path="/tmp/cache",
        noaa_cache_path="/tmp/noaa",
    )
    classifier_job = SimpleNamespace(
        hydrophone_id="rpi_orcasound_lab",
        local_cache_path=None,
        start_timestamp=1000.0,
        end_timestamp=1100.0,
    )
    region_job = SimpleNamespace(
        hydrophone_id="rpi_orcasound_lab",
        local_cache_path=None,
        start_timestamp=1000.0,
        end_timestamp=1100.0,
    )

    assert (
        TimelineSourceRef.from_job(classifier_job, settings).span_key
        == TimelineSourceRef.from_job(region_job, settings).span_key
    )


def test_source_ref_key_changes_with_identity_fields():
    base = _source_ref()

    assert base.span_key != _source_ref(start=1001.0).span_key
    assert base.span_key != _source_ref(end=1110.0).span_key


def test_repository_path_includes_renderer_frequency_and_geometry(tmp_path: Path):
    from humpback.processing.timeline_repository import TimelineTileRepository

    repo = TimelineTileRepository(tmp_path / "timeline_cache")
    source = _source_ref()
    request = _request()

    path = repo.tile_path(source, "lifted-ocean", 1, request)

    assert path.parts[-6:] == (
        "lifted-ocean",
        "v1",
        "1m",
        "f0-3000",
        "w512_h256",
        "tile_0007.png",
    )


def test_repository_path_changes_for_renderer_and_frequency(tmp_path: Path):
    from humpback.processing.timeline_repository import TimelineTileRepository

    repo = TimelineTileRepository(tmp_path / "timeline_cache")
    source = _source_ref()

    lifted = repo.tile_path(source, "lifted-ocean", 1, _request())
    ocean = repo.tile_path(source, "ocean-depth", 7, _request())
    narrow = repo.tile_path(source, "lifted-ocean", 1, _request(freq_max=1000))

    assert lifted != ocean
    assert lifted != narrow


def test_repository_put_get_and_count_roundtrip(tmp_path: Path):
    from humpback.processing.timeline_repository import TimelineTileRepository

    repo = TimelineTileRepository(tmp_path / "timeline_cache")
    source = _source_ref()
    request = _request()

    repo.put(source, "lifted-ocean", 1, request, b"png")

    assert repo.get(source, "lifted-ocean", 1, request) == b"png"
    assert repo.count_cached_tiles(source, "lifted-ocean", 1, [request]) == 1
    assert (
        repo.tile_count_for_zoom(
            source,
            "lifted-ocean",
            1,
            "1m",
            freq_min=0,
            freq_max=3000,
            width_px=512,
            height_px=256,
        )
        == 1
    )


def test_audio_manifest_roundtrip(tmp_path: Path):
    from humpback.processing.timeline_repository import TimelineTileRepository

    repo = TimelineTileRepository(tmp_path / "timeline_cache")
    source = _source_ref()
    manifest = {"entries": [{"segment_path": "a.ts"}]}

    repo.put_audio_manifest(source, manifest)

    assert repo.get_audio_manifest(source) == manifest
