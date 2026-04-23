"""Unit tests for call_parsing parquet I/O helpers."""

from pathlib import Path

import pytest

from humpback.call_parsing.storage import (
    classification_job_dir,
    read_embeddings,
    read_events,
    read_regions,
    read_trace,
    read_typed_events,
    region_job_dir,
    segmentation_job_dir,
    write_embeddings,
    write_events,
    write_regions,
    write_trace,
    write_typed_events,
)
from humpback.call_parsing.types import (
    Event,
    Region,
    TypedEvent,
    WindowEmbedding,
    WindowScore,
    new_uuid,
)


# ---- Trace -------------------------------------------------------------


def test_trace_roundtrip(tmp_path: Path) -> None:
    scores = [WindowScore(time_sec=float(i), score=0.1 * i) for i in range(50)]
    path = tmp_path / "trace.parquet"
    write_trace(path, scores)
    loaded = read_trace(path)
    assert loaded == scores


def test_trace_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty_trace.parquet"
    write_trace(path, [])
    assert read_trace(path) == []


# ---- Regions -----------------------------------------------------------


def _sample_region(start: float = 0.0, end: float = 5.0) -> Region:
    return Region(
        region_id=new_uuid(),
        start_sec=start,
        end_sec=end,
        padded_start_sec=max(0.0, start - 0.5),
        padded_end_sec=end + 0.5,
        max_score=0.92,
        mean_score=0.71,
        n_windows=4,
    )


def test_regions_roundtrip(tmp_path: Path) -> None:
    regions = [_sample_region(float(i * 10), float(i * 10 + 5)) for i in range(5)]
    path = tmp_path / "regions.parquet"
    write_regions(path, regions)
    loaded = read_regions(path)
    assert loaded == regions


def test_regions_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty_regions.parquet"
    write_regions(path, [])
    assert read_regions(path) == []


# ---- Events ------------------------------------------------------------


def _sample_event(region_id: str, start: float, end: float) -> Event:
    return Event(
        event_id=new_uuid(),
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2,
        segmentation_confidence=0.87,
    )


def test_events_roundtrip(tmp_path: Path) -> None:
    region_id = new_uuid()
    events = [_sample_event(region_id, float(i), float(i + 1.5)) for i in range(10)]
    path = tmp_path / "events.parquet"
    write_events(path, events)
    loaded = read_events(path)
    assert loaded == events


def test_events_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty_events.parquet"
    write_events(path, [])
    assert read_events(path) == []


# ---- Typed events ------------------------------------------------------


def _sample_typed_event(event_id: str, type_name: str, score: float) -> TypedEvent:
    return TypedEvent(
        event_id=event_id,
        start_sec=0.0,
        end_sec=1.0,
        type_name=type_name,
        score=score,
        above_threshold=score >= 0.5,
    )


def test_typed_events_roundtrip_preserves_sort_order(tmp_path: Path) -> None:
    event_id = new_uuid()
    typed = [
        _sample_typed_event(event_id, name, score)
        for name, score in [("whup", 0.9), ("moan", 0.3), ("ascending_moan", 0.65)]
    ]
    path = tmp_path / "typed_events.parquet"
    write_typed_events(path, typed)
    loaded = read_typed_events(path)
    assert loaded == typed  # order preserved


def test_typed_events_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty_typed.parquet"
    write_typed_events(path, [])
    assert read_typed_events(path) == []


# ---- Schema mismatch ---------------------------------------------------


def test_read_with_wrong_schema_raises(tmp_path: Path) -> None:
    regions_path = tmp_path / "regions.parquet"
    write_regions(regions_path, [_sample_region()])
    with pytest.raises(ValueError, match="Schema mismatch"):
        read_events(regions_path)


# ---- Directory helpers -------------------------------------------------


def test_directory_layout_helpers(tmp_path: Path) -> None:
    root = tmp_path / "storage"
    assert region_job_dir(root, "job-1") == (
        root / "call_parsing" / "regions" / "job-1"
    )
    assert segmentation_job_dir(root, "job-2") == (
        root / "call_parsing" / "segmentation" / "job-2"
    )
    assert classification_job_dir(root, "job-3") == (
        root / "call_parsing" / "classification" / "job-3"
    )


# ---- Atomic write ------------------------------------------------------


def test_atomic_write_no_tmp_file_left_behind(tmp_path: Path) -> None:
    path = tmp_path / "regions.parquet"
    write_regions(path, [_sample_region(), _sample_region(10.0, 15.0)])
    assert path.exists()
    tmp = path.with_suffix(".parquet.tmp")
    assert not tmp.exists()


# ---- Embeddings ----------------------------------------------------------


def test_embeddings_roundtrip(tmp_path: Path) -> None:
    embs = [
        WindowEmbedding(time_sec=float(i), embedding=[0.1 * j for j in range(8)])
        for i in range(10)
    ]
    path = tmp_path / "embeddings.parquet"
    write_embeddings(path, embs)
    loaded = read_embeddings(path)
    assert len(loaded) == len(embs)
    for orig, got in zip(embs, loaded):
        assert orig.time_sec == got.time_sec
        assert len(got.embedding) == 8
        for a, b in zip(orig.embedding, got.embedding):
            assert abs(a - b) < 1e-6


def test_embeddings_empty_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "empty_embeddings.parquet"
    write_embeddings(path, [])
    assert read_embeddings(path) == []
