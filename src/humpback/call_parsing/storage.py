"""Parquet I/O helpers and storage layout for the call parsing pipeline.

Per-job artifacts live under ``<storage_root>/call_parsing/<pass>/<job_id>/``:

- ``regions/<job_id>/{trace,regions}.parquet``
- ``segmentation/<job_id>/events.parquet``
- ``classification/<job_id>/typed_events.parquet``

Read/write helpers are symmetric — every writer is paired with a reader
that reconstructs the dataclass list. Writes are atomic via temp-file
rename so partial artifacts never appear to readers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar


import pyarrow as pa
import pyarrow.parquet as pq

from humpback.call_parsing.types import (
    EVENT_SCHEMA,
    REGION_SCHEMA,
    TRACE_SCHEMA,
    TYPED_EVENT_SCHEMA,
    Event,
    Region,
    TypedEvent,
    WindowScore,
)

T = TypeVar("T")


# ---- Directory layout ---------------------------------------------------


def region_job_dir(storage_root: Path, job_id: str) -> Path:
    return Path(storage_root) / "call_parsing" / "regions" / job_id


def segmentation_job_dir(storage_root: Path, job_id: str) -> Path:
    return Path(storage_root) / "call_parsing" / "segmentation" / job_id


def classification_job_dir(storage_root: Path, job_id: str) -> Path:
    return Path(storage_root) / "call_parsing" / "classification" / job_id


# ---- Internals ----------------------------------------------------------


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    """Write a pyarrow Table to ``path`` atomically via temp-file rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _table_from_rows(
    rows: Sequence[Any], schema: pa.Schema, dataclass_type: type[T]
) -> pa.Table:
    # cast: T is a dataclass at every call site, but pyright can't prove it
    # from the bare TypeVar bound.
    field_names = [f.name for f in fields(dataclass_type)]  # pyright: ignore[reportArgumentType]
    columns: dict[str, list[Any]] = {name: [] for name in field_names}
    for row in rows:
        for name in field_names:
            columns[name].append(getattr(row, name))
    arrays = [
        pa.array(columns[name], type=schema.field(name).type) for name in field_names
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _rows_from_table(
    table: pa.Table, schema: pa.Schema, dataclass_type: type[T]
) -> list[T]:
    if table.schema.names != schema.names:
        raise ValueError(
            f"Schema mismatch when reading {dataclass_type.__name__}: "
            f"expected fields {schema.names}, got {table.schema.names}"
        )
    # pyarrow to_pylist preserves schema order; construct dataclasses by name.
    result: list[T] = []
    py_rows = table.to_pylist()
    for py_row in py_rows:
        result.append(dataclass_type(**py_row))  # type: ignore[call-arg]
    return result


# ---- Trace (Pass 1) -----------------------------------------------------


def write_trace(path: Path, scores: Iterable[WindowScore]) -> None:
    table = _table_from_rows(list(scores), TRACE_SCHEMA, WindowScore)
    _atomic_write_parquet(table, path)


def read_trace(path: Path) -> list[WindowScore]:
    table = pq.read_table(path)
    return _rows_from_table(table, TRACE_SCHEMA, WindowScore)


# ---- Regions (Pass 1) ---------------------------------------------------


def write_regions(path: Path, regions: Iterable[Region]) -> None:
    table = _table_from_rows(list(regions), REGION_SCHEMA, Region)
    _atomic_write_parquet(table, path)


def read_regions(path: Path) -> list[Region]:
    table = pq.read_table(path)
    return _rows_from_table(table, REGION_SCHEMA, Region)


# ---- Events (Pass 2) ----------------------------------------------------


def write_events(path: Path, events: Iterable[Event]) -> None:
    table = _table_from_rows(list(events), EVENT_SCHEMA, Event)
    _atomic_write_parquet(table, path)


def read_events(path: Path) -> list[Event]:
    table = pq.read_table(path)
    return _rows_from_table(table, EVENT_SCHEMA, Event)


# ---- Typed events (Pass 3) ----------------------------------------------


def write_typed_events(path: Path, typed_events: Iterable[TypedEvent]) -> None:
    table = _table_from_rows(list(typed_events), TYPED_EVENT_SCHEMA, TypedEvent)
    _atomic_write_parquet(table, path)


def read_typed_events(path: Path) -> list[TypedEvent]:
    table = pq.read_table(path)
    return _rows_from_table(table, TYPED_EVENT_SCHEMA, TypedEvent)


# ---- Chunk manifest (Pass 1 hydrophone progress) -------------------------


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(data, indent=2))
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_manifest(job_dir: Path, manifest: dict[str, Any]) -> None:
    _atomic_write_json(manifest, job_dir / "manifest.json")


def read_manifest(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def update_manifest_chunk(
    job_dir: Path, chunk_index: int, update: dict[str, Any]
) -> None:
    manifest = read_manifest(job_dir)
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json in {job_dir}")
    manifest["chunks"][chunk_index].update(update)
    write_manifest(job_dir, manifest)


# ---- Chunk parquet I/O (Pass 1 hydrophone progress) ----------------------


def chunk_parquet_path(job_dir: Path, chunk_index: int) -> Path:
    return job_dir / "chunks" / f"{chunk_index:04d}.parquet"


def write_chunk_trace(
    job_dir: Path, chunk_index: int, scores: Sequence[WindowScore]
) -> None:
    table = _table_from_rows(list(scores), TRACE_SCHEMA, WindowScore)
    _atomic_write_parquet(table, chunk_parquet_path(job_dir, chunk_index))


def read_all_chunk_traces(job_dir: Path, total_chunks: int) -> list[WindowScore]:
    all_scores: list[WindowScore] = []
    for i in range(total_chunks):
        path = chunk_parquet_path(job_dir, i)
        table = pq.read_table(path)
        all_scores.extend(_rows_from_table(table, TRACE_SCHEMA, WindowScore))
    return all_scores
