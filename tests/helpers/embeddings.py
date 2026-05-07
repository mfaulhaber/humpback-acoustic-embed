"""Shared parquet embedding fixture writers for tests."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq


def _embedding_type(
    rows: Sequence[Sequence[float]],
    *,
    fixed_size: bool,
) -> pa.DataType:
    if fixed_size and rows:
        return pa.list_(pa.float32(), len(rows[0]))
    return pa.list_(pa.float32())


def write_legacy_embedding_set_parquet(
    path: Path,
    rows: Sequence[Sequence[float]],
    *,
    fixed_size: bool = False,
) -> None:
    """Write a legacy embedding-set parquet keyed by row_index."""
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(rows))), type=pa.int32()),
            "embedding": pa.array(
                [list(row) for row in rows],
                type=_embedding_type(rows, fixed_size=fixed_size),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def write_detection_embeddings_parquet(
    path: Path,
    row_ids: Sequence[str],
    rows: Sequence[Sequence[float]],
    *,
    fixed_size: bool = False,
) -> None:
    """Write a detection embeddings parquet keyed by row_id."""
    table = pa.table(
        {
            "row_id": pa.array(list(row_ids), type=pa.string()),
            "embedding": pa.array(
                [list(row) for row in rows],
                type=_embedding_type(rows, fixed_size=fixed_size),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))
