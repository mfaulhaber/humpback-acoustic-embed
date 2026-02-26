from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.storage import atomic_rename, ensure_dir


class IncrementalParquetWriter:
    """Writes embedding vectors incrementally to a temp Parquet file,
    then atomically renames to the final path on close."""

    def __init__(self, final_path: Path, vector_dim: int, batch_size: int = 100):
        self.final_path = final_path
        self.tmp_path = final_path.with_suffix(".tmp.parquet")
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        self._buffer: list[np.ndarray] = []
        self._row_index = 0
        self._writer: pq.ParquetWriter | None = None
        self._schema = pa.schema(
            [
                ("row_index", pa.int32()),
                ("embedding", pa.list_(pa.float32(), self.vector_dim)),
            ]
        )
        ensure_dir(self.tmp_path.parent)

    def add(self, vector: np.ndarray) -> None:
        self._buffer.append(vector)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        indices = list(range(self._row_index, self._row_index + len(self._buffer)))
        embeddings = [v.tolist() for v in self._buffer]
        table = pa.table(
            {"row_index": indices, "embedding": embeddings},
            schema=self._schema,
        )
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.tmp_path), self._schema)
        self._writer.write_table(table)
        self._row_index += len(self._buffer)
        self._buffer.clear()

    def close(self) -> Path:
        """Flush remaining data and atomically rename to final path."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
        if self.tmp_path.exists():
            atomic_rename(self.tmp_path, self.final_path)
        return self.final_path

    @property
    def total_rows(self) -> int:
        return self._row_index + len(self._buffer)


def read_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read embeddings from Parquet. Returns (row_indices, embeddings) arrays."""
    table = pq.read_table(str(path))
    row_indices = table["row_index"].to_numpy()
    # Convert list column to 2D numpy array
    embeddings = np.array([v.as_py() for v in table["embedding"]], dtype=np.float32)
    return row_indices, embeddings
