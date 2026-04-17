"""Service for creating and extending training dataset snapshots."""

import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.audio import AudioFile
from humpback.models.labeling import VocalizationLabel
from humpback.models.processing import EmbeddingSet
from humpback.models.training_dataset import TrainingDataset, TrainingDatasetLabel
from humpback.storage import (
    detection_embeddings_path,
    detection_row_store_path,
    ensure_dir,
    training_dataset_dir,
    training_dataset_parquet_path,
)

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        ("row_index", pa.int32()),
        ("embedding", pa.list_(pa.float32())),
        ("source_type", pa.string()),
        ("source_id", pa.string()),
        ("filename", pa.string()),
        ("start_sec", pa.float32()),
        ("end_sec", pa.float32()),
        ("confidence", pa.float32()),
    ]
)


async def create_training_dataset_snapshot(
    session: AsyncSession,
    source_config: dict,
    storage_root: Path,
    name: str | None = None,
) -> TrainingDataset:
    """Snapshot source embeddings and labels into a unified training dataset.

    Creates a TrainingDataset record, writes a unified parquet file, and copies
    labels into training_dataset_labels rows.
    """
    embedding_set_ids: list[str] = source_config.get("embedding_set_ids", [])
    detection_job_ids: list[str] = source_config.get("detection_job_ids", [])

    rows, label_sets = await _collect_from_sources(
        session, embedding_set_ids, detection_job_ids, storage_root, row_offset=0
    )

    if not rows:
        raise ValueError("No embeddings collected from source config")

    # Create dataset record
    dataset = TrainingDataset(
        name=name or f"dataset-{len(rows)}-rows",
        source_config=json.dumps(source_config),
        parquet_path="",  # set after write
        total_rows=len(rows),
    )
    session.add(dataset)
    await session.flush()

    # Write parquet
    parquet_path = training_dataset_parquet_path(storage_root, dataset.id)
    ensure_dir(training_dataset_dir(storage_root, dataset.id))
    _write_parquet(rows, parquet_path)

    # Collect vocabulary and create label records
    vocab: set[str] = set()
    for row_idx, labels in label_sets.items():
        for label in labels:
            if label != "(Negative)":
                vocab.add(label)
            session.add(
                TrainingDatasetLabel(
                    training_dataset_id=dataset.id,
                    row_index=row_idx,
                    label=label,
                    source="snapshot",
                )
            )

    dataset.parquet_path = str(parquet_path)
    dataset.vocabulary = json.dumps(sorted(vocab))
    await session.flush()

    return dataset


async def extend_training_dataset(
    session: AsyncSession,
    dataset: TrainingDataset,
    extend_config: dict,
    storage_root: Path,
) -> TrainingDataset:
    """Append new source rows to an existing training dataset."""
    embedding_set_ids: list[str] = extend_config.get("embedding_set_ids", [])
    detection_job_ids: list[str] = extend_config.get("detection_job_ids", [])

    row_offset = dataset.total_rows

    rows, label_sets = await _collect_from_sources(
        session, embedding_set_ids, detection_job_ids, storage_root, row_offset
    )

    if not rows:
        logger.warning("No new rows collected during extend")
        return dataset

    # Read existing parquet and append
    parquet_path = training_dataset_parquet_path(storage_root, dataset.id)
    existing_table = pq.read_table(str(parquet_path))
    new_table = _rows_to_table(rows)
    combined = pa.concat_tables([existing_table, new_table])
    pq.write_table(combined, str(parquet_path))

    # Create label records
    existing_vocab: set[str] = set(json.loads(dataset.vocabulary))
    for row_idx, labels in label_sets.items():
        for label in labels:
            if label != "(Negative)":
                existing_vocab.add(label)
            session.add(
                TrainingDatasetLabel(
                    training_dataset_id=dataset.id,
                    row_index=row_idx,
                    label=label,
                    source="snapshot",
                )
            )

    dataset.total_rows = combined.num_rows
    dataset.vocabulary = json.dumps(sorted(existing_vocab))

    # Merge source_config
    old_config = json.loads(dataset.source_config)
    old_es = set(old_config.get("embedding_set_ids", []))
    old_dj = set(old_config.get("detection_job_ids", []))
    old_es.update(embedding_set_ids)
    old_dj.update(detection_job_ids)
    dataset.source_config = json.dumps(
        {
            "embedding_set_ids": sorted(old_es),
            "detection_job_ids": sorted(old_dj),
        }
    )

    await session.flush()
    return dataset


# ---- Internal helpers ----


def _filename_from_utc(epoch: float) -> str:
    """Generate a synthetic filename from an absolute UTC epoch timestamp."""
    from datetime import datetime, timezone

    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ.wav")


def _write_parquet(rows: list[dict], path: Path) -> None:
    table = _rows_to_table(rows)
    pq.write_table(table, str(path))


def _rows_to_table(rows: list[dict]) -> pa.Table:
    return pa.table(
        {
            "row_index": pa.array([r["row_index"] for r in rows], type=pa.int32()),
            "embedding": pa.array(
                [r["embedding"] for r in rows], type=pa.list_(pa.float32())
            ),
            "source_type": pa.array([r["source_type"] for r in rows], type=pa.string()),
            "source_id": pa.array([r["source_id"] for r in rows], type=pa.string()),
            "filename": pa.array([r["filename"] for r in rows], type=pa.string()),
            "start_sec": pa.array([r["start_sec"] for r in rows], type=pa.float32()),
            "end_sec": pa.array([r["end_sec"] for r in rows], type=pa.float32()),
            "confidence": pa.array([r["confidence"] for r in rows], type=pa.float32()),
        },
        schema=PARQUET_SCHEMA,
    )


async def _collect_from_sources(
    session: AsyncSession,
    embedding_set_ids: list[str],
    detection_job_ids: list[str],
    storage_root: Path,
    row_offset: int,
) -> tuple[list[dict], dict[int, set[str]]]:
    """Collect embedding rows and label sets from sources.

    Returns (rows, label_sets) where label_sets maps row_index -> set of label names.
    """
    rows: list[dict] = []
    label_sets: dict[int, set[str]] = {}
    seen_keys: set[str] = set()
    next_idx = row_offset

    # Source 1: Embedding sets with folder-inferred type
    for es_id in embedding_set_ids:
        es_result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id == es_id)
        )
        es = es_result.scalar_one_or_none()
        if es is None:
            logger.warning("Embedding set %s not found, skipping", es_id)
            continue

        af_result = await session.execute(
            select(AudioFile).where(AudioFile.id == es.audio_file_id)
        )
        af = af_result.scalar_one_or_none()
        if af is None:
            continue

        # Infer type from folder path leaf
        type_name = None
        if af.folder_path:
            parts = Path(af.folder_path).parts
            if parts:
                type_name = parts[-1].strip().title()

        if not type_name:
            logger.warning("No folder-based type for embedding set %s, skipping", es_id)
            continue

        parquet_path = Path(es.parquet_path)
        if not parquet_path.exists():
            logger.warning("Parquet %s not found, skipping", parquet_path)
            continue

        table = pq.read_table(str(parquet_path))
        embeddings_col = table.column("embedding")
        window_size = es.window_size_seconds

        for i in range(table.num_rows):
            key = f"es:{es_id}:{i}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            vec = np.array(embeddings_col[i].as_py(), dtype=np.float32).tolist()
            start_sec = float(i * window_size)
            end_sec = start_sec + window_size

            rows.append(
                {
                    "row_index": next_idx,
                    "embedding": vec,
                    "source_type": "embedding_set",
                    "source_id": es_id,
                    "filename": af.filename,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "confidence": None,
                }
            )
            label_sets[next_idx] = {type_name}
            next_idx += 1

    # Source 2: Detection job vocalization labels
    from humpback.services.classifier_service.models import (
        resolve_detection_job_model_version,
    )

    for det_job_id in detection_job_ids:
        try:
            model_version = await resolve_detection_job_model_version(
                session, det_job_id
            )
        except ValueError:
            logger.warning(
                "No classifier/model_version for detection job %s, skipping", det_job_id
            )
            continue
        emb_path = detection_embeddings_path(storage_root, det_job_id, model_version)
        if not emb_path.exists():
            logger.warning("No embeddings for detection job %s, skipping", det_job_id)
            continue

        # Get vocalization labels for this detection job
        result = await session.execute(
            select(VocalizationLabel).where(
                VocalizationLabel.detection_job_id == det_job_id
            )
        )
        voc_labels = result.scalars().all()

        # Build multi-hot label index by row_id
        labels_by_row_id: dict[str, set[str]] = {}
        for vl in voc_labels:
            if vl.row_id not in labels_by_row_id:
                labels_by_row_id[vl.row_id] = set()
            labels_by_row_id[vl.row_id].add(vl.label)

        # Build row_id -> (filename, start_sec, end_sec) from row store
        row_meta: dict[str, tuple[str, float, float]] = {}
        rs_path = detection_row_store_path(storage_root, det_job_id)
        if rs_path.exists():
            rs_table = pq.read_table(
                str(rs_path), columns=["row_id", "start_utc", "end_utc"]
            )
            for j in range(rs_table.num_rows):
                rs_rid = rs_table.column("row_id")[j].as_py()
                s_utc = float(rs_table.column("start_utc")[j].as_py())
                e_utc = float(rs_table.column("end_utc")[j].as_py())
                fname = _filename_from_utc(s_utc)
                row_meta[rs_rid] = (fname, 0.0, e_utc - s_utc)
        else:
            logger.warning(
                "No row store for detection job %s, metadata will be empty",
                det_job_id,
            )

        table = pq.read_table(str(emb_path))
        row_ids = table.column("row_id").to_pylist()
        embeddings_col = table.column("embedding")
        conf_col = (
            table.column("confidence").to_pylist()
            if "confidence" in table.schema.names
            else None
        )

        for i in range(table.num_rows):
            rid = row_ids[i]

            # Only include explicitly labeled windows
            if rid not in labels_by_row_id:
                continue

            dedup_key = f"det:{det_job_id}:{rid}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            label_set = labels_by_row_id[rid]
            vec = np.array(embeddings_col[i].as_py(), dtype=np.float32).tolist()
            confidence = (
                float(conf_col[i]) if conf_col and conf_col[i] is not None else None
            )
            fname, s_sec, e_sec = row_meta.get(rid, ("", 0.0, 0.0))

            rows.append(
                {
                    "row_index": next_idx,
                    "embedding": vec,
                    "source_type": "detection_job",
                    "source_id": det_job_id,
                    "filename": fname,
                    "start_sec": s_sec,
                    "end_sec": e_sec,
                    "confidence": confidence,
                }
            )
            label_sets[next_idx] = label_set
            next_idx += 1

    return rows, label_sets
