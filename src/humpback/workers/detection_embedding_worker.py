"""Worker for post-hoc detection embedding generation.

Embeds detection windows using the embedding model specified by
``DetectionEmbeddingJob.model_version``. The source classifier's
``window_size_seconds`` and ``target_sample_rate`` are used for audio framing,
but the embedding model itself comes from the registry entry named
``job.model_version`` — not the classifier's.
"""

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detector import (
    _build_file_timeline,
    diff_row_store_vs_embeddings,
    resolve_audio_for_window,
    resolve_audio_for_window_hydrophone,
)
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.processing.inference import EmbeddingModel
from humpback.storage import (
    detection_dir,
    detection_embeddings_path,
    detection_row_store_path,
    ensure_dir,
)
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import (
    complete_detection_embedding_job,
    fail_detection_embedding_job,
)

logger = logging.getLogger(__name__)

_PROGRESS_FLUSH_EVERY = 5


async def run_detection_embedding_job(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Generate or sync detection embeddings for an existing detection job.

    Both modes iterate over the detection row store and (re-)embed windows
    using the embedding model resolved from ``job.model_version``.
    """
    job_id = job.id
    try:
        if job.mode == "sync":
            await _run_sync_mode(session, job, settings)
        else:
            await _run_full_mode(session, job, settings)
    except Exception as e:
        logger.exception("Detection embedding job %s failed", job_id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_embedding_job(session, job_id, str(e))
        except Exception:
            logger.exception("Failed to mark detection embedding job as failed")


async def _load_context(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> tuple[DetectionJob, ClassifierModel, EmbeddingModel, str, dict | None]:
    """Resolve the detection job, its source classifier, and the target model.

    The embedding model is resolved via ``job.model_version`` — the classifier's
    ``model_version`` is ignored for this purpose.
    """
    det_result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job.detection_job_id)
    )
    det_job = det_result.scalar_one_or_none()
    if det_job is None:
        raise ValueError(f"Detection job {job.detection_job_id} not found")
    if det_job.status != "complete":
        raise ValueError(
            f"Detection job {job.detection_job_id} status is {det_job.status}, "
            "expected complete"
        )

    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == det_job.classifier_model_id)
    )
    cm = cm_result.scalar_one_or_none()
    if cm is None:
        raise ValueError(
            f"Source classifier {det_job.classifier_model_id} not found for "
            f"detection job {det_job.id}"
        )

    emb_model, input_format = await get_model_by_version(
        session, job.model_version, settings
    )
    feature_config = json.loads(cm.feature_config) if cm.feature_config else None
    return det_job, cm, emb_model, input_format, feature_config


def _embed_window(
    audio: np.ndarray,
    emb_model: EmbeddingModel,
    input_format: str,
    target_sample_rate: int,
    normalization: str,
) -> list[float]:
    from humpback.processing.features import extract_logmel_batch

    if input_format == "waveform":
        batch = np.expand_dims(audio, 0)
    else:
        specs = extract_logmel_batch(
            [audio],
            target_sample_rate,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
            normalization=normalization,
        )
        batch = np.stack(specs)
    embedding = emb_model.embed(batch)
    return embedding[0].tolist()


async def _embed_rows(
    rows: list[dict],
    det_job: DetectionJob,
    cm: ClassifierModel,
    emb_model: EmbeddingModel,
    input_format: str,
    feature_config: dict | None,
    settings: Settings,
    *,
    session: AsyncSession,
    job_id: str,
    progress_start: int = 0,
) -> tuple[list[dict], list[str]]:
    """Embed each row in ``rows`` using the selected model.

    Returns ``(records, skipped_reasons)``. ``records`` are dicts with keys
    ``row_id``, ``embedding``, ``confidence``. Updates ``rows_processed``
    periodically.
    """
    from humpback.classifier.providers import build_archive_detection_provider

    normalization = (feature_config or {}).get("normalization", "per_window_max")
    target_sample_rate = cm.target_sample_rate

    is_hydrophone = bool(det_job.hydrophone_id) and not det_job.audio_folder
    provider = None
    file_timeline = None

    if is_hydrophone:
        assert det_job.hydrophone_id is not None
        provider = build_archive_detection_provider(
            det_job.hydrophone_id,
            local_cache_path=det_job.local_cache_path,
            s3_cache_path=settings.s3_cache_path,
            noaa_cache_path=settings.noaa_cache_path,
            force_refresh=False,
        )
    elif det_job.audio_folder:
        file_timeline = await asyncio.to_thread(
            _build_file_timeline, Path(det_job.audio_folder), target_sample_rate
        )

    records: list[dict] = []
    skipped_reasons: list[str] = []
    audio_cache: dict[str, np.ndarray] = {}
    processed = progress_start

    for row in rows:
        start_utc = float(row["start_utc"])
        end_utc = float(row["end_utc"])

        if is_hydrophone and provider is not None:
            audio, reason = await asyncio.to_thread(
                resolve_audio_for_window_hydrophone,
                start_utc,
                end_utc,
                provider,
                target_sample_rate,
            )
        elif det_job.audio_folder:
            audio, reason = await asyncio.to_thread(
                resolve_audio_for_window,
                start_utc,
                end_utc,
                Path(det_job.audio_folder),
                target_sample_rate,
                _file_timeline=file_timeline,
                _audio_cache=audio_cache,
            )
        else:
            audio = None
            reason = "no audio source (no audio_folder or hydrophone_id)"

        processed += 1

        if audio is None:
            skipped_reasons.append(f"[{start_utc:.1f}, {end_utc:.1f}]: {reason}")
        else:
            try:
                emb_vec = await asyncio.to_thread(
                    _embed_window,
                    audio,
                    emb_model,
                    input_format,
                    target_sample_rate,
                    normalization,
                )
                records.append(
                    {
                        "row_id": row.get("row_id", ""),
                        "embedding": emb_vec,
                        "confidence": None,
                    }
                )
            except Exception as exc:
                skipped_reasons.append(
                    f"[{start_utc:.1f}, {end_utc:.1f}]: embedding failed: {exc}"
                )

        if processed % _PROGRESS_FLUSH_EVERY == 0 or processed == len(rows):
            await session.execute(
                update(DetectionEmbeddingJob)
                .where(DetectionEmbeddingJob.id == job_id)
                .values(rows_processed=processed, progress_current=processed)
            )
            await session.commit()

    return records, skipped_reasons


async def _run_full_mode(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Re-embed every row in the detection job's row store under ``job.model_version``."""
    from humpback.classifier.detection_rows import read_detection_row_store

    det_job, cm, emb_model, input_format, feature_config = await _load_context(
        session, job, settings
    )

    rs_path = detection_row_store_path(settings.storage_root, det_job.id)
    if not rs_path.exists():
        raise ValueError(f"Row store not found for detection job {det_job.id}")

    _, rows = await asyncio.to_thread(read_detection_row_store, rs_path)
    total = len(rows)

    ensure_dir(detection_dir(settings.storage_root, det_job.id))
    emb_path = detection_embeddings_path(
        settings.storage_root, det_job.id, job.model_version
    )
    ensure_dir(emb_path.parent)

    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(
            rows_total=total,
            rows_processed=0,
            progress_total=total,
            progress_current=0,
        )
    )
    await session.commit()

    records, skipped_reasons = await _embed_rows(
        rows,
        det_job,
        cm,
        emb_model,
        input_format,
        feature_config,
        settings,
        session=session,
        job_id=job.id,
    )

    await asyncio.to_thread(_write_embeddings_parquet, emb_path, records)

    result_summary = json.dumps(
        {
            "total": len(records),
            "skipped": len(skipped_reasons),
            "skipped_reasons": skipped_reasons,
        }
    )
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(
            rows_processed=total,
            progress_current=total,
            result_summary=result_summary,
        )
    )
    await session.commit()

    logger.info(
        "Detection embedding job %s complete: %d embeddings for detection %s (model=%s)",
        job.id,
        len(records),
        job.detection_job_id,
        job.model_version,
    )
    await complete_detection_embedding_job(session, job.id)


async def _run_sync_mode(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Diff row store against existing embeddings; patch the delta."""
    det_job, cm, emb_model, input_format, feature_config = await _load_context(
        session, job, settings
    )

    rs_path = detection_row_store_path(settings.storage_root, det_job.id)
    emb_path = detection_embeddings_path(
        settings.storage_root, det_job.id, job.model_version
    )
    if not rs_path.exists():
        raise ValueError("Row store not found")
    if not emb_path.exists():
        raise ValueError("Embeddings parquet not found — run full generation first")

    diff = await asyncio.to_thread(diff_row_store_vs_embeddings, rs_path, emb_path)

    if not diff.missing and not diff.orphaned_indices:
        summary = json.dumps(
            {
                "added": 0,
                "removed": 0,
                "unchanged": diff.matched_count,
                "skipped": 0,
                "skipped_reasons": [],
            }
        )
        await session.execute(
            update(DetectionEmbeddingJob)
            .where(DetectionEmbeddingJob.id == job.id)
            .values(
                rows_total=0,
                rows_processed=0,
                progress_total=0,
                progress_current=0,
                result_summary=summary,
            )
        )
        await session.commit()
        logger.info(
            "Sync job %s: already in sync (%d matched)", job.id, diff.matched_count
        )
        await complete_detection_embedding_job(session, job.id)
        return

    total = len(diff.missing)
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(
            rows_total=total,
            rows_processed=0,
            progress_total=total,
            progress_current=0,
        )
    )
    await session.commit()

    new_records, skipped_reasons = await _embed_rows(
        list(diff.missing),
        det_job,
        cm,
        emb_model,
        input_format,
        feature_config,
        settings,
        session=session,
        job_id=job.id,
    )

    await asyncio.to_thread(
        _rewrite_embeddings, emb_path, diff.orphaned_indices, new_records
    )

    summary = json.dumps(
        {
            "added": len(new_records),
            "removed": len(diff.orphaned_indices),
            "unchanged": diff.matched_count,
            "skipped": len(skipped_reasons),
            "skipped_reasons": skipped_reasons,
        }
    )
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(
            rows_processed=total,
            progress_current=total,
            result_summary=summary,
        )
    )
    await session.commit()

    logger.info(
        "Sync job %s complete: +%d -%d =%d skip=%d for detection %s (model=%s)",
        job.id,
        len(new_records),
        len(diff.orphaned_indices),
        diff.matched_count,
        len(skipped_reasons),
        job.detection_job_id,
        job.model_version,
    )
    await complete_detection_embedding_job(session, job.id)


def _write_embeddings_parquet(emb_path: Path, records: list[dict]) -> None:
    """Write a fresh embeddings parquet keyed by row_id."""
    if not records:
        # Still write an empty table so downstream code can rely on the file.
        schema = pa.schema(
            [
                ("row_id", pa.string()),
                ("embedding", pa.list_(pa.float32())),
                ("confidence", pa.float32()),
            ]
        )
        pq.write_table(
            pa.table({"row_id": [], "embedding": [], "confidence": []}, schema=schema),
            str(emb_path),
        )
        return

    vector_dim = len(records[0]["embedding"])
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), vector_dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": [r["row_id"] for r in records],
            "embedding": [r["embedding"] for r in records],
            "confidence": [r.get("confidence") for r in records],
        },
        schema=schema,
    )
    pq.write_table(table, str(emb_path))


def _rewrite_embeddings(
    emb_path: Path,
    orphaned_indices: list[int],
    new_records: list[dict],
) -> None:
    """Rewrite the embeddings parquet, removing orphans and appending new records."""
    table = pq.read_table(str(emb_path))

    if orphaned_indices:
        keep_mask = [True] * table.num_rows
        for idx in orphaned_indices:
            keep_mask[idx] = False
        table = table.filter(pa.array(keep_mask))

    # If the existing table uses the old schema (filename/start_sec/end_sec),
    # drop those columns and keep only row_id/embedding/confidence.
    col_names = set(table.column_names)
    if "filename" in col_names and "row_id" not in col_names:
        table = pa.table(
            {"row_id": [], "embedding": [], "confidence": []},
        )

    if new_records:
        vector_dim = len(new_records[0]["embedding"])
        new_schema = pa.schema(
            [
                ("row_id", pa.string()),
                ("embedding", pa.list_(pa.float32(), vector_dim)),
                ("confidence", pa.float32()),
            ]
        )
        new_table = pa.table(
            {
                "row_id": [r["row_id"] for r in new_records],
                "embedding": [r["embedding"] for r in new_records],
                "confidence": [r.get("confidence") for r in new_records],
            },
            schema=new_schema,
        )
        table = pa.concat_tables([table, new_table], promote_options="default")

    pq.write_table(table, str(emb_path))
