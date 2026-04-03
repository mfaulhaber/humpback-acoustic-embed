"""Worker for post-hoc detection embedding generation."""

import asyncio
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detector import (
    _build_file_timeline,
    diff_row_store_vs_embeddings,
    match_embedding_records_to_row_store,
    resolve_audio_for_window,
    resolve_audio_for_window_hydrophone,
    run_detection,
    write_detection_embeddings,
)
from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob
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


async def run_detection_embedding_job(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Generate detection embeddings for an existing detection job."""
    try:
        if job.mode == "sync":
            await _run_sync_mode(session, job, settings)
        else:
            await _run_full_mode(session, job, settings)
    except Exception as e:
        logger.exception("Detection embedding job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_embedding_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark detection embedding job as failed")


async def _run_full_mode(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Full embedding generation — re-runs detection to collect embeddings."""
    from sqlalchemy import select

    # Load the detection job
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job.detection_job_id)
    )
    det_job = result.scalar_one_or_none()
    if det_job is None:
        raise ValueError(f"Detection job {job.detection_job_id} not found")
    if det_job.status != "complete":
        raise ValueError(
            f"Detection job {job.detection_job_id} status is {det_job.status}, "
            "expected complete"
        )
    if not det_job.audio_folder:
        raise ValueError(
            f"Detection job {job.detection_job_id} has no audio_folder "
            "(hydrophone jobs already generate embeddings during detection)"
        )

    # Load classifier model and embedding model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == det_job.classifier_model_id)
    )
    cm = cm_result.scalar_one()
    pipeline = joblib.load(cm.model_path)
    model, input_format = await get_model_by_version(
        session, cm.model_version, settings
    )
    feature_config = json.loads(cm.feature_config) if cm.feature_config else None

    audio_folder = Path(det_job.audio_folder)

    # Count audio files for progress tracking
    audio_extensions = {".wav", ".flac", ".mp3", ".aif", ".aiff"}
    audio_files = sorted(
        p for p in audio_folder.rglob("*") if p.suffix.lower() in audio_extensions
    )
    files_total = len(audio_files)

    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(progress_total=files_total, progress_current=0)
    )
    await session.commit()

    def on_file_complete(_detections: list[dict], files_done: int, total: int) -> None:
        pass  # Progress updated after run_detection completes

    # Re-run detection to collect embeddings (reuses existing infrastructure)
    (
        _detections,
        _summary,
        _diagnostics,
        detection_embeddings,
    ) = await asyncio.to_thread(
        run_detection,
        audio_folder,
        pipeline,
        model,
        cm.window_size_seconds,
        cm.target_sample_rate,
        det_job.confidence_threshold,
        input_format,
        feature_config,
        False,  # emit_diagnostics — not needed
        det_job.hop_seconds,
        det_job.high_threshold,
        det_job.low_threshold,
        on_file_complete,
        det_job.detection_mode,
        True,  # emit_embeddings
    )

    # Cross-reference embeddings with row store to assign row_ids.
    ddir = ensure_dir(detection_dir(settings.storage_root, det_job.id))
    emb_path = ddir / "detection_embeddings.parquet"
    if detection_embeddings:
        rs_path = detection_row_store_path(settings.storage_root, det_job.id)
        if rs_path.exists():
            from humpback.classifier.detection_rows import read_detection_row_store

            _, rs_rows = read_detection_row_store(rs_path)
            detection_embeddings = match_embedding_records_to_row_store(
                detection_embeddings, rs_rows
            )

        write_detection_embeddings(detection_embeddings, emb_path)

    count = len(detection_embeddings) if detection_embeddings else 0
    result_summary = json.dumps({"total": count})
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(progress_current=files_total, result_summary=result_summary)
    )
    await session.commit()

    logger.info(
        "Detection embedding job %s complete: %d embeddings for detection %s",
        job.id,
        count,
        job.detection_job_id,
    )
    await complete_detection_embedding_job(session, job.id)


async def _run_sync_mode(
    session: AsyncSession,
    job: DetectionEmbeddingJob,
    settings: Settings,
) -> None:
    """Sync mode — diff row store vs embeddings, patch the delta."""
    from humpback.classifier.providers import build_archive_detection_provider
    from humpback.processing.features import extract_logmel_batch
    from sqlalchemy import select

    # Load the detection job
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job.detection_job_id)
    )
    det_job = result.scalar_one_or_none()
    if det_job is None:
        raise ValueError(f"Detection job {job.detection_job_id} not found")
    if det_job.status != "complete":
        raise ValueError(
            f"Detection job {job.detection_job_id} status is {det_job.status}, "
            "expected complete"
        )

    # Paths
    rs_path = detection_row_store_path(settings.storage_root, det_job.id)
    emb_path = detection_embeddings_path(settings.storage_root, det_job.id)
    if not rs_path.exists():
        raise ValueError("Row store not found")
    if not emb_path.exists():
        raise ValueError("Embeddings parquet not found — run full generation first")

    # Load classifier model and embedding model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == det_job.classifier_model_id)
    )
    cm = cm_result.scalar_one()
    emb_model, input_format = await get_model_by_version(
        session, cm.model_version, settings
    )
    feature_config = json.loads(cm.feature_config) if cm.feature_config else None
    normalization = (feature_config or {}).get("normalization", "per_window_max")

    # Run diff
    diff = await asyncio.to_thread(diff_row_store_vs_embeddings, rs_path, emb_path)

    if not diff.missing and not diff.orphaned_indices:
        # Nothing to do — already in sync.
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

    # Set progress to number of missing rows to generate
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job.id)
        .values(progress_total=len(diff.missing), progress_current=0)
    )
    await session.commit()

    # Prepare audio resolution
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
        )
    elif det_job.audio_folder:
        file_timeline = await asyncio.to_thread(
            _build_file_timeline, Path(det_job.audio_folder), cm.target_sample_rate
        )

    # Generate embeddings for missing rows
    new_records: list[dict] = []
    skipped_reasons: list[str] = []
    progress = 0
    audio_cache: dict[str, np.ndarray] = {}

    for row in diff.missing:
        start_utc = float(row["start_utc"])
        end_utc = float(row["end_utc"])

        # Resolve audio
        if is_hydrophone and provider is not None:
            audio, reason = await asyncio.to_thread(
                resolve_audio_for_window_hydrophone,
                start_utc,
                end_utc,
                provider,
                cm.target_sample_rate,
            )
        elif det_job.audio_folder:
            audio, reason = await asyncio.to_thread(
                resolve_audio_for_window,
                start_utc,
                end_utc,
                Path(det_job.audio_folder),
                cm.target_sample_rate,
                _file_timeline=file_timeline,
                _audio_cache=audio_cache,
            )
        else:
            audio = None
            reason = "no audio source (no audio_folder or hydrophone_id)"

        if audio is None:
            skipped_reasons.append(f"[{start_utc:.1f}, {end_utc:.1f}]: {reason}")
            progress += 1
            continue

        # Extract features and embed
        def _embed_window(audio_data: np.ndarray) -> list[float]:
            if input_format == "waveform":
                batch = np.expand_dims(audio_data, 0)
            else:
                specs = extract_logmel_batch(
                    [audio_data],
                    cm.target_sample_rate,
                    n_mels=128,
                    hop_length=1252,
                    target_frames=128,
                    normalization=normalization,
                )
                batch = np.stack(specs)
            embedding = emb_model.embed(batch)
            return embedding[0].tolist()

        try:
            emb_vec = await asyncio.to_thread(_embed_window, audio)
        except Exception as exc:
            skipped_reasons.append(
                f"[{start_utc:.1f}, {end_utc:.1f}]: embedding failed: {exc}"
            )
            progress += 1
            continue

        new_records.append(
            {
                "row_id": row.get("row_id", ""),
                "embedding": emb_vec,
                "confidence": None,
            }
        )

        progress += 1
        if progress % 5 == 0 or progress == len(diff.missing):
            await session.execute(
                update(DetectionEmbeddingJob)
                .where(DetectionEmbeddingJob.id == job.id)
                .values(progress_current=progress)
            )
            await session.commit()

    # Rewrite embeddings parquet: keep matched, drop orphaned, add new
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
            progress_current=len(diff.missing),
            result_summary=summary,
        )
    )
    await session.commit()

    logger.info(
        "Sync job %s complete: +%d -%d =%d skip=%d for detection %s",
        job.id,
        len(new_records),
        len(diff.orphaned_indices),
        diff.matched_count,
        len(skipped_reasons),
        job.detection_job_id,
    )
    await complete_detection_embedding_job(session, job.id)


def _rewrite_embeddings(
    emb_path: Path,
    orphaned_indices: list[int],
    new_records: list[dict],
) -> None:
    """Rewrite the embeddings parquet, removing orphans and appending new records."""
    import pyarrow as pa

    table = pq.read_table(str(emb_path))

    # Remove orphaned rows
    if orphaned_indices:
        keep_mask = [True] * table.num_rows
        for idx in orphaned_indices:
            keep_mask[idx] = False
        table = table.filter(pa.array(keep_mask))

    # If the existing table uses the old schema (filename/start_sec/end_sec),
    # drop those columns and keep only row_id/embedding/confidence.
    col_names = set(table.column_names)
    if "filename" in col_names and "row_id" not in col_names:
        # Legacy schema — cannot merge with new records; drop entire old table.
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
