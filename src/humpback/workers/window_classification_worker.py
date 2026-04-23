"""Window classification sidecar worker.

Scores cached Perch embeddings from a completed Pass 1 region detection
job through existing multi-label vocalization classifiers, producing
dense per-window probability vectors in wide-format parquet.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import (
    read_embeddings,
    read_regions,
    region_job_dir,
    window_classification_job_dir,
)
from humpback.classifier.vocalization_inference import (
    load_vocalization_model,
    score_embeddings,
)
from humpback.config import Settings
from humpback.models.call_parsing import WindowClassificationJob
from humpback.storage import ensure_dir
from humpback.workers.queue import claim_window_classification_job

logger = logging.getLogger(__name__)

WINDOW_HALF_SEC = 2.5


def _select_windows_for_regions(
    embeddings_time_sec: list[float],
    embeddings_arr: np.ndarray,
    regions: list,
) -> tuple[list[float], list[str], np.ndarray]:
    """Select embeddings whose window center falls within padded region bounds.

    Returns ``(time_secs, region_ids, selected_embeddings)`` — one entry
    per (window, region) pair. A window may appear in multiple regions if
    padded bounds overlap.
    """
    out_times: list[float] = []
    out_regions: list[str] = []
    out_indices: list[int] = []

    for region in regions:
        pad_start = region.padded_start_sec
        pad_end = region.padded_end_sec
        for i, t in enumerate(embeddings_time_sec):
            center = t + WINDOW_HALF_SEC
            if pad_start <= center <= pad_end:
                out_times.append(t)
                out_regions.append(region.region_id)
                out_indices.append(i)

    if not out_indices:
        return (
            out_times,
            out_regions,
            np.zeros(
                (0, embeddings_arr.shape[1] if embeddings_arr.ndim == 2 else 0),
                dtype=np.float32,
            ),
        )

    return out_times, out_regions, embeddings_arr[out_indices]


def _write_window_scores(
    path: Path,
    time_secs: list[float],
    region_ids: list[str],
    scores: dict[str, np.ndarray],
    vocabulary: list[str],
) -> None:
    """Write wide-format window_scores.parquet atomically."""
    columns: dict[str, list] = {
        "time_sec": time_secs,
        "region_id": region_ids,
    }
    for type_name in vocabulary:
        arr = scores.get(type_name, np.zeros(len(time_secs)))
        columns[type_name] = [float(v) for v in arr]

    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


async def run_window_classification_job(
    session: AsyncSession,
    job: WindowClassificationJob,
    settings: Settings,
) -> None:
    """Execute a window classification sidecar job end-to-end."""
    job_id = job.id
    region_detection_job_id = job.region_detection_job_id
    vocalization_model_id = job.vocalization_model_id

    job_dir = ensure_dir(window_classification_job_dir(settings.storage_root, job_id))
    try:
        from humpback.models.vocalization import VocalizationClassifierModel

        vm = await session.get(VocalizationClassifierModel, vocalization_model_id)
        if vm is None:
            raise ValueError(
                f"VocalizationClassifierModel {vocalization_model_id} not found"
            )

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        rd_dir = region_job_dir(settings.storage_root, region_detection_job_id)
        embeddings_path = rd_dir / "embeddings.parquet"
        regions_path = rd_dir / "regions.parquet"

        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"embeddings.parquet not found in region job {region_detection_job_id}"
            )
        if not regions_path.exists():
            raise FileNotFoundError(
                f"regions.parquet not found in region job {region_detection_job_id}"
            )

        raw_embeddings = read_embeddings(embeddings_path)
        regions = read_regions(regions_path)

        if not raw_embeddings or not regions:
            _write_window_scores(job_dir / "window_scores.parquet", [], [], {}, [])
            now = datetime.now(timezone.utc)
            refreshed = await session.get(WindowClassificationJob, job_id)
            target = refreshed if refreshed is not None else job
            target.status = "complete"
            target.window_count = 0
            target.vocabulary_snapshot = "[]"
            target.completed_at = now
            target.updated_at = now
            await session.commit()
            return

        emb_times = [we.time_sec for we in raw_embeddings]
        emb_arr = np.array([we.embedding for we in raw_embeddings], dtype=np.float32)

        sel_times, sel_regions, sel_embeddings = _select_windows_for_regions(
            emb_times, emb_arr, regions
        )

        pipelines, vocabulary, _thresholds = load_vocalization_model(
            Path(vm.model_dir_path)
        )

        if len(sel_embeddings) > 0:
            scores = score_embeddings(pipelines, vocabulary, sel_embeddings)
        else:
            scores = {t: np.zeros(0) for t in vocabulary}

        _write_window_scores(
            job_dir / "window_scores.parquet",
            sel_times,
            sel_regions,
            scores,
            vocabulary,
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(WindowClassificationJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.window_count = len(sel_times)
        target.vocabulary_snapshot = json.dumps(vocabulary)
        target.completed_at = now
        target.updated_at = now
        await session.commit()

        logger.info(
            "window_classification | job=%s | complete | %d windows, %d types",
            job_id,
            len(sel_times),
            len(vocabulary),
        )

    except Exception as exc:
        logger.exception("Window classification job %s failed", job_id)
        scores_path = job_dir / "window_scores.parquet"
        if scores_path.exists():
            scores_path.unlink()
        for tmp in job_dir.glob("*.tmp"):
            tmp.unlink()
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(WindowClassificationJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark window classification job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> WindowClassificationJob | None:
    """Claim and process at most one window classification job."""
    job = await claim_window_classification_job(session)
    if job is None:
        return None
    await run_window_classification_job(session, job, settings)
    return job
