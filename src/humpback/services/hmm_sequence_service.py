"""Service layer for HMM sequence jobs.

Validates that the source ``ContinuousEmbeddingJob`` is complete before
creating an ``HMMSequenceJob``. No idempotency key — HMM training is
stochastic and comparing configs requires multiple runs.

PR 3 adds interpretation artifact generation (overlay, exemplars,
label distribution) for completed jobs.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.classifier import DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.schemas.sequence_models import HMMSequenceJobCreate
from humpback.sequence_models.exemplars import select_exemplars
from humpback.sequence_models.label_distribution import (
    DetectionWindow,
    LabelRecord,
    compute_label_distribution,
)
from humpback.sequence_models.loaders import get_loader
from humpback.sequence_models.overlay import compute_overlay
from humpback.services.continuous_embedding_service import source_kind_for
from humpback.storage import (
    atomic_rename,
    detection_row_store_path,
    ensure_dir,
    hmm_sequence_dir,
    hmm_sequence_exemplars_dir,
    hmm_sequence_exemplars_path,
    hmm_sequence_label_distribution_path,
    hmm_sequence_overlay_path,
)

logger = logging.getLogger(__name__)


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a job in a terminal state."""


async def create_hmm_sequence_job(
    session: AsyncSession,
    payload: HMMSequenceJobCreate,
) -> HMMSequenceJob:
    source = await session.get(
        ContinuousEmbeddingJob, payload.continuous_embedding_job_id
    )
    if source is None:
        raise ValueError(
            f"continuous_embedding_job not found: {payload.continuous_embedding_job_id}"
        )
    if source.status != JobStatus.complete.value:
        raise ValueError(
            "HMM sequence job requires a completed continuous_embedding_job "
            f"(current status: {source.status!r})"
        )

    from humpback.services.continuous_embedding_service import (
        SOURCE_KIND_REGION_CRNN,
        source_kind_for,
    )

    is_crnn = source_kind_for(source.model_version) == SOURCE_KIND_REGION_CRNN
    crnn_only_fields = {
        "training_mode": payload.training_mode,
        "event_core_overlap_threshold": payload.event_core_overlap_threshold,
        "near_event_window_seconds": payload.near_event_window_seconds,
        "event_balanced_proportions": payload.event_balanced_proportions,
        "subsequence_length_chunks": payload.subsequence_length_chunks,
        "subsequence_stride_chunks": payload.subsequence_stride_chunks,
        "target_train_chunks": payload.target_train_chunks,
        "min_region_length_seconds": payload.min_region_length_seconds,
    }
    if not is_crnn:
        present = [
            name for name, value in crnn_only_fields.items() if value is not None
        ]
        if present:
            raise ValueError(
                "training_mode and tier configuration are only valid for "
                "CRNN-source HMM jobs: " + ", ".join(present)
            )

    proportions_json: Optional[str] = None
    if payload.event_balanced_proportions is not None:
        proportions_json = json.dumps(
            payload.event_balanced_proportions,
            sort_keys=True,
            separators=(",", ":"),
        )

    job = HMMSequenceJob(
        continuous_embedding_job_id=payload.continuous_embedding_job_id,
        n_states=payload.n_states,
        pca_dims=payload.pca_dims,
        pca_whiten=payload.pca_whiten,
        l2_normalize=payload.l2_normalize,
        covariance_type=payload.covariance_type,
        n_iter=payload.n_iter,
        random_seed=payload.random_seed,
        min_sequence_length_frames=payload.min_sequence_length_frames,
        tol=payload.tol,
        training_mode=payload.training_mode,
        event_core_overlap_threshold=payload.event_core_overlap_threshold,
        near_event_window_seconds=payload.near_event_window_seconds,
        event_balanced_proportions=proportions_json,
        subsequence_length_chunks=payload.subsequence_length_chunks,
        subsequence_stride_chunks=payload.subsequence_stride_chunks,
        target_train_chunks=payload.target_train_chunks,
        min_region_length_seconds=payload.min_region_length_seconds,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def list_hmm_sequence_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
    continuous_embedding_job_id: Optional[str] = None,
) -> list[HMMSequenceJob]:
    stmt = select(HMMSequenceJob).order_by(HMMSequenceJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(HMMSequenceJob.status == status)
    if continuous_embedding_job_id is not None:
        stmt = stmt.where(
            HMMSequenceJob.continuous_embedding_job_id == continuous_embedding_job_id
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_hmm_sequence_job(
    session: AsyncSession, job_id: str
) -> Optional[HMMSequenceJob]:
    return await session.get(HMMSequenceJob, job_id)


async def cancel_hmm_sequence_job(
    session: AsyncSession, job_id: str
) -> Optional[HMMSequenceJob]:
    """Flip ``queued`` or ``running`` to ``canceled``.

    Returns the (possibly updated) job, ``None`` if not found,
    or raises ``CancelTerminalJobError`` for terminal states.
    """
    job = await get_hmm_sequence_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"hmm_sequence_job {job_id} is in terminal state {job.status!r}"
    )


async def delete_hmm_sequence_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(HMMSequenceJob, job_id)
    if job is None:
        return False
    artifact_dir = hmm_sequence_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True


# ---------------------------------------------------------------------------
# Interpretation artifact generation (PR 3)
# ---------------------------------------------------------------------------


def generate_interpretations(
    storage_root: Path,
    job: HMMSequenceJob,
    cej: ContinuousEmbeddingJob,
) -> None:
    """Generate PCA/UMAP overlay and state exemplars for a completed HMM job.

    Source-agnostic: dispatches to the registered loader for the upstream
    embedding source family (SurfPerch event-padded vs. CRNN region-based;
    see ADR-059). Does NOT generate label distribution (that requires DB
    access to vocalization_labels, which changes over time).
    """
    loader = get_loader(source_kind_for(cej.model_version))
    inputs = loader.load(storage_root, job, cej)

    overlay_table, pca_full = compute_overlay(
        inputs.pca_model,
        inputs.raw_sequences,
        inputs.viterbi_states,
        inputs.max_probs,
        inputs.metadata,
        l2_normalize=job.l2_normalize,
        random_state=job.random_seed,
    )
    overlay_dst = hmm_sequence_overlay_path(storage_root, job.id)
    overlay_tmp = overlay_dst.with_suffix(".parquet.tmp")
    overlay_dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(overlay_table, overlay_tmp)
    atomic_rename(overlay_tmp, overlay_dst)

    all_states_flat = np.concatenate(inputs.viterbi_states)

    exemplars_result = select_exemplars(
        pca_full, all_states_flat, inputs.window_metas, job.n_states
    )
    exemplars_dir = ensure_dir(hmm_sequence_exemplars_dir(storage_root, job.id))
    exemplars_dst = hmm_sequence_exemplars_path(storage_root, job.id)
    exemplars_tmp = exemplars_dir / "exemplars.json.tmp"
    exemplars_tmp.write_text(
        json.dumps(exemplars_result, sort_keys=True, indent=2), encoding="utf-8"
    )
    atomic_rename(exemplars_tmp, exemplars_dst)


async def generate_label_distribution(
    session: AsyncSession,
    storage_root: Path,
    job: HMMSequenceJob,
) -> dict[str, Any]:
    """Compute and persist state-to-label distribution for a completed HMM job.

    Source-agnostic: dispatches to the registered loader for the upstream
    embedding source (SurfPerch event-padded vs. CRNN region-based; see
    ADR-060) for hydrophone resolution and per-row tier extraction. The
    DetectionJob + VocalizationLabel SQL fetch is shared across sources.
    Joins HMM window timestamps with detection-window extents and labels
    via center-time-in-window semantics.
    """
    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            f"ContinuousEmbeddingJob not found: {job.continuous_embedding_job_id}"
        )

    loader = get_loader(source_kind_for(cej.model_version))
    inputs = await loader.load_label_distribution_inputs(
        session, storage_root, job, cej
    )

    detection_windows: list[DetectionWindow] = []
    label_records: list[LabelRecord] = []

    if inputs.hydrophone_id:
        stmt = (
            select(DetectionJob)
            .where(DetectionJob.hydrophone_id == inputs.hydrophone_id)
            .where(DetectionJob.status == "complete")
        )
        result = await session.execute(stmt)
        det_jobs = list(result.scalars().all())

        from humpback.classifier.detection_rows import read_detection_row_store

        for dj in det_jobs:
            rs_path = detection_row_store_path(storage_root, dj.id)
            if not rs_path.exists():
                continue

            _, rows = read_detection_row_store(rs_path)
            for r in rows:
                rid = r.get("row_id", "")
                s_utc = r.get("start_utc", "")
                e_utc = r.get("end_utc", "")
                if rid and s_utc and e_utc:
                    detection_windows.append(
                        DetectionWindow(
                            row_id=rid,
                            start_utc=float(s_utc),
                            end_utc=float(e_utc),
                        )
                    )

            lbl_stmt = (
                select(VocalizationLabel)
                .where(VocalizationLabel.detection_job_id == dj.id)
                .where(VocalizationLabel.source == "manual")
            )
            lbl_result = await session.execute(lbl_stmt)
            for lbl in lbl_result.scalars().all():
                label_records.append(LabelRecord(row_id=lbl.row_id, label=lbl.label))

    dist = compute_label_distribution(
        inputs.state_rows,
        detection_windows,
        label_records,
        job.n_states,
        tier_per_row=inputs.tier_per_row,
    )

    dist_dst = hmm_sequence_label_distribution_path(storage_root, job.id)
    dist_tmp = dist_dst.with_suffix(".json.tmp")
    dist_dst.parent.mkdir(parents=True, exist_ok=True)
    dist_tmp.write_text(json.dumps(dist, sort_keys=True, indent=2), encoding="utf-8")
    atomic_rename(dist_tmp, dist_dst)

    return dist
