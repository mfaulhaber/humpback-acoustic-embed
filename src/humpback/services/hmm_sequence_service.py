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

import joblib
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
from humpback.sequence_models.exemplars import WindowMeta, select_exemplars
from humpback.sequence_models.label_distribution import (
    DetectionWindow,
    LabelRecord,
    compute_label_distribution,
)
from humpback.sequence_models.overlay import OverlayMetadata, compute_overlay
from humpback.storage import (
    atomic_rename,
    continuous_embedding_parquet_path,
    detection_row_store_path,
    ensure_dir,
    hmm_sequence_dir,
    hmm_sequence_exemplars_dir,
    hmm_sequence_exemplars_path,
    hmm_sequence_label_distribution_path,
    hmm_sequence_overlay_path,
    hmm_sequence_pca_model_path,
    hmm_sequence_states_path,
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


def _load_overlay_inputs(
    storage_root: Path, job: HMMSequenceJob, cej: ContinuousEmbeddingJob
) -> tuple[
    Any,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    OverlayMetadata,
    list[WindowMeta],
]:
    """Load PCA model, raw embeddings, states, and window metadata."""
    pca_model = joblib.load(hmm_sequence_pca_model_path(storage_root, job.id))
    emb_table = pq.read_table(continuous_embedding_parquet_path(storage_root, cej.id))
    states_table = pq.read_table(hmm_sequence_states_path(storage_root, job.id))

    span_col = emb_table.column("merged_span_id").to_pylist()
    unique_spans = sorted(set(span_col))

    raw_sequences: list[np.ndarray] = []
    viterbi_states: list[np.ndarray] = []
    max_probs: list[np.ndarray] = []
    all_span_ids: list[int] = []
    all_win_indices: list[int] = []
    all_starts: list[float] = []
    all_ends: list[float] = []

    states_dict = states_table.to_pydict()
    states_by_span: dict[int, dict[int, dict[str, Any]]] = {}
    audio_file_by_span_win: dict[tuple[int, int], int] = {}
    for i in range(states_table.num_rows):
        sid = states_dict["merged_span_id"][i]
        widx = states_dict["window_index_in_span"][i]
        states_by_span.setdefault(sid, {})[widx] = {
            "viterbi_state": states_dict["viterbi_state"][i],
            "max_state_probability": states_dict["max_state_probability"][i],
        }
        audio_file_by_span_win[(sid, widx)] = states_dict["audio_file_id"][i]

    emb_span_vals = emb_table.column("merged_span_id").to_pylist()
    window_metas: list[WindowMeta] = []
    for span_id in unique_spans:
        indices = [i for i, v in enumerate(emb_span_vals) if v == span_id]
        sub = emb_table.take(indices).sort_by("window_index_in_span")
        embeddings = np.array(sub.column("embedding").to_pylist(), dtype=np.float32)
        raw_sequences.append(embeddings)

        span_states = states_by_span.get(span_id, {})
        n_rows = sub.num_rows
        v_states = np.zeros(n_rows, dtype=np.int16)
        m_probs = np.zeros(n_rows, dtype=np.float32)
        for row_i in range(n_rows):
            widx = sub.column("window_index_in_span")[row_i].as_py()
            st = span_states.get(widx, {})
            prob = st.get("max_state_probability", 0.0)
            v_states[row_i] = st.get("viterbi_state", 0)
            m_probs[row_i] = prob
            start = sub.column("start_timestamp")[row_i].as_py()
            end = sub.column("end_timestamp")[row_i].as_py()
            all_span_ids.append(span_id)
            all_win_indices.append(widx)
            all_starts.append(start)
            all_ends.append(end)
            window_metas.append(
                WindowMeta(
                    merged_span_id=span_id,
                    window_index_in_span=widx,
                    audio_file_id=audio_file_by_span_win.get((span_id, widx), 0),
                    start_timestamp=start,
                    end_timestamp=end,
                    max_state_probability=prob,
                )
            )

        viterbi_states.append(v_states)
        max_probs.append(m_probs)

    meta = OverlayMetadata(
        merged_span_ids=all_span_ids,
        window_indices=all_win_indices,
        start_timestamps=all_starts,
        end_timestamps=all_ends,
    )
    return pca_model, raw_sequences, viterbi_states, max_probs, meta, window_metas


def generate_interpretations(
    storage_root: Path,
    job: HMMSequenceJob,
    cej: ContinuousEmbeddingJob,
) -> None:
    """Generate PCA/UMAP overlay and state exemplars for a completed HMM job.

    Does NOT generate label distribution (that requires DB access to
    vocalization_labels, which changes over time).
    """
    pca_model, raw_sequences, viterbi_states, max_probs, meta, window_metas = (
        _load_overlay_inputs(storage_root, job, cej)
    )

    overlay_table, pca_full = compute_overlay(
        pca_model,
        raw_sequences,
        viterbi_states,
        max_probs,
        meta,
        l2_normalize=job.l2_normalize,
        random_state=job.random_seed,
    )
    overlay_dst = hmm_sequence_overlay_path(storage_root, job.id)
    overlay_tmp = overlay_dst.with_suffix(".parquet.tmp")
    overlay_dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(overlay_table, overlay_tmp)
    atomic_rename(overlay_tmp, overlay_dst)

    all_states_flat = np.concatenate(viterbi_states)

    exemplars_result = select_exemplars(
        pca_full, all_states_flat, window_metas, job.n_states
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

    Traces HMM → CEJ → RegionDetectionJob → hydrophone source, finds
    overlapping DetectionJobs with vocalization labels, and joins via
    center-time-in-window semantics.
    """
    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            f"ContinuousEmbeddingJob not found: {job.continuous_embedding_job_id}"
        )

    from humpback.models.call_parsing import RegionDetectionJob

    rdj = await session.get(RegionDetectionJob, cej.region_detection_job_id)
    if rdj is None:
        raise ValueError(f"RegionDetectionJob not found: {cej.region_detection_job_id}")

    states_table = pq.read_table(hmm_sequence_states_path(storage_root, job.id))
    state_rows: list[dict[str, Any]] = []
    for i in range(states_table.num_rows):
        start_time = float(states_table.column("start_timestamp")[i].as_py())
        end_time = float(states_table.column("end_timestamp")[i].as_py())
        state_rows.append(
            {
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "viterbi_state": states_table.column("viterbi_state")[i].as_py(),
            }
        )

    detection_windows: list[DetectionWindow] = []
    label_records: list[LabelRecord] = []

    if rdj.hydrophone_id:
        stmt = (
            select(DetectionJob)
            .where(DetectionJob.hydrophone_id == rdj.hydrophone_id)
            .where(DetectionJob.status == "complete")
        )
        result = await session.execute(stmt)
        det_jobs = list(result.scalars().all())

        for dj in det_jobs:
            rs_path = detection_row_store_path(storage_root, dj.id)
            if not rs_path.exists():
                continue
            from humpback.classifier.detection_rows import read_detection_row_store

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
        state_rows, detection_windows, label_records, job.n_states
    )

    dist_dst = hmm_sequence_label_distribution_path(storage_root, job.id)
    dist_tmp = dist_dst.with_suffix(".json.tmp")
    dist_dst.parent.mkdir(parents=True, exist_ok=True)
    dist_tmp.write_text(json.dumps(dist, sort_keys=True, indent=2), encoding="utf-8")
    atomic_rename(dist_tmp, dist_dst)

    return dist
