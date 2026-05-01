"""Worker for HMM sequence jobs.

Reads a completed continuous-embedding parquet, fits PCA + GaussianHMM,
decodes Viterbi states for every window/chunk, and persists model
artifacts atomically.

ADR-057 added a CRNN-source path: the upstream parquet groups by
``region_id`` instead of ``merged_span_id``, the training set is built
via ``region_sampling.build_training_set``, and ``states.parquet``
carries ``tier`` + ``was_used_for_training`` columns.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.hmm_decoder import decode_sequences
from humpback.sequence_models.hmm_trainer import fit_hmm
from humpback.sequence_models.pca_pipeline import fit_pca, transform_sequences
from humpback.sequence_models.region_sampling import (
    RegionSequence,
    SamplingConfig,
    TierConfig,
    build_training_set,
)
from humpback.sequence_models.summary import compute_summary
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.storage import (
    atomic_rename,
    continuous_embedding_parquet_path,
    ensure_dir,
    hmm_sequence_dir,
    hmm_sequence_hmm_model_path,
    hmm_sequence_pca_model_path,
    hmm_sequence_states_path,
    hmm_sequence_summary_path,
    hmm_sequence_training_log_path,
    hmm_sequence_transition_matrix_path,
)
from humpback.workers.queue import claim_hmm_sequence_job

logger = logging.getLogger(__name__)


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    """Remove temp files left by an incomplete run."""
    for p in job_dir.glob("*.tmp"):
        try:
            p.unlink()
        except OSError:
            logger.debug("failed to remove temp file %s", p, exc_info=True)
    for p in job_dir.glob("*.tmp.*"):
        try:
            p.unlink()
        except OSError:
            logger.debug("failed to remove temp file %s", p, exc_info=True)


def _atomic_write_joblib(obj: object, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, tmp)
    atomic_rename(tmp, dst)


def _atomic_write_npy(arr: np.ndarray, dst: Path) -> None:
    tmp = dst.parent / (dst.stem + ".tmp.npy")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, arr)
    atomic_rename(tmp, dst)


def _atomic_write_json(payload: dict, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    atomic_rename(tmp, dst)


def _atomic_write_parquet(table: pa.Table, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    atomic_rename(tmp, dst)


def _group_sequences(
    table: pa.Table,
) -> tuple[list[int], list[np.ndarray], list[pa.Table]]:
    """Group parquet rows by merged_span_id, returning sequences sorted by window index."""
    span_col = table.column("merged_span_id").to_pylist()
    unique_spans = sorted(set(span_col))
    sequences: list[np.ndarray] = []
    span_tables: list[pa.Table] = []

    span_values = table.column("merged_span_id").to_pylist()
    for span_id in unique_spans:
        indices = [i for i, v in enumerate(span_values) if v == span_id]
        sub = table.take(indices).sort_by("window_index_in_span")
        embeddings = sub.column("embedding").to_pylist()
        sequences.append(np.array(embeddings, dtype=np.float32))
        span_tables.append(sub)

    return unique_spans, sequences, span_tables


STATES_SCHEMA = pa.schema(
    [
        pa.field("merged_span_id", pa.int32()),
        pa.field("window_index_in_span", pa.int32()),
        pa.field("audio_file_id", pa.int32()),
        pa.field("start_timestamp", pa.float64()),
        pa.field("end_timestamp", pa.float64()),
        pa.field("is_in_pad", pa.bool_()),
        pa.field("event_id", pa.string()),
        pa.field("viterbi_state", pa.int16()),
        pa.field("state_posterior", pa.list_(pa.float32())),
        pa.field("max_state_probability", pa.float32()),
        pa.field("was_used_for_training", pa.bool_()),
    ]
)


REGION_STATES_SCHEMA = pa.schema(
    [
        pa.field("region_id", pa.string()),
        pa.field("chunk_index_in_region", pa.int32()),
        pa.field("audio_file_id", pa.int32(), nullable=True),
        pa.field("start_timestamp", pa.float64()),
        pa.field("end_timestamp", pa.float64()),
        pa.field("is_in_pad", pa.bool_()),
        pa.field("tier", pa.string()),
        pa.field("viterbi_state", pa.int16()),
        pa.field("state_posterior", pa.list_(pa.float32())),
        pa.field("max_state_probability", pa.float32()),
        pa.field("was_used_for_training", pa.bool_()),
    ]
)


_DEFAULT_TIER_PROPORTIONS = {"event_core": 0.40, "near_event": 0.35, "background": 0.25}


def _group_region_sequences(
    table: pa.Table,
) -> tuple[list[str], list[np.ndarray], list[pa.Table], list[np.ndarray]]:
    """Group CRNN-source rows by ``region_id``.

    Returns ``(region_ids, embeddings, region_tables, tiers)`` aligned by
    region order; rows within a region are sorted by chunk index.
    """
    region_col = table.column("region_id").to_pylist()
    unique_regions = sorted(set(region_col))
    sequences: list[np.ndarray] = []
    region_tables: list[pa.Table] = []
    tiers_per_region: list[np.ndarray] = []

    for region_id in unique_regions:
        indices = [i for i, v in enumerate(region_col) if v == region_id]
        sub = table.take(indices).sort_by("chunk_index_in_region")
        embeddings = sub.column("embedding").to_pylist()
        sequences.append(np.asarray(embeddings, dtype=np.float32))
        region_tables.append(sub)
        tiers_per_region.append(
            np.asarray(sub.column("tier").to_pylist(), dtype=object)
        )

    return unique_regions, sequences, region_tables, tiers_per_region


def _parse_tier_proportions(payload: Optional[str]) -> dict[str, float]:
    if not payload:
        return dict(_DEFAULT_TIER_PROPORTIONS)
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("event_balanced_proportions JSON must be an object")
    return {str(k): float(v) for k, v in parsed.items()}


def _compute_tier_composition(rows: list[dict], n_states: int) -> list[dict]:
    """Aggregate per-state ``tier`` percentages from a row list."""
    counts: dict[int, dict[str, int]] = {
        s: {"event_core": 0, "near_event": 0, "background": 0} for s in range(n_states)
    }
    for row in rows:
        s = int(row["viterbi_state"])
        tier = str(row["tier"])
        if tier not in counts[s]:
            counts[s][tier] = 0
        counts[s][tier] += 1
    composition: list[dict] = []
    for s in range(n_states):
        bucket = counts[s]
        total = sum(bucket.values())
        if total == 0:
            composition.append(
                {
                    "state": s,
                    "event_core": 0.0,
                    "near_event": 0.0,
                    "background": 0.0,
                }
            )
        else:
            composition.append(
                {
                    "state": s,
                    "event_core": bucket.get("event_core", 0) / total,
                    "near_event": bucket.get("near_event", 0) / total,
                    "background": bucket.get("background", 0) / total,
                }
            )
    return composition


async def run_hmm_sequence_job(
    session: AsyncSession,
    job: HMMSequenceJob,
    settings: Settings,
) -> None:
    """Execute one HMM sequence job end-to-end.

    Dispatches on the upstream embedding job's source kind. The
    PCA / HMM trainer / decoder are source-agnostic; only the parquet
    grouping, training-set selection, and persisted ``states.parquet``
    schema differ between SurfPerch and CRNN sources.
    """
    job_id = job.id
    job_dir = ensure_dir(hmm_sequence_dir(settings.storage_root, job_id))

    try:
        job = await session.merge(job)
        # Validate source
        source = await session.get(
            ContinuousEmbeddingJob, job.continuous_embedding_job_id
        )
        if source is None or source.status != JobStatus.complete.value:
            raise ValueError(
                f"source continuous_embedding_job "
                f"{job.continuous_embedding_job_id} not complete"
            )

        embeddings_path = continuous_embedding_parquet_path(
            settings.storage_root, source.id
        )
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"embeddings.parquet not found for continuous_embedding_job {source.id}"
            )

        table = pq.read_table(embeddings_path)

        is_crnn_source = (
            source_kind_for(source.model_version) == SOURCE_KIND_REGION_CRNN
        )

        if is_crnn_source:
            await _run_region_crnn_hmm(
                session=session,
                job=job,
                source=source,
                job_dir=job_dir,
                table=table,
                settings=settings,
            )
            return

        span_ids, sequences, span_tables = _group_sequences(table)

        if not sequences:
            raise ValueError("no embedding sequences found in source parquet")

        # --- Cancellation check ---
        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        # --- PCA ---
        pca_model, preproc_seqs = fit_pca(
            sequences,
            pca_dims=job.pca_dims,
            whiten=job.pca_whiten,
            l2_norm=job.l2_normalize,
            random_state=job.random_seed,
        )
        transformed = transform_sequences(pca_model, preproc_seqs)

        # --- Cancellation check ---
        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        # --- HMM fit ---
        train_result = fit_hmm(
            transformed,
            n_states=job.n_states,
            covariance_type=job.covariance_type,
            n_iter=job.n_iter,
            tol=job.tol,
            random_state=job.random_seed,
            min_sequence_length_frames=job.min_sequence_length_frames,
        )

        # --- Cancellation check ---
        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        # --- Decode ---
        decoded = decode_sequences(train_result.model, transformed)

        # --- Summary ---
        viterbi_per_seq = [d.viterbi_states.astype(np.intp) for d in decoded]
        summary = compute_summary(viterbi_per_seq, job.n_states)

        # --- Build states parquet ---
        rows: list[dict] = []
        for i, (span_id, span_tbl, dec) in enumerate(
            zip(span_ids, span_tables, decoded)
        ):
            was_trained = train_result.training_mask[i]
            for row_idx in range(span_tbl.num_rows):
                rows.append(
                    {
                        "merged_span_id": span_tbl.column("merged_span_id")[
                            row_idx
                        ].as_py(),
                        "window_index_in_span": span_tbl.column("window_index_in_span")[
                            row_idx
                        ].as_py(),
                        "audio_file_id": span_tbl.column("audio_file_id")[
                            row_idx
                        ].as_py(),
                        "start_timestamp": span_tbl.column("start_timestamp")[
                            row_idx
                        ].as_py(),
                        "end_timestamp": span_tbl.column("end_timestamp")[
                            row_idx
                        ].as_py(),
                        "is_in_pad": span_tbl.column("is_in_pad")[row_idx].as_py(),
                        "event_id": span_tbl.column("event_id")[row_idx].as_py(),
                        "viterbi_state": int(dec.viterbi_states[row_idx]),
                        "state_posterior": dec.posteriors[row_idx].tolist(),
                        "max_state_probability": float(
                            dec.max_state_probability[row_idx]
                        ),
                        "was_used_for_training": was_trained,
                    }
                )

        states_table = pa.Table.from_pylist(rows, schema=STATES_SCHEMA)

        # --- Persist artifacts atomically ---
        _atomic_write_parquet(
            states_table, hmm_sequence_states_path(settings.storage_root, job_id)
        )
        _atomic_write_joblib(
            pca_model, hmm_sequence_pca_model_path(settings.storage_root, job_id)
        )
        _atomic_write_joblib(
            train_result.model,
            hmm_sequence_hmm_model_path(settings.storage_root, job_id),
        )
        _atomic_write_npy(
            summary.transition_matrix,
            hmm_sequence_transition_matrix_path(settings.storage_root, job_id),
        )

        summary_payload = {
            "n_states": job.n_states,
            "states": [
                {
                    "state": s.state,
                    "occupancy": s.occupancy,
                    "mean_dwell_frames": s.mean_dwell_frames,
                    "dwell_histogram": s.dwell_histogram,
                }
                for s in summary.states
            ],
        }
        _atomic_write_json(
            summary_payload,
            hmm_sequence_summary_path(settings.storage_root, job_id),
        )

        training_log = {
            "library": job.library,
            "n_states": job.n_states,
            "pca_dims": pca_model.n_components_,
            "pca_whiten": job.pca_whiten,
            "l2_normalize": job.l2_normalize,
            "covariance_type": job.covariance_type,
            "n_iter": job.n_iter,
            "tol": job.tol,
            "random_seed": job.random_seed,
            "min_sequence_length_frames": job.min_sequence_length_frames,
            "train_log_likelihood": train_result.train_log_likelihood,
            "n_train_sequences": train_result.n_train_sequences,
            "n_train_frames": train_result.n_train_frames,
            "n_decoded_sequences": len(decoded),
        }
        _atomic_write_json(
            training_log,
            hmm_sequence_training_log_path(settings.storage_root, job_id),
        )

        # --- Generate interpretation artifacts (overlay + exemplars) ---
        try:
            from humpback.services.hmm_sequence_service import generate_interpretations

            generate_interpretations(settings.storage_root, job, source)
        except Exception:
            logger.warning(
                "hmm_sequence | job=%s | interpretation generation failed, continuing",
                job_id,
                exc_info=True,
            )

        # --- Update job row ---
        now = datetime.now(timezone.utc)
        refreshed = await session.get(HMMSequenceJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.train_log_likelihood = train_result.train_log_likelihood
        target.n_train_sequences = train_result.n_train_sequences
        target.n_train_frames = train_result.n_train_frames
        target.n_decoded_sequences = len(decoded)
        target.artifact_dir = str(job_dir)
        target.error_message = None
        target.updated_at = now
        await session.commit()

        logger.info(
            "hmm_sequence | job=%s | complete | states=%d train_seqs=%d frames=%d",
            job_id,
            job.n_states,
            train_result.n_train_sequences,
            train_result.n_train_frames,
        )

    except Exception as exc:
        logger.exception("hmm_sequence job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(HMMSequenceJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = JobStatus.failed.value
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                await session.commit()
        except Exception:
            logger.exception("failed to mark hmm_sequence job %s as failed", job_id)


async def run_one_iteration(
    session: AsyncSession,
    settings: Settings,
) -> Optional[HMMSequenceJob]:
    """Claim and process at most one HMM sequence job."""
    job = await claim_hmm_sequence_job(session)
    if job is None:
        return None
    await run_hmm_sequence_job(session, job, settings)
    return job


async def _run_region_crnn_hmm(
    *,
    session: AsyncSession,
    job: HMMSequenceJob,
    source: ContinuousEmbeddingJob,
    job_dir: Path,
    table: pa.Table,
    settings: Settings,
) -> None:
    """CRNN-source HMM training + decode (ADR-057 consumer path)."""
    job_id = job.id

    region_ids, sequences, region_tables, tiers_per_region = _group_region_sequences(
        table
    )
    if not sequences:
        raise ValueError("no embedding sequences found in source parquet")

    await session.refresh(job)
    if job.status == JobStatus.canceled.value:
        _cleanup_partial_artifacts(job_dir)
        return

    training_mode = job.training_mode or "event_balanced"
    tier_proportions = _parse_tier_proportions(job.event_balanced_proportions)
    tier_config = TierConfig(event_balanced_proportions=tier_proportions)
    sampling_config = SamplingConfig(
        subsequence_length_chunks=int(job.subsequence_length_chunks or 32),
        subsequence_stride_chunks=int(job.subsequence_stride_chunks or 16),
        target_train_chunks=int(job.target_train_chunks or 200_000),
        min_sequence_length_frames=job.min_sequence_length_frames,
        random_seed=job.random_seed,
    )

    region_sequences = [
        RegionSequence(region_id=rid, chunks=seq, tiers=tiers)
        for rid, seq, tiers in zip(region_ids, sequences, tiers_per_region)
    ]

    training_set = build_training_set(
        region_sequences=region_sequences,
        mode=training_mode,
        tier_config=tier_config,
        sampling=sampling_config,
    )
    if not training_set.sub_sequences:
        raise ValueError(
            "no training sub-sequences produced "
            f"(mode={training_mode!r}; check region lengths and tier coverage)"
        )

    pca_model, preproc_seqs = fit_pca(
        training_set.sub_sequences,
        pca_dims=job.pca_dims,
        whiten=job.pca_whiten,
        l2_norm=job.l2_normalize,
        random_state=job.random_seed,
    )

    await session.refresh(job)
    if job.status == JobStatus.canceled.value:
        _cleanup_partial_artifacts(job_dir)
        return

    train_result = fit_hmm(
        transform_sequences(pca_model, preproc_seqs),
        n_states=job.n_states,
        covariance_type=job.covariance_type,
        n_iter=job.n_iter,
        tol=job.tol,
        random_state=job.random_seed,
        # Sub-sequences are uniform-length by construction; min-length
        # filtering would just drop everything unexpectedly. Use 1 here
        # so any non-empty training sub-sequence trains the HMM.
        min_sequence_length_frames=1,
    )

    await session.refresh(job)
    if job.status == JobStatus.canceled.value:
        _cleanup_partial_artifacts(job_dir)
        return

    # Decode whole regions, not training sub-sequences. Each whole-region
    # sequence goes through L2 + PCA the same way as training data.
    full_region_seqs = transform_sequences(
        pca_model,
        [
            np.asarray(s, dtype=np.float32)
            for s in (_l2_if_needed(seq, job.l2_normalize) for seq in sequences)
        ],
    )
    decoded = decode_sequences(train_result.model, full_region_seqs)

    viterbi_per_seq = [d.viterbi_states.astype(np.intp) for d in decoded]
    summary = compute_summary(viterbi_per_seq, job.n_states)

    rows: list[dict] = []
    for region_id, region_tbl, dec in zip(region_ids, region_tables, decoded):
        was_used = training_set.was_used_for_training_per_region.get(
            region_id, np.zeros(region_tbl.num_rows, dtype=bool)
        )
        for row_idx in range(region_tbl.num_rows):
            audio_file_id = region_tbl.column("audio_file_id")[row_idx].as_py()
            rows.append(
                {
                    "region_id": region_id,
                    "chunk_index_in_region": region_tbl.column("chunk_index_in_region")[
                        row_idx
                    ].as_py(),
                    "audio_file_id": audio_file_id,
                    "start_timestamp": region_tbl.column("start_timestamp")[
                        row_idx
                    ].as_py(),
                    "end_timestamp": region_tbl.column("end_timestamp")[
                        row_idx
                    ].as_py(),
                    "is_in_pad": region_tbl.column("is_in_pad")[row_idx].as_py(),
                    "tier": region_tbl.column("tier")[row_idx].as_py(),
                    "viterbi_state": int(dec.viterbi_states[row_idx]),
                    "state_posterior": dec.posteriors[row_idx].tolist(),
                    "max_state_probability": float(dec.max_state_probability[row_idx]),
                    "was_used_for_training": bool(was_used[row_idx]),
                }
            )

    states_table = pa.Table.from_pylist(rows, schema=REGION_STATES_SCHEMA)

    _atomic_write_parquet(
        states_table, hmm_sequence_states_path(settings.storage_root, job_id)
    )
    _atomic_write_joblib(
        pca_model, hmm_sequence_pca_model_path(settings.storage_root, job_id)
    )
    _atomic_write_joblib(
        train_result.model,
        hmm_sequence_hmm_model_path(settings.storage_root, job_id),
    )
    _atomic_write_npy(
        summary.transition_matrix,
        hmm_sequence_transition_matrix_path(settings.storage_root, job_id),
    )

    tier_composition = _compute_tier_composition(rows, job.n_states)
    summary_payload = {
        "n_states": job.n_states,
        "source_kind": "region_crnn",
        "training_mode": training_mode,
        "tier_composition": tier_composition,
        "states": [
            {
                "state": s.state,
                "occupancy": s.occupancy,
                "mean_dwell_frames": s.mean_dwell_frames,
                "dwell_histogram": s.dwell_histogram,
            }
            for s in summary.states
        ],
    }
    _atomic_write_json(
        summary_payload,
        hmm_sequence_summary_path(settings.storage_root, job_id),
    )

    training_log = {
        "library": job.library,
        "n_states": job.n_states,
        "pca_dims": pca_model.n_components_,
        "pca_whiten": job.pca_whiten,
        "l2_normalize": job.l2_normalize,
        "covariance_type": job.covariance_type,
        "n_iter": job.n_iter,
        "tol": job.tol,
        "random_seed": job.random_seed,
        "min_sequence_length_frames": job.min_sequence_length_frames,
        "training_mode": training_mode,
        "tier_proportions": tier_proportions,
        "subsequence_length_chunks": sampling_config.subsequence_length_chunks,
        "subsequence_stride_chunks": sampling_config.subsequence_stride_chunks,
        "target_train_chunks": sampling_config.target_train_chunks,
        "train_log_likelihood": train_result.train_log_likelihood,
        "n_train_sequences": train_result.n_train_sequences,
        "n_train_frames": train_result.n_train_frames,
        "n_decoded_sequences": len(decoded),
    }
    _atomic_write_json(
        training_log,
        hmm_sequence_training_log_path(settings.storage_root, job_id),
    )

    # --- Generate interpretation artifacts (overlay + exemplars) ---
    # Run BEFORE the status flip so a UI poll that sees status=complete
    # never 404s on GET /overlay or /exemplars. Failure is non-fatal:
    # the job still completes.
    try:
        from humpback.services.hmm_sequence_service import generate_interpretations

        generate_interpretations(settings.storage_root, job, source)
    except Exception:
        logger.warning(
            "hmm_sequence | job=%s | interpretation generation failed, continuing",
            job_id,
            exc_info=True,
        )

    now = datetime.now(timezone.utc)
    refreshed = await session.get(HMMSequenceJob, job_id)
    target = refreshed if refreshed is not None else job
    target.status = JobStatus.complete.value
    target.train_log_likelihood = train_result.train_log_likelihood
    target.n_train_sequences = train_result.n_train_sequences
    target.n_train_frames = train_result.n_train_frames
    target.n_decoded_sequences = len(decoded)
    target.artifact_dir = str(job_dir)
    target.error_message = None
    target.updated_at = now
    await session.commit()

    logger.info(
        "hmm_sequence | job=%s | complete (CRNN) | states=%d regions=%d chunks=%d",
        job_id,
        job.n_states,
        len(region_ids),
        len(rows),
    )


def _l2_if_needed(seq: np.ndarray, do_l2: bool) -> np.ndarray:
    """L2-normalize rows of ``seq`` when ``do_l2`` is true."""
    if not do_l2:
        return seq
    from sklearn.preprocessing import normalize

    return np.asarray(normalize(seq, norm="l2"), dtype=np.float32)
