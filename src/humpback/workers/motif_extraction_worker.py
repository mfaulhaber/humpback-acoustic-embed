"""Worker for motif extraction jobs."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import segmentation_job_dir
from humpback.config import Settings
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MaskedTransformerJob,
    MotifExtractionJob,
)
from humpback.sequence_models.loaders import read_decoded_parquet
from humpback.sequence_models.motifs import (
    MotifExtractionConfig,
    extract_motifs,
    write_motif_artifacts,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.storage import (
    continuous_embedding_parquet_path,
    hmm_sequence_decoded_path,
    hmm_sequence_legacy_states_path,
    masked_transformer_k_decoded_path,
    motif_extraction_dir,
)
from humpback.workers.queue import claim_motif_extraction_job

logger = logging.getLogger(__name__)


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)


def _config_from_job(job: MotifExtractionJob) -> MotifExtractionConfig:
    return MotifExtractionConfig(
        min_ngram=job.min_ngram,
        max_ngram=job.max_ngram,
        minimum_occurrences=job.minimum_occurrences,
        minimum_event_sources=job.minimum_event_sources,
        frequency_weight=job.frequency_weight,
        event_source_weight=job.event_source_weight,
        event_core_weight=job.event_core_weight,
        low_background_weight=job.low_background_weight,
        call_probability_weight=job.call_probability_weight,
    )


def _load_event_lookup(storage_root: Path, cej: ContinuousEmbeddingJob) -> dict:
    if not cej.event_segmentation_job_id:
        return {}
    events_path = (
        segmentation_job_dir(storage_root, cej.event_segmentation_job_id)
        / "events.parquet"
    )
    if not events_path.exists():
        return {}
    table = pq.read_table(events_path)
    if not {"event_id", "start_sec", "end_sec"}.issubset(table.column_names):
        return {}
    rows = table.to_pylist()
    return {
        str(row["event_id"]): (float(row["start_sec"]), float(row["end_sec"]))
        for row in rows
        if row.get("event_id") is not None
    }


async def _raise_if_canceled(
    session: AsyncSession,
    job: MotifExtractionJob,
    job_dir: Path,
) -> bool:
    await session.refresh(job)
    if job.status == JobStatus.canceled.value:
        _cleanup_partial_artifacts(job_dir)
        return True
    return False


def _resolve_decoded_path(
    storage_root: Path,
    job: MotifExtractionJob,
    hmm_job: Optional[HMMSequenceJob],
    mt_job: Optional[MaskedTransformerJob],
) -> Path:
    """Resolve the decoded.parquet path based on parent_kind.

    HMM parent: prefers ``decoded.parquet`` and falls back to the legacy
    ``states.parquet`` for jobs that completed before ADR-061.
    Masked-transformer parent: reads ``k<N>/decoded.parquet`` from the
    masked-transformer job dir.
    """
    if job.parent_kind == "masked_transformer":
        if mt_job is None or job.k is None:
            raise ValueError(
                "masked-transformer parent requires masked_transformer_job_id and k"
            )
        path = masked_transformer_k_decoded_path(storage_root, mt_job.id, int(job.k))
        if not path.exists():
            raise FileNotFoundError(
                f"decoded.parquet not found for masked_transformer_job "
                f"{mt_job.id} k={int(job.k)}"
            )
        return path

    if hmm_job is None:
        raise ValueError("HMM parent requires hmm_sequence_job_id")
    decoded_path = hmm_sequence_decoded_path(storage_root, hmm_job.id)
    legacy_path = hmm_sequence_legacy_states_path(storage_root, hmm_job.id)
    if not decoded_path.exists() and not legacy_path.exists():
        raise FileNotFoundError(f"decoded.parquet not found for HMM job {hmm_job.id}")
    return decoded_path


async def run_motif_extraction_job(
    session: AsyncSession,
    job: MotifExtractionJob,
    settings: Settings,
) -> None:
    """Execute one motif extraction job end-to-end."""
    job_id = job.id
    job_dir = motif_extraction_dir(settings.storage_root, job_id)

    try:
        job = await session.merge(job)
        hmm_job: Optional[HMMSequenceJob] = None
        mt_job: Optional[MaskedTransformerJob] = None
        cej: Optional[ContinuousEmbeddingJob] = None

        if job.parent_kind == "masked_transformer":
            if not job.masked_transformer_job_id:
                raise ValueError(
                    "masked-transformer parent missing masked_transformer_job_id"
                )
            mt_job = await session.get(
                MaskedTransformerJob, job.masked_transformer_job_id
            )
            if mt_job is None or mt_job.status != JobStatus.complete.value:
                raise ValueError(
                    f"source masked_transformer_job {job.masked_transformer_job_id}"
                    " is not complete"
                )
            cej = await session.get(
                ContinuousEmbeddingJob, mt_job.continuous_embedding_job_id
            )
            if cej is None:
                raise ValueError(
                    "source continuous_embedding_job not found: "
                    f"{mt_job.continuous_embedding_job_id}"
                )
        else:
            if not job.hmm_sequence_job_id:
                raise ValueError("HMM parent missing hmm_sequence_job_id")
            hmm_job = await session.get(HMMSequenceJob, job.hmm_sequence_job_id)
            if hmm_job is None or hmm_job.status != JobStatus.complete.value:
                raise ValueError(
                    f"source hmm_sequence_job {job.hmm_sequence_job_id} is not complete"
                )
            cej = await session.get(
                ContinuousEmbeddingJob, hmm_job.continuous_embedding_job_id
            )
            if cej is None:
                raise ValueError(
                    "source continuous_embedding_job not found: "
                    f"{hmm_job.continuous_embedding_job_id}"
                )

        source_kind = source_kind_for(cej.model_version)
        decoded_path = _resolve_decoded_path(
            settings.storage_root, job, hmm_job, mt_job
        )

        # ``read_decoded_parquet`` applies the legacy backwards shim and
        # always returns a table whose label column is named ``label``.
        states_table = read_decoded_parquet(decoded_path)
        if await _raise_if_canceled(session, job, job_dir):
            return

        embedding_table: Optional[pa.Table] = None
        if source_kind == SOURCE_KIND_REGION_CRNN:
            embeddings_path = continuous_embedding_parquet_path(
                settings.storage_root, cej.id
            )
            if not embeddings_path.exists():
                raise FileNotFoundError(
                    f"embeddings.parquet not found for continuous embedding {cej.id}"
                )
            embedding_table = pq.read_table(embeddings_path)

        event_lookup = _load_event_lookup(settings.storage_root, cej)
        if await _raise_if_canceled(session, job, job_dir):
            return

        result = extract_motifs(
            states_table,
            source_kind=source_kind,
            config=_config_from_job(job),
            hmm_sequence_job_id=(hmm_job.id if hmm_job else ""),
            continuous_embedding_job_id=cej.id,
            event_lookup=event_lookup,
            embedding_table=embedding_table,
            parent_kind=job.parent_kind,
            masked_transformer_job_id=(mt_job.id if mt_job else ""),
            k=int(job.k) if job.k is not None else None,
        )

        if await _raise_if_canceled(session, job, job_dir):
            return

        write_motif_artifacts(result, job_dir, motif_extraction_job_id=job_id)

        now = datetime.now(timezone.utc)
        refreshed = await session.get(MotifExtractionJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.source_kind = source_kind
        target.total_groups = result.total_groups
        target.total_collapsed_tokens = result.total_collapsed_tokens
        target.total_candidate_occurrences = result.total_candidate_occurrences
        target.total_motifs = result.total_motifs
        target.artifact_dir = str(job_dir)
        target.error_message = None
        target.updated_at = now
        await session.commit()

        logger.info(
            "motif_extraction | job=%s | complete | motifs=%d occurrences=%d",
            job_id,
            result.total_motifs,
            len(result.occurrences),
        )

    except Exception as exc:
        logger.exception("motif_extraction job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(MotifExtractionJob, job_id)
            if refreshed is not None:
                refreshed.status = JobStatus.failed.value
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = datetime.now(timezone.utc)
                await session.commit()
        except Exception:
            logger.exception("failed to mark motif_extraction job %s as failed", job_id)


async def run_one_iteration(
    session: AsyncSession,
    settings: Settings,
) -> Optional[MotifExtractionJob]:
    job = await claim_motif_extraction_job(session)
    if job is None:
        return None
    await run_motif_extraction_job(session, job, settings)
    return job
