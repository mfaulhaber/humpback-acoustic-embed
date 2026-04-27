"""Sequence Models continuous embedding worker.

Reads regions from a completed Pass-1 ``RegionDetectionJob``, produces
1-second-hop SurfPerch embeddings padded around each region, and writes
``embeddings.parquet`` + ``manifest.json`` atomically into the per-job
storage directory.

The actual SurfPerch model invocation lives behind an injected
``EmbedderProtocol`` so the worker can be exercised under tests with a
deterministic stub. The default production embedder resolves the model
via the registry — wiring is intentionally narrow so PR 1 lands the
plumbing without taking a runtime dependency on hydrophone audio
streaming code paths.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import read_regions, region_job_dir
from humpback.config import Settings
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.processing.region_windowing import (
    AudioEnvelope,
    MergedSpan,
    Region as WindowRegion,
    WindowRecord,
    iter_windows,
    merge_padded_regions,
)
from humpback.storage import (
    continuous_embedding_dir,
    continuous_embedding_manifest_path,
    continuous_embedding_parquet_path,
    ensure_dir,
)
from humpback.workers.queue import claim_continuous_embedding_job

logger = logging.getLogger(__name__)


CONTINUOUS_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("merged_span_id", pa.int32(), nullable=False),
        pa.field("window_index_in_span", pa.int32(), nullable=False),
        pa.field("audio_file_id", pa.int32(), nullable=True),
        pa.field("start_time_sec", pa.float64(), nullable=False),
        pa.field("end_time_sec", pa.float64(), nullable=False),
        pa.field("is_in_pad", pa.bool_(), nullable=False),
        pa.field("source_region_ids", pa.list_(pa.int32()), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
)


@dataclass(slots=True)
class EmbedderResult:
    """One window's embedding produced by the injected embedder."""

    window_index_in_span: int
    embedding: np.ndarray  # shape (vector_dim,), float32


class EmbedderProtocol(Protocol):
    """Pluggable SurfPerch invocation surface.

    Implementations decode audio for ``span``, run the embedding model at
    ``hop_seconds`` / ``window_size_seconds``, and return one embedding
    per window in span order. The sequence length must match the count
    yielded by ``iter_windows`` for the same span and parameters.
    """

    def __call__(
        self,
        *,
        span: MergedSpan,
        region_job: RegionDetectionJob,
        model_version: str,
        hop_seconds: float,
        window_size_seconds: float,
        target_sample_rate: int,
        settings: Settings,
    ) -> list[np.ndarray]: ...


def _default_embedder(
    *,
    span: MergedSpan,
    region_job: RegionDetectionJob,
    model_version: str,
    hop_seconds: float,
    window_size_seconds: float,
    target_sample_rate: int,
    settings: Settings,
) -> list[np.ndarray]:
    """Production embedder placeholder.

    The full SurfPerch + hydrophone-streaming integration is intentionally
    out of scope for PR 1 plumbing. This raises a clear error so the
    queue surface is testable today and the wiring can be filled in
    behind the same interface in a follow-up.
    """
    raise NotImplementedError(
        "ContinuousEmbeddingWorker requires an EmbedderProtocol implementation; "
        "the default production embedder for SurfPerch is not yet wired up. "
        "Inject an embedder when running this worker."
    )


def _regions_to_window_geometry(regions) -> list[WindowRegion]:
    return [
        WindowRegion(
            region_id=r.region_id,
            start_time_sec=float(r.padded_start_sec),
            end_time_sec=float(r.padded_end_sec),
        )
        for r in regions
    ]


def _audio_envelope(region_job: RegionDetectionJob, regions) -> AudioEnvelope:
    if region_job.start_timestamp is not None and region_job.end_timestamp is not None:
        start = float(region_job.start_timestamp)
        end = float(region_job.end_timestamp)
        return AudioEnvelope(start_time_sec=0.0, end_time_sec=max(0.0, end - start))

    # File-mode jobs: regions are in seconds-from-file-start. Bound the
    # envelope by the largest region end seen in regions.parquet — the
    # producer never extrapolates past the file end the source job saw.
    if regions:
        max_end = max(float(r.padded_end_sec) for r in regions)
    else:
        max_end = 0.0
    return AudioEnvelope(start_time_sec=0.0, end_time_sec=max_end)


def _atomic_write_parquet(table: pa.Table, dst: Path) -> None:
    """Write ``table`` to ``dst`` atomically via a same-dir temp file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _atomic_write_text(text: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    for name in ("embeddings.parquet", "manifest.json"):
        path = job_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("failed to delete %s", path, exc_info=True)
    for tmp in job_dir.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            logger.warning("failed to delete %s", tmp, exc_info=True)


def _row_dict(
    span: MergedSpan,
    record: WindowRecord,
    embedding: np.ndarray,
) -> dict:
    region_ids = []
    for rid in record.source_region_ids:
        try:
            region_ids.append(int(rid))
        except (TypeError, ValueError):
            # Region ids in this project are UUID strings; the parquet
            # schema requires int32, so non-numeric ids are skipped here
            # and surfaced via the ``source_regions`` manifest section.
            continue
    emb = np.asarray(embedding, dtype=np.float32)
    return {
        "merged_span_id": int(span.merged_span_id),
        "window_index_in_span": int(record.window_index_in_span),
        "audio_file_id": None,
        "start_time_sec": float(record.start_time_sec),
        "end_time_sec": float(record.end_time_sec),
        "is_in_pad": bool(record.is_in_pad),
        "source_region_ids": region_ids,
        "embedding": emb.tolist(),
    }


@dataclass(slots=True)
class _SpanSummary:
    merged_span_id: int
    start_time_sec: float
    end_time_sec: float
    window_count: int
    source_region_ids: list[str]


def _build_manifest_payload(
    *,
    job: ContinuousEmbeddingJob,
    vector_dim: int,
    total_regions: int,
    spans: Iterable[_SpanSummary],
) -> dict:
    spans_list = list(spans)
    return {
        "job_id": job.id,
        "model_version": job.model_version,
        "vector_dim": int(vector_dim),
        "window_size_seconds": float(job.window_size_seconds),
        "hop_seconds": float(job.hop_seconds),
        "pad_seconds": float(job.pad_seconds),
        "target_sample_rate": int(job.target_sample_rate),
        "total_regions": int(total_regions),
        "merged_spans": len(spans_list),
        "total_windows": int(sum(s.window_count for s in spans_list)),
        "spans": [
            {
                "merged_span_id": s.merged_span_id,
                "start_time_sec": s.start_time_sec,
                "end_time_sec": s.end_time_sec,
                "window_count": s.window_count,
                "source_region_ids": s.source_region_ids,
            }
            for s in spans_list
        ],
    }


async def run_continuous_embedding_job(
    session: AsyncSession,
    job: ContinuousEmbeddingJob,
    settings: Settings,
    *,
    embedder: Optional[EmbedderProtocol] = None,
) -> None:
    """Execute one continuous-embedding job end-to-end.

    The worker assumes ``job`` is the canonical row to execute — the
    idempotency check is the service layer's responsibility.
    """
    embed = embedder or _default_embedder
    job_id = job.id
    region_detection_job_id = job.region_detection_job_id

    job_dir = ensure_dir(continuous_embedding_dir(settings.storage_root, job_id))
    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job_id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job_id)

    try:
        region_job = await session.get(RegionDetectionJob, region_detection_job_id)
        if region_job is None:
            raise ValueError(
                f"region_detection_job not found: {region_detection_job_id}"
            )
        if region_job.status != JobStatus.complete.value:
            raise ValueError(
                f"region_detection_job {region_detection_job_id} not complete "
                f"(status={region_job.status!r})"
            )

        regions_path = (
            region_job_dir(settings.storage_root, region_detection_job_id)
            / "regions.parquet"
        )
        if not regions_path.exists():
            raise FileNotFoundError(
                f"regions.parquet not found for {region_detection_job_id}"
            )

        regions = read_regions(regions_path)
        regions_sorted = sorted(regions, key=lambda r: r.padded_start_sec)
        envelope = _audio_envelope(region_job, regions_sorted)
        window_regions = _regions_to_window_geometry(regions_sorted)

        spans = merge_padded_regions(
            window_regions,
            pad_seconds=float(job.pad_seconds),
            audio_envelope=envelope,
        )

        rows: list[dict] = []
        span_summaries: list[_SpanSummary] = []
        vector_dim: Optional[int] = None

        for span in spans:
            await session.refresh(job)
            if job.status == JobStatus.canceled.value:
                _cleanup_partial_artifacts(job_dir)
                logger.info(
                    "continuous_embedding | job=%s | canceled before span %d",
                    job_id,
                    span.merged_span_id,
                )
                return

            window_records = list(
                iter_windows(
                    span,
                    hop_seconds=float(job.hop_seconds),
                    window_size_seconds=float(job.window_size_seconds),
                )
            )
            if not window_records:
                span_summaries.append(
                    _SpanSummary(
                        merged_span_id=span.merged_span_id,
                        start_time_sec=span.start_time_sec,
                        end_time_sec=span.end_time_sec,
                        window_count=0,
                        source_region_ids=[r.region_id for r in span.source_regions],
                    )
                )
                continue

            embeddings = embed(
                span=span,
                region_job=region_job,
                model_version=job.model_version,
                hop_seconds=float(job.hop_seconds),
                window_size_seconds=float(job.window_size_seconds),
                target_sample_rate=int(job.target_sample_rate),
                settings=settings,
            )
            if len(embeddings) != len(window_records):
                raise ValueError(
                    f"embedder returned {len(embeddings)} vectors but expected "
                    f"{len(window_records)} for span {span.merged_span_id}"
                )

            for record, vec in zip(window_records, embeddings):
                arr = np.asarray(vec, dtype=np.float32)
                if vector_dim is None:
                    vector_dim = int(arr.shape[0])
                elif arr.shape[0] != vector_dim:
                    raise ValueError(
                        "embedding vector_dim mismatch: "
                        f"got {arr.shape[0]}, expected {vector_dim}"
                    )
                rows.append(_row_dict(span, record, arr))

            span_summaries.append(
                _SpanSummary(
                    merged_span_id=span.merged_span_id,
                    start_time_sec=span.start_time_sec,
                    end_time_sec=span.end_time_sec,
                    window_count=len(window_records),
                    source_region_ids=[r.region_id for r in span.source_regions],
                )
            )

        rows.sort(key=lambda r: (r["merged_span_id"], r["window_index_in_span"]))

        if vector_dim is None:
            vector_dim = 0
        table = pa.Table.from_pylist(rows, schema=CONTINUOUS_EMBEDDING_SCHEMA)
        _atomic_write_parquet(table, parquet_path)

        manifest = _build_manifest_payload(
            job=job,
            vector_dim=vector_dim,
            total_regions=len(regions_sorted),
            spans=span_summaries,
        )
        _atomic_write_text(
            json.dumps(manifest, sort_keys=True, indent=2), manifest_path
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(ContinuousEmbeddingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.vector_dim = vector_dim
        target.total_regions = len(regions_sorted)
        target.merged_spans = len(span_summaries)
        target.total_windows = len(rows)
        target.parquet_path = str(parquet_path)
        target.error_message = None
        target.updated_at = now
        await session.commit()

        logger.info(
            "continuous_embedding | job=%s | complete | spans=%d windows=%d dim=%d",
            job_id,
            len(span_summaries),
            len(rows),
            vector_dim,
        )

    except Exception as exc:
        logger.exception("continuous_embedding job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(ContinuousEmbeddingJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = JobStatus.failed.value
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "failed to mark continuous_embedding job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession,
    settings: Settings,
    *,
    embedder: Optional[EmbedderProtocol] = None,
) -> Optional[ContinuousEmbeddingJob]:
    """Claim and process at most one continuous-embedding job."""
    job = await claim_continuous_embedding_job(session)
    if job is None:
        return None
    await run_continuous_embedding_job(session, job, settings, embedder=embedder)
    return job
