"""Sequence Models continuous embedding worker.

Dispatches on ``model_version`` family (per ADR-057) to one of two
producer paths:

- ``surfperch-tensorflow2``: events from a completed Pass-2
  ``EventSegmentationJob`` are padded into independent spans and embedded
  at 1-second hops.
- ``crnn-call-parsing-pytorch``: regions from a completed Pass-1
  ``RegionDetectionJob`` are embedded as 250 ms chunks via the Pass 2
  segmentation CRNN's BiGRU activations, with per-chunk metadata joined
  against the matching Pass 2 ``EventSegmentationJob`` events.

Both paths write ``embeddings.parquet`` + ``manifest.json`` atomically.
The SurfPerch invocation lives behind an injected ``EmbedderProtocol``
so the worker can be exercised under tests with a deterministic stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_region_audio_loader
from humpback.call_parsing.regions_overlay import load_corrected_regions
from humpback.call_parsing.storage import (
    read_events,
    region_job_dir,
    segmentation_job_dir,
)
from humpback.call_parsing.types import Event, Region
from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers import build_archive_detection_provider
from humpback.classifier.s3_stream import resolve_audio_slice
from humpback.config import Settings
from humpback.ml.device import select_and_validate_device
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.region_windowing import (
    AudioEnvelope,
    MergedSpan,
    WindowRecord,
    iter_windows,
)
from humpback.processing.windowing import window_sample_count
from humpback.sequence_models.chunk_projection import (
    ChunkProjection,
    IdentityProjection,
    PCAProjection,
    RandomProjection,
)
from humpback.sequence_models.crnn_features import (
    EXPECTED_BIGRU_WIDTH,
    extract_chunk_embeddings,
    load_crnn_for_extraction,
)
from humpback.sequence_models.event_overlap_join import (
    ChunkBounds,
    EventBounds,
    compute_chunk_event_metadata,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.storage import (
    continuous_embedding_dir,
    continuous_embedding_manifest_path,
    continuous_embedding_parquet_path,
    ensure_dir,
)
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import claim_continuous_embedding_job
from sqlalchemy import select

logger = logging.getLogger(__name__)


CONTINUOUS_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("merged_span_id", pa.int32(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("window_index_in_span", pa.int32(), nullable=False),
        pa.field("audio_file_id", pa.int32(), nullable=True),
        pa.field("start_timestamp", pa.float64(), nullable=False),
        pa.field("end_timestamp", pa.float64(), nullable=False),
        pa.field("is_in_pad", pa.bool_(), nullable=False),
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
    raise RuntimeError(
        "_default_embedder should be replaced by a prepared production embedder"
    )


def _event_to_span(
    event: Event,
    span_id: int,
    pad_seconds: float,
    envelope: AudioEnvelope,
) -> MergedSpan:
    raw_start = float(event.start_sec) - pad_seconds
    raw_end = float(event.end_sec) + pad_seconds
    clamped_start = max(envelope.start_offset_sec, raw_start)
    clamped_end = min(envelope.end_offset_sec, raw_end)
    from humpback.processing.region_windowing import Region as WindowRegion

    region = WindowRegion(
        region_id=event.event_id,
        start_offset_sec=float(event.start_sec),
        end_offset_sec=float(event.end_sec),
    )
    return MergedSpan(
        merged_span_id=span_id,
        start_offset_sec=clamped_start,
        end_offset_sec=clamped_end,
        source_regions=[region],
    )


def _audio_envelope_from_region_job(region_job: RegionDetectionJob) -> AudioEnvelope:
    start = float(region_job.start_timestamp or 0.0)
    end = float(region_job.end_timestamp or 0.0)
    return AudioEnvelope(start_offset_sec=0.0, end_offset_sec=max(0.0, end - start))


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
    event_id: str,
    record: WindowRecord,
    embedding: np.ndarray,
    timestamp_offset: float,
) -> dict:
    emb = np.asarray(embedding, dtype=np.float32)
    return {
        "merged_span_id": int(span.merged_span_id),
        "event_id": event_id,
        "window_index_in_span": int(record.window_index_in_span),
        "audio_file_id": None,
        "start_timestamp": float(record.start_offset_sec + timestamp_offset),
        "end_timestamp": float(record.end_offset_sec + timestamp_offset),
        "is_in_pad": bool(record.is_in_pad),
        "embedding": emb.tolist(),
    }


@dataclass(slots=True)
class _SpanSummary:
    merged_span_id: int
    event_id: str
    region_id: str
    start_timestamp: float
    end_timestamp: float
    window_count: int


@dataclass(slots=True)
class _ProductionEmbedder:
    model: EmbeddingModel
    input_format: str
    normalization: str
    provider: ArchiveProvider
    timeline: list[StreamSegment]
    stream_start_ts: float
    stream_end_ts: float
    target_sample_rate: int

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
    ) -> list[np.ndarray]:
        del region_job, model_version, settings
        if target_sample_rate != self.target_sample_rate:
            raise ValueError(
                "continuous embedding target_sample_rate mismatch between job "
                f"({target_sample_rate}) and prepared embedder ({self.target_sample_rate})"
            )

        window_records = list(
            iter_windows(
                span,
                hop_seconds=hop_seconds,
                window_size_seconds=window_size_seconds,
            )
        )
        if not window_records:
            return []

        span_start_utc = self.stream_start_ts + span.start_offset_sec
        span_duration = span.end_offset_sec - span.start_offset_sec
        span_audio = resolve_audio_slice(
            self.provider,
            self.stream_start_ts,
            self.stream_end_ts,
            span_start_utc,
            span_duration,
            target_sr=self.target_sample_rate,
            timeline=self.timeline,
        )
        window_batch = _build_window_batch(
            span_audio=span_audio,
            span=span,
            window_records=window_records,
            window_size_seconds=window_size_seconds,
            target_sample_rate=self.target_sample_rate,
        )
        return _embed_window_batch(
            model=self.model,
            input_format=self.input_format,
            normalization=self.normalization,
            target_sample_rate=self.target_sample_rate,
            window_batch=window_batch,
        )


def _build_manifest_payload(
    *,
    job: ContinuousEmbeddingJob,
    vector_dim: int,
    total_events: int,
    spans: Iterable[_SpanSummary],
    window_size_seconds: float,
    hop_seconds: float,
    pad_seconds: float,
) -> dict:
    spans_list = list(spans)
    return {
        "job_id": job.id,
        "model_version": job.model_version,
        "source_kind": "surfperch",
        "vector_dim": int(vector_dim),
        "window_size_seconds": window_size_seconds,
        "hop_seconds": hop_seconds,
        "pad_seconds": pad_seconds,
        "target_sample_rate": int(job.target_sample_rate),
        "total_events": int(total_events),
        "merged_spans": len(spans_list),
        "total_windows": int(sum(s.window_count for s in spans_list)),
        "spans": [
            {
                "merged_span_id": s.merged_span_id,
                "event_id": s.event_id,
                "region_id": s.region_id,
                "start_timestamp": s.start_timestamp,
                "end_timestamp": s.end_timestamp,
                "window_count": s.window_count,
            }
            for s in spans_list
        ],
    }


def _build_window_batch(
    *,
    span_audio: np.ndarray,
    span: MergedSpan,
    window_records: list[WindowRecord],
    window_size_seconds: float,
    target_sample_rate: int,
) -> np.ndarray:
    window_samples = window_sample_count(target_sample_rate, window_size_seconds)
    batch = np.zeros((len(window_records), window_samples), dtype=np.float32)

    for idx, record in enumerate(window_records):
        rel_start_sec = record.start_offset_sec - span.start_offset_sec
        start_sample = int(round(rel_start_sec * target_sample_rate))
        end_sample = start_sample + window_samples
        window_audio = span_audio[start_sample:end_sample]
        if len(window_audio) > 0:
            batch[idx, : min(len(window_audio), window_samples)] = window_audio[
                :window_samples
            ]

    return batch


def _embed_window_batch(
    *,
    model: EmbeddingModel,
    input_format: str,
    normalization: str,
    target_sample_rate: int,
    window_batch: np.ndarray,
) -> list[np.ndarray]:
    if len(window_batch) == 0:
        return []

    if input_format == "waveform":
        batch = window_batch
    else:
        spectrograms = extract_logmel_batch(
            list(window_batch),
            target_sample_rate,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
            normalization=normalization,
        )
        batch = np.stack(spectrograms)

    embeddings = model.embed(batch)
    return [np.asarray(row, dtype=np.float32) for row in embeddings]


async def _prepare_production_embedder(
    session: AsyncSession,
    job: ContinuousEmbeddingJob,
    region_job: RegionDetectionJob,
    settings: Settings,
) -> EmbedderProtocol:
    if (
        not region_job.hydrophone_id
        or region_job.start_timestamp is None
        or region_job.end_timestamp is None
    ):
        raise ValueError(
            "continuous embedding currently supports hydrophone-backed "
            "region_detection_job rows only"
        )

    model, input_format = await get_model_by_version(
        session, job.model_version, settings
    )
    feature_config = (
        json.loads(job.feature_config_json) if job.feature_config_json else {}
    )
    normalization = feature_config.get("normalization", "per_window_max")
    provider = build_archive_detection_provider(
        region_job.hydrophone_id,
        local_cache_path=None,
        s3_cache_path=settings.s3_cache_path,
        noaa_cache_path=settings.noaa_cache_path,
        force_refresh=False,
    )
    timeline = await asyncio.to_thread(
        provider.build_timeline,
        float(region_job.start_timestamp),
        float(region_job.end_timestamp),
    )
    return _ProductionEmbedder(
        model=model,
        input_format=input_format,
        normalization=normalization,
        provider=provider,
        timeline=timeline,
        stream_start_ts=float(region_job.start_timestamp),
        stream_end_ts=float(region_job.end_timestamp),
        target_sample_rate=int(job.target_sample_rate),
    )


async def run_continuous_embedding_job(
    session: AsyncSession,
    job: ContinuousEmbeddingJob,
    settings: Settings,
    *,
    embedder: Optional[EmbedderProtocol] = None,
) -> None:
    """Execute one continuous-embedding job end-to-end.

    Dispatches on ``model_version`` family (per ADR-057). The worker
    assumes ``job`` is the canonical row to execute — the idempotency
    check is the service layer's responsibility.
    """
    if source_kind_for(job.model_version) == SOURCE_KIND_REGION_CRNN:
        await _run_region_crnn(session, job, settings)
        return
    await _run_event_padded_surfperch(session, job, settings, embedder=embedder)


async def _run_event_padded_surfperch(
    session: AsyncSession,
    job: ContinuousEmbeddingJob,
    settings: Settings,
    *,
    embedder: Optional[EmbedderProtocol] = None,
) -> None:
    """SurfPerch event-padded source — original behavior, byte-identical."""
    job_id = job.id
    event_segmentation_job_id = job.event_segmentation_job_id

    job_dir = ensure_dir(continuous_embedding_dir(settings.storage_root, job_id))
    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job_id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job_id)

    # Migration 061 made the SurfPerch-only configuration columns
    # nullable so future CRNN-source rows can leave them unset. The
    # SurfPerch path narrows them up front and fails fast if a row is
    # missing them.
    if event_segmentation_job_id is None:
        raise ValueError(
            f"continuous_embedding_job {job_id} missing event_segmentation_job_id"
        )
    if (
        job.window_size_seconds is None
        or job.hop_seconds is None
        or job.pad_seconds is None
    ):
        raise ValueError(
            f"continuous_embedding_job {job_id} missing required "
            "window/hop/pad configuration"
        )
    window_size_seconds = float(job.window_size_seconds)
    hop_seconds = float(job.hop_seconds)
    pad_seconds = float(job.pad_seconds)

    try:
        job = await session.merge(job)
        seg_job = await session.get(EventSegmentationJob, event_segmentation_job_id)
        if seg_job is None:
            raise ValueError(
                f"event_segmentation_job not found: {event_segmentation_job_id}"
            )
        if seg_job.status != JobStatus.complete.value:
            raise ValueError(
                f"event_segmentation_job {event_segmentation_job_id} not complete "
                f"(status={seg_job.status!r})"
            )
        region_job = await session.get(
            RegionDetectionJob, seg_job.region_detection_job_id
        )
        if region_job is None:
            raise ValueError(
                f"region_detection_job not found via segmentation FK: "
                f"{seg_job.region_detection_job_id}"
            )

        embed = embedder or await _prepare_production_embedder(
            session, job, region_job, settings
        )

        events_path = (
            segmentation_job_dir(settings.storage_root, event_segmentation_job_id)
            / "events.parquet"
        )
        if not events_path.exists():
            raise FileNotFoundError(
                f"events.parquet not found for {event_segmentation_job_id}"
            )

        events = read_events(events_path)
        events_sorted = sorted(events, key=lambda e: e.start_sec)
        envelope = _audio_envelope_from_region_job(region_job)
        timestamp_offset = float(region_job.start_timestamp or 0.0)

        spans_and_events: list[tuple[MergedSpan, Event]] = [
            (
                _event_to_span(
                    event,
                    span_id=idx,
                    pad_seconds=pad_seconds,
                    envelope=envelope,
                ),
                event,
            )
            for idx, event in enumerate(events_sorted)
        ]

        rows: list[dict] = []
        span_summaries: list[_SpanSummary] = []
        vector_dim: Optional[int] = None

        for span, event in spans_and_events:
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
                    hop_seconds=hop_seconds,
                    window_size_seconds=window_size_seconds,
                )
            )
            if not window_records:
                span_summaries.append(
                    _SpanSummary(
                        merged_span_id=span.merged_span_id,
                        event_id=event.event_id,
                        region_id=event.region_id,
                        start_timestamp=span.start_offset_sec + timestamp_offset,
                        end_timestamp=span.end_offset_sec + timestamp_offset,
                        window_count=0,
                    )
                )
                continue

            embeddings = await asyncio.to_thread(
                embed,
                span=span,
                region_job=region_job,
                model_version=job.model_version,
                hop_seconds=hop_seconds,
                window_size_seconds=window_size_seconds,
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
                rows.append(
                    _row_dict(span, event.event_id, record, arr, timestamp_offset)
                )

            span_summaries.append(
                _SpanSummary(
                    merged_span_id=span.merged_span_id,
                    event_id=event.event_id,
                    region_id=event.region_id,
                    start_timestamp=span.start_offset_sec + timestamp_offset,
                    end_timestamp=span.end_offset_sec + timestamp_offset,
                    window_count=len(window_records),
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
            total_events=len(events_sorted),
            spans=span_summaries,
            window_size_seconds=window_size_seconds,
            hop_seconds=hop_seconds,
            pad_seconds=pad_seconds,
        )
        _atomic_write_text(
            json.dumps(manifest, sort_keys=True, indent=2), manifest_path
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(ContinuousEmbeddingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.vector_dim = vector_dim
        target.total_events = len(events_sorted)
        target.merged_spans = len(span_summaries)
        target.total_windows = len(rows)
        target.parquet_path = str(parquet_path)
        target.error_message = None
        target.updated_at = now
        await session.commit()

        logger.info(
            "continuous_embedding | job=%s | complete | events=%d spans=%d windows=%d dim=%d",
            job_id,
            len(events_sorted),
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


# ---------------------------------------------------------------------------
# CRNN region-based source (ADR-057)
# ---------------------------------------------------------------------------


REGION_CRNN_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("region_id", pa.string(), nullable=False),
        pa.field("audio_file_id", pa.int32(), nullable=True),
        pa.field("hydrophone_id", pa.string(), nullable=True),
        pa.field("chunk_index_in_region", pa.int32(), nullable=False),
        pa.field("start_timestamp", pa.float64(), nullable=False),
        pa.field("end_timestamp", pa.float64(), nullable=False),
        pa.field("is_in_pad", pa.bool_(), nullable=False),
        pa.field("call_probability", pa.float32(), nullable=False),
        pa.field("event_overlap_fraction", pa.float32(), nullable=False),
        pa.field("nearest_event_id", pa.string(), nullable=True),
        pa.field("distance_to_nearest_event_seconds", pa.float32(), nullable=True),
        pa.field("tier", pa.string(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
)


def _build_chunk_projection(kind: str, dim: int) -> ChunkProjection:
    """Construct a ``ChunkProjection`` from persisted job-row config."""
    if kind == "identity":
        return IdentityProjection(input_dim=dim)
    if kind == "random":
        return RandomProjection(output_dim=dim, seed=0)
    if kind == "pca":
        return PCAProjection(output_dim=dim, whiten=False)
    raise ValueError(f"unsupported projection_kind {kind!r}")


def _events_for_region(
    events: list[Event], region_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ids, starts, ends)`` arrays for events in a region.

    The events parquet is in absolute audio-source seconds; chunk
    timestamps the producer emits are also absolute. The arrays returned
    here align with that timeline.
    """
    ids: list[str] = []
    starts: list[float] = []
    ends: list[float] = []
    for ev in events:
        if ev.region_id != region_id:
            continue
        ids.append(ev.event_id)
        starts.append(float(ev.start_sec))
        ends.append(float(ev.end_sec))
    return (
        np.asarray(ids, dtype=object),
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
    )


def _select_device() -> torch.device:
    """Pick a PyTorch device; CPU fallback on MPS/CUDA validation failure."""
    # Validation on a tiny dummy module so we don't pay the CRNN load
    # cost twice. ``select_and_validate_device`` exercises a forward pass
    # against a Module, which is enough for device selection.
    probe = torch.nn.Linear(8, 8)
    sample = torch.zeros(1, 8)
    device, _ = select_and_validate_device(probe, sample)
    return device


def _run_region_inference(
    *,
    region: Region,
    audio: np.ndarray,
    feature_config: Any,
    model: Any,
    chunk_size_seconds: float,
    chunk_hop_seconds: float,
    projection: ChunkProjection,
    device: torch.device,
) -> Any:
    """Blocking helper: extract per-chunk embeddings for one region."""
    return extract_chunk_embeddings(
        model=model,
        audio=audio,
        feature_config=feature_config,
        chunk_size_seconds=chunk_size_seconds,
        chunk_hop_seconds=chunk_hop_seconds,
        projection=projection,
        device=device,
    )


def _build_region_crnn_manifest(
    *,
    job: ContinuousEmbeddingJob,
    vector_dim: int,
    crnn_checkpoint_sha256: str,
    chunk_size_seconds: float,
    chunk_hop_seconds: float,
    projection_kind: str,
    projection_dim: int,
    region_summaries: list[dict],
) -> dict:
    return {
        "job_id": job.id,
        "model_version": job.model_version,
        "source_kind": "region_crnn",
        "vector_dim": int(vector_dim),
        "target_sample_rate": int(job.target_sample_rate),
        "region_detection_job_id": job.region_detection_job_id,
        "event_segmentation_job_id": job.event_segmentation_job_id,
        "crnn_checkpoint_sha256": crnn_checkpoint_sha256,
        "chunk_size_seconds": chunk_size_seconds,
        "chunk_hop_seconds": chunk_hop_seconds,
        "projection_kind": projection_kind,
        "projection_dim": int(projection_dim),
        "total_regions": len(region_summaries),
        "total_chunks": int(sum(r["chunk_count"] for r in region_summaries)),
        "regions": region_summaries,
    }


async def _run_region_crnn(
    session: AsyncSession,
    job: ContinuousEmbeddingJob,
    settings: Settings,
) -> None:
    """CRNN region-based source — ADR-057 producer path."""
    job_id = job.id
    job_dir = ensure_dir(continuous_embedding_dir(settings.storage_root, job_id))
    parquet_path = continuous_embedding_parquet_path(settings.storage_root, job_id)
    manifest_path = continuous_embedding_manifest_path(settings.storage_root, job_id)

    if (
        job.region_detection_job_id is None
        or job.event_segmentation_job_id is None
        or job.crnn_segmentation_model_id is None
        or job.chunk_size_seconds is None
        or job.chunk_hop_seconds is None
        or job.projection_kind is None
        or job.projection_dim is None
    ):
        raise ValueError(
            f"continuous_embedding_job {job_id} missing required CRNN "
            "configuration columns"
        )
    chunk_size_seconds = float(job.chunk_size_seconds)
    chunk_hop_seconds = float(job.chunk_hop_seconds)
    projection_kind = str(job.projection_kind)
    projection_dim = int(job.projection_dim)

    try:
        job = await session.merge(job)

        seg_job = await session.get(EventSegmentationJob, job.event_segmentation_job_id)
        if seg_job is None:
            raise ValueError(
                f"event_segmentation_job not found: {job.event_segmentation_job_id}"
            )
        if seg_job.status != JobStatus.complete.value:
            raise ValueError(
                f"event_segmentation_job is not complete (status={seg_job.status!r})"
            )

        region_job = await session.get(RegionDetectionJob, job.region_detection_job_id)
        if region_job is None:
            raise ValueError(
                f"region_detection_job not found: {job.region_detection_job_id}"
            )
        if region_job.status != JobStatus.complete.value:
            raise ValueError(
                f"region_detection_job not complete (status={region_job.status!r})"
            )

        seg_model = await session.get(SegmentationModel, job.crnn_segmentation_model_id)
        if seg_model is None:
            raise ValueError(
                f"segmentation_model not found: {job.crnn_segmentation_model_id}"
            )
        checkpoint_path = Path(seg_model.model_path)
        if not checkpoint_path.exists():
            raise ValueError(
                f"segmentation_model checkpoint missing at {checkpoint_path}"
            )

        # Load CRNN + feature config; recompute sha256 for sanity even
        # though the service already stamped one onto the row.
        device = await asyncio.to_thread(_select_device)
        loaded = await asyncio.to_thread(
            load_crnn_for_extraction, checkpoint_path, device
        )
        if (
            job.crnn_checkpoint_sha256
            and job.crnn_checkpoint_sha256 != loaded.checkpoint_sha256
        ):
            raise ValueError(
                "stored crnn_checkpoint_sha256 does not match the on-disk "
                f"checkpoint at {checkpoint_path}"
            )

        # Read regions and events from upstream artifacts.
        regions_path = (
            region_job_dir(settings.storage_root, region_job.id) / "regions.parquet"
        )
        if not regions_path.exists():
            raise FileNotFoundError(
                f"regions.parquet not found for region job {region_job.id}"
            )
        regions = await load_corrected_regions(session, region_job.id, regions_path)

        events_path = (
            segmentation_job_dir(settings.storage_root, seg_job.id) / "events.parquet"
        )
        if not events_path.exists():
            raise FileNotFoundError(
                f"events.parquet not found for segmentation job {seg_job.id}"
            )
        events = read_events(events_path)

        # Build the audio loader. The CRNN feature config dictates the
        # sample rate; we always run via hydrophone or audio-file (same
        # contract as Pass 2).
        sample_rate = int(loaded.feature_config.sample_rate)
        if region_job.audio_file_id:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id == region_job.audio_file_id)
            )
            audio_file = af_result.scalar_one_or_none()
            if audio_file is None:
                raise ValueError(f"AudioFile {region_job.audio_file_id} not found")
            audio_loader = build_region_audio_loader(
                target_sr=sample_rate,
                settings=settings,
                audio_file=audio_file,
                storage_root=settings.storage_root,
            )
        elif region_job.hydrophone_id:
            audio_loader = build_region_audio_loader(
                target_sr=sample_rate,
                settings=settings,
                hydrophone_id=region_job.hydrophone_id,
                job_start_ts=region_job.start_timestamp or 0.0,
                job_end_ts=region_job.end_timestamp or 0.0,
            )
        else:
            raise ValueError(
                f"region_detection_job {region_job.id} has no audio source"
            )

        projection = _build_chunk_projection(projection_kind, projection_dim)

        min_region_seconds = 2.0  # spec §8 default — producer-side filter.
        rows: list[dict] = []
        region_summaries: list[dict] = []
        vector_dim: Optional[int] = None
        timestamp_offset = float(region_job.start_timestamp or 0.0)
        audio_file_id_int: Optional[int] = None

        for region in regions:
            await session.refresh(job)
            if job.status == JobStatus.canceled.value:
                _cleanup_partial_artifacts(job_dir)
                logger.info(
                    "continuous_embedding | job=%s | canceled before region %s",
                    job_id,
                    region.region_id,
                )
                return

            region_duration = float(region.padded_end_sec - region.padded_start_sec)
            if region_duration < min_region_seconds:
                logger.info(
                    "continuous_embedding | job=%s | skipping short region %s "
                    "(%.2fs < %.2fs)",
                    job_id,
                    region.region_id,
                    region_duration,
                    min_region_seconds,
                )
                continue

            audio = await asyncio.to_thread(audio_loader, region)
            result = await asyncio.to_thread(
                _run_region_inference,
                region=region,
                audio=audio,
                feature_config=loaded.feature_config,
                model=loaded.model,
                chunk_size_seconds=chunk_size_seconds,
                chunk_hop_seconds=chunk_hop_seconds,
                projection=projection,
                device=device,
            )

            n_chunks = int(result.embeddings.shape[0])
            if n_chunks == 0:
                continue

            if vector_dim is None:
                vector_dim = int(result.embeddings.shape[1])
            elif int(result.embeddings.shape[1]) != vector_dim:
                raise ValueError(
                    "embedding vector_dim mismatch within job: "
                    f"{result.embeddings.shape[1]} vs {vector_dim}"
                )

            # Convert chunk timestamps from region-local seconds to the
            # absolute audio-source timeline and join against events.
            absolute_starts = (
                result.chunk_starts.astype(np.float64)
                + float(region.padded_start_sec)
                + timestamp_offset
            )
            absolute_ends = (
                result.chunk_ends.astype(np.float64)
                + float(region.padded_start_sec)
                + timestamp_offset
            )

            event_ids, event_starts, event_ends = _events_for_region(
                events, region.region_id
            )
            event_starts_abs = event_starts + timestamp_offset
            event_ends_abs = event_ends + timestamp_offset
            metadata = compute_chunk_event_metadata(
                ChunkBounds(starts=absolute_starts, ends=absolute_ends),
                EventBounds(
                    ids=event_ids, starts=event_starts_abs, ends=event_ends_abs
                ),
            )

            region_start_abs = float(region.start_sec) + timestamp_offset
            region_end_abs = float(region.end_sec) + timestamp_offset
            chunk_centers = (absolute_starts + absolute_ends) / 2.0
            is_in_pad = (chunk_centers < region_start_abs) | (
                chunk_centers > region_end_abs
            )

            for i in range(n_chunks):
                rows.append(
                    {
                        "region_id": region.region_id,
                        "audio_file_id": audio_file_id_int,
                        "hydrophone_id": region_job.hydrophone_id,
                        "chunk_index_in_region": i,
                        "start_timestamp": float(absolute_starts[i]),
                        "end_timestamp": float(absolute_ends[i]),
                        "is_in_pad": bool(is_in_pad[i]),
                        "call_probability": float(result.call_probabilities[i]),
                        "event_overlap_fraction": float(
                            metadata.event_overlap_fraction[i]
                        ),
                        "nearest_event_id": metadata.nearest_event_id[i],
                        "distance_to_nearest_event_seconds": (
                            None
                            if metadata.distance_to_nearest_event_seconds[i] is None
                            else float(metadata.distance_to_nearest_event_seconds[i])
                        ),
                        "tier": str(metadata.tier[i]),
                        "embedding": result.embeddings[i].tolist(),
                    }
                )
            region_summaries.append(
                {
                    "region_id": region.region_id,
                    "start_timestamp": float(region.padded_start_sec)
                    + timestamp_offset,
                    "end_timestamp": float(region.padded_end_sec) + timestamp_offset,
                    "chunk_count": n_chunks,
                }
            )

        if vector_dim is None:
            vector_dim = projection_dim

        rows.sort(key=lambda r: (r["region_id"], r["chunk_index_in_region"]))
        table = pa.Table.from_pylist(rows, schema=REGION_CRNN_EMBEDDING_SCHEMA)
        _atomic_write_parquet(table, parquet_path)

        manifest = _build_region_crnn_manifest(
            job=job,
            vector_dim=vector_dim,
            crnn_checkpoint_sha256=loaded.checkpoint_sha256,
            chunk_size_seconds=chunk_size_seconds,
            chunk_hop_seconds=chunk_hop_seconds,
            projection_kind=projection_kind,
            projection_dim=projection_dim,
            region_summaries=region_summaries,
        )
        _atomic_write_text(
            json.dumps(manifest, sort_keys=True, indent=2), manifest_path
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(ContinuousEmbeddingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.vector_dim = vector_dim
        target.total_regions = len(region_summaries)
        target.total_chunks = len(rows)
        target.parquet_path = str(parquet_path)
        target.crnn_checkpoint_sha256 = loaded.checkpoint_sha256
        target.error_message = None
        target.updated_at = now
        await session.commit()

        # Silence unused-import lint when the module ships without using
        # the BiGRU width constant directly — we keep it to support
        # future load-time guards close to the worker.
        _ = EXPECTED_BIGRU_WIDTH

        logger.info(
            "continuous_embedding | job=%s | complete (CRNN) | regions=%d chunks=%d dim=%d",
            job_id,
            len(region_summaries),
            len(rows),
            vector_dim,
        )

    except Exception as exc:
        logger.exception("continuous_embedding job %s failed (CRNN)", job_id)
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
