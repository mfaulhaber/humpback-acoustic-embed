"""Worker for Sequence Models Event Encoder jobs."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_event_audio_loader
from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.call_parsing.storage import read_events, segmentation_job_dir
from humpback.call_parsing.types import Event
from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, EventEncoderJob
from humpback.sequence_models.event_encoder import (
    ChunkEmbedding,
    DESCRIPTOR_ORDER,
    EventInterval,
    build_event_embedding,
    compute_acoustic_descriptors,
    compute_gap_to_previous,
    descriptor_vector,
)
from humpback.sequence_models.event_tokenization import (
    fit_kmeans_tokenizers,
    preprocess_event_features,
    tokenization_summary,
)
from humpback.storage import (
    ensure_dir,
    event_encoder_dir,
    event_encoder_kmeans_path,
    event_encoder_manifest_path,
    event_encoder_preprocess_path,
    event_encoder_report_path,
    event_encoder_sequences_path,
    event_encoder_tokens_path,
    event_encoder_vectors_path,
)

logger = logging.getLogger(__name__)


def _event_vector_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("region_id", pa.string(), nullable=False),
            pa.field("source_sequence_key", pa.string(), nullable=False),
            pa.field("sequence_index", pa.int32(), nullable=False),
            pa.field("start_timestamp", pa.float64(), nullable=False),
            pa.field("end_timestamp", pa.float64(), nullable=False),
            pa.field("segmentation_confidence", pa.float64(), nullable=False),
            *[
                pa.field(name, pa.float32(), nullable=False)
                for name in DESCRIPTOR_ORDER
            ],
            pa.field("chunk_count", pa.int32(), nullable=False),
            pa.field("coverage_fraction", pa.float32(), nullable=False),
            pa.field("embedding_vector", pa.list_(pa.float32()), nullable=False),
            pa.field("descriptor_vector", pa.list_(pa.float32()), nullable=False),
            pa.field("event_vector", pa.list_(pa.float32()), nullable=False),
        ]
    )


def _event_token_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("k", pa.int32(), nullable=False),
            pa.field("event_id", pa.string(), nullable=False),
            pa.field("region_id", pa.string(), nullable=False),
            pa.field("source_sequence_key", pa.string(), nullable=False),
            pa.field("sequence_index", pa.int32(), nullable=False),
            pa.field("start_timestamp", pa.float64(), nullable=False),
            pa.field("end_timestamp", pa.float64(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("token_label", pa.string(), nullable=False),
            pa.field("distance_to_centroid", pa.float32(), nullable=False),
            pa.field("second_centroid_distance", pa.float32(), nullable=True),
            pa.field("token_confidence", pa.float32(), nullable=False),
            *[
                pa.field(name, pa.float32(), nullable=False)
                for name in DESCRIPTOR_ORDER
            ],
        ]
    )


TOKEN_SEQUENCE_SCHEMA = pa.schema(
    [
        pa.field("k", pa.int32(), nullable=False),
        pa.field("source_sequence_key", pa.string(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("token_id", pa.int32(), nullable=False),
        pa.field("token_label", pa.string(), nullable=False),
        pa.field("start_timestamp", pa.float64(), nullable=False),
        pa.field("end_timestamp", pa.float64(), nullable=False),
        pa.field("gap_to_previous", pa.float32(), nullable=False),
    ]
)


EventAudioProvider = Callable[[Event, int], np.ndarray]


@dataclass(slots=True)
class _EncodedEvent:
    event: EventInterval
    sequence_index: int
    chunk_count: int
    coverage_fraction: float
    pool_vector: np.ndarray
    descriptors: dict[str, float]


def _atomic_write_parquet(table: pa.Table, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
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
            tmp.unlink()
        raise


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    for path in job_dir.glob("*.tmp"):
        try:
            path.unlink()
        except OSError:
            logger.warning("failed to delete %s", path, exc_info=True)
    for name in (
        "manifest.json",
        "report.json",
        "event_vectors.parquet",
        "event_tokens.parquet",
        "token_sequences.parquet",
        "preprocess.joblib",
    ):
        path = job_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("failed to delete %s", path, exc_info=True)
    for path in job_dir.glob("kmeans_k*.joblib"):
        try:
            path.unlink()
        except OSError:
            logger.warning("failed to delete %s", path, exc_info=True)


async def run_event_encoder_job(
    session: AsyncSession,
    job: EventEncoderJob,
    settings: Settings,
    *,
    audio_provider: Optional[EventAudioProvider] = None,
) -> None:
    """Execute one Event Encoder job end-to-end."""
    job_id = job.id
    job_dir = ensure_dir(event_encoder_dir(settings.storage_root, job_id))

    try:
        job = await session.merge(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        seg_job = await session.get(EventSegmentationJob, job.event_segmentation_job_id)
        if seg_job is None:
            raise ValueError(
                f"event_segmentation_job not found: {job.event_segmentation_job_id}"
            )
        if seg_job.status != JobStatus.complete.value:
            raise ValueError(
                f"event_segmentation_job is not complete (status={seg_job.status!r})"
            )

        continuous = await session.get(
            ContinuousEmbeddingJob, job.continuous_embedding_job_id
        )
        if continuous is None:
            raise ValueError(
                f"continuous_embedding_job not found: {job.continuous_embedding_job_id}"
            )
        if continuous.status != JobStatus.complete.value:
            raise ValueError(
                "continuous_embedding_job is not complete "
                f"(status={continuous.status!r})"
            )
        if not continuous.parquet_path:
            raise ValueError("continuous_embedding_job is missing parquet_path")

        region_job = await session.get(
            RegionDetectionJob, seg_job.region_detection_job_id
        )
        if region_job is None:
            raise ValueError(
                f"region_detection_job not found: {seg_job.region_detection_job_id}"
            )

        events = await _load_events(session, job, settings)
        chunks = _load_chunks(Path(continuous.parquet_path))
        if not events:
            raise ValueError("event encoder found no source events")
        if not chunks:
            raise ValueError("event encoder found no CRNN chunks")

        descriptor_config = json.loads(job.descriptor_config_json)
        target_sr = int(descriptor_config.get("target_sample_rate", 16000))
        load_audio = audio_provider or await _build_audio_provider(
            session,
            settings,
            region_job,
            events,
            target_sr,
        )

        encoded_events, skip_reasons = await _build_encoded_events(
            session,
            job,
            events,
            chunks,
            region_job,
            load_audio,
            descriptor_config,
        )
        if not encoded_events:
            raise ValueError(
                "event encoder could not encode any events "
                f"(skip_reasons={skip_reasons})"
            )

        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        preprocessing_config = json.loads(job.preprocessing_config_json)
        pooling_config = json.loads(job.pooling_config_json)
        pool_count = len(pooling_config["enabled_pools"])
        pool_dim = int(encoded_events[0].pool_vector.shape[0] / pool_count)
        pool_matrix = np.stack([row.pool_vector for row in encoded_events]).astype(
            np.float32
        )
        descriptor_matrix = np.stack(
            [descriptor_vector(row.descriptors) for row in encoded_events]
        ).astype(np.float32)
        preprocess = preprocess_event_features(
            pool_matrix,
            descriptor_matrix,
            pool_dim=pool_dim,
            pool_count=pool_count,
            l2_normalize_pools=bool(
                preprocessing_config.get("l2_normalize_pools", True)
            ),
            pca_dim=int(preprocessing_config.get("pca_dim", 128)),
            embedding_weight=float(preprocessing_config.get("embedding_weight", 1.0)),
            descriptor_weight=float(
                preprocessing_config.get("descriptor_weight", 0.571)
            ),
            descriptor_clip_value=preprocessing_config.get(
                "descriptor_clip_value", 3.0
            ),
            random_seed=int(job.random_seed),
        )
        k_values = list(json.loads(job.k_values_json))
        tokenization = fit_kmeans_tokenizers(
            preprocess.event_vectors,
            k_values,
            random_seed=int(job.random_seed),
        )
        if not tokenization.tokenizations:
            raise ValueError(
                "event encoder had no valid k_values for "
                f"{len(encoded_events)} encoded event(s): {k_values}"
            )

        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            _cleanup_partial_artifacts(job_dir)
            return

        vector_rows = _build_event_vector_rows(encoded_events, preprocess)
        token_rows, sequence_rows = _build_token_rows(encoded_events, tokenization)
        vector_path = event_encoder_vectors_path(settings.storage_root, job_id)
        token_path = event_encoder_tokens_path(settings.storage_root, job_id)
        sequence_path = event_encoder_sequences_path(settings.storage_root, job_id)
        manifest_path = event_encoder_manifest_path(settings.storage_root, job_id)
        report_path = event_encoder_report_path(settings.storage_root, job_id)
        preprocess_path = event_encoder_preprocess_path(settings.storage_root, job_id)

        _atomic_write_parquet(
            pa.Table.from_pylist(
                vector_rows,
                schema=_event_vector_schema(),
            ),
            vector_path,
        )
        _atomic_write_parquet(
            pa.Table.from_pylist(
                token_rows,
                schema=_event_token_schema(),
            ),
            token_path,
        )
        _atomic_write_parquet(
            pa.Table.from_pylist(sequence_rows, schema=TOKEN_SEQUENCE_SCHEMA),
            sequence_path,
        )
        joblib.dump(
            {
                "pca_model": preprocess.pca_model,
                "effective_pca_dim": preprocess.effective_pca_dim,
                "descriptor_median": preprocess.descriptor_median,
                "descriptor_scale": preprocess.descriptor_scale,
            },
            preprocess_path,
        )
        for k, result in tokenization.tokenizations.items():
            joblib.dump(
                result.model,
                event_encoder_kmeans_path(settings.storage_root, job_id, k),
            )

        manifest = _build_manifest(
            job=job,
            continuous=continuous,
            total_events=len(events),
            encoded_events=len(encoded_events),
            skip_reasons=skip_reasons,
            event_vector_dim=int(preprocess.event_vectors.shape[1]),
            effective_pca_dim=preprocess.effective_pca_dim,
            valid_k_values=sorted(tokenization.tokenizations),
            invalid_k_values=tokenization.invalid_k_values,
        )
        report = _build_report(
            manifest=manifest,
            encoded_events=encoded_events,
            tokenization=tokenization,
        )
        _atomic_write_text(
            json.dumps(manifest, sort_keys=True, indent=2), manifest_path
        )
        _atomic_write_text(json.dumps(report, sort_keys=True, indent=2), report_path)

        refreshed = await session.get(EventEncoderJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.event_vector_dim = int(preprocess.event_vectors.shape[1])
        target.total_events = len(events)
        target.encoded_events = len(encoded_events)
        target.skipped_events = len(events) - len(encoded_events)
        target.event_vectors_path = str(vector_path)
        target.event_tokens_path = str(token_path)
        target.token_sequences_path = str(sequence_path)
        target.manifest_path = str(manifest_path)
        target.report_path = str(report_path)
        target.error_message = None
        await session.commit()

        logger.info(
            "event_encoder | job=%s | complete | events=%d encoded=%d dim=%d",
            job_id,
            len(events),
            len(encoded_events),
            int(preprocess.event_vectors.shape[1]),
        )
    except Exception as exc:
        logger.exception("event_encoder job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        failed = await session.get(EventEncoderJob, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.error_message = str(exc)
        target.event_vectors_path = None
        target.event_tokens_path = None
        target.token_sequences_path = None
        target.manifest_path = None
        target.report_path = None
        await session.commit()


async def _load_events(
    session: AsyncSession,
    job: EventEncoderJob,
    settings: Settings,
) -> list[Event]:
    if job.event_source_mode == "effective":
        return await load_effective_events(
            session,
            event_segmentation_job_id=job.event_segmentation_job_id,
            storage_root=settings.storage_root,
        )
    events_path = (
        segmentation_job_dir(settings.storage_root, job.event_segmentation_job_id)
        / "events.parquet"
    )
    if not events_path.exists():
        raise FileNotFoundError(
            f"events.parquet not found for segmentation job {job.event_segmentation_job_id}"
        )
    return read_events(events_path)


def _load_chunks(path: Path) -> list[ChunkEmbedding]:
    if not path.exists():
        raise FileNotFoundError(f"CRNN embeddings parquet not found: {path}")
    table = pq.read_table(path)
    rows = table.to_pylist()
    chunks: list[ChunkEmbedding] = []
    for row in rows:
        chunks.append(
            ChunkEmbedding(
                region_id=str(row["region_id"]),
                start_timestamp=float(row["start_timestamp"]),
                end_timestamp=float(row["end_timestamp"]),
                call_probability=float(row.get("call_probability") or 0.0),
                embedding=np.asarray(row["embedding"], dtype=np.float32),
            )
        )
    return chunks


async def _build_audio_provider(
    session: AsyncSession,
    settings: Settings,
    region_job: RegionDetectionJob,
    events: list[Event],
    target_sr: int,
) -> EventAudioProvider:
    if region_job.audio_file_id:
        result = await session.execute(
            select(AudioFile).where(AudioFile.id == region_job.audio_file_id)
        )
        audio_file = result.scalar_one_or_none()
        if audio_file is None:
            raise ValueError(f"AudioFile {region_job.audio_file_id} not found")
        event_loader = build_event_audio_loader(
            target_sr=target_sr,
            settings=settings,
            audio_file=audio_file,
            storage_root=settings.storage_root,
        )
    elif region_job.hydrophone_id:
        event_loader = build_event_audio_loader(
            target_sr=target_sr,
            settings=settings,
            hydrophone_id=region_job.hydrophone_id,
            job_start_ts=region_job.start_timestamp or 0.0,
            job_end_ts=region_job.end_timestamp or 0.0,
            preload_events=events,
        )
    else:
        raise ValueError(f"region_detection_job {region_job.id} has no audio source")

    def _provider(event: Event, sample_rate: int) -> np.ndarray:
        del sample_rate
        buffer, buffer_start = event_loader(event)
        start_idx = int(round((float(event.start_sec) - buffer_start) * target_sr))
        end_idx = int(round((float(event.end_sec) - buffer_start) * target_sr))
        start_idx = max(0, min(start_idx, buffer.shape[0]))
        end_idx = max(start_idx, min(end_idx, buffer.shape[0]))
        return np.asarray(buffer[start_idx:end_idx], dtype=np.float32)

    return _provider


async def _build_encoded_events(
    session: AsyncSession,
    job: EventEncoderJob,
    events: list[Event],
    chunks: list[ChunkEmbedding],
    region_job: RegionDetectionJob,
    audio_provider: EventAudioProvider,
    descriptor_config: dict[str, Any],
) -> tuple[list[_EncodedEvent], dict[str, int]]:
    pooling_config = json.loads(job.pooling_config_json)
    timestamp_offset = float(region_job.start_timestamp or 0.0)
    source_key = _source_sequence_key(region_job)
    intervals = [
        EventInterval(
            event_id=event.event_id,
            region_id=event.region_id,
            start_timestamp=float(event.start_sec) + timestamp_offset,
            end_timestamp=float(event.end_sec) + timestamp_offset,
            segmentation_confidence=float(event.segmentation_confidence),
            source_sequence_key=source_key,
        )
        for event in events
    ]
    gaps = compute_gap_to_previous(intervals)
    events_with_intervals = sorted(
        zip(events, intervals),
        key=lambda pair: (
            pair[1].source_sequence_key,
            pair[1].start_timestamp,
            pair[1].end_timestamp,
            pair[1].event_id,
        ),
    )

    encoded: list[_EncodedEvent] = []
    skip_reasons: dict[str, int] = {}
    target_sr = int(descriptor_config.get("target_sample_rate", 16000))
    for sequence_index, (raw_event, interval) in enumerate(events_with_intervals):
        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            return encoded, skip_reasons
        try:
            embedding = build_event_embedding(
                interval,
                chunks,
                enabled_pools=list(pooling_config["enabled_pools"]),
                top_k_fraction=float(pooling_config.get("top_k_fraction", 0.25)),
                min_overlap_fraction=float(
                    pooling_config.get("min_overlap_fraction", 0.25)
                ),
                min_chunks_per_event=int(pooling_config.get("min_chunks_per_event", 1)),
            )
        except ValueError as exc:
            reason = str(exc)
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        audio = audio_provider(raw_event, target_sr)
        descriptors = compute_acoustic_descriptors(
            audio,
            sample_rate=target_sr,
            gap_to_previous=gaps[interval.event_id],
            n_fft=int(descriptor_config.get("n_fft", 1024)),
            hop_length=int(descriptor_config.get("hop_length", 512)),
            eps=float(descriptor_config.get("eps", 1e-12)),
            ridge_min_frequency_hz=float(
                descriptor_config.get("ridge_min_frequency_hz", 100.0)
            ),
            ridge_max_frequency_hz=float(
                descriptor_config.get("ridge_max_frequency_hz", 3000.0)
            ),
            ridge_candidate_count=int(
                descriptor_config.get("ridge_candidate_count", 5)
            ),
            ridge_smoothness_penalty=float(
                descriptor_config.get("ridge_smoothness_penalty", 8.0)
            ),
            ridge_peak_prominence_ratio=float(
                descriptor_config.get("ridge_peak_prominence_ratio", 0.0)
            ),
            ridge_summary_low_percentile=float(
                descriptor_config.get("ridge_summary_low_percentile", 10.0)
            ),
            ridge_summary_high_percentile=float(
                descriptor_config.get("ridge_summary_high_percentile", 90.0)
            ),
            band_peak_min_frequency_hz=float(
                descriptor_config.get("band_peak_min_frequency_hz", 100.0)
            ),
            band_peak_max_frequency_hz=_optional_float(
                descriptor_config.get("band_peak_max_frequency_hz")
            ),
            high_band_min_frequency_hz=float(
                descriptor_config.get("high_band_min_frequency_hz", 1000.0)
            ),
            f0_fmin=float(descriptor_config.get("f0_fmin", 70.0)),
            f0_fmax=float(descriptor_config.get("f0_fmax", 1200.0)),
            pulse_min_rate_hz=float(descriptor_config.get("pulse_min_rate_hz", 2.0)),
            pulse_max_rate_hz=float(descriptor_config.get("pulse_max_rate_hz", 200.0)),
            pulse_confidence_threshold=float(
                descriptor_config.get("pulse_confidence_threshold", 0.3)
            ),
            pulse_envelope_smooth_ms=float(
                descriptor_config.get("pulse_envelope_smooth_ms", 5.0)
            ),
        )
        encoded.append(
            _EncodedEvent(
                event=interval,
                sequence_index=sequence_index,
                chunk_count=embedding.chunk_count,
                coverage_fraction=embedding.coverage_fraction,
                pool_vector=embedding.pool_vector,
                descriptors=descriptors,
            )
        )
    return encoded, skip_reasons


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _source_sequence_key(region_job: RegionDetectionJob) -> str:
    if region_job.audio_file_id:
        return f"audio:{region_job.audio_file_id}"
    if region_job.hydrophone_id:
        return (
            f"hydrophone:{region_job.hydrophone_id}:"
            f"{region_job.start_timestamp}:{region_job.end_timestamp}"
        )
    return f"region:{region_job.id}"


def _build_event_vector_rows(
    encoded_events: list[_EncodedEvent],
    preprocess,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, encoded in enumerate(encoded_events):
        descriptors = encoded.descriptors
        row = {
            "event_id": encoded.event.event_id,
            "region_id": encoded.event.region_id,
            "source_sequence_key": encoded.event.source_sequence_key,
            "sequence_index": encoded.sequence_index,
            "start_timestamp": encoded.event.start_timestamp,
            "end_timestamp": encoded.event.end_timestamp,
            "segmentation_confidence": encoded.event.segmentation_confidence,
            "chunk_count": encoded.chunk_count,
            "coverage_fraction": encoded.coverage_fraction,
            "embedding_vector": preprocess.embedding_vectors[i].tolist(),
            "descriptor_vector": preprocess.descriptor_vectors[i].tolist(),
            "event_vector": preprocess.event_vectors[i].tolist(),
        }
        for name in DESCRIPTOR_ORDER:
            row[name] = float(descriptors[name])
        rows.append(row)
    return rows


def _build_token_rows(
    encoded_events: list[_EncodedEvent],
    tokenization,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    token_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    positions_by_source: dict[tuple[int, str], int] = {}
    for k, result in sorted(tokenization.tokenizations.items()):
        for i, encoded in enumerate(encoded_events):
            token_id = int(result.token_ids[i])
            label = f"T{token_id:02d}"
            second = float(result.second_distances[i])
            if np.isnan(second):
                second_value = None
            else:
                second_value = second
            base = {
                "k": int(k),
                "event_id": encoded.event.event_id,
                "region_id": encoded.event.region_id,
                "source_sequence_key": encoded.event.source_sequence_key,
                "sequence_index": encoded.sequence_index,
                "start_timestamp": encoded.event.start_timestamp,
                "end_timestamp": encoded.event.end_timestamp,
                "token_id": token_id,
                "token_label": label,
            }
            token_row = {
                **base,
                "distance_to_centroid": float(result.distances[i]),
                "second_centroid_distance": second_value,
                "token_confidence": float(result.confidences[i]),
            }
            for name in DESCRIPTOR_ORDER:
                token_row[name] = float(encoded.descriptors[name])
            token_rows.append(token_row)

            pos_key = (int(k), encoded.event.source_sequence_key)
            position = positions_by_source.get(pos_key, 0)
            positions_by_source[pos_key] = position + 1
            sequence_rows.append(
                {
                    "k": int(k),
                    "source_sequence_key": encoded.event.source_sequence_key,
                    "position": position,
                    "event_id": encoded.event.event_id,
                    "token_id": token_id,
                    "token_label": label,
                    "start_timestamp": encoded.event.start_timestamp,
                    "end_timestamp": encoded.event.end_timestamp,
                    "gap_to_previous": float(encoded.descriptors["gap_to_previous"]),
                }
            )
    return token_rows, sequence_rows


def _build_manifest(
    *,
    job: EventEncoderJob,
    continuous: ContinuousEmbeddingJob,
    total_events: int,
    encoded_events: int,
    skip_reasons: dict[str, int],
    event_vector_dim: int,
    effective_pca_dim: int,
    valid_k_values: list[int],
    invalid_k_values: list[int],
) -> dict[str, Any]:
    return {
        "job_id": job.id,
        "tokenizer_version": job.tokenizer_version,
        "event_segmentation_job_id": job.event_segmentation_job_id,
        "event_source_mode": job.event_source_mode,
        "continuous_embedding_job_id": job.continuous_embedding_job_id,
        "continuous_embedding_signature": job.continuous_embedding_signature,
        "continuous_embedding": {
            "model_version": continuous.model_version,
            "crnn_checkpoint_sha256": continuous.crnn_checkpoint_sha256,
            "chunk_size_seconds": continuous.chunk_size_seconds,
            "chunk_hop_seconds": continuous.chunk_hop_seconds,
            "projection_kind": continuous.projection_kind,
            "projection_dim": continuous.projection_dim,
        },
        "pooling_config": json.loads(job.pooling_config_json),
        "descriptor_config": json.loads(job.descriptor_config_json),
        "preprocessing_config": json.loads(job.preprocessing_config_json),
        "descriptor_feature_names": list(DESCRIPTOR_ORDER),
        "k_values": json.loads(job.k_values_json),
        "valid_k_values": valid_k_values,
        "invalid_k_values": invalid_k_values,
        "random_seed": job.random_seed,
        "event_vector_dim": event_vector_dim,
        "effective_pca_dim": effective_pca_dim,
        "total_events": total_events,
        "encoded_events": encoded_events,
        "skipped_events": total_events - encoded_events,
        "skip_reasons": skip_reasons,
    }


def _build_report(
    *,
    manifest: dict[str, Any],
    encoded_events: list[_EncodedEvent],
    tokenization,
) -> dict[str, Any]:
    sequence_preview: dict[str, list[str]] = {}
    for k, result in sorted(tokenization.tokenizations.items()):
        key = str(k)
        sequence_preview[key] = [
            f"T{int(token_id):02d}" for token_id in result.token_ids[:100]
        ]
    return {
        "summary": {
            "total_events": manifest["total_events"],
            "encoded_events": manifest["encoded_events"],
            "skipped_events": manifest["skipped_events"],
            "valid_k_values": manifest["valid_k_values"],
            "invalid_k_values": manifest["invalid_k_values"],
        },
        "tokenization": tokenization_summary(tokenization),
        "token_examples": _token_examples(encoded_events, tokenization),
        "descriptor_summary": _descriptor_summary(encoded_events),
        "descriptor_feature_names": list(DESCRIPTOR_ORDER),
        "sequence_preview": sequence_preview,
    }


def _token_examples(
    encoded_events: list[_EncodedEvent], tokenization, limit: int = 5
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    examples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for k, result in sorted(tokenization.tokenizations.items()):
        k_examples: dict[str, list[dict[str, Any]]] = {}
        for token_id in sorted(result.token_counts):
            indices = np.where(result.token_ids == token_id)[0]
            ordered = sorted(
                indices,
                key=lambda idx: (
                    float(result.distances[idx]),
                    encoded_events[int(idx)].event.event_id,
                ),
            )[:limit]
            label = f"T{int(token_id):02d}"
            k_examples[label] = [
                {
                    "event_id": encoded_events[int(idx)].event.event_id,
                    "distance_to_centroid": float(result.distances[idx]),
                    "token_confidence": float(result.confidences[idx]),
                }
                for idx in ordered
            ]
        examples[str(k)] = k_examples
    return examples


def _descriptor_summary(
    encoded_events: list[_EncodedEvent],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    if not encoded_events:
        return summary
    for name in DESCRIPTOR_ORDER:
        values = np.asarray([event.descriptors[name] for event in encoded_events])
        summary[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return summary
