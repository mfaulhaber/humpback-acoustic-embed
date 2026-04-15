"""Pass 3 feedback training worker.

Claims a queued ``EventClassifierTrainingJob``, collects typed events
and human type corrections from source classification jobs, assembles
training samples with corrected labels, resolves audio through the full
job chain, and trains an ``EventClassifierCNN`` model via
``train_event_classifier``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig,
    EventClassifierTrainingResult,
    train_event_classifier,
)
from humpback.call_parsing.segmentation.extraction import load_corrected_events
from humpback.call_parsing.storage import (
    classification_job_dir,
    read_typed_events,
)
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.feedback_training import (
    EventClassifierTrainingJob,
    EventTypeCorrection,
)
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.schemas.call_parsing import SegmentationFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class _ClassifierSample:
    """Synthetic sample compatible with the event classifier trainer."""

    start_sec: float
    end_sec: float
    type_index: int
    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float


def _resolve_event_labels(
    typed_events_by_event: dict[str, list[Any]],
    corrections: dict[str, str | None],
    *,
    corrections_only: bool = True,
) -> dict[str, str | None]:
    """For each event, resolve the final single type label.

    Returns ``{event_id: type_name}`` where type_name is None for negatives.
    Corrected events use the correction; when *corrections_only* is False,
    uncorrected events fall back to the highest-scoring above-threshold type
    from inference output.  When True, uncorrected events are excluded.
    """
    labels: dict[str, str | None] = {}

    all_event_ids = set(typed_events_by_event.keys()) | set(corrections.keys())

    for event_id in all_event_ids:
        if event_id in corrections:
            labels[event_id] = corrections[event_id]
        elif corrections_only:
            labels[event_id] = None
        else:
            rows = typed_events_by_event.get(event_id, [])
            above = [r for r in rows if r.above_threshold]
            if above:
                best = max(above, key=lambda r: r.score)
                labels[event_id] = best.type_name
            else:
                labels[event_id] = None

    return labels


async def _collect_samples(
    session: AsyncSession,
    source_job_ids: list[str],
    settings: Settings,
    *,
    corrections_only: bool = True,
) -> tuple[list[_ClassifierSample], list[str]]:
    """Collect training samples from source classification jobs.

    Returns ``(samples, vocabulary)`` where vocabulary is the sorted
    list of all type names seen in the resolved labels.
    """
    all_labeled: list[tuple[str, float, float, str, float, float]] = []

    for cls_job_id in source_job_ids:
        cls_job = await session.get(EventClassificationJob, cls_job_id)
        if cls_job is None:
            raise ValueError(f"Classification job {cls_job_id} not found")

        seg_job = await session.get(
            EventSegmentationJob, cls_job.event_segmentation_job_id
        )
        if seg_job is None:
            raise ValueError(
                f"Segmentation job {cls_job.event_segmentation_job_id} not found"
            )

        upstream = await session.get(
            RegionDetectionJob, seg_job.region_detection_job_id
        )
        if upstream is None:
            raise ValueError(
                f"Region detection job {seg_job.region_detection_job_id} not found"
            )
        if not upstream.hydrophone_id:
            raise ValueError(
                f"Region detection job {upstream.id} is not hydrophone-sourced"
            )

        hydro_id = upstream.hydrophone_id
        job_start_ts = upstream.start_timestamp or 0.0
        job_end_ts = upstream.end_timestamp or 0.0

        typed_path = (
            classification_job_dir(settings.storage_root, cls_job_id)
            / "typed_events.parquet"
        )
        if not typed_path.exists():
            logger.warning(
                "typed_events.parquet missing for classification job %s", cls_job_id
            )
            continue

        typed_events = read_typed_events(typed_path)
        typed_by_event: dict[str, list[Any]] = defaultdict(list)
        for te in typed_events:
            typed_by_event[te.event_id].append(te)

        # Use corrected boundaries from the upstream segmentation job so
        # that human boundary edits (adjust/add/delete) are reflected in
        # the audio crops used for classifier training.
        corrected_events = await load_corrected_events(
            session, cls_job.event_segmentation_job_id, settings.storage_root
        )
        event_bounds: dict[str, tuple[float, float]] = {
            e.event_id: (e.start_sec, e.end_sec) for e in corrected_events
        }

        corr_result = await session.execute(
            select(EventTypeCorrection).where(
                EventTypeCorrection.event_classification_job_id == cls_job_id
            )
        )
        corrections_raw = list(corr_result.scalars().all())
        corrections = {c.event_id: c.type_name for c in corrections_raw}

        labels = _resolve_event_labels(
            typed_by_event, corrections, corrections_only=corrections_only
        )

        for event_id, type_name in labels.items():
            if type_name is None:
                continue
            bounds = event_bounds.get(event_id)
            if bounds is None:
                continue
            start_sec, end_sec = bounds
            all_labeled.append(
                (type_name, start_sec, end_sec, hydro_id, job_start_ts, job_end_ts)
            )

    vocabulary = sorted({label[0] for label in all_labeled})
    type_to_idx = {name: i for i, name in enumerate(vocabulary)}

    samples = [
        _ClassifierSample(
            start_sec=start,
            end_sec=end,
            type_index=type_to_idx[tname],
            hydrophone_id=hydro,
            start_timestamp=job_start,
            end_timestamp=job_end,
        )
        for tname, start, end, hydro, job_start, job_end in all_labeled
    ]

    return samples, vocabulary


def _build_audio_loader(settings: Settings) -> Any:
    """Return a callable that fetches full-range audio for a sample via hydrophone."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    target_sr = 16000

    def _load(sample: Any) -> np.ndarray:
        hydro_id = sample.hydrophone_id
        start_ts = float(sample.start_timestamp)
        end_ts = float(sample.end_timestamp)
        duration = sample.end_sec - sample.start_sec
        context_sec = max(10.0, duration)
        pad = (context_sec - duration) / 2.0
        ctx_start = max(start_ts, start_ts + sample.start_sec - pad)
        ctx_end = min(end_ts, start_ts + sample.end_sec + pad)
        ctx_duration = ctx_end - ctx_start

        audio = resolve_timeline_audio(
            hydrophone_id=hydro_id,
            local_cache_path=str(settings.s3_cache_path or ""),
            job_start_timestamp=start_ts,
            job_end_timestamp=end_ts,
            start_sec=ctx_start,
            duration_sec=ctx_duration,
            target_sr=target_sr,
            noaa_cache_path=str(settings.noaa_cache_path)
            if settings.noaa_cache_path
            else None,
        )
        return audio

    return _load


async def run_event_classifier_feedback_training(
    session: AsyncSession,
    job: EventClassifierTrainingJob,
    settings: Settings,
) -> None:
    """Execute one Pass 3 feedback training job end-to-end."""
    job_id = job.id
    source_job_ids = json.loads(job.source_job_ids)
    config_json = job.config_json

    model_dir = settings.storage_root / "vocalization_models" / job_id

    try:
        training_params = json.loads(config_json) if config_json else {}
        config = EventClassifierTrainingConfig(**training_params)

        samples, vocabulary = await _collect_samples(
            session,
            source_job_ids,
            settings,
            corrections_only=config.corrections_only,
        )
        if not samples:
            raise ValueError("No training samples collected from source jobs")

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        feature_config = SegmentationFeatureConfig()
        audio_loader = _build_audio_loader(settings)
        device = select_device()

        result: EventClassifierTrainingResult = await asyncio.to_thread(
            train_event_classifier,
            samples=samples,
            vocabulary=vocabulary,
            feature_config=feature_config,
            audio_loader=audio_loader,
            config=config,
            model_dir=model_dir,
            device=device,
        )

        model = VocalizationClassifierModel(
            name=f"event-classifier-fb-{job_id[:8]}",
            model_dir_path=str(model_dir),
            vocabulary_snapshot=json.dumps(result.vocabulary),
            per_class_thresholds=json.dumps(result.per_type_thresholds),
            per_class_metrics=json.dumps(result.per_type_metrics),
            training_summary=json.dumps(result.to_summary()),
            model_family="pytorch_event_cnn",
            input_mode="segmented_event",
        )
        session.add(model)
        await session.flush()

        now = datetime.now(timezone.utc)
        refreshed = await session.get(EventClassifierTrainingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.vocalization_model_id = model.id
        target.result_summary = json.dumps(result.to_summary())
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Event classifier feedback training job %s complete (model_id=%s)",
            job_id,
            model.id,
        )

    except Exception as exc:
        logger.exception("Event classifier feedback training job %s failed", job_id)
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(EventClassifierTrainingJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark event classifier feedback training job %s as failed",
                job_id,
            )
