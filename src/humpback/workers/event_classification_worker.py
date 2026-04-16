"""Pass 3 — event classification worker.

Claims a queued ``EventClassificationJob``, loads the upstream
``events.parquet`` from a completed Pass 2 job, runs the trained
``EventClassifierCNN`` on each event crop, writes
``typed_events.parquet``, and marks the job complete. Audio source is
resolved transitively from the Pass 1 job through Pass 2. Crash safety
mirrors the Pass 2 worker: partial artifacts are cleaned up and the row
flips to ``failed``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_event_audio_loader
from humpback.call_parsing.event_classifier.inference import (
    EventAudioLoader,
    classify_events,
    load_event_classifier,
)
from humpback.call_parsing.event_classifier.model import EventClassifierCNN
from humpback.call_parsing.segmentation.extraction import load_corrected_events
from humpback.call_parsing.storage import (
    classification_job_dir,
    write_typed_events,
)
from humpback.call_parsing.types import Event, TypedEvent
from humpback.config import Settings
from humpback.ml.device import select_and_validate_device
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.vocalization import VocalizationClassifierModel
from humpback.schemas.call_parsing import SegmentationFeatureConfig
from humpback.storage import ensure_dir
from humpback.workers.queue import claim_event_classification_job

logger = logging.getLogger(__name__)

# Sample input length for the load-time device validation. The event
# classifier operates on per-event crops; ~150 frames matches a ~1.5s
# event at the default hop and is small enough to validate cheaply.
_VALIDATION_FRAMES: int = 150


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    for name in ("typed_events.parquet",):
        p = job_dir / name
        if p.exists():
            try:
                p.unlink()
            except OSError:
                logger.warning("Failed to delete %s", p, exc_info=True)
    for tmp in job_dir.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            logger.warning("Failed to delete %s", tmp, exc_info=True)


def _load_and_validate_model(
    *,
    model_dir: Path,
) -> tuple[
    EventClassifierCNN,
    list[str],
    dict[str, float],
    SegmentationFeatureConfig,
    torch.device,
    str | None,
]:
    """Load the classifier and pick a device via load-time validation."""
    model, vocabulary, thresholds, feature_config = load_event_classifier(model_dir)
    model.eval()
    sample_input = torch.zeros(1, 1, feature_config.n_mels, _VALIDATION_FRAMES)
    device, fallback_reason = select_and_validate_device(model, sample_input)
    return model, vocabulary, thresholds, feature_config, device, fallback_reason


def _run_classification_pipeline(
    *,
    model: EventClassifierCNN,
    vocabulary: list[str],
    thresholds: dict[str, float],
    feature_config: SegmentationFeatureConfig,
    device: torch.device,
    events: list[Event],
    audio_loader: EventAudioLoader,
    out_path: Path,
) -> list[TypedEvent]:
    typed_events = classify_events(
        model=model,
        events=events,
        audio_loader=audio_loader,
        feature_config=feature_config,
        vocabulary=vocabulary,
        thresholds=thresholds,
        device=device,
    )
    write_typed_events(out_path, typed_events)
    return typed_events


async def run_event_classification_job(
    session: AsyncSession,
    job: EventClassificationJob,
    settings: Settings,
) -> None:
    """Execute one Pass 3 event classification job end-to-end."""
    job_id = job.id
    upstream_seg_id = job.event_segmentation_job_id
    model_id = job.vocalization_model_id

    job_dir = ensure_dir(classification_job_dir(settings.storage_root, job_id))
    try:
        if not upstream_seg_id:
            raise ValueError(
                "event classification job missing event_segmentation_job_id"
            )
        if not model_id:
            raise ValueError("event classification job missing vocalization_model_id")

        upstream_seg = await session.get(EventSegmentationJob, upstream_seg_id)
        if upstream_seg is None:
            raise ValueError(f"EventSegmentationJob {upstream_seg_id} not found")
        if upstream_seg.status != "complete":
            raise ValueError(
                f"upstream EventSegmentationJob {upstream_seg_id} not complete "
                f"(status={upstream_seg.status})"
            )

        voc_model = await session.get(VocalizationClassifierModel, model_id)
        if voc_model is None:
            raise ValueError(f"VocalizationClassifierModel {model_id} not found")
        if voc_model.model_family != "pytorch_event_cnn":
            raise ValueError(
                f"VocalizationClassifierModel {model_id} has model_family="
                f"'{voc_model.model_family}', expected 'pytorch_event_cnn'"
            )

        model_dir = Path(voc_model.model_dir_path)
        if not (model_dir / "model.pt").exists():
            raise ValueError(f"model checkpoint missing at {model_dir / 'model.pt'}")

        events = await load_corrected_events(
            session, upstream_seg_id, settings.storage_root
        )

        upstream_region_id = upstream_seg.region_detection_job_id
        if not upstream_region_id:
            raise ValueError(
                f"EventSegmentationJob {upstream_seg_id} missing region_detection_job_id"
            )
        upstream_region = await session.get(RegionDetectionJob, upstream_region_id)
        if upstream_region is None:
            raise ValueError(f"RegionDetectionJob {upstream_region_id} not found")

        audio_file_id = upstream_region.audio_file_id
        hydrophone_id = upstream_region.hydrophone_id
        if audio_file_id:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id == audio_file_id)
            )
            audio_file = af_result.scalar_one_or_none()
            if audio_file is None:
                raise ValueError(f"AudioFile {audio_file_id} not found")
            audio_loader = build_event_audio_loader(
                target_sr=16000,
                settings=settings,
                audio_file=audio_file,
                storage_root=settings.storage_root,
            )
        elif hydrophone_id:
            audio_loader = await asyncio.to_thread(
                build_event_audio_loader,
                target_sr=16000,
                settings=settings,
                hydrophone_id=hydrophone_id,
                job_start_ts=upstream_region.start_timestamp or 0.0,
                job_end_ts=upstream_region.end_timestamp or 0.0,
                preload_events=events,
            )
        else:
            raise ValueError(
                f"upstream RegionDetectionJob {upstream_region_id} has no audio source"
            )

        (
            model,
            vocabulary,
            thresholds,
            feature_config,
            device,
            fallback_reason,
        ) = await asyncio.to_thread(
            _load_and_validate_model,
            model_dir=model_dir,
        )
        if fallback_reason is not None:
            logger.warning(
                "Pass 3 job %s falling back to CPU (reason=%s)",
                job_id,
                fallback_reason,
            )

        job.started_at = datetime.now(timezone.utc)
        job.compute_device = device.type
        job.gpu_fallback_reason = fallback_reason
        await session.commit()

        typed_events = await asyncio.to_thread(
            _run_classification_pipeline,
            model=model,
            vocabulary=vocabulary,
            thresholds=thresholds,
            feature_config=feature_config,
            device=device,
            events=events,
            audio_loader=audio_loader,
            out_path=job_dir / "typed_events.parquet",
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(EventClassificationJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.typed_event_count = len(typed_events)
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Event classification job %s complete (typed_event_count=%d)",
            job_id,
            len(typed_events),
        )

    except Exception as exc:
        logger.exception("Event classification job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(EventClassificationJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark event classification job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> EventClassificationJob | None:
    """Claim and process at most one event classification job. Returns it or None."""
    job = await claim_event_classification_job(session)
    if job is None:
        return None
    await run_event_classification_job(session, job, settings)
    return job
