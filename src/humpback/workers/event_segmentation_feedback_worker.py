"""Pass 2 feedback training worker.

Claims a queued ``EventSegmentationTrainingJob``, collects events and
human boundary corrections from the source segmentation jobs, applies
corrections per region, builds framewise training crops, and trains a
``SegmentationCRNN`` model via ``train_model``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.trainer import (
    SegmentationTrainingResult,
    train_model,
)
from humpback.call_parsing.storage import (
    read_events,
    read_regions,
    segmentation_job_dir,
)
from humpback.call_parsing.types import Event, Region
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.feedback_training import (
    EventBoundaryCorrection,
    EventSegmentationTrainingJob,
)
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
    SegmentationTrainingConfig,
)

logger = logging.getLogger(__name__)


def _segmentation_model_dir(storage_root: Path, model_id: str) -> Path:
    return storage_root / "segmentation_models" / model_id


def _cleanup_model_dir(model_dir: Path) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)


@dataclass
class _FeedbackSample:
    """Synthetic sample compatible with the segmentation trainer's contract."""

    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float
    crop_start_sec: float
    crop_end_sec: float
    events_json: str


def _apply_corrections(
    original_events: list[Event],
    corrections: list[EventBoundaryCorrection],
) -> list[dict[str, float]]:
    """Apply boundary corrections to a region's events, return corrected event dicts."""
    events_by_id: dict[str, dict[str, float]] = {
        e.event_id: {"start_sec": e.start_sec, "end_sec": e.end_sec}
        for e in original_events
    }

    for c in corrections:
        if c.correction_type == "delete":
            events_by_id.pop(c.event_id, None)
        elif c.correction_type == "adjust":
            if (
                c.event_id in events_by_id
                and c.start_sec is not None
                and c.end_sec is not None
            ):
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }
        elif c.correction_type == "add":
            if c.start_sec is not None and c.end_sec is not None:
                events_by_id[c.event_id] = {
                    "start_sec": c.start_sec,
                    "end_sec": c.end_sec,
                }

    return list(events_by_id.values())


async def _collect_samples(
    session: AsyncSession,
    source_job_ids: list[str],
    settings: Settings,
) -> tuple[list[_FeedbackSample], dict[str, tuple[str, float, float]]]:
    """Collect per-region training samples from source segmentation jobs.

    Returns ``(samples, audio_context)`` where ``audio_context`` maps
    ``segmentation_job_id`` → ``(hydrophone_id, job_start_ts, job_end_ts)``
    for audio resolution.
    """
    samples: list[_FeedbackSample] = []
    audio_context: dict[str, tuple[str, float, float]] = {}

    for seg_job_id in source_job_ids:
        seg_job = await session.get(EventSegmentationJob, seg_job_id)
        if seg_job is None:
            raise ValueError(f"Segmentation job {seg_job_id} not found")

        upstream = await session.get(
            RegionDetectionJob, seg_job.region_detection_job_id
        )
        if upstream is None:
            raise ValueError(
                f"Upstream region detection job "
                f"{seg_job.region_detection_job_id} not found"
            )
        if not upstream.hydrophone_id:
            raise ValueError(
                f"Region detection job {upstream.id} is not hydrophone-sourced"
            )

        hydro_id = upstream.hydrophone_id
        job_start_ts = upstream.start_timestamp or 0.0
        job_end_ts = upstream.end_timestamp or 0.0
        audio_context[seg_job_id] = (hydro_id, job_start_ts, job_end_ts)

        seg_dir = segmentation_job_dir(settings.storage_root, seg_job_id)
        events_path = seg_dir / "events.parquet"
        from humpback.call_parsing.storage import region_job_dir

        regions_path = (
            region_job_dir(settings.storage_root, upstream.id) / "regions.parquet"
        )

        if not events_path.exists():
            logger.warning("events.parquet missing for segmentation job %s", seg_job_id)
            continue
        if not regions_path.exists():
            logger.warning("regions.parquet missing for region job %s", upstream.id)
            continue

        all_events = read_events(events_path)
        all_regions = read_regions(regions_path)
        regions_by_id: dict[str, Region] = {r.region_id: r for r in all_regions}

        events_by_region: dict[str, list[Event]] = defaultdict(list)
        for e in all_events:
            events_by_region[e.region_id].append(e)

        corr_result = await session.execute(
            select(EventBoundaryCorrection).where(
                EventBoundaryCorrection.event_segmentation_job_id == seg_job_id
            )
        )
        corrections = list(corr_result.scalars().all())
        corrections_by_region: dict[str, list[EventBoundaryCorrection]] = defaultdict(
            list
        )
        for c in corrections:
            corrections_by_region[c.region_id].append(c)

        for region_id, region in regions_by_id.items():
            region_events = events_by_region.get(region_id, [])
            region_corrections = corrections_by_region.get(region_id, [])
            corrected = _apply_corrections(region_events, region_corrections)

            crop_start = region.padded_start_sec
            crop_end = region.padded_end_sec

            # Events stay in absolute (job-relative) coordinates —
            # build_framewise_target subtracts crop_start_sec itself.
            samples.append(
                _FeedbackSample(
                    hydrophone_id=hydro_id,
                    start_timestamp=job_start_ts,
                    end_timestamp=job_end_ts,
                    crop_start_sec=crop_start,
                    crop_end_sec=crop_end,
                    events_json=json.dumps(corrected),
                )
            )

    return samples, audio_context


def _build_audio_loader(
    feature_config: SegmentationFeatureConfig,
    settings: Settings,
) -> Any:
    """Return a callable that fetches region-crop audio via resolve_timeline_audio."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    target_sr = feature_config.sample_rate

    def _load(sample: Any) -> np.ndarray:
        hydro_id = sample.hydrophone_id
        start_ts = float(sample.start_timestamp)
        end_ts = float(sample.end_timestamp)
        crop_start = float(sample.crop_start_sec)
        crop_end = float(sample.crop_end_sec)
        duration = crop_end - crop_start
        return resolve_timeline_audio(
            hydrophone_id=hydro_id,
            local_cache_path=str(settings.s3_cache_path or ""),
            job_start_timestamp=start_ts,
            job_end_timestamp=end_ts,
            start_sec=start_ts + crop_start,
            duration_sec=duration,
            target_sr=target_sr,
            noaa_cache_path=str(settings.noaa_cache_path)
            if settings.noaa_cache_path
            else None,
        )

    return _load


def _summary_for_model(result: SegmentationTrainingResult) -> dict[str, Any]:
    return {
        "framewise_f1": result.framewise_f1,
        "event_f1_iou_0_3": result.event_f1,
        "pos_weight": result.pos_weight,
        "n_train_samples": result.n_train_samples,
        "n_val_samples": result.n_val_samples,
    }


async def run_event_segmentation_feedback_training(
    session: AsyncSession,
    job: EventSegmentationTrainingJob,
    settings: Settings,
) -> None:
    """Execute one Pass 2 feedback training job end-to-end."""
    job_id = job.id
    source_job_ids = json.loads(job.source_job_ids)
    config_json = job.config_json

    model_id = uuid.uuid4().hex
    model_dir = _segmentation_model_dir(settings.storage_root, model_id)
    checkpoint_path = model_dir / "checkpoint.pt"
    model_config_path = model_dir / "config.json"

    try:
        if not config_json:
            raise ValueError("feedback training job missing config_json")

        training_config = SegmentationTrainingConfig.model_validate_json(config_json)

        samples, _audio_ctx = await _collect_samples(session, source_job_ids, settings)
        if not samples:
            raise ValueError("No training samples collected from source jobs")

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        feature_config = SegmentationFeatureConfig(n_mels=training_config.n_mels)
        decoder_config = SegmentationDecoderConfig()
        audio_loader = _build_audio_loader(feature_config, settings)

        model_dir.mkdir(parents=True, exist_ok=True)
        device = select_device()

        result: SegmentationTrainingResult = await asyncio.to_thread(
            train_model,
            samples=samples,
            feature_config=feature_config,
            decoder_config=decoder_config,
            audio_loader=audio_loader,
            config=training_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        model_config_payload = {
            "model_type": "SegmentationCRNN",
            "n_mels": training_config.n_mels,
            "conv_channels": list(training_config.conv_channels),
            "gru_hidden": training_config.gru_hidden,
            "gru_layers": training_config.gru_layers,
            "feature_config": feature_config.model_dump(),
            "metrics": _summary_for_model(result),
        }
        model_config_path.write_text(json.dumps(model_config_payload, indent=2))

        seg_model = SegmentationModel(
            id=model_id,
            name=f"segmentation-fb-{job_id[:8]}",
            model_family="pytorch_crnn",
            model_path=str(checkpoint_path),
            config_json=json.dumps(model_config_payload),
            training_job_id=job_id,
        )
        session.add(seg_model)
        await session.flush()

        now = datetime.now(timezone.utc)
        refreshed = await session.get(EventSegmentationTrainingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.segmentation_model_id = seg_model.id
        target.result_summary = json.dumps(result.to_summary())
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Segmentation feedback training job %s complete (model_id=%s)",
            job_id,
            seg_model.id,
        )

    except Exception as exc:
        logger.exception("Segmentation feedback training job %s failed", job_id)
        _cleanup_model_dir(model_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(EventSegmentationTrainingJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark segmentation feedback training job %s as failed",
                job_id,
            )
