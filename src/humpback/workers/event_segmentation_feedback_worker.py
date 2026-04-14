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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.extraction import (
    CorrectedSample,
    collect_corrected_samples,
)
from humpback.call_parsing.segmentation.trainer import (
    SegmentationTrainingResult,
    train_model,
)
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.call_parsing import (
    EventSegmentationJob,
    SegmentationModel,
)
from humpback.models.feedback_training import (
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


async def _collect_samples(
    session: AsyncSession,
    source_job_ids: list[str],
    settings: Settings,
) -> list[CorrectedSample]:
    """Collect per-region training samples from source segmentation jobs."""
    samples: list[CorrectedSample] = []
    for seg_job_id in source_job_ids:
        job_samples = await collect_corrected_samples(
            session, seg_job_id, settings.storage_root
        )
        samples.extend(job_samples)
    return samples


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

        samples = await _collect_samples(session, source_job_ids, settings)
        if not samples:
            raise ValueError("No training samples collected from source jobs")

        # Resolve the source model checkpoint for fine-tuning.
        pretrained_checkpoint: Path | None = None
        first_seg_job = await session.get(EventSegmentationJob, source_job_ids[0])
        if first_seg_job and first_seg_job.segmentation_model_id:
            source_model = await session.get(
                SegmentationModel, first_seg_job.segmentation_model_id
            )
            if source_model and source_model.model_path:
                candidate = Path(source_model.model_path)
                if candidate.exists():
                    pretrained_checkpoint = candidate
                    logger.info(
                        "Will fine-tune from source model %s (%s)",
                        source_model.name,
                        candidate,
                    )

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
            pretrained_checkpoint=pretrained_checkpoint,
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
