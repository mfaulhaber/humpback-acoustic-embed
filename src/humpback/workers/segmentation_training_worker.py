"""Worker for dataset-based segmentation training jobs.

Claims a queued ``SegmentationTrainingJob``, loads its training dataset
samples, trains a ``SegmentationCRNN`` from scratch, and registers the
resulting ``SegmentationModel``.
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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_training_audio_loader
from humpback.call_parsing.segmentation.trainer import (
    SegmentationTrainingResult,
    train_model,
)
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.call_parsing import SegmentationModel
from humpback.models.segmentation_training import (
    SegmentationTrainingJob,
    SegmentationTrainingSample,
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


def _summary_for_model(result: SegmentationTrainingResult) -> dict[str, Any]:
    return {
        "framewise_f1": result.framewise_f1,
        "event_f1_iou_0_3": result.event_f1,
        "pos_weight": result.pos_weight,
        "n_train_samples": result.n_train_samples,
        "n_val_samples": result.n_val_samples,
    }


async def run_segmentation_training(
    session: AsyncSession,
    job: SegmentationTrainingJob,
    settings: Settings,
) -> None:
    """Execute one dataset-based segmentation training job."""
    job_id = job.id
    dataset_id = job.training_dataset_id
    config_json = job.config_json

    model_id = uuid.uuid4().hex
    model_dir = _segmentation_model_dir(settings.storage_root, model_id)
    checkpoint_path = model_dir / "checkpoint.pt"
    model_config_path = model_dir / "config.json"

    try:
        if not config_json:
            raise ValueError("training job missing config_json")

        training_config = SegmentationTrainingConfig.model_validate_json(config_json)

        result = await session.execute(
            select(SegmentationTrainingSample).where(
                SegmentationTrainingSample.training_dataset_id == dataset_id
            )
        )
        samples = list(result.scalars().all())
        if not samples:
            raise ValueError(f"No samples found for dataset {dataset_id}")

        logger.info(
            "Training segmentation model from dataset %s (%d samples)",
            dataset_id,
            len(samples),
        )

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        feature_config = SegmentationFeatureConfig(n_mels=training_config.n_mels)
        decoder_config = SegmentationDecoderConfig()
        audio_loader = build_training_audio_loader(
            target_sr=feature_config.sample_rate,
            settings=settings,
            samples=samples,
        )

        model_dir.mkdir(parents=True, exist_ok=True)
        device = select_device()

        train_result: SegmentationTrainingResult = await asyncio.to_thread(
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
            "metrics": _summary_for_model(train_result),
        }
        model_config_path.write_text(json.dumps(model_config_payload, indent=2))

        seg_model = SegmentationModel(
            id=model_id,
            name=f"segmentation-{job_id[:8]}",
            model_family="pytorch_crnn",
            model_path=str(checkpoint_path),
            config_json=json.dumps(model_config_payload),
            training_job_id=job_id,
        )
        session.add(seg_model)
        await session.flush()

        now = datetime.now(timezone.utc)
        refreshed = await session.get(SegmentationTrainingJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.segmentation_model_id = seg_model.id
        target.result_summary = json.dumps(train_result.to_summary())
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Segmentation training job %s complete (model_id=%s)",
            job_id,
            seg_model.id,
        )

    except Exception as exc:
        logger.exception("Segmentation training job %s failed", job_id)
        _cleanup_model_dir(model_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(SegmentationTrainingJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark segmentation training job %s as failed",
                job_id,
            )
