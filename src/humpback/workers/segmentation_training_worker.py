"""Pass 2 segmentation training worker.

Claims a queued ``SegmentationTrainingJob``, loads its dataset's samples,
runs ``train_model`` end-to-end on a background thread, persists the
resulting checkpoint under ``storage_root/segmentation_models/<model_id>``,
and registers a ``SegmentationModel`` row pointing at the checkpoint.
Mirrors the failure-cleanup pattern used by the Pass 1 worker: on any
exception the partial checkpoint directory is removed, the job row flips
to ``failed``, and no half-written ``segmentation_models`` row is left
behind.
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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.trainer import (
    SegmentationTrainingResult,
    train_model,
)
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import SegmentationModel
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingJob,
    SegmentationTrainingSample,
)
from humpback.processing.audio_io import decode_audio, resample
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
    SegmentationTrainingConfig,
)
from humpback.storage import resolve_audio_path
from humpback.workers.queue import claim_segmentation_training_job

logger = logging.getLogger(__name__)


def _segmentation_model_dir(storage_root: Path, model_id: str) -> Path:
    return storage_root / "segmentation_models" / model_id


def _cleanup_model_dir(model_dir: Path) -> None:
    if model_dir.exists():
        shutil.rmtree(model_dir, ignore_errors=True)


def _summary_for_model(result: SegmentationTrainingResult) -> dict[str, Any]:
    """Condensed metrics snapshot stored on the ``SegmentationModel`` row.

    Tight by design — enough for list endpoints to show framewise/event F1
    without loading the full ``result_summary`` JSON from the training job
    row.
    """
    return {
        "framewise_f1": result.framewise_f1,
        "event_f1_iou_0_3": result.event_f1,
        "pos_weight": result.pos_weight,
        "n_train_samples": result.n_train_samples,
        "n_val_samples": result.n_val_samples,
    }


def _build_audio_loader(
    audio_files_by_id: dict[str, AudioFile],
    feature_config: SegmentationFeatureConfig,
    storage_root: Path,
    settings: Settings,
) -> Any:
    """Return a callable that fetches ``[crop_start_sec, crop_end_sec]`` audio.

    Handles both audio-file-sourced and hydrophone-sourced samples.
    Audio files are resolved off-disk via ``resolve_audio_path`` + the
    standard decode/resample helpers. Hydrophone samples are fetched via
    ``resolve_timeline_audio`` using the sample's hydrophone_id and
    absolute UTC timestamp bounds.
    """
    from humpback.processing.timeline_audio import resolve_timeline_audio

    target_sr = feature_config.sample_rate
    cache: dict[str, tuple[np.ndarray, int]] = {}

    def _load(sample: Any) -> np.ndarray:
        audio_file_id = getattr(sample, "audio_file_id", None)
        hydrophone_id = getattr(sample, "hydrophone_id", None)
        if audio_file_id is None and hydrophone_id is None:
            raise ValueError(f"sample {getattr(sample, 'id', '?')} has no audio source")

        if hydrophone_id is not None and audio_file_id is None:
            start_ts = float(sample.start_timestamp)
            end_ts = float(sample.end_timestamp)
            crop_start = float(sample.crop_start_sec)
            crop_end = float(sample.crop_end_sec)
            duration = crop_end - crop_start
            return resolve_timeline_audio(
                hydrophone_id=hydrophone_id,
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

        assert audio_file_id is not None
        audio_file = audio_files_by_id.get(audio_file_id)
        if audio_file is None:
            raise ValueError(
                f"sample {getattr(sample, 'id', '?')} references missing "
                f"audio_file_id={audio_file_id}"
            )

        if audio_file_id not in cache:
            path = resolve_audio_path(audio_file, storage_root)
            raw, sr = decode_audio(path)
            resampled = resample(raw, sr, target_sr)
            cache[audio_file_id] = (np.asarray(resampled, dtype=np.float32), target_sr)

        audio, _ = cache[audio_file_id]
        start = int(round(float(sample.crop_start_sec) * target_sr))
        end = int(round(float(sample.crop_end_sec) * target_sr))
        start = max(0, min(start, audio.shape[0]))
        end = max(start, min(end, audio.shape[0]))
        return audio[start:end].copy()

    return _load


async def _load_samples(
    session: AsyncSession, training_dataset_id: str
) -> list[SegmentationTrainingSample]:
    result = await session.execute(
        select(SegmentationTrainingSample).where(
            SegmentationTrainingSample.training_dataset_id == training_dataset_id
        )
    )
    return list(result.scalars().all())


async def _load_audio_files(
    session: AsyncSession, samples: list[SegmentationTrainingSample]
) -> dict[str, AudioFile]:
    ids = sorted({s.audio_file_id for s in samples if s.audio_file_id})
    if not ids:
        return {}
    result = await session.execute(select(AudioFile).where(AudioFile.id.in_(ids)))
    return {af.id: af for af in result.scalars().all()}


async def run_segmentation_training_job(
    session: AsyncSession,
    job: SegmentationTrainingJob,
    settings: Settings,
) -> None:
    """Execute one Pass 2 segmentation training job end-to-end."""
    job_id = job.id
    training_dataset_id = job.training_dataset_id
    config_json = job.config_json

    model_id = uuid.uuid4().hex
    model_dir = _segmentation_model_dir(settings.storage_root, model_id)
    checkpoint_path = model_dir / "checkpoint.pt"
    model_config_path = model_dir / "config.json"

    try:
        if not config_json:
            raise ValueError("segmentation training job missing config_json")

        training_config = SegmentationTrainingConfig.model_validate_json(config_json)

        dataset_row = await session.get(
            SegmentationTrainingDataset, training_dataset_id
        )
        if dataset_row is None:
            raise ValueError(
                f"SegmentationTrainingDataset {training_dataset_id} not found"
            )

        samples = await _load_samples(session, training_dataset_id)
        if not samples:
            raise ValueError(f"training dataset {training_dataset_id} has no samples")

        audio_files_by_id = await _load_audio_files(session, samples)

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        feature_config = SegmentationFeatureConfig(n_mels=training_config.n_mels)
        decoder_config = SegmentationDecoderConfig()
        audio_loader = _build_audio_loader(
            audio_files_by_id, feature_config, settings.storage_root, settings
        )

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
        target.result_summary = json.dumps(result.to_summary())
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Segmentation training job %s complete (model_id=%s)", job_id, seg_model.id
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
                "Failed to mark segmentation training job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> SegmentationTrainingJob | None:
    """Claim and process at most one training job. Returns it or None."""
    job = await claim_segmentation_training_job(session)
    if job is None:
        return None
    await run_segmentation_training_job(session, job, settings)
    return job
