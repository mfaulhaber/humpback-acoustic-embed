"""Pass 2 — event segmentation worker.

Runs the trained ``SegmentationCRNN`` on every ``Region`` from a
completed upstream ``RegionDetectionJob`` and writes ``events.parquet``
to the per-job storage directory. The audio source is resolved from
the upstream Pass 1 job's columns — Pass 2 rows never carry their own
source identity. Mirrors the Pass 1 worker's crash-safety pattern: on
any exception the partial parquet and ``.tmp`` sidecars are deleted,
the job row flips to ``failed``, and nothing stale is left on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_region_audio_loader
from humpback.call_parsing.segmentation.inference import run_inference
from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.storage import (
    read_regions,
    region_job_dir,
    segmentation_job_dir,
    write_events,
)
from humpback.call_parsing.types import Event, Region
from humpback.config import Settings
from humpback.ml.checkpointing import load_checkpoint
from humpback.ml.device import select_and_validate_device
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
)
from humpback.storage import ensure_dir
from humpback.workers.queue import claim_event_segmentation_job

logger = logging.getLogger(__name__)

# Sample input length for the load-time device validation. The CRNN
# trains on 30-second crops; ~500 frames of log-mel matches that order
# of magnitude and is small enough to be cheap on either device.
_VALIDATION_FRAMES: int = 500


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    """Delete any ``events.parquet`` + ``.tmp`` sidecars left by a failed run."""
    if not job_dir.exists():
        return
    events_path = job_dir / "events.parquet"
    if events_path.exists():
        try:
            events_path.unlink()
        except OSError:
            logger.warning("Failed to delete %s", events_path, exc_info=True)
    for tmp in job_dir.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            logger.warning("Failed to delete %s", tmp, exc_info=True)


def _instantiate_model(model_config: dict[str, Any]) -> SegmentationCRNN:
    """Build a ``SegmentationCRNN`` from the persisted architecture config."""
    n_mels = int(model_config.get("n_mels", 64))
    conv_channels_raw = model_config.get("conv_channels", [32, 64, 96, 128])
    conv_channels = [int(c) for c in conv_channels_raw]
    gru_hidden = int(model_config.get("gru_hidden", 64))
    gru_layers = int(model_config.get("gru_layers", 2))
    return SegmentationCRNN(
        n_mels=n_mels,
        conv_channels=conv_channels,
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
    )


def _feature_config_from(model_config: dict[str, Any]) -> SegmentationFeatureConfig:
    """Rehydrate the feature config saved alongside the checkpoint."""
    raw = model_config.get("feature_config") or {}
    if not isinstance(raw, dict):
        raise ValueError("checkpoint feature_config must be a dict")
    return SegmentationFeatureConfig(**raw)


def _load_and_validate_model(
    *,
    checkpoint_path: Path,
    model_config: dict[str, Any],
    feature_config: SegmentationFeatureConfig,
) -> tuple[SegmentationCRNN, torch.device, str | None]:
    """Load the checkpoint and pick a device via load-time validation.

    Runs under ``asyncio.to_thread``: building the model, loading
    weights, and the two validation forward passes all hit torch and
    must stay off the event loop. Returns the model (left on the
    chosen device), the chosen ``torch.device``, and a fallback reason
    (``None`` on success or when no GPU was attempted).
    """
    model = _instantiate_model(model_config)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    sample_input = torch.zeros(1, 1, feature_config.n_mels, _VALIDATION_FRAMES)
    device, fallback_reason = select_and_validate_device(model, sample_input)
    return model, device, fallback_reason


def _run_inference_pipeline(
    *,
    model: SegmentationCRNN,
    device: torch.device,
    feature_config: SegmentationFeatureConfig,
    decoder_config: SegmentationDecoderConfig,
    regions: list[Region],
    audio_loader,
    out_events_path: Path,
) -> int:
    """Blocking work: decode each region with the prepared model, write parquet.

    The model has already been loaded and moved to ``device`` by
    ``_load_and_validate_model``. Runs under ``asyncio.to_thread`` so
    torch, librosa, and parquet I/O never block the worker event loop.
    Returns the decoded event count for the caller to stamp onto the
    row.
    """
    all_events: list[Event] = []
    for region in regions:
        events = run_inference(
            model=model,
            region=region,
            audio_loader=audio_loader,
            feature_config=feature_config,
            decoder_config=decoder_config,
            device=device,
        )
        all_events.extend(events)

    write_events(out_events_path, all_events)
    return len(all_events)


async def run_event_segmentation_job(
    session: AsyncSession,
    job: EventSegmentationJob,
    settings: Settings,
) -> None:
    """Execute one Pass 2 event segmentation job end-to-end."""
    job_id = job.id
    upstream_id = job.region_detection_job_id
    seg_model_id = job.segmentation_model_id
    config_json = job.config_json

    job_dir = ensure_dir(segmentation_job_dir(settings.storage_root, job_id))
    try:
        if not config_json:
            raise ValueError("event segmentation job missing config_json")
        decoder_config = SegmentationDecoderConfig.model_validate_json(config_json)

        if not seg_model_id:
            raise ValueError("event segmentation job missing segmentation_model_id")
        if not upstream_id:
            raise ValueError("event segmentation job missing region_detection_job_id")

        upstream = await session.get(RegionDetectionJob, upstream_id)
        if upstream is None:
            raise ValueError(f"RegionDetectionJob {upstream_id} not found")
        if upstream.status != "complete":
            raise ValueError(
                f"upstream RegionDetectionJob {upstream_id} not complete "
                f"(status={upstream.status})"
            )

        seg_model = await session.get(SegmentationModel, seg_model_id)
        if seg_model is None:
            raise ValueError(f"SegmentationModel {seg_model_id} not found")

        checkpoint_path = Path(seg_model.model_path)
        if not checkpoint_path.exists():
            raise ValueError(
                f"SegmentationModel {seg_model_id} checkpoint missing at "
                f"{checkpoint_path}"
            )

        model_config_raw = json.loads(seg_model.config_json or "{}")
        if not isinstance(model_config_raw, dict):
            raise ValueError(
                f"SegmentationModel {seg_model_id} config_json is not a JSON object"
            )
        feature_config = _feature_config_from(model_config_raw)

        regions_path = (
            region_job_dir(settings.storage_root, upstream_id) / "regions.parquet"
        )
        if not regions_path.exists():
            raise ValueError(f"upstream regions.parquet not found at {regions_path}")
        regions = read_regions(regions_path)

        upstream_audio_file_id = upstream.audio_file_id
        upstream_hydrophone_id = upstream.hydrophone_id
        if upstream_audio_file_id:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id == upstream_audio_file_id)
            )
            audio_file = af_result.scalar_one_or_none()
            if audio_file is None:
                raise ValueError(
                    f"AudioFile {upstream_audio_file_id} referenced by upstream "
                    f"RegionDetectionJob {upstream_id} not found"
                )
            audio_loader = build_region_audio_loader(
                target_sr=feature_config.sample_rate,
                settings=settings,
                audio_file=audio_file,
                storage_root=settings.storage_root,
            )
        elif upstream_hydrophone_id:
            audio_loader = build_region_audio_loader(
                target_sr=feature_config.sample_rate,
                settings=settings,
                hydrophone_id=upstream_hydrophone_id,
                job_start_ts=upstream.start_timestamp or 0.0,
                job_end_ts=upstream.end_timestamp or 0.0,
            )
        else:
            raise ValueError(
                f"upstream RegionDetectionJob {upstream_id} has no audio source"
            )

        model, device, fallback_reason = await asyncio.to_thread(
            _load_and_validate_model,
            checkpoint_path=checkpoint_path,
            model_config=model_config_raw,
            feature_config=feature_config,
        )
        if fallback_reason is not None:
            logger.warning(
                "Pass 2 job %s falling back to CPU (reason=%s)",
                job_id,
                fallback_reason,
            )

        job.started_at = datetime.now(timezone.utc)
        job.compute_device = device.type
        job.gpu_fallback_reason = fallback_reason
        await session.commit()

        event_count = await asyncio.to_thread(
            _run_inference_pipeline,
            model=model,
            device=device,
            feature_config=feature_config,
            decoder_config=decoder_config,
            regions=regions,
            audio_loader=audio_loader,
            out_events_path=job_dir / "events.parquet",
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(EventSegmentationJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = "complete"
        target.event_count = event_count
        target.completed_at = now
        target.updated_at = now
        await session.commit()
        logger.info(
            "Event segmentation job %s complete (event_count=%d)", job_id, event_count
        )

    except Exception as exc:
        logger.exception("Event segmentation job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(EventSegmentationJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception(
                "Failed to mark event segmentation job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> EventSegmentationJob | None:
    """Claim and process at most one event segmentation job. Returns it or None."""
    job = await claim_event_segmentation_job(session)
    if job is None:
        return None
    await run_event_segmentation_job(session, job, settings)
    return job
