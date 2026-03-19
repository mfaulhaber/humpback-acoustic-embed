"""Worker function for search jobs: encode a detection audio window."""

import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.search import SearchJob
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import complete_search_job, fail_search_job

logger = logging.getLogger(__name__)


async def run_search_job(
    session: AsyncSession,
    job: SearchJob,
    settings: Settings,
) -> None:
    """Encode a single audio window from a detection job and store the embedding."""
    try:
        # 1. Look up detection job -> classifier model -> model config
        det_result = await session.execute(
            select(DetectionJob).where(DetectionJob.id == job.detection_job_id)
        )
        det_job = det_result.scalar_one_or_none()
        if det_job is None:
            await fail_search_job(
                session, job.id, f"Detection job {job.detection_job_id} not found"
            )
            return

        cm_result = await session.execute(
            select(ClassifierModel).where(
                ClassifierModel.id == det_job.classifier_model_id
            )
        )
        classifier_model = cm_result.scalar_one_or_none()
        if classifier_model is None:
            await fail_search_job(
                session,
                job.id,
                f"Classifier model {det_job.classifier_model_id} not found",
            )
            return

        model_version = classifier_model.model_version
        target_sr = classifier_model.target_sample_rate
        window_size = classifier_model.window_size_seconds

        # 2. Resolve and decode audio
        audio, sr = await _resolve_audio(
            det_job, settings, job.filename, job.start_sec, job.end_sec
        )

        # 3. Resample
        audio = resample(audio, sr, target_sr)

        # 4. Trim to window_size if span > window size (take first window)
        window_samples = int(window_size * target_sr)
        if len(audio) > window_samples:
            audio = audio[:window_samples]
        elif len(audio) < window_samples:
            # Pad with zeros if too short
            audio = np.pad(audio, (0, window_samples - len(audio)))

        # 5. Load model and embed
        model, input_format = await get_model_by_version(
            session, model_version, settings
        )

        if input_format == "waveform":
            batch = [audio]
        else:
            batch = extract_logmel_batch([audio], target_sr)

        embeddings = model.embed(np.array(batch))
        vector = embeddings[0].tolist()

        # 6. Complete the job
        await complete_search_job(
            session,
            job.id,
            model_version=model_version,
            embedding_vector=json.dumps(vector),
        )
        logger.info("Search job %s complete (dim=%d)", job.id, len(vector))

    except Exception as exc:
        logger.exception("Search job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        await fail_search_job(session, job.id, str(exc))


async def _resolve_audio(
    det_job: DetectionJob,
    settings: Settings,
    filename: str,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, int]:
    """Decode audio for a detection row. Returns (audio_array, sample_rate)."""
    import asyncio

    duration_sec = end_sec - start_sec

    if det_job.hydrophone_id:
        from humpback.classifier.providers import build_archive_playback_provider
        from humpback.classifier.s3_stream import resolve_audio_slice

        if det_job.start_timestamp is None or det_job.end_timestamp is None:
            raise ValueError("Hydrophone job missing start/end timestamps")

        cache_path = det_job.local_cache_path or settings.s3_cache_path
        provider = build_archive_playback_provider(
            det_job.hydrophone_id,
            cache_path=cache_path,
            noaa_cache_path=settings.noaa_cache_path,
        )

        target_sr = 32000
        segment = await asyncio.to_thread(
            resolve_audio_slice,
            provider,
            det_job.start_timestamp,
            det_job.end_timestamp,
            filename,
            start_sec,
            duration_sec,
            target_sr,
            det_job.start_timestamp,
        )
        return segment, target_sr

    # Local audio folder
    if det_job.audio_folder is None:
        raise ValueError("Detection job has no audio_folder")

    audio_folder = Path(det_job.audio_folder)
    file_path = audio_folder / filename
    audio, sr = await asyncio.to_thread(decode_audio, file_path)

    # Slice to [start_sec, end_sec]
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    audio = audio[start_sample:end_sample]

    return audio, sr
