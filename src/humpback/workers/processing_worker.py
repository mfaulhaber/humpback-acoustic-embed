"""Processing worker: decode audio → resample → window → embed → Parquet."""

import asyncio
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet, ProcessingJob
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.embeddings import IncrementalParquetWriter
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel, FakeTF2Model, FakeTFLiteModel
from humpback.processing.windowing import slice_windows
from humpback.services.model_registry_service import get_model_by_name
from humpback.storage import embedding_path, resolve_audio_path
from humpback.workers.queue import complete_processing_job, fail_processing_job

logger = logging.getLogger(__name__)

# Cache of loaded models keyed by (model_name, use_real_model)
_model_cache: dict[str, EmbeddingModel] = {}


def get_model(settings: Settings) -> EmbeddingModel:
    if settings.use_real_model:
        from humpback.processing.inference import TFLiteModel

        return TFLiteModel(settings.model_path, settings.vector_dim)
    return FakeTFLiteModel(settings.vector_dim)


async def get_model_for_job(
    session, job: ProcessingJob, settings: Settings
) -> tuple[EmbeddingModel, str]:
    """Load model based on job's model_version, using registry and cache.

    Returns (model, input_format) where input_format is "spectrogram" or "waveform".
    """
    cache_key = job.model_version
    model_config = await get_model_by_name(session, job.model_version)
    input_format = model_config.input_format if model_config else "spectrogram"
    model_type = model_config.model_type if model_config else "tflite"

    if cache_key in _model_cache:
        logger.info(
            "Using cached model for %s (type=%s)",
            cache_key,
            type(_model_cache[cache_key]).__name__,
        )
        return _model_cache[cache_key], input_format

    if not settings.use_real_model:
        vector_dim = model_config.vector_dim if model_config else settings.vector_dim
        if input_format == "waveform":
            model = FakeTF2Model(vector_dim)
        else:
            model = FakeTFLiteModel(vector_dim)
        _model_cache[cache_key] = model
        return model, input_format

    # Real model: look up path and dims from registry
    if model_config:
        if model_type == "tf2_saved_model":
            from humpback.processing.inference import TF2SavedModel

            logger.info(
                "Loading TF2SavedModel: path=%s, dim=%d",
                model_config.path,
                model_config.vector_dim,
            )
            model = TF2SavedModel(
                model_config.path,
                model_config.vector_dim,
                force_cpu=settings.tf_force_cpu,
            )
        else:
            from humpback.processing.inference import TFLiteModel

            logger.info(
                "Loading TFLiteModel: path=%s, dim=%d",
                model_config.path,
                model_config.vector_dim,
            )
            model = TFLiteModel(model_config.path, model_config.vector_dim)
    else:
        # Fallback to settings for unregistered model versions
        from humpback.processing.inference import TFLiteModel

        model = TFLiteModel(settings.model_path, settings.vector_dim)

    _model_cache[cache_key] = model
    return model, input_format


async def run_processing_job(
    session: AsyncSession,
    job: ProcessingJob,
    settings: Settings,
    model: EmbeddingModel | None = None,
) -> None:
    """Execute a processing job end-to-end."""
    try:
        # Check if embedding set already exists
        existing = await session.execute(
            select(EmbeddingSet).where(
                EmbeddingSet.audio_file_id == job.audio_file_id,
                EmbeddingSet.encoding_signature == job.encoding_signature,
            )
        )
        if existing.scalar_one_or_none():
            logger.info(
                f"Embedding set already exists for signature {job.encoding_signature}, skipping"
            )
            await complete_processing_job(session, job.id)
            return

        # Load audio file info
        af_result = await session.execute(
            select(AudioFile).where(AudioFile.id == job.audio_file_id)
        )
        af = af_result.scalar_one()

        # Find the audio file on disk
        audio_path = resolve_audio_path(af, settings.storage_root)
        if not audio_path.exists():
            raise FileNotFoundError(f"No audio file found at {audio_path}")

        # Run CPU-bound processing in thread
        input_format = "spectrogram"
        if model is None:
            model, input_format = await get_model_for_job(session, job, settings)

        final_path = embedding_path(
            settings.storage_root,
            job.model_version,
            job.audio_file_id,
            job.encoding_signature,
        )

        total_rows = await asyncio.to_thread(
            _process_audio, audio_path, job, model, final_path, input_format
        )

        # Check for GPU fallback warning
        warning = None
        if getattr(model, "gpu_failed", False):
            warning = "GPU inference failed; used CPU fallback. Results are correct but processing may be slower."
            logger.warning("Job %s: %s", job.id, warning)

        if total_rows == 0:
            # Audio too short for window size — no embeddings produced
            audio_data, sr = decode_audio(audio_path)
            duration = len(audio_data) / sr
            warning = (
                f"Audio too short for window size ({duration:.1f}s < "
                f"{job.window_size_seconds:.1f}s) — no embeddings produced"
            )
            logger.warning("Job %s: %s", job.id, warning)
            # Clean up empty parquet if it was somehow written
            if final_path.exists():
                final_path.unlink()
        else:
            # Create EmbeddingSet record only when embeddings were produced
            es = EmbeddingSet(
                audio_file_id=job.audio_file_id,
                encoding_signature=job.encoding_signature,
                model_version=job.model_version,
                window_size_seconds=job.window_size_seconds,
                target_sample_rate=job.target_sample_rate,
                vector_dim=model.vector_dim,
                parquet_path=str(final_path),
            )
            session.add(es)
            await session.flush()

        # Update audio file metadata if not set
        if af.duration_seconds is None:
            audio_data, sr = decode_audio(audio_path)
            af.duration_seconds = len(audio_data) / sr
            af.sample_rate_original = sr
            await session.flush()

        await complete_processing_job(session, job.id, warning_message=warning)

    except Exception as e:
        logger.exception(f"Processing job {job.id} failed")
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_processing_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark processing job as failed")


def _process_audio(
    audio_path: Path,
    job: ProcessingJob,
    model: EmbeddingModel,
    final_path: Path,
    input_format: str = "spectrogram",
) -> int:
    """CPU-bound audio processing (runs in thread). Returns number of embeddings written."""
    import json as _json
    import time

    # Decode + resample
    t0 = time.monotonic()
    audio, sr = decode_audio(audio_path)
    audio = resample(audio, sr, job.target_sample_rate)
    t_decode = time.monotonic() - t0

    # Write embeddings incrementally
    writer = IncrementalParquetWriter(
        final_path, vector_dim=model.vector_dim, batch_size=50
    )

    # Parse feature_config for normalization setting
    feature_config = _json.loads(job.feature_config) if job.feature_config else {}
    normalization = feature_config.get("normalization", "per_window_max")

    window_samples = int(job.target_sample_rate * job.window_size_seconds)
    if len(audio) < window_samples:
        logger.warning(
            "Audio too short for window size (%.3fs < %.1fs), producing 0 embeddings for job %s",
            len(audio) / job.target_sample_rate,
            job.window_size_seconds,
            job.id,
        )

    t_features = 0.0
    t_inference = 0.0

    # Phase 1: Collect all windows
    raw_windows: list[np.ndarray] = []
    for window in slice_windows(audio, job.target_sample_rate, job.window_size_seconds):
        raw_windows.append(window)

    n_windows = len(raw_windows)

    if raw_windows:
        # Phase 2: Feature extraction (batch for spectrogram, pass-through for waveform)
        if input_format == "waveform":
            batch_items: list[np.ndarray] = raw_windows
        else:
            t0 = time.monotonic()
            batch_items = extract_logmel_batch(
                raw_windows,
                job.target_sample_rate,
                n_mels=128,
                hop_length=1252,
                target_frames=128,
                normalization=normalization,
            )
            t_features = time.monotonic() - t0

        # Phase 3: Batch embed (groups of 64 — optimal for TFLite on M-series)
        batch_size = 64
        for i in range(0, len(batch_items), batch_size):
            batch = np.stack(batch_items[i : i + batch_size])
            t0 = time.monotonic()
            embeddings = model.embed(batch)
            t_inference += time.monotonic() - t0
            for emb in embeddings:
                writer.add(emb)

    writer.close()

    logger.info(
        "Processing timing: decode=%.3fs, features=%.3fs (%d windows), inference=%.3fs",
        t_decode,
        t_features,
        n_windows,
        t_inference,
    )

    return writer.total_rows
