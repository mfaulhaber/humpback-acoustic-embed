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
from humpback.processing.features import extract_logmel
from humpback.processing.inference import EmbeddingModel, FakeTF2Model, FakeTFLiteModel
from humpback.processing.windowing import slice_windows
from humpback.services.model_registry_service import get_model_by_name
from humpback.storage import audio_raw_dir, embedding_path, ensure_dir
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
    input_format = (
        model_config.input_format if model_config else "spectrogram"
    )
    model_type = model_config.model_type if model_config else "tflite"

    if cache_key in _model_cache:
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

            model = TF2SavedModel(model_config.path, model_config.vector_dim)
        else:
            from humpback.processing.inference import TFLiteModel

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
            logger.info(f"Embedding set already exists for signature {job.encoding_signature}, skipping")
            await complete_processing_job(session, job.id)
            return

        # Load audio file info
        af_result = await session.execute(
            select(AudioFile).where(AudioFile.id == job.audio_file_id)
        )
        af = af_result.scalar_one()

        # Find the audio file on disk
        raw_dir = audio_raw_dir(settings.storage_root, af.id)
        audio_files = list(raw_dir.glob("original.*"))
        if not audio_files:
            raise FileNotFoundError(f"No audio file found in {raw_dir}")
        audio_path = audio_files[0]

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

        await asyncio.to_thread(
            _process_audio, audio_path, job, model, final_path, input_format
        )

        # Create EmbeddingSet record
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
            audio, sr = decode_audio(audio_path)
            af.duration_seconds = len(audio) / sr
            af.sample_rate_original = sr

        await complete_processing_job(session, job.id)

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
) -> None:
    """CPU-bound audio processing (runs in thread)."""
    # Decode + resample
    audio, sr = decode_audio(audio_path)
    audio = resample(audio, sr, job.target_sample_rate)

    # Write embeddings incrementally
    writer = IncrementalParquetWriter(
        final_path, vector_dim=model.vector_dim, batch_size=50
    )

    batch_items = []
    batch_size = 32

    for window in slice_windows(audio, job.target_sample_rate, job.window_size_seconds):
        if input_format == "waveform":
            # Feed raw audio directly (TF2 SavedModel path)
            batch_items.append(window)
        else:
            # Extract spectrogram (TFLite path)
            spec = extract_logmel(
                window,
                job.target_sample_rate,
                n_mels=128,
                hop_length=1252,
                target_frames=128,
            )
            batch_items.append(spec)

        if len(batch_items) >= batch_size:
            batch = np.stack(batch_items)
            embeddings = model.embed(batch)
            for emb in embeddings:
                writer.add(emb)
            batch_items.clear()

    # Process remaining windows
    if batch_items:
        batch = np.stack(batch_items)
        embeddings = model.embed(batch)
        for emb in embeddings:
            writer.add(emb)

    writer.close()
