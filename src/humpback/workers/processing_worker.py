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
from humpback.processing.inference import EmbeddingModel, FakeTFLiteModel
from humpback.processing.windowing import slice_windows
from humpback.storage import audio_raw_dir, embedding_path, ensure_dir
from humpback.workers.queue import complete_processing_job, fail_processing_job

logger = logging.getLogger(__name__)


def get_model(settings: Settings) -> EmbeddingModel:
    if settings.use_real_model:
        from humpback.processing.inference import TFLiteModel

        return TFLiteModel(settings.model_path, settings.vector_dim)
    return FakeTFLiteModel(settings.vector_dim)


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
        if model is None:
            model = get_model(settings)

        final_path = embedding_path(
            settings.storage_root,
            job.model_version,
            job.audio_file_id,
            job.encoding_signature,
        )

        await asyncio.to_thread(
            _process_audio, audio_path, job, model, final_path
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
) -> None:
    """CPU-bound audio processing (runs in thread)."""
    # Decode + resample
    audio, sr = decode_audio(audio_path)
    audio = resample(audio, sr, job.target_sample_rate)

    # Write embeddings incrementally
    writer = IncrementalParquetWriter(
        final_path, vector_dim=model.vector_dim, batch_size=50
    )

    batch_specs = []
    batch_size = 32

    for window in slice_windows(audio, job.target_sample_rate, job.window_size_seconds):
        spec = extract_logmel(
            window,
            job.target_sample_rate,
            n_mels=128,
            hop_length=1252,
            target_frames=128,
        )
        batch_specs.append(spec)
        if len(batch_specs) >= batch_size:
            batch = np.stack(batch_specs)
            embeddings = model.embed(batch)
            for emb in embeddings:
                writer.add(emb)
            batch_specs.clear()

    # Process remaining windows
    if batch_specs:
        batch = np.stack(batch_specs)
        embeddings = model.embed(batch)
        for emb in embeddings:
            writer.add(emb)

    writer.close()
