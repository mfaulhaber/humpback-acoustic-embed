import asyncio
import io
import json
import struct
from pathlib import Path

import numpy as np
import re

from fastapi import APIRouter, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.processing import EmbeddingSet
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.embeddings import read_embeddings
from humpback.processing.features import extract_logmel
from humpback.processing.windowing import slice_windows, count_windows
from humpback.schemas.audio import (
    AudioFileOut,
    AudioMetadataIn,
    AudioMetadataOut,
    EmbeddingSimilarityOut,
    SpectrogramOut,
)
from humpback.services import audio_service
from humpback.storage import audio_raw_dir

router = APIRouter(prefix="/audio", tags=["audio"])


def _audio_to_out(af) -> AudioFileOut:
    meta = None
    if af.metadata_:
        m = af.metadata_
        meta = AudioMetadataOut(
            id=m.id,
            audio_file_id=m.audio_file_id,
            tag_data=json.loads(m.tag_data) if m.tag_data else None,
            visual_observations=json.loads(m.visual_observations) if m.visual_observations else None,
            group_composition=json.loads(m.group_composition) if m.group_composition else None,
            prey_density_proxy=json.loads(m.prey_density_proxy) if m.prey_density_proxy else None,
        )
    return AudioFileOut(
        id=af.id,
        filename=af.filename,
        folder_path=af.folder_path,
        checksum_sha256=af.checksum_sha256,
        duration_seconds=af.duration_seconds,
        sample_rate_original=af.sample_rate_original,
        created_at=af.created_at,
        metadata=meta,
    )


def _normalize_folder_path(raw: str) -> str:
    """Strip leading/trailing slashes, collapse doubles, normalize."""
    path = raw.strip()
    path = re.sub(r"[\\/]+", "/", path)  # normalize separators
    path = path.strip("/")
    return path


@router.post("/upload", status_code=201)
async def upload_audio(
    file: UploadFile,
    session: SessionDep,
    settings: SettingsDep,
    folder_path: str = Form(default=""),
) -> AudioFileOut:
    data = await file.read()
    normalized_path = _normalize_folder_path(folder_path)
    af, created = await audio_service.upload_audio(
        session, settings.storage_root, file.filename or "unknown.wav", data,
        folder_path=normalized_path,
    )
    return _audio_to_out(af)


@router.get("/")
async def list_audio(session: SessionDep) -> list[AudioFileOut]:
    files = await audio_service.list_audio(session)
    return [_audio_to_out(af) for af in files]


@router.get("/{audio_id}")
async def get_audio(audio_id: str, session: SessionDep) -> AudioFileOut:
    af = await audio_service.get_audio(session, audio_id)
    if af is None:
        raise HTTPException(404, "Audio file not found")
    return _audio_to_out(af)


@router.put("/{audio_id}/metadata")
async def update_metadata(
    audio_id: str,
    body: AudioMetadataIn,
    session: SessionDep,
) -> AudioMetadataOut:
    meta = await audio_service.update_metadata(
        session,
        audio_id,
        tag_data=body.tag_data,
        visual_observations=body.visual_observations,
        group_composition=body.group_composition,
        prey_density_proxy=body.prey_density_proxy,
    )
    if meta is None:
        raise HTTPException(404, "Audio file not found")
    return AudioMetadataOut(
        id=meta.id,
        audio_file_id=meta.audio_file_id,
        tag_data=json.loads(meta.tag_data) if meta.tag_data else None,
        visual_observations=json.loads(meta.visual_observations) if meta.visual_observations else None,
        group_composition=json.loads(meta.group_composition) if meta.group_composition else None,
        prey_density_proxy=json.loads(meta.prey_density_proxy) if meta.prey_density_proxy else None,
    )


@router.get("/{audio_id}/download")
async def download_audio(
    audio_id: str,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
):
    af = await audio_service.get_audio(session, audio_id)
    if af is None:
        raise HTTPException(404, "Audio file not found")
    suffix = Path(af.filename).suffix or ".wav"
    file_path = audio_raw_dir(settings.storage_root, af.id) / f"original{suffix}"
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found on disk")
    media_types = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
    media_type = media_types.get(suffix.lower(), "application/octet-stream")
    file_size = file_path.stat().st_size

    range_header = request.headers.get("range")
    if range_header:
        range_spec = range_header.strip().lower().removeprefix("bytes=")
        parts = range_spec.split("-", 1)
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(length)

        return Response(
            content=data,
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(length),
                "Accept-Ranges": "bytes",
            },
        )

    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@router.get("/{audio_id}/window")
async def get_audio_window(
    audio_id: str,
    session: SessionDep,
    settings: SettingsDep,
    start_seconds: float = Query(..., ge=0),
    duration_seconds: float = Query(..., gt=0),
):
    """Return a WAV segment of the audio file for the given time range."""
    af = await audio_service.get_audio(session, audio_id)
    if af is None:
        raise HTTPException(404, "Audio file not found")
    suffix = Path(af.filename).suffix or ".wav"
    file_path = audio_raw_dir(settings.storage_root, af.id) / f"original{suffix}"
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found on disk")

    audio, sr = decode_audio(file_path)

    start_sample = int(start_seconds * sr)
    end_sample = int((start_seconds + duration_seconds) * sr)
    start_sample = min(start_sample, len(audio))
    end_sample = min(end_sample, len(audio))
    segment = audio[start_sample:end_sample]

    # Encode as 16-bit PCM WAV in memory
    import numpy as np

    pcm = (segment * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    n_samples = len(pcm)
    data_size = n_samples * 2  # 16-bit = 2 bytes per sample
    # Write WAV header manually for single-channel 16-bit PCM
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())

    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={"Content-Length": str(buf.tell())},
    )


def _compute_spectrogram(
    file_path: Path,
    window_index: int,
    window_size_seconds: float,
    target_sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_frames: int,
) -> tuple[np.ndarray, int, int, list[float], list[float]]:
    """CPU-bound spectrogram computation. Returns (spectrogram, sr, total_windows, y_hz, x_seconds)."""
    import librosa

    audio, sr = decode_audio(file_path)
    audio = resample(audio, sr, target_sample_rate)
    n_samples = len(audio)
    total = count_windows(n_samples, target_sample_rate, window_size_seconds)
    if window_index >= total:
        raise ValueError(f"window_index {window_index} >= total windows {total}")

    # Extract the raw window (without zero-padding short final windows)
    window_samples = int(target_sample_rate * window_size_seconds)
    start = window_index * window_samples
    end = min(start + window_samples, n_samples)
    raw_window = audio[start:end]

    # For the spectrogram visualization, only pad to target_frames if the
    # window is full-length; short final windows use their natural frame count
    is_short = len(raw_window) < window_samples
    tf = None if is_short else target_frames

    spec = extract_logmel(
        raw_window, target_sample_rate,
        n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, target_frames=tf,
    )
    # Compute mel center frequencies in Hz for y-axis
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels + 2, fmin=0, fmax=target_sample_rate / 2)
    y_hz = mel_frequencies[1:-1].tolist()
    # Compute time axis in seconds for x-axis
    n_frames = spec.shape[1]
    time_offset = window_index * window_size_seconds
    x_seconds = [time_offset + (j * hop_length / target_sample_rate) for j in range(n_frames)]
    return spec, target_sample_rate, total, y_hz, x_seconds


@router.get("/{audio_id}/spectrogram")
async def get_spectrogram(
    audio_id: str,
    session: SessionDep,
    settings: SettingsDep,
    window_index: int = Query(..., ge=0),
    window_size_seconds: float = Query(5.0, gt=0),
    target_sample_rate: int = Query(32000, gt=0),
    n_mels: int = Query(128, gt=0),
    n_fft: int = Query(2048, gt=0),
    hop_length: int = Query(1252, gt=0),
    target_frames: int = Query(128, gt=0),
) -> SpectrogramOut:
    """Return log-mel spectrogram for a single window of the audio file."""
    af = await audio_service.get_audio(session, audio_id)
    if af is None:
        raise HTTPException(404, "Audio file not found")
    suffix = Path(af.filename).suffix or ".wav"
    file_path = audio_raw_dir(settings.storage_root, af.id) / f"original{suffix}"
    if not file_path.exists():
        raise HTTPException(404, "Audio file not found on disk")

    try:
        spec, sr, total, y_hz, x_seconds = await asyncio.to_thread(
            _compute_spectrogram, file_path, window_index,
            window_size_seconds, target_sample_rate,
            n_mels, n_fft, hop_length, target_frames,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return SpectrogramOut(
        window_index=window_index,
        sample_rate=sr,
        window_size_seconds=window_size_seconds,
        shape=list(spec.shape),
        data=spec.tolist(),
        total_windows=total,
        min_db=float(spec.min()),
        max_db=float(spec.max()),
        y_axis_hz=y_hz,
        x_axis_seconds=x_seconds,
    )


def _cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute mean-centered pairwise cosine similarity between all rows.

    Subtracting the global mean embedding before normalisation removes
    the shared "baseline" direction that makes raw cosine similarity
    uninformatively high for non-negative (post-ReLU) embeddings.
    """
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # avoid division by zero
    normed = centered / norms
    return normed @ normed.T


@router.get("/{audio_id}/embeddings")
async def get_embeddings(
    audio_id: str,
    session: SessionDep,
    settings: SettingsDep,
    embedding_set_id: str = Query(...),
) -> EmbeddingSimilarityOut:
    """Return cosine similarity matrix between all window embeddings."""
    af = await audio_service.get_audio(session, audio_id)
    if af is None:
        raise HTTPException(404, "Audio file not found")

    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id == embedding_set_id)
    )
    es = result.scalar_one_or_none()
    if es is None or es.audio_file_id != audio_id:
        raise HTTPException(404, "Embedding set not found for this audio")

    parquet_path = Path(es.parquet_path)

    if not parquet_path.exists():
        raise HTTPException(404, "Embedding parquet file not found on disk")

    row_indices, embeddings = await asyncio.to_thread(read_embeddings, parquet_path)
    sim_matrix = _cosine_similarity_matrix(embeddings)

    return EmbeddingSimilarityOut(
        embedding_set_id=es.id,
        vector_dim=es.vector_dim,
        num_windows=len(row_indices),
        row_indices=row_indices.tolist(),
        similarity_matrix=sim_matrix.tolist(),
    )
