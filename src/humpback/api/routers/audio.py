import io
import json
import struct
from pathlib import Path

import re

from fastapi import APIRouter, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from humpback.api.deps import SessionDep, SettingsDep
from humpback.processing.audio_io import decode_audio
from humpback.schemas.audio import AudioFileOut, AudioMetadataIn, AudioMetadataOut
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
    media_types = {".wav": "audio/wav", ".mp3": "audio/mpeg"}
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
