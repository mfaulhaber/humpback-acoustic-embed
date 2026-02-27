import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from humpback.models.audio import AudioFile, AudioMetadata
from humpback.storage import audio_raw_dir, ensure_dir


async def upload_audio(
    session: AsyncSession,
    storage_root: Path,
    filename: str,
    file_data: bytes,
    folder_path: str = "",
) -> tuple[AudioFile, bool]:
    """Upload an audio file. Returns (AudioFile, created). Deduplicates by SHA-256."""
    checksum = hashlib.sha256(file_data).hexdigest()

    existing = await session.execute(
        select(AudioFile)
        .options(selectinload(AudioFile.metadata_))
        .where(AudioFile.checksum_sha256 == checksum)
    )
    if row := existing.scalar_one_or_none():
        return row, False

    af = AudioFile(filename=filename, folder_path=folder_path, checksum_sha256=checksum)
    session.add(af)
    await session.flush()

    # Save file to storage
    raw_dir = ensure_dir(audio_raw_dir(storage_root, af.id))
    suffix = Path(filename).suffix or ".wav"
    dest = raw_dir / f"original{suffix}"
    dest.write_bytes(file_data)

    await session.commit()

    # Re-fetch with eager load to avoid lazy-load issues in async context
    result = await session.execute(
        select(AudioFile)
        .options(selectinload(AudioFile.metadata_))
        .where(AudioFile.id == af.id)
    )
    return result.scalar_one(), True


async def list_audio(session: AsyncSession) -> list[AudioFile]:
    result = await session.execute(
        select(AudioFile).options(selectinload(AudioFile.metadata_)).order_by(AudioFile.created_at.desc())
    )
    return list(result.scalars().all())


async def get_audio(session: AsyncSession, audio_id: str) -> Optional[AudioFile]:
    result = await session.execute(
        select(AudioFile)
        .options(selectinload(AudioFile.metadata_))
        .where(AudioFile.id == audio_id)
    )
    return result.scalar_one_or_none()


async def update_metadata(
    session: AsyncSession,
    audio_id: str,
    tag_data: Optional[dict] = None,
    visual_observations: Optional[dict] = None,
    group_composition: Optional[dict] = None,
    prey_density_proxy: Optional[dict] = None,
) -> Optional[AudioMetadata]:
    af = await get_audio(session, audio_id)
    if af is None:
        return None

    meta = af.metadata_
    if meta is None:
        meta = AudioMetadata(audio_file_id=audio_id)
        session.add(meta)

    if tag_data is not None:
        meta.tag_data = json.dumps(tag_data)
    if visual_observations is not None:
        meta.visual_observations = json.dumps(visual_observations)
    if group_composition is not None:
        meta.group_composition = json.dumps(group_composition)
    if prey_density_proxy is not None:
        meta.prey_density_proxy = json.dumps(prey_density_proxy)

    await session.commit()
    return meta
