import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from humpback.models.audio import AudioFile, AudioMetadata
from humpback.models.clustering import ClusteringJob
from humpback.models.processing import EmbeddingSet, ProcessingJob
from humpback.schemas.audio import (
    AffectedClusteringJob,
    FolderDeletePreview,
    FolderDeleteResult,
)
from humpback.storage import audio_raw_dir, cluster_dir, ensure_dir


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


async def preview_folder_delete(
    session: AsyncSession,
    folder_path: str,
) -> FolderDeletePreview:
    """Build a preview of what would be deleted for a top-level folder."""
    # Find all audio files in this folder (exact match or sub-folders)
    result = await session.execute(
        select(AudioFile).where(
            or_(
                AudioFile.folder_path == folder_path,
                AudioFile.folder_path.like(f"{folder_path}/%"),
            )
        )
    )
    audio_files = list(result.scalars().all())
    audio_ids = [af.id for af in audio_files]

    if not audio_ids:
        return FolderDeletePreview(
            folder_path=folder_path,
            audio_file_count=0,
            embedding_set_count=0,
            processing_job_count=0,
            affected_clustering_jobs=[],
            has_clustering_conflicts=False,
        )

    # Find related embedding sets
    es_result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.audio_file_id.in_(audio_ids))
    )
    embedding_sets = list(es_result.scalars().all())
    es_ids = {es.id for es in embedding_sets}

    # Find related processing jobs
    pj_result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.audio_file_id.in_(audio_ids))
    )
    processing_jobs = list(pj_result.scalars().all())

    # Find clustering jobs that reference any of these embedding sets
    cj_result = await session.execute(select(ClusteringJob))
    all_clustering_jobs = list(cj_result.scalars().all())

    affected: list[AffectedClusteringJob] = []
    for cj in all_clustering_jobs:
        cj_es_ids = set(json.loads(cj.embedding_set_ids))
        overlap = cj_es_ids & es_ids
        if overlap:
            affected.append(
                AffectedClusteringJob(
                    id=cj.id,
                    status=cj.status,
                    overlapping_embedding_set_ids=sorted(overlap),
                )
            )

    return FolderDeletePreview(
        folder_path=folder_path,
        audio_file_count=len(audio_files),
        embedding_set_count=len(embedding_sets),
        processing_job_count=len(processing_jobs),
        affected_clustering_jobs=affected,
        has_clustering_conflicts=len(affected) > 0,
    )


async def execute_folder_delete(
    session: AsyncSession,
    storage_root: Path,
    folder_path: str,
    confirm_clustering_delete: bool = False,
) -> FolderDeleteResult:
    """Delete a folder and all related records + files."""
    preview = await preview_folder_delete(session, folder_path)

    if preview.has_clustering_conflicts and not confirm_clustering_delete:
        raise ValueError(
            f"Folder has {len(preview.affected_clustering_jobs)} affected clustering "
            f"job(s). Set confirm_clustering_delete=true to proceed."
        )

    # Gather IDs
    af_result = await session.execute(
        select(AudioFile).where(
            or_(
                AudioFile.folder_path == folder_path,
                AudioFile.folder_path.like(f"{folder_path}/%"),
            )
        )
    )
    audio_files = list(af_result.scalars().all())
    audio_ids = [af.id for af in audio_files]

    es_result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.audio_file_id.in_(audio_ids))
    )
    embedding_sets = list(es_result.scalars().all())
    es_ids = {es.id for es in embedding_sets}

    pj_result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.audio_file_id.in_(audio_ids))
    )
    processing_jobs = list(pj_result.scalars().all())

    # 1. Delete affected clustering jobs (cascade handles clusters + assignments)
    cj_result = await session.execute(select(ClusteringJob))
    all_cjs = list(cj_result.scalars().all())
    deleted_cj_count = 0
    for cj in all_cjs:
        cj_es_ids = set(json.loads(cj.embedding_set_ids))
        if cj_es_ids & es_ids:
            # Remove cluster output dir from disk
            cj_dir = cluster_dir(storage_root, cj.id)
            if cj_dir.exists():
                shutil.rmtree(cj_dir)
            await session.delete(cj)
            deleted_cj_count += 1

    # 2. Delete processing jobs
    for pj in processing_jobs:
        await session.delete(pj)

    # 3. Delete embedding sets + parquet files
    for es in embedding_sets:
        parquet = Path(es.parquet_path)
        if parquet.exists():
            parquet.unlink()
        # Clean up parent dir if empty
        if parquet.parent.exists() and not any(parquet.parent.iterdir()):
            parquet.parent.rmdir()
        await session.delete(es)

    # 4. Delete audio files (cascade handles AudioMetadata) + raw audio dirs
    for af in audio_files:
        raw_dir = audio_raw_dir(storage_root, af.id)
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        await session.delete(af)

    await session.commit()

    return FolderDeleteResult(
        folder_path=folder_path,
        deleted_audio_files=len(audio_files),
        deleted_embedding_sets=len(embedding_sets),
        deleted_processing_jobs=len(processing_jobs),
        deleted_clustering_jobs=deleted_cj_count,
    )
