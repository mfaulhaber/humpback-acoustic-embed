import os
import shutil
from pathlib import Path


def audio_raw_dir(storage_root: Path, audio_file_id: str) -> Path:
    return storage_root / "audio" / "raw" / audio_file_id


def embedding_dir(storage_root: Path, model_version: str, audio_file_id: str) -> Path:
    return storage_root / "embeddings" / model_version / audio_file_id


def embedding_path(
    storage_root: Path, model_version: str, audio_file_id: str, encoding_signature: str
) -> Path:
    return embedding_dir(storage_root, model_version, audio_file_id) / f"{encoding_signature}.parquet"


def embedding_tmp_path(
    storage_root: Path, model_version: str, audio_file_id: str, encoding_signature: str
) -> Path:
    return embedding_dir(storage_root, model_version, audio_file_id) / f"{encoding_signature}.tmp.parquet"


def cluster_dir(storage_root: Path, clustering_job_id: str) -> Path:
    return storage_root / "clusters" / clustering_job_id


def atomic_rename(src: Path, dst: Path) -> None:
    """Move src to dst atomically (same filesystem). Creates parent dirs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
