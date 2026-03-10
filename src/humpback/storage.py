import os
from pathlib import Path


def audio_raw_dir(storage_root: Path, audio_file_id: str) -> Path:
    return storage_root / "audio" / "raw" / audio_file_id


def embedding_dir(storage_root: Path, model_version: str, audio_file_id: str) -> Path:
    return storage_root / "embeddings" / model_version / audio_file_id


def embedding_path(
    storage_root: Path, model_version: str, audio_file_id: str, encoding_signature: str
) -> Path:
    return (
        embedding_dir(storage_root, model_version, audio_file_id)
        / f"{encoding_signature}.parquet"
    )


def embedding_tmp_path(
    storage_root: Path, model_version: str, audio_file_id: str, encoding_signature: str
) -> Path:
    return (
        embedding_dir(storage_root, model_version, audio_file_id)
        / f"{encoding_signature}.tmp.parquet"
    )


def cluster_dir(storage_root: Path, clustering_job_id: str) -> Path:
    return storage_root / "clusters" / clustering_job_id


def atomic_rename(src: Path, dst: Path) -> None:
    """Move src to dst atomically (same filesystem). Creates parent dirs."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)


def classifier_dir(storage_root: Path, classifier_model_id: str) -> Path:
    return storage_root / "classifiers" / classifier_model_id


def detection_dir(storage_root: Path, detection_job_id: str) -> Path:
    return storage_root / "detections" / detection_job_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_audio_path(af, storage_root: Path) -> Path:
    """Return the on-disk path for an AudioFile.

    If the file was imported from a source folder, read from there directly.
    Otherwise fall back to the uploaded copy in audio/raw/.
    """
    if af.source_folder:
        return Path(af.source_folder) / af.filename
    # Uploaded file — find original.* in the raw directory
    raw_dir = audio_raw_dir(storage_root, af.id)
    candidates = list(raw_dir.glob("original.*"))
    if candidates:
        return candidates[0]
    # Fall back to suffix from filename
    suffix = Path(af.filename).suffix or ".wav"
    return raw_dir / f"original{suffix}"
