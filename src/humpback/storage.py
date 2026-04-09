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


def detection_row_store_path(storage_root: Path, detection_job_id: str) -> Path:
    return detection_dir(storage_root, detection_job_id) / "detection_rows.parquet"


def detection_tsv_path(storage_root: Path, detection_job_id: str) -> Path:
    return detection_dir(storage_root, detection_job_id) / "detections.tsv"


def detection_diagnostics_path(storage_root: Path, detection_job_id: str) -> Path:
    return detection_dir(storage_root, detection_job_id) / "window_diagnostics.parquet"


def label_processing_dir(storage_root: Path, job_id: str) -> Path:
    return storage_root / "label_processing" / job_id


def detection_embeddings_path(storage_root: Path, detection_job_id: str) -> Path:
    return (
        detection_dir(storage_root, detection_job_id) / "detection_embeddings.parquet"
    )


def training_dataset_dir(storage_root: Path, dataset_id: str) -> Path:
    return storage_root / "training_datasets" / dataset_id


def training_dataset_parquet_path(storage_root: Path, dataset_id: str) -> Path:
    return training_dataset_dir(storage_root, dataset_id) / "embeddings.parquet"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def hyperparameter_manifest_dir(storage_root: Path, manifest_id: str) -> Path:
    return storage_root / "hyperparameter" / "manifests" / manifest_id


def hyperparameter_manifest_path(storage_root: Path, manifest_id: str) -> Path:
    return hyperparameter_manifest_dir(storage_root, manifest_id) / "manifest.json"


def hyperparameter_search_results_dir(storage_root: Path, search_id: str) -> Path:
    return storage_root / "hyperparameter" / "searches" / search_id


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
