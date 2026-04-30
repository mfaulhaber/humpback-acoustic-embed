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


def cleanup_manifests_dir(storage_root: Path) -> Path:
    return storage_root / "cleanup-manifests"


def path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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


def detection_embeddings_path(
    storage_root: Path, detection_job_id: str, model_version: str
) -> Path:
    return (
        detection_dir(storage_root, detection_job_id)
        / "embeddings"
        / model_version
        / "detection_embeddings.parquet"
    )


def training_dataset_dir(storage_root: Path, dataset_id: str) -> Path:
    return storage_root / "training_datasets" / dataset_id


def training_dataset_parquet_path(storage_root: Path, dataset_id: str) -> Path:
    return training_dataset_dir(storage_root, dataset_id) / "embeddings.parquet"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def continuous_embedding_dir(storage_root: Path, job_id: str) -> Path:
    return storage_root / "continuous_embeddings" / job_id


def continuous_embedding_parquet_path(storage_root: Path, job_id: str) -> Path:
    return continuous_embedding_dir(storage_root, job_id) / "embeddings.parquet"


def continuous_embedding_manifest_path(storage_root: Path, job_id: str) -> Path:
    return continuous_embedding_dir(storage_root, job_id) / "manifest.json"


def hmm_sequence_dir(storage_root: Path, job_id: str) -> Path:
    return storage_root / "hmm_sequences" / job_id


def hmm_sequence_states_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "states.parquet"


def hmm_sequence_pca_model_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "pca_model.joblib"


def hmm_sequence_hmm_model_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "hmm_model.joblib"


def hmm_sequence_transition_matrix_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "transition_matrix.npy"


def hmm_sequence_summary_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "state_summary.json"


def hmm_sequence_training_log_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "training_log.json"


def hmm_sequence_overlay_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "pca_overlay.parquet"


def hmm_sequence_label_distribution_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "label_distribution.json"


def hmm_sequence_exemplars_dir(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_dir(storage_root, job_id) / "exemplars"


def hmm_sequence_exemplars_path(storage_root: Path, job_id: str) -> Path:
    return hmm_sequence_exemplars_dir(storage_root, job_id) / "exemplars.json"


def motif_extraction_dir(storage_root: Path, job_id: str) -> Path:
    return storage_root / "motif_extractions" / job_id


def motif_extraction_manifest_path(storage_root: Path, job_id: str) -> Path:
    return motif_extraction_dir(storage_root, job_id) / "manifest.json"


def motif_extraction_motifs_path(storage_root: Path, job_id: str) -> Path:
    return motif_extraction_dir(storage_root, job_id) / "motifs.parquet"


def motif_extraction_occurrences_path(storage_root: Path, job_id: str) -> Path:
    return motif_extraction_dir(storage_root, job_id) / "occurrences.parquet"


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
