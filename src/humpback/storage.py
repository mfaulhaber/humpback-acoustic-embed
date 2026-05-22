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


def event_encoder_dir(storage_root: Path, job_id: str) -> Path:
    return storage_root / "event_encoders" / job_id


def event_encoder_manifest_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "manifest.json"


def event_encoder_report_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "report.json"


def event_encoder_vectors_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "event_vectors.parquet"


def event_encoder_tokens_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "event_tokens.parquet"


def event_encoder_sequences_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "token_sequences.parquet"


def event_encoder_preprocess_path(storage_root: Path, job_id: str) -> Path:
    return event_encoder_dir(storage_root, job_id) / "preprocess.joblib"


def event_encoder_kmeans_path(storage_root: Path, job_id: str, k: int) -> Path:
    return event_encoder_dir(storage_root, job_id) / f"kmeans_k{k}.joblib"


def event_encoder_notes_path(
    storage_root: Path, job_id: str, extractor_version: str
) -> Path:
    """Path to the Piano Roll Notes parquet sidecar for an Event Encoder job."""
    return (
        event_encoder_dir(storage_root, job_id)
        / f"event_notes_{extractor_version}.parquet"
    )


def event_encoder_ridges_path(
    storage_root: Path, job_id: str, tokenizer_version: str
) -> Path:
    """Path to the per-event ridge sidecar parquet for an Event Encoder job.

    Written by the encoder worker (ADR-069 / spec §3.1) so the Piano Roll
    Notes v3 extractor consumes the same per-frame ridge contour the
    encoder's descriptor summaries were derived from.
    """
    return (
        event_encoder_dir(storage_root, job_id)
        / f"event_ridges_{tokenizer_version}.parquet"
    )


def event_encoder_note_contours_path(
    storage_root: Path, job_id: str, extractor_version: str
) -> Path:
    """Path to the per-frame note contour sidecar (Piano Roll Notes v3+).

    One row per (note_uid, frame_index) per ADR-069 §6.3 — feeds the MPE
    pitch-bend stream and the frontend ribbon renderer. Co-located with
    the ``event_notes_{version}.parquet`` sidecar.
    """
    return (
        event_encoder_dir(storage_root, job_id)
        / f"event_note_contours_{extractor_version}.parquet"
    )


def exports_root(storage_root: Path) -> Path:
    """Root directory for export artifacts (e.g. MIDI files)."""
    return storage_root / "exports"


def event_encoder_midi_export_path(
    storage_root: Path, job_id: str, extractor_version: str
) -> Path:
    """Path to the Piano Roll Notes MIDI export for an Event Encoder job."""
    return (
        exports_root(storage_root)
        / "event_encoders"
        / job_id
        / f"notes_{extractor_version}.mid"
    )


def event_encoder_audio_export_path(
    storage_root: Path, job_id: str, extractor_version: str
) -> Path:
    """Path to the FLAC clip co-exported with the Piano Roll MIDI file.

    The FLAC covers the same UTC window as the sibling ``.mid`` artifact
    produced by ``event_encoder_midi_export_path``.
    """
    return (
        exports_root(storage_root)
        / "event_encoders"
        / job_id
        / f"audio_{extractor_version}.flac"
    )


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
