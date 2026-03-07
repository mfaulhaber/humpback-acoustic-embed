"""Train a binary whale vocalization classifier on embeddings."""

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import slice_windows

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def embed_audio_folder(
    folder: Path,
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
) -> np.ndarray:
    """Recursively scan folder for audio files, embed each, return stacked vectors."""
    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    all_embeddings: list[np.ndarray] = []
    audio_files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {folder}")

    logger.info("Embedding %d audio files from %s", len(audio_files), folder)

    for audio_path in audio_files:
        try:
            audio, sr = decode_audio(audio_path)
            audio = resample(audio, sr, target_sample_rate)

            window_samples = int(target_sample_rate * window_size_seconds)
            if len(audio) < window_samples:
                logger.warning(
                    "Skipping %s: audio too short (%.3fs < %.1fs window)",
                    audio_path.name, len(audio) / target_sample_rate, window_size_seconds,
                )
                continue

            batch_items: list[np.ndarray] = []
            batch_size = 32

            for window in slice_windows(audio, target_sample_rate, window_size_seconds):
                if input_format == "waveform":
                    batch_items.append(window)
                else:
                    spec = extract_logmel(
                        window,
                        target_sample_rate,
                        n_mels=128,
                        hop_length=1252,
                        target_frames=128,
                        normalization=normalization,
                    )
                    batch_items.append(spec)

                if len(batch_items) >= batch_size:
                    batch = np.stack(batch_items)
                    embeddings = model.embed(batch)
                    all_embeddings.append(embeddings)
                    batch_items.clear()

            if batch_items:
                batch = np.stack(batch_items)
                embeddings = model.embed(batch)
                all_embeddings.append(embeddings)

        except Exception:
            logger.warning("Failed to process %s, skipping", audio_path, exc_info=True)
            continue

    if not all_embeddings:
        raise ValueError(f"No embeddings produced from {folder}")

    return np.vstack(all_embeddings)


def train_binary_classifier(
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
    parameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Train StandardScaler + LogisticRegression pipeline.

    Returns (pipeline, summary_dict) where summary includes CV metrics.
    """
    parameters = parameters or {}

    if len(positive_embeddings) < 2:
        raise ValueError(
            f"Need at least 2 positive samples, got {len(positive_embeddings)}"
        )
    if len(negative_embeddings) < 2:
        raise ValueError(
            f"Need at least 2 negative samples, got {len(negative_embeddings)}"
        )

    X = np.vstack([positive_embeddings, negative_embeddings])
    y = np.concatenate([
        np.ones(len(positive_embeddings), dtype=int),
        np.zeros(len(negative_embeddings), dtype=int),
    ])

    class_weight = parameters.get("class_weight", "balanced")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver=parameters.get("solver", "lbfgs"),
            max_iter=parameters.get("max_iter", 1000),
            C=parameters.get("C", 1.0),
            class_weight=class_weight,
        )),
    ])

    # Cross-validation for honest estimates.
    # n_splits must not exceed the size of the minority class so that
    # every fold contains at least one sample from each class.
    minority_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = min(5, minority_count)
    n_splits = max(2, n_splits)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "roc_auc"],
        return_train_score=False,
        error_score="raise",
    )

    # Final fit on all data
    pipeline.fit(X, y)

    n_pos = len(positive_embeddings)
    n_neg = len(negative_embeddings)
    balance_ratio = round(n_pos / n_neg, 2) if n_neg > 0 else float("inf")

    summary = {
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "balance_ratio": balance_ratio,
        "cv_accuracy": float(np.nanmean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.nanstd(cv_results["test_accuracy"])),
        "cv_roc_auc": float(np.nanmean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.nanstd(cv_results["test_roc_auc"])),
        "n_cv_folds": n_splits,
    }

    if balance_ratio > 3.0 or balance_ratio < 1 / 3.0:
        summary["imbalance_warning"] = (
            f"Class imbalance detected: {n_pos} positive vs {n_neg} negative "
            f"(ratio {balance_ratio}:1). Results may be unreliable without "
            f"class_weight='balanced'."
        )
        logger.warning(
            "Class imbalance: %d positive vs %d negative (ratio %.1f:1)",
            n_pos, n_neg, balance_ratio,
        )

    logger.info(
        "Trained classifier: accuracy=%.3f (+-%.3f), AUC=%.3f (+-%.3f)",
        summary["cv_accuracy"], summary["cv_accuracy_std"],
        summary["cv_roc_auc"], summary["cv_roc_auc_std"],
    )

    return pipeline, summary
