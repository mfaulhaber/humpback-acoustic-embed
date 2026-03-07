"""Train a binary whale vocalization classifier on embeddings."""

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

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

    classifier_type = parameters.get("classifier_type", "logistic_regression")
    l2_normalize = parameters.get("l2_normalize", False)
    class_weight = parameters.get("class_weight", "balanced")

    # Build pipeline steps
    steps: list[tuple] = []
    if l2_normalize:
        steps.append(("l2_norm", Normalizer(norm="l2")))
    steps.append(("scaler", StandardScaler()))

    if classifier_type == "mlp":
        steps.append(("classifier", MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=parameters.get("max_iter", 500),
            early_stopping=True,
            random_state=42,
        )))
    else:
        steps.append(("classifier", LogisticRegression(
            solver=parameters.get("solver", "lbfgs"),
            max_iter=parameters.get("max_iter", 1000),
            C=parameters.get("C", 1.0),
            class_weight=class_weight,
        )))

    pipeline = Pipeline(steps)

    # Cross-validation for honest estimates.
    # n_splits must not exceed the size of the minority class so that
    # every fold contains at least one sample from each class.
    minority_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = min(5, minority_count)
    n_splits = max(2, n_splits)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["accuracy", "roc_auc", "precision", "recall", "f1"],
        return_train_score=False,
        error_score="raise",
    )

    # Final fit on all data
    pipeline.fit(X, y)

    n_pos = len(positive_embeddings)
    n_neg = len(negative_embeddings)
    balance_ratio = round(n_pos / n_neg, 2) if n_neg > 0 else float("inf")

    # Decision boundary diagnostics on training set
    if hasattr(pipeline.named_steps["classifier"], "decision_function"):
        scores = pipeline.decision_function(X)
    else:
        scores = pipeline.predict_proba(X)[:, 1]

    pos_scores = scores[y == 1]
    neg_scores = scores[y == 0]
    pos_mean = float(np.mean(pos_scores))
    neg_mean = float(np.mean(neg_scores))
    pooled_std = float(np.sqrt(
        (np.var(pos_scores) * len(pos_scores) + np.var(neg_scores) * len(neg_scores))
        / (len(pos_scores) + len(neg_scores))
    ))
    score_separation = float((pos_mean - neg_mean) / pooled_std) if pooled_std > 0 else 0.0

    train_preds = pipeline.predict(X)
    cm = confusion_matrix(y, train_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Effective class weights
    clf = pipeline.named_steps["classifier"]
    effective_class_weights = None
    if hasattr(clf, "class_weight") and clf.class_weight == "balanced":
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        effective_class_weights = {"0": round(float(weights[0]), 4), "1": round(float(weights[1]), 4)}

    summary = {
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "balance_ratio": balance_ratio,
        "classifier_type": classifier_type,
        "l2_normalize": l2_normalize,
        "class_weight_strategy": str(class_weight),
        "effective_class_weights": effective_class_weights,
        "cv_accuracy": float(np.nanmean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.nanstd(cv_results["test_accuracy"])),
        "cv_roc_auc": float(np.nanmean(cv_results["test_roc_auc"])),
        "cv_roc_auc_std": float(np.nanstd(cv_results["test_roc_auc"])),
        "cv_precision": float(np.nanmean(cv_results["test_precision"])),
        "cv_precision_std": float(np.nanstd(cv_results["test_precision"])),
        "cv_recall": float(np.nanmean(cv_results["test_recall"])),
        "cv_recall_std": float(np.nanstd(cv_results["test_recall"])),
        "cv_f1": float(np.nanmean(cv_results["test_f1"])),
        "cv_f1_std": float(np.nanstd(cv_results["test_f1"])),
        "n_cv_folds": n_splits,
        "positive_mean_score": pos_mean,
        "negative_mean_score": neg_mean,
        "score_separation": score_separation,
        "train_confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
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
