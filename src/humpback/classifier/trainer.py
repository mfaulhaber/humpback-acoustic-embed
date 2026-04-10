"""Train a binary whale vocalization classifier on embeddings."""

import dataclasses as _dataclasses
import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from humpback.processing.embeddings import read_embeddings
from humpback.processing.audio_io import decode_audio, resample
from humpback.processing.features import extract_logmel_batch
from humpback.processing.inference import EmbeddingModel
from humpback.processing.windowing import (
    format_short_audio_window_message,
    slice_windows,
    window_sample_count,
)

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
    import time

    feature_config = feature_config or {}
    normalization = feature_config.get("normalization", "per_window_max")

    all_embeddings: list[np.ndarray] = []
    audio_files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {folder}")

    logger.info("Embedding %d audio files from %s", len(audio_files), folder)
    t_decode_total = 0.0
    t_features_total = 0.0
    t_inference_total = 0.0
    n_windows_total = 0

    for audio_path in audio_files:
        try:
            t0 = time.monotonic()
            audio, sr = decode_audio(audio_path)
            audio = resample(audio, sr, target_sample_rate)
            t_decode_total += time.monotonic() - t0

            window_samples = window_sample_count(
                target_sample_rate, window_size_seconds
            )
            if len(audio) < window_samples:
                logger.warning(
                    "Skipping %s: audio too short (%s)",
                    audio_path.name,
                    format_short_audio_window_message(
                        len(audio), target_sample_rate, window_size_seconds
                    ),
                )
                continue

            # Phase 1: Collect all windows
            raw_windows: list[np.ndarray] = []
            for window in slice_windows(audio, target_sample_rate, window_size_seconds):
                raw_windows.append(window)

            if not raw_windows:
                continue

            # Phase 2: Feature extraction (batch for spectrogram, pass-through for waveform)
            n_windows_total += len(raw_windows)
            if input_format == "waveform":
                batch_items: list[np.ndarray] = raw_windows
            else:
                t0 = time.monotonic()
                batch_items = extract_logmel_batch(
                    raw_windows,
                    target_sample_rate,
                    n_mels=128,
                    hop_length=1252,
                    target_frames=128,
                    normalization=normalization,
                )
                t_features_total += time.monotonic() - t0

            # Phase 3: Batch embed (groups of 64 — optimal for TFLite on M-series)
            batch_size = 64
            for i in range(0, len(batch_items), batch_size):
                batch = np.stack(batch_items[i : i + batch_size])
                t0 = time.monotonic()
                embeddings = model.embed(batch)
                t_inference_total += time.monotonic() - t0
                all_embeddings.append(embeddings)

        except Exception:
            logger.warning("Failed to process %s, skipping", audio_path, exc_info=True)
            continue

    if not all_embeddings:
        raise ValueError(f"No embeddings produced from {folder}")

    logger.info(
        "Embedding timing: decode=%.3fs, features=%.3fs (%d windows), inference=%.3fs",
        t_decode_total,
        t_features_total,
        n_windows_total,
        t_inference_total,
    )

    return np.vstack(all_embeddings)


def map_autoresearch_config_to_training_parameters(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Map a promotable autoresearch config into trainer parameters."""
    classifier = config.get("classifier", "logreg")
    if classifier == "logreg":
        classifier_type = "logistic_regression"
    elif classifier == "mlp":
        classifier_type = "mlp"
    elif classifier == "linear_svm":
        classifier_type = "linear_svm"
    else:
        raise ValueError(f"Unsupported autoresearch classifier: {classifier!r}")

    feature_norm = config.get("feature_norm", "standard")
    if feature_norm not in {"none", "l2", "standard"}:
        raise ValueError(f"Unsupported autoresearch feature_norm: {feature_norm!r}")

    class_weight_pos = float(config.get("class_weight_pos", 1.0))
    class_weight_neg = float(config.get("class_weight_neg", 1.0))

    parameters: dict[str, Any] = {
        "classifier_type": classifier_type,
        "feature_norm": feature_norm,
        "random_state": int(config.get("seed", 42)),
    }
    # Both logreg and MLP need the class_weight dict; logreg passes it
    # natively, MLP uses it to compute sample_weight in train_binary_classifier.
    parameters["class_weight"] = {0: class_weight_neg, 1: class_weight_pos}

    return parameters


def _load_manifest_examples(manifest_path: Path | str) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        raise ValueError(f"Manifest examples missing or invalid: {path}")
    return [ex for ex in examples if isinstance(ex, dict)]


def _load_manifest_parquet_cache(
    examples: list[dict[str, Any]],
) -> dict[str, tuple[np.ndarray | None, np.ndarray, list[str] | None]]:
    """Load each unique manifest parquet once.

    Cache entries are `(row_indices, embeddings, row_ids)`.
    """
    cache: dict[str, tuple[np.ndarray | None, np.ndarray, list[str] | None]] = {}
    for parquet_path in sorted({str(ex["parquet_path"]) for ex in examples}):
        schema = pq.read_schema(parquet_path)
        col_names = set(schema.names)
        if "row_index" in col_names:
            row_indices, embeddings = read_embeddings(Path(parquet_path))
            cache[parquet_path] = (row_indices, embeddings, None)
            continue

        table = pq.read_table(parquet_path)
        embeddings = np.array(
            [value.as_py() for value in table["embedding"]],
            dtype=np.float32,
        )

        if "row_id" in col_names:
            row_ids = [
                str(row_id or "").strip() for row_id in table["row_id"].to_pylist()
            ]
            cache[parquet_path] = (None, embeddings, row_ids)
            continue

        if "filename" in col_names:
            row_indices = np.arange(len(embeddings), dtype=np.int32)
            cache[parquet_path] = (row_indices, embeddings, None)
            continue

        raise ValueError(
            f"Unknown manifest parquet format in {parquet_path}: columns {sorted(col_names)}"
        )

    return cache


def load_manifest_split_embeddings(
    manifest_path: Path | str,
    *,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load positive/negative embeddings for a manifest split."""
    examples = _load_manifest_examples(manifest_path)
    split_examples = [ex for ex in examples if ex.get("split") == split]
    if not split_examples:
        raise ValueError(
            f"Manifest has no examples for split {split!r}: {manifest_path}"
        )

    cache = _load_manifest_parquet_cache(split_examples)
    row_index_lookups: dict[str, dict[int, int]] = {}
    row_id_lookups: dict[str, dict[str, int]] = {}
    for path, (row_indices, _embeddings, row_ids) in cache.items():
        if row_indices is not None:
            row_index_lookups[path] = {
                int(row_index): idx for idx, row_index in enumerate(row_indices)
            }
        if row_ids is not None:
            row_id_lookups[path] = {
                row_id: idx for idx, row_id in enumerate(row_ids) if row_id
            }

    positive: list[np.ndarray] = []
    negative: list[np.ndarray] = []
    missing_example_ids: list[str] = []

    for example in split_examples:
        path = str(example["parquet_path"])
        row_indices, embeddings, row_ids = cache[path]
        idx: int | None = None

        row_id = str(example.get("row_id") or "").strip()
        if row_id and row_ids is not None:
            idx = row_id_lookups.get(path, {}).get(row_id)
        elif example.get("row_index") is not None and row_indices is not None:
            idx = row_index_lookups.get(path, {}).get(int(example["row_index"]))

        if idx is None:
            missing_example_ids.append(str(example.get("id") or "unknown"))
            continue

        vector = embeddings[idx]
        if int(example.get("label", 0)) == 1:
            positive.append(vector)
        else:
            negative.append(vector)

    if missing_example_ids:
        raise ValueError(
            "Manifest examples missing embeddings for split "
            f"{split!r}: {len(missing_example_ids)} missing"
        )
    if len(positive) < 2:
        raise ValueError(
            f"Need at least 2 positive manifest examples for split {split!r}, got {len(positive)}"
        )
    if len(negative) < 2:
        raise ValueError(
            f"Need at least 2 negative manifest examples for split {split!r}, got {len(negative)}"
        )

    positive_embeddings = np.array(positive, dtype=np.float32)
    negative_embeddings = np.array(negative, dtype=np.float32)
    source_summary = {
        "manifest_path": str(Path(manifest_path)),
        "split": split,
        "example_count": len(split_examples),
        "positive_count": int(positive_embeddings.shape[0]),
        "negative_count": int(negative_embeddings.shape[0]),
        "vector_dim": int(
            positive_embeddings.shape[1]
            if positive_embeddings.size
            else negative_embeddings.shape[1]
        ),
    }
    return positive_embeddings, negative_embeddings, source_summary


@_dataclasses.dataclass
class ManifestSplitData:
    """All data needed for replay training from a manifest split."""

    X: np.ndarray
    y: np.ndarray
    examples: list[dict[str, Any]]
    parquet_cache: dict[str, Any]
    manifest: dict[str, Any]
    source_summary: dict[str, Any]


def load_manifest_split_data(
    manifest_path: Path | str,
    *,
    split: str = "train",
) -> ManifestSplitData:
    """Load manifest split with full metadata for context pooling.

    Unlike ``load_manifest_split_embeddings`` which returns flat pos/neg arrays,
    this returns the full manifest, parquet cache, and per-example metadata needed
    for context pooling and replay verification.
    """
    from humpback.classifier.replay import (
        build_embedding_lookup,
        collect_split_arrays,
        load_manifest as _load_manifest,
        load_parquet_cache,
    )

    manifest = _load_manifest(manifest_path)
    parquet_cache = load_parquet_cache(manifest)
    embedding_lookup = build_embedding_lookup(manifest, parquet_cache)

    split_examples = [ex for ex in manifest["examples"] if ex.get("split") == split]
    if not split_examples:
        raise ValueError(
            f"Manifest has no examples for split {split!r}: {manifest_path}"
        )

    _example_ids, y, X, _negative_groups = collect_split_arrays(
        manifest, embedding_lookup, split
    )

    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))
    if positive_count < 2:
        raise ValueError(
            f"Need at least 2 positive manifest examples for split {split!r}, "
            f"got {positive_count}"
        )
    if negative_count < 2:
        raise ValueError(
            f"Need at least 2 negative manifest examples for split {split!r}, "
            f"got {negative_count}"
        )

    vector_dim = int(X.shape[1]) if X.size else 0
    source_summary = {
        "manifest_path": str(Path(manifest_path)),
        "split": split,
        "example_count": len(split_examples),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "vector_dim": vector_dim,
    }

    return ManifestSplitData(
        X=X,
        y=y,
        examples=split_examples,
        parquet_cache=parquet_cache,
        manifest=manifest,
        source_summary=source_summary,
    )


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
    y = np.concatenate(
        [
            np.ones(len(positive_embeddings), dtype=int),
            np.zeros(len(negative_embeddings), dtype=int),
        ]
    )

    classifier_type = parameters.get("classifier_type", "logistic_regression")
    feature_norm = parameters.get("feature_norm")
    use_l2_norm = False
    use_standard_scaler = False
    if feature_norm is None:
        l2_normalize = parameters.get("l2_normalize", False)
        use_l2_norm = bool(l2_normalize)
        use_standard_scaler = True
        feature_norm_summary = "l2_then_standard" if use_l2_norm else "standard"
    elif feature_norm == "none":
        l2_normalize = False
        feature_norm_summary = "none"
    elif feature_norm == "l2":
        l2_normalize = True
        use_l2_norm = True
        feature_norm_summary = "l2"
    elif feature_norm == "standard":
        l2_normalize = False
        use_standard_scaler = True
        feature_norm_summary = "standard"
    else:
        raise ValueError(f"Unsupported feature_norm: {feature_norm!r}")
    class_weight = parameters.get("class_weight", "balanced")
    if isinstance(class_weight, dict):
        class_weight = {
            int(label): float(weight) for label, weight in class_weight.items()
        }

    # Build pipeline steps
    steps: list[tuple] = []
    if use_l2_norm:
        steps.append(("l2_norm", Normalizer(norm="l2")))
    if use_standard_scaler:
        steps.append(("scaler", StandardScaler()))

    if classifier_type == "mlp":
        steps.append(
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    max_iter=parameters.get("max_iter", 500),
                    early_stopping=True,
                    random_state=parameters.get("random_state", 42),
                ),
            )
        )
    else:
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    solver=parameters.get("solver", "lbfgs"),
                    max_iter=parameters.get("max_iter", 1000),
                    C=parameters.get("C", 1.0),
                    class_weight=class_weight,
                ),
            )
        )

    pipeline = Pipeline(steps)

    # MLP has no native class_weight param; pass sample_weight through the
    # pipeline's fit_params instead.
    fit_params: dict[str, Any] = {}
    if classifier_type == "mlp" and isinstance(class_weight, dict):
        from humpback.classifier.replay import compute_sample_weight

        sw = compute_sample_weight(class_weight, y)
        if sw is not None:
            fit_params["classifier__sample_weight"] = sw

    # Cross-validation for honest estimates.
    # n_splits must not exceed the size of the minority class so that
    # every fold contains at least one sample from each class.
    minority_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = min(5, minority_count)
    n_splits = max(2, n_splits)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "roc_auc", "precision", "recall", "f1"],
        return_train_score=False,
        error_score=cast(Any, "raise"),
        params=fit_params if fit_params else None,
    )

    # Final fit on all data
    pipeline.fit(X, y, **fit_params)

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
    pooled_std = float(
        np.sqrt(
            (
                np.var(pos_scores) * len(pos_scores)
                + np.var(neg_scores) * len(neg_scores)
            )
            / (len(pos_scores) + len(neg_scores))
        )
    )
    score_separation = (
        float((pos_mean - neg_mean) / pooled_std) if pooled_std > 0 else 0.0
    )

    train_preds = pipeline.predict(X)
    cm = confusion_matrix(y, train_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Effective class weights
    clf = pipeline.named_steps["classifier"]
    effective_class_weights = None
    if hasattr(clf, "class_weight") and clf.class_weight == "balanced":
        from sklearn.utils.class_weight import compute_class_weight

        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
        effective_class_weights = {
            "0": round(float(weights[0]), 4),
            "1": round(float(weights[1]), 4),
        }

    summary = {
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
        "balance_ratio": balance_ratio,
        "classifier_type": classifier_type,
        "l2_normalize": l2_normalize,
        "feature_norm": feature_norm_summary,
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
            n_pos,
            n_neg,
            balance_ratio,
        )

    logger.info(
        "Trained classifier: accuracy=%.3f (+-%.3f), AUC=%.3f (+-%.3f)",
        summary["cv_accuracy"],
        summary["cv_accuracy_std"],
        summary["cv_roc_auc"],
        summary["cv_roc_auc_std"],
    )

    return pipeline, summary
