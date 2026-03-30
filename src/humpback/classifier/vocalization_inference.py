"""Score embedding windows through per-type vocalization classifiers."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_vocalization_model(
    model_dir: Path,
) -> tuple[dict[str, Pipeline], list[str], dict[str, float]]:
    """Load per-type pipelines, vocabulary, and thresholds from model directory.

    Returns (pipelines, vocabulary, thresholds).
    """
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json in {model_dir}")

    metadata = json.loads(metadata_path.read_text())
    vocabulary: list[str] = metadata["vocabulary"]
    thresholds: dict[str, float] = metadata["thresholds"]

    pipelines: dict[str, Pipeline] = {}
    for type_name in vocabulary:
        joblib_path = model_dir / f"{type_name}.joblib"
        if not joblib_path.exists():
            logger.warning("Missing %s.joblib in %s, skipping", type_name, model_dir)
            continue
        pipelines[type_name] = joblib.load(joblib_path)

    return pipelines, vocabulary, thresholds


def score_embeddings(
    pipelines: dict[str, Pipeline],
    vocabulary: list[str],
    embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score each embedding through all per-type classifiers.

    Returns {type_name: array of shape (N,) with positive-class probabilities}.
    """
    scores: dict[str, np.ndarray] = {}
    for type_name in vocabulary:
        if type_name not in pipelines:
            scores[type_name] = np.zeros(len(embeddings))
            continue
        probs = pipelines[type_name].predict_proba(embeddings)
        # Column 1 is the positive-class probability
        scores[type_name] = probs[:, 1]
    return scores


def run_inference(
    model_dir: Path,
    embeddings: np.ndarray,
    filenames: list[str],
    start_secs: list[float],
    end_secs: list[float],
    output_path: Path,
    start_utcs: list[float] | None = None,
    end_utcs: list[float] | None = None,
    confidences: list[float] | None = None,
) -> dict[str, Any]:
    """Run vocalization inference and write predictions parquet.

    Returns result_summary with per-type tag counts at stored thresholds.
    """
    pipelines, vocabulary, thresholds = load_vocalization_model(model_dir)

    if len(embeddings) == 0:
        # Write empty parquet with correct schema
        _write_empty_parquet(output_path, vocabulary)
        return {"total_windows": 0, "per_type_counts": {}, "vocabulary": vocabulary}

    scores = score_embeddings(pipelines, vocabulary, embeddings)

    # Build parquet table
    columns: dict[str, list[Any]] = {
        "filename": filenames,
        "start_sec": start_secs,
        "end_sec": end_secs,
    }
    if start_utcs is not None:
        columns["start_utc"] = start_utcs
    if end_utcs is not None:
        columns["end_utc"] = end_utcs
    if confidences is not None:
        columns["confidence"] = confidences

    # Add per-type score columns
    for type_name in vocabulary:
        columns[type_name] = [round(float(s), 4) for s in scores[type_name]]

    table = pa.table(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))

    # Compute tag counts at stored thresholds
    per_type_counts: dict[str, int] = {}
    for type_name in vocabulary:
        t = thresholds.get(type_name, 0.5)
        per_type_counts[type_name] = int((scores[type_name] >= t).sum())

    result_summary: dict[str, Any] = {
        "total_windows": len(embeddings),
        "per_type_counts": per_type_counts,
        "vocabulary": vocabulary,
    }

    logger.info(
        "Inference complete: %d windows, %d types, counts=%s",
        len(embeddings),
        len(vocabulary),
        per_type_counts,
    )

    return result_summary


def _write_empty_parquet(output_path: Path, vocabulary: list[str]) -> None:
    """Write an empty parquet with the correct schema."""
    columns: dict[str, list[Any]] = {
        "filename": [],
        "start_sec": [],
        "end_sec": [],
    }
    for type_name in vocabulary:
        columns[type_name] = []
    table = pa.table(columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))


def read_predictions(
    output_path: Path,
    vocabulary: list[str],
    thresholds: dict[str, float],
    threshold_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Read predictions parquet and apply thresholds to produce tagged rows.

    Returns list of dicts with window identity, scores, and tags.
    """
    if not output_path.exists():
        return []

    table = pq.read_table(str(output_path))
    effective_thresholds = dict(thresholds)
    if threshold_overrides:
        effective_thresholds.update(threshold_overrides)

    rows: list[dict[str, Any]] = []
    for i in range(table.num_rows):
        row: dict[str, Any] = {
            "filename": table.column("filename")[i].as_py(),
            "start_sec": float(table.column("start_sec")[i].as_py()),
            "end_sec": float(table.column("end_sec")[i].as_py()),
        }
        if "start_utc" in table.column_names:
            row["start_utc"] = float(table.column("start_utc")[i].as_py())
        if "end_utc" in table.column_names:
            row["end_utc"] = float(table.column("end_utc")[i].as_py())
        if "confidence" in table.column_names:
            val = table.column("confidence")[i].as_py()
            row["confidence"] = float(val) if val is not None else None

        scores: dict[str, float] = {}
        tags: list[str] = []
        for type_name in vocabulary:
            if type_name in table.column_names:
                score = float(table.column(type_name)[i].as_py())
                scores[type_name] = score
                t = effective_thresholds.get(type_name, 0.5)
                if score >= t:
                    tags.append(type_name)

        row["scores"] = scores
        row["tags"] = tags
        rows.append(row)

    return rows
