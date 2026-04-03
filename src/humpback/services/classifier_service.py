"""Service layer for binary classifier training and detection."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pyarrow.parquet as pq
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.trainer import map_autoresearch_config_to_training_parameters
from humpback.models.audio import AudioFile
from humpback.models.classifier import (
    AutoresearchCandidate,
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.models.processing import EmbeddingSet
from humpback.models.retrain import RetrainWorkflow

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
AUTORESEARCH_CANDIDATE_DIRNAME = "autoresearch_candidates"
AUTORESEARCH_CANDIDATE_STATUS_PROMOTABLE = "promotable"
AUTORESEARCH_CANDIDATE_STATUS_BLOCKED = "blocked"


def _load_json_file(path_str: str, *, label: str) -> tuple[Path, Any]:
    path = Path(path_str).expanduser()
    if not path.is_file():
        raise ValueError(f"{label} not found: {path_str}")
    try:
        return path.resolve(), json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path_str}") from exc


def _validate_manifest_json(data: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"manifest.json must contain an object: {path}")
    metadata = data.get("metadata")
    examples = data.get("examples")
    if not isinstance(metadata, dict) or not isinstance(examples, list):
        raise ValueError(
            f"manifest.json must contain 'metadata' and 'examples': {path}"
        )
    return data


def _validate_best_run_json(data: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"best_run.json must contain an object: {path}")
    if not isinstance(data.get("config"), dict) or not isinstance(
        data.get("metrics"), dict
    ):
        raise ValueError(f"best_run.json must contain 'config' and 'metrics': {path}")
    return data


def _validate_comparison_json(data: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"comparison JSON must contain an object: {path}")
    if isinstance(data.get("splits"), dict):
        return data

    # Also accept lightweight comparison summaries that omit split-level
    # production detail. These can still be imported for review, but they do
    # not drive the richer delta/disagreement previews used by full comparison
    # artifacts like phase*/lr-v12-comparison.json.
    has_best_run_summary = any(
        key.endswith("_best")
        and isinstance(value, dict)
        and isinstance(value.get("config"), dict)
        and isinstance(value.get("metrics"), dict)
        for key, value in data.items()
    )
    has_metric_summary = any(
        key.endswith("_metrics") and isinstance(value, dict)
        for key, value in data.items()
    )
    has_false_positive_summary = any(
        key.endswith("_top_false_positives") and isinstance(value, list)
        for key, value in data.items()
    )
    if has_best_run_summary and (has_metric_summary or has_false_positive_summary):
        return data

    raise ValueError(
        "comparison JSON must contain 'splits' or a recognized comparison summary: "
        f"{path}"
    )
    return data


def _validate_top_false_positives_json(
    data: Any, *, path: Path
) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError(f"top_false_positives JSON must contain a list: {path}")
    preview: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                f"top_false_positives JSON entries must be objects: {path}"
            )
        preview.append(item)
    return preview


def _derive_candidate_phase(best_run_path: Path) -> str | None:
    phase_name = best_run_path.parent.name.strip()
    return phase_name if phase_name.lower().startswith("phase") else None


def _summarize_manifest_sources(manifest: dict[str, Any]) -> dict[str, Any]:
    metadata = manifest["metadata"]
    examples: list[dict[str, Any]] = [
        item for item in manifest["examples"] if isinstance(item, dict)
    ]

    split_counts = Counter(str(item.get("split") or "unknown") for item in examples)
    label_counts = Counter(str(item.get("label") or "unknown") for item in examples)
    label_source_counts = Counter(
        str(item.get("label_source") or "unknown") for item in examples
    )
    negative_group_counts = Counter(
        str(item.get("negative_group") or "none") for item in examples
    )
    source_type_counts = Counter(
        str(item.get("source_type") or "unknown") for item in examples
    )

    return {
        "example_count": len(examples),
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "label_source_counts": dict(sorted(label_source_counts.items())),
        "negative_group_counts": dict(sorted(negative_group_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "detection_job_ids": metadata.get("detection_job_ids", []),
        "source_job_ids": metadata.get("source_job_ids", []),
        "positive_embedding_set_ids": metadata.get("positive_embedding_set_ids", []),
        "negative_embedding_set_ids": metadata.get("negative_embedding_set_ids", []),
    }


def _extract_replay_summary(
    manifest: dict[str, Any],
    best_run: dict[str, Any],
    comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_summary = comparison.get("manifest_summary") if comparison else None
    replay = (
        manifest_summary.get("replay") if isinstance(manifest_summary, dict) else None
    )
    if isinstance(replay, dict):
        return replay

    metrics = best_run.get("metrics", {})
    metadata = manifest.get("metadata", {})
    return {
        "available_hard_negatives": metrics.get("available_hard_negatives"),
        "replayed_hard_negatives": metrics.get("replayed_hard_negatives"),
        "used_replay_manifest": bool(metrics.get("replayed_hard_negatives")),
        "hard_negative_from": metadata.get("detection_job_ids", []),
    }


def _extract_split_metrics(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if comparison is None:
        return None

    split_metrics: dict[str, Any] = {}
    for split_name, payload in comparison.get("splits", {}).items():
        if not isinstance(payload, dict):
            continue
        split_summary: dict[str, Any] = {}
        autoresearch = payload.get("autoresearch")
        production = payload.get("production")
        if isinstance(autoresearch, dict) and isinstance(
            autoresearch.get("metrics"), dict
        ):
            split_summary["autoresearch"] = autoresearch["metrics"]
        if isinstance(production, dict) and isinstance(production.get("metrics"), dict):
            split_summary["production"] = production["metrics"]
        if split_summary:
            split_metrics[str(split_name)] = split_summary

    return split_metrics or None


def _extract_metric_deltas(comparison: dict[str, Any] | None) -> dict[str, Any] | None:
    if comparison is None:
        return None

    metric_deltas: dict[str, Any] = {}
    for split_name, payload in comparison.get("splits", {}).items():
        if isinstance(payload, dict) and isinstance(payload.get("delta"), dict):
            metric_deltas[str(split_name)] = payload["delta"]
    return metric_deltas or None


def _extract_prediction_disagreements_preview(
    comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if comparison is None:
        return None

    preview: dict[str, Any] = {}
    for split_name, payload in comparison.get("splits", {}).items():
        disagreements = (
            payload.get("prediction_disagreements")
            if isinstance(payload, dict)
            else None
        )
        if isinstance(disagreements, list) and disagreements:
            preview[str(split_name)] = disagreements[:10]
    return preview or None


def _extract_top_false_positives_preview(
    top_false_positives: list[dict[str, Any]] | None,
    comparison: dict[str, Any] | None,
) -> dict[str, Any] | None:
    preview: dict[str, Any] = {}
    if top_false_positives:
        preview["imported"] = top_false_positives[:10]

    if comparison is not None:
        for split_name, payload in comparison.get("splits", {}).items():
            if not isinstance(payload, dict):
                continue
            split_preview: dict[str, Any] = {}
            autoresearch = payload.get("autoresearch")
            production = payload.get("production")
            if isinstance(autoresearch, dict) and isinstance(
                autoresearch.get("top_false_positives"), list
            ):
                split_preview["autoresearch"] = autoresearch["top_false_positives"][:10]
            if isinstance(production, dict) and isinstance(
                production.get("top_false_positives"), list
            ):
                split_preview["production"] = production["top_false_positives"][:10]
            if split_preview:
                preview[str(split_name)] = split_preview

    return preview or None


def _comparison_has_split_details(comparison: dict[str, Any] | None) -> bool:
    return isinstance(comparison, dict) and isinstance(comparison.get("splits"), dict)


def _assess_reproducibility(config: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []

    classifier = config.get("classifier", "logreg")
    if classifier not in {"logreg", "mlp"}:
        blockers.append(
            f"classifier={classifier!r} is not supported by the production trainer"
        )

    feature_norm = config.get("feature_norm", "standard")
    if feature_norm not in {"none", "l2", "standard"}:
        blockers.append(
            f"feature_norm={feature_norm!r} is not supported by the production trainer"
        )

    hard_negative_fraction = float(config.get("hard_negative_fraction", 0.0))
    if hard_negative_fraction != 0.0:
        blockers.append(
            "Replay-adjusted hard-negative sampling is not yet supported for promotion"
        )

    if classifier == "mlp":
        if (
            float(config.get("class_weight_pos", 1.0)) != 1.0
            or float(config.get("class_weight_neg", 1.0)) != 1.0
        ):
            blockers.append("MLP promotion cannot yet reproduce explicit class weights")

    return not blockers, blockers


def _build_source_model_metadata(
    comparison: dict[str, Any] | None,
    *,
    source_model_id_override: str | None,
    source_model_name_override: str | None,
) -> dict[str, Any] | None:
    production = comparison.get("production") if comparison else None
    source_model_metadata = dict(production) if isinstance(production, dict) else {}
    if source_model_id_override:
        source_model_metadata["id"] = source_model_id_override
    if source_model_name_override:
        source_model_metadata["name"] = source_model_name_override
    return source_model_metadata or None


def _default_candidate_name(
    *,
    provided_name: str | None,
    phase: str | None,
    source_model_name: str | None,
    config_hash: str | None,
) -> str:
    if provided_name and provided_name.strip():
        return provided_name.strip()
    if phase and source_model_name:
        return f"{phase} vs {source_model_name}"
    if phase and config_hash:
        return f"{phase} {config_hash}"
    if source_model_name and config_hash:
        return f"{source_model_name} {config_hash}"
    return config_hash or "autoresearch-candidate"


def _copy_candidate_artifact(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _load_candidate_manifest(candidate: AutoresearchCandidate) -> dict[str, Any]:
    manifest_path, manifest_raw = _load_json_file(
        candidate.manifest_path,
        label="candidate manifest",
    )
    return _validate_manifest_json(manifest_raw, path=manifest_path)


def _extract_detection_job_id_from_parquet_path(parquet_path: str) -> str | None:
    parts = Path(parquet_path).parts
    for idx, part in enumerate(parts):
        if (
            part == "detections"
            and idx + 2 < len(parts)
            and parts[idx + 2] == "detection_embeddings.parquet"
        ):
            return parts[idx + 1]
    return None


async def _resolve_candidate_training_runtime(
    session: AsyncSession,
    candidate: AutoresearchCandidate,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Infer model/runtime metadata for a candidate-backed training job."""
    runtimes: list[dict[str, Any]] = []
    train_examples = [
        ex
        for ex in manifest["examples"]
        if isinstance(ex, dict) and ex.get("split") == "train"
    ]
    if not train_examples:
        raise ValueError("Candidate manifest has no train split examples")
    parquet_paths = sorted({str(ex["parquet_path"]) for ex in train_examples})

    for parquet_path in parquet_paths:
        result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.parquet_path == parquet_path)
        )
        embedding_set = result.scalar_one_or_none()
        if embedding_set is not None:
            runtimes.append(
                {
                    "model_version": embedding_set.model_version,
                    "window_size_seconds": embedding_set.window_size_seconds,
                    "target_sample_rate": embedding_set.target_sample_rate,
                    "vector_dim": embedding_set.vector_dim,
                }
            )
            continue

        detection_job_id = _extract_detection_job_id_from_parquet_path(parquet_path)
        if detection_job_id is None:
            continue

        result = await session.execute(
            select(DetectionJob).where(DetectionJob.id == detection_job_id)
        )
        detection_job = result.scalar_one_or_none()
        if detection_job is None:
            continue

        result = await session.execute(
            select(ClassifierModel).where(
                ClassifierModel.id == detection_job.classifier_model_id
            )
        )
        source_model = result.scalar_one_or_none()
        if source_model is None:
            continue

        runtimes.append(
            {
                "model_version": source_model.model_version,
                "window_size_seconds": source_model.window_size_seconds,
                "target_sample_rate": source_model.target_sample_rate,
                "vector_dim": source_model.vector_dim,
            }
        )

    if not runtimes and candidate.source_model_id:
        result = await session.execute(
            select(ClassifierModel).where(
                ClassifierModel.id == candidate.source_model_id
            )
        )
        source_model = result.scalar_one_or_none()
        if source_model is not None:
            runtimes.append(
                {
                    "model_version": source_model.model_version,
                    "window_size_seconds": source_model.window_size_seconds,
                    "target_sample_rate": source_model.target_sample_rate,
                    "vector_dim": source_model.vector_dim,
                }
            )

    if not runtimes:
        raise ValueError(
            "Could not infer training runtime metadata for candidate manifest"
        )

    reference = runtimes[0]
    if any(runtime != reference for runtime in runtimes[1:]):
        raise ValueError("Candidate manifest mixes incompatible model/runtime metadata")
    return reference


def _build_candidate_promotion_context(
    candidate: AutoresearchCandidate,
    *,
    trainer_parameters: dict[str, Any],
    runtime: dict[str, Any],
    notes: str | None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "candidate_name": candidate.name,
        "source_model_id": candidate.source_model_id,
        "source_model_name": candidate.source_model_name,
        "comparison_target": candidate.comparison_target,
        "objective_name": candidate.objective_name,
        "threshold": candidate.threshold,
        "promoted_config": json.loads(candidate.promoted_config),
        "split_metrics": json.loads(candidate.split_metrics)
        if candidate.split_metrics
        else None,
        "metric_deltas": json.loads(candidate.metric_deltas)
        if candidate.metric_deltas
        else None,
        "warnings": json.loads(candidate.warnings) if candidate.warnings else [],
        "trainer_parameters": trainer_parameters,
        "training_runtime": runtime,
        "notes": notes,
    }


async def import_autoresearch_candidate(
    session: AsyncSession,
    storage_root: Path,
    manifest_path: str,
    best_run_path: str,
    comparison_path: str | None = None,
    top_false_positives_path: str | None = None,
    name: str | None = None,
    source_model_id_override: str | None = None,
    source_model_name_override: str | None = None,
) -> AutoresearchCandidate:
    """Import an autoresearch result bundle as a persisted promotion candidate."""
    manifest_file, manifest_raw = _load_json_file(manifest_path, label="manifest.json")
    best_run_file, best_run_raw = _load_json_file(best_run_path, label="best_run.json")
    manifest = _validate_manifest_json(manifest_raw, path=manifest_file)
    best_run = _validate_best_run_json(best_run_raw, path=best_run_file)

    comparison_file: Path | None = None
    comparison: dict[str, Any] | None = None
    if comparison_path:
        comparison_file, comparison_raw = _load_json_file(
            comparison_path, label="comparison JSON"
        )
        comparison = _validate_comparison_json(comparison_raw, path=comparison_file)

    top_false_positives_file: Path | None = None
    top_false_positives: list[dict[str, Any]] | None = None
    if top_false_positives_path:
        top_false_positives_file, top_false_positives_raw = _load_json_file(
            top_false_positives_path,
            label="top_false_positives JSON",
        )
        top_false_positives = _validate_top_false_positives_json(
            top_false_positives_raw,
            path=top_false_positives_file,
        )

    phase = _derive_candidate_phase(best_run_file)
    promoted_config = best_run["config"]
    reproducible_exact, blockers = _assess_reproducibility(promoted_config)
    warnings = list(blockers)
    if comparison is None:
        warnings.append(
            "No comparison artifact was imported, so production deltas are unavailable"
        )
    elif not _comparison_has_split_details(comparison):
        warnings.append(
            "Comparison summary imported without split-level production deltas; "
            "use phase*/lr-v12-comparison.json for detailed comparison previews"
        )

    source_model_metadata = _build_source_model_metadata(
        comparison,
        source_model_id_override=source_model_id_override,
        source_model_name_override=source_model_name_override,
    )
    source_model_id_value = (
        source_model_metadata.get("id") if source_model_metadata else None
    )
    source_model_name_value = (
        source_model_metadata.get("name") if source_model_metadata else None
    )
    source_model_id = (
        str(source_model_id_value) if source_model_id_value is not None else None
    )
    source_model_name = (
        str(source_model_name_value) if source_model_name_value is not None else None
    )
    comparison_target = source_model_name
    if comparison_target is None and comparison_file is not None:
        comparison_target = comparison_file.stem

    split_metrics = _extract_split_metrics(comparison)
    metric_deltas = _extract_metric_deltas(comparison)
    top_false_positives_preview = _extract_top_false_positives_preview(
        top_false_positives,
        comparison,
    )
    prediction_disagreements_preview = _extract_prediction_disagreements_preview(
        comparison
    )
    threshold_raw = best_run.get("metrics", {}).get(
        "threshold",
        best_run.get("config", {}).get("threshold"),
    )
    threshold = float(threshold_raw) if threshold_raw is not None else None

    candidate_id = str(uuid4())
    candidate_dir = (
        storage_root / AUTORESEARCH_CANDIDATE_DIRNAME / candidate_id / "artifacts"
    )

    stored_manifest_path = _copy_candidate_artifact(
        manifest_file, candidate_dir / "manifest.json"
    )
    stored_best_run_path = _copy_candidate_artifact(
        best_run_file, candidate_dir / "best_run.json"
    )
    stored_comparison_path = (
        _copy_candidate_artifact(comparison_file, candidate_dir / "comparison.json")
        if comparison_file is not None
        else None
    )
    stored_top_false_positives_path = (
        _copy_candidate_artifact(
            top_false_positives_file,
            candidate_dir / "top_false_positives.json",
        )
        if top_false_positives_file is not None
        else None
    )

    candidate = AutoresearchCandidate(
        id=candidate_id,
        name=_default_candidate_name(
            provided_name=name,
            phase=phase,
            source_model_name=source_model_name,
            config_hash=best_run.get("config_hash"),
        ),
        status=(
            AUTORESEARCH_CANDIDATE_STATUS_PROMOTABLE
            if reproducible_exact
            else AUTORESEARCH_CANDIDATE_STATUS_BLOCKED
        ),
        manifest_path=stored_manifest_path,
        best_run_path=stored_best_run_path,
        comparison_path=stored_comparison_path,
        top_false_positives_path=stored_top_false_positives_path,
        phase=phase,
        objective_name=(
            str(comparison.get("objective_name"))
            if comparison and comparison.get("objective_name") is not None
            else "default"
        ),
        threshold=threshold,
        promoted_config=json.dumps(promoted_config),
        best_run_metrics=json.dumps(best_run["metrics"]),
        split_metrics=json.dumps(split_metrics) if split_metrics else None,
        metric_deltas=json.dumps(metric_deltas) if metric_deltas else None,
        replay_summary=json.dumps(
            _extract_replay_summary(manifest, best_run, comparison)
        ),
        source_counts=json.dumps(_summarize_manifest_sources(manifest)),
        warnings=json.dumps(warnings),
        source_model_id=source_model_id,
        source_model_name=source_model_name,
        source_model_metadata=json.dumps(source_model_metadata)
        if source_model_metadata
        else None,
        comparison_target=comparison_target,
        top_false_positives_preview=json.dumps(top_false_positives_preview)
        if top_false_positives_preview
        else None,
        prediction_disagreements_preview=json.dumps(prediction_disagreements_preview)
        if prediction_disagreements_preview
        else None,
        is_reproducible_exact=reproducible_exact,
    )
    session.add(candidate)
    await session.commit()
    return candidate


async def create_training_job_from_autoresearch_candidate(
    session: AsyncSession,
    candidate_id: str,
    new_model_name: str,
    *,
    notes: str | None = None,
) -> ClassifierTrainingJob:
    """Create a manifest-backed classifier training job from a promotable candidate."""
    candidate = await get_autoresearch_candidate(session, candidate_id)
    if candidate is None:
        raise ValueError(f"Autoresearch candidate not found: {candidate_id}")
    if candidate.status != AUTORESEARCH_CANDIDATE_STATUS_PROMOTABLE:
        raise ValueError(f"Candidate is not promotable (status={candidate.status})")
    if not candidate.is_reproducible_exact:
        raise ValueError("Candidate is not exactly reproducible by the current trainer")

    manifest = _load_candidate_manifest(candidate)
    runtime = await _resolve_candidate_training_runtime(session, candidate, manifest)
    promoted_config = json.loads(candidate.promoted_config)
    trainer_parameters = map_autoresearch_config_to_training_parameters(promoted_config)
    promotion_context = _build_candidate_promotion_context(
        candidate,
        trainer_parameters=trainer_parameters,
        runtime=runtime,
        notes=notes,
    )

    job = ClassifierTrainingJob(
        name=new_model_name,
        positive_embedding_set_ids=json.dumps([]),
        negative_embedding_set_ids=json.dumps([]),
        model_version=runtime["model_version"],
        window_size_seconds=runtime["window_size_seconds"],
        target_sample_rate=runtime["target_sample_rate"],
        feature_config=None,
        parameters=json.dumps(trainer_parameters),
        source_mode="autoresearch_candidate",
        source_candidate_id=candidate.id,
        source_model_id=candidate.source_model_id,
        manifest_path=candidate.manifest_path,
        training_split_name="train",
        promoted_config=candidate.promoted_config,
        source_comparison_context=json.dumps(promotion_context),
    )
    session.add(job)
    await session.flush()

    candidate.status = "training"
    candidate.training_job_id = job.id
    candidate.error_message = None

    await session.commit()
    return job


async def create_training_job(
    session: AsyncSession,
    name: str,
    positive_embedding_set_ids: list[str],
    negative_embedding_set_ids: list[str],
    parameters: Optional[dict[str, Any]] = None,
) -> ClassifierTrainingJob:
    """Create a classifier training job after validating inputs."""
    if not positive_embedding_set_ids:
        raise ValueError("At least one positive embedding set is required")
    if not negative_embedding_set_ids:
        raise ValueError("At least one negative embedding set is required")

    # Reject overlap between positive and negative sets
    overlap = set(positive_embedding_set_ids) & set(negative_embedding_set_ids)
    if overlap:
        raise ValueError(
            f"Embedding sets cannot be both positive and negative: {overlap}"
        )

    # Load and validate positive embedding sets
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(positive_embedding_set_ids))
    )
    pos_sets = list(result.scalars().all())
    if len(pos_sets) != len(positive_embedding_set_ids):
        found_ids = {es.id for es in pos_sets}
        missing = set(positive_embedding_set_ids) - found_ids
        raise ValueError(f"Positive embedding sets not found: {missing}")

    # Load and validate negative embedding sets
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id.in_(negative_embedding_set_ids))
    )
    neg_sets = list(result.scalars().all())
    if len(neg_sets) != len(negative_embedding_set_ids):
        found_ids = {es.id for es in neg_sets}
        missing = set(negative_embedding_set_ids) - found_ids
        raise ValueError(f"Negative embedding sets not found: {missing}")

    # Validate all sets share same model_version and vector_dim
    all_sets = pos_sets + neg_sets
    model_versions = {es.model_version for es in all_sets}
    if len(model_versions) > 1:
        raise ValueError(
            f"Embedding sets use different model versions: {model_versions}"
        )

    vector_dims = {es.vector_dim for es in all_sets}
    if len(vector_dims) > 1:
        raise ValueError(
            f"Embedding sets have different vector dimensions: {vector_dims}"
        )

    # Check encoding signature consistency
    encoding_sigs = {es.encoding_signature for es in all_sets if es.encoding_signature}
    if len(encoding_sigs) > 1:
        if parameters is None:
            parameters = {}
        parameters["_config_mismatch_warning"] = (
            f"Embedding sets use {len(encoding_sigs)} different encoding signatures. "
            "Results may be unreliable when mixing different processing configurations."
        )

    # Use first positive embedding set's config
    ref = pos_sets[0]

    job = ClassifierTrainingJob(
        name=name,
        positive_embedding_set_ids=json.dumps(positive_embedding_set_ids),
        negative_embedding_set_ids=json.dumps(negative_embedding_set_ids),
        model_version=ref.model_version,
        window_size_seconds=ref.window_size_seconds,
        target_sample_rate=ref.target_sample_rate,
        feature_config=None,  # inherit from embedding sets
        parameters=json.dumps(parameters) if parameters else None,
    )
    session.add(job)
    await session.commit()
    return job


async def create_detection_job(
    session: AsyncSession,
    classifier_model_id: str,
    audio_folder: str,
    confidence_threshold: float = 0.5,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
) -> DetectionJob:
    """Create a detection job after validating inputs."""
    # Validate classifier model exists
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate audio folder
    folder = Path(audio_folder)
    if not folder.is_dir():
        raise ValueError(f"Audio folder not found: {audio_folder}")

    audio_files = [p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_folder}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    if hop_seconds > cm.window_size_seconds:
        raise ValueError(
            f"hop_seconds ({hop_seconds}) must be <= window_size_seconds ({cm.window_size_seconds})"
        )

    job = DetectionJob(
        classifier_model_id=classifier_model_id,
        audio_folder=audio_folder,
        confidence_threshold=confidence_threshold,
        hop_seconds=hop_seconds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        detection_mode="windowed",
    )
    session.add(job)
    await session.commit()
    return job


async def create_hydrophone_detection_job(
    session: AsyncSession,
    classifier_model_id: str,
    hydrophone_id: str,
    start_timestamp: float,
    end_timestamp: float,
    confidence_threshold: float = 0.5,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    local_cache_path: str | None = None,
) -> DetectionJob:
    """Create a hydrophone detection job after validating inputs."""
    from humpback.config import (
        ARCHIVE_SOURCE_IDS,
        ORCASOUND_S3_BUCKET,
        get_archive_source,
    )

    # Validate classifier model exists
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        raise ValueError(f"Classifier model not found: {classifier_model_id}")

    # Validate archive source (legacy hydrophone_id field name retained)
    if hydrophone_id not in ARCHIVE_SOURCE_IDS:
        raise ValueError(f"Unknown hydrophone: {hydrophone_id}")

    hydrophone = get_archive_source(hydrophone_id)
    if hydrophone is None:
        raise ValueError(f"Unknown hydrophone: {hydrophone_id}")

    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")

    if hop_seconds > cm.window_size_seconds:
        raise ValueError(
            f"hop_seconds ({hop_seconds}) must be <= window_size_seconds ({cm.window_size_seconds})"
        )

    # Validate local cache path if provided
    if local_cache_path:
        if hydrophone["provider_kind"] != "orcasound_hls":
            raise ValueError(
                "local_cache_path is only supported for Orcasound HLS sources"
            )
        from pathlib import Path

        cache_dir = Path(local_cache_path) / ORCASOUND_S3_BUCKET / hydrophone_id / "hls"
        if not cache_dir.is_dir():
            raise ValueError(
                f"Local cache path does not contain expected HLS structure: "
                f"{cache_dir} not found"
            )

    job = DetectionJob(
        classifier_model_id=classifier_model_id,
        hydrophone_id=hydrophone_id,
        hydrophone_name=hydrophone["name"],
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        confidence_threshold=confidence_threshold,
        hop_seconds=hop_seconds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        local_cache_path=local_cache_path,
        detection_mode="windowed",
    )
    session.add(job)
    await session.commit()
    return job


async def list_hydrophone_detection_jobs(session: AsyncSession) -> list[DetectionJob]:
    """List detection jobs that are hydrophone-based."""
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.hydrophone_id.isnot(None))
        .order_by(DetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def cancel_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Cancel a running or paused hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status not in ("running", "paused", "queued"):
        raise ValueError(f"Job is not running, paused, or queued (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="canceled", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job


async def pause_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Pause a running hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status != "running":
        raise ValueError(f"Job is not running (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="paused", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job


async def resume_hydrophone_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    """Resume a paused hydrophone detection job. Returns job if found."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    if job.status != "paused":
        raise ValueError(f"Job is not paused (status={job.status})")

    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="running", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    return job


async def list_training_jobs(session: AsyncSession) -> list[ClassifierTrainingJob]:
    result = await session.execute(
        select(ClassifierTrainingJob).order_by(ClassifierTrainingJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_training_job(
    session: AsyncSession, job_id: str
) -> Optional[ClassifierTrainingJob]:
    result = await session.execute(
        select(ClassifierTrainingJob).where(ClassifierTrainingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def list_autoresearch_candidates(
    session: AsyncSession,
) -> list[AutoresearchCandidate]:
    result = await session.execute(
        select(AutoresearchCandidate).order_by(AutoresearchCandidate.created_at.desc())
    )
    return list(result.scalars().all())


async def get_autoresearch_candidate(
    session: AsyncSession, candidate_id: str
) -> Optional[AutoresearchCandidate]:
    result = await session.execute(
        select(AutoresearchCandidate).where(AutoresearchCandidate.id == candidate_id)
    )
    return result.scalar_one_or_none()


async def list_classifier_models(session: AsyncSession) -> list[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).order_by(ClassifierModel.created_at.desc())
    )
    return list(result.scalars().all())


async def get_classifier_model(
    session: AsyncSession, model_id: str
) -> Optional[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    return result.scalar_one_or_none()


async def delete_classifier_model(
    session: AsyncSession, model_id: str, storage_root: Path
) -> bool:
    """Delete a classifier model and its files. Returns True if found."""
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        return False

    # Delete files
    from humpback.storage import classifier_dir

    cdir = classifier_dir(storage_root, model_id)
    if cdir.is_dir():
        shutil.rmtree(cdir)

    await session.delete(cm)
    await session.commit()
    return True


async def list_detection_jobs(session: AsyncSession) -> list[DetectionJob]:
    """List local (non-hydrophone) detection jobs."""
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.hydrophone_id.is_(None))
        .order_by(DetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[DetectionJob]:
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def delete_training_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> bool:
    """Delete a training job. If it produced a model, cascade-delete the model too."""
    result = await session.execute(
        select(ClassifierTrainingJob).where(ClassifierTrainingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return False

    # Cascade-delete the associated classifier model if any
    if job.classifier_model_id:
        await delete_classifier_model(session, job.classifier_model_id, storage_root)

    await session.delete(job)
    await session.commit()
    return True


async def bulk_delete_training_jobs(
    session: AsyncSession, job_ids: list[str], storage_root: Path
) -> int:
    """Delete multiple training jobs. Returns count of deleted jobs."""
    count = 0
    for job_id in job_ids:
        if await delete_training_job(session, job_id, storage_root):
            count += 1
    return count


class DetectionJobDependencyError(Exception):
    """Raised when a detection job cannot be deleted due to downstream deps."""

    def __init__(self, job_id: str, message: str) -> None:
        self.job_id = job_id
        self.message = message
        super().__init__(message)


async def _check_detection_job_dependencies(
    session: AsyncSession, job_id: str
) -> str | None:
    """Return a dependency message if the job cannot be deleted, else None."""
    from sqlalchemy import func

    from humpback.models.labeling import VocalizationLabel
    from humpback.models.training_dataset import TrainingDataset

    # Check vocalization labels
    vl_result = await session.execute(
        select(func.count()).where(VocalizationLabel.detection_job_id == job_id)
    )
    vl_count = vl_result.scalar() or 0

    # Check training datasets referencing this job in source_config JSON
    td_result = await session.execute(select(TrainingDataset))
    td_count = 0
    for td in td_result.scalars().all():
        if job_id in (td.source_config or ""):
            td_count += 1

    parts: list[str] = []
    if vl_count:
        parts.append(f"{vl_count} vocalization label{'s' if vl_count != 1 else ''}")
    if td_count:
        parts.append(f"{td_count} training dataset{'s' if td_count != 1 else ''}")

    if parts:
        return (
            f"Cannot delete detection job: used by {' and '.join(parts)}. "
            "Remove these associations first."
        )
    return None


async def delete_detection_job(
    session: AsyncSession, job_id: str, storage_root: Path
) -> bool:
    """Delete a detection job and its output files.

    Raises DetectionJobDependencyError if the job has downstream dependencies.
    """
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return False

    dep_msg = await _check_detection_job_dependencies(session, job_id)
    if dep_msg:
        raise DetectionJobDependencyError(job_id, dep_msg)

    # Delete detection output directory
    from humpback.storage import detection_dir

    ddir = detection_dir(storage_root, job_id)
    if ddir.is_dir():
        shutil.rmtree(ddir)

    await session.delete(job)
    await session.commit()
    return True


async def bulk_delete_detection_jobs(
    session: AsyncSession, job_ids: list[str], storage_root: Path
) -> tuple[int, list[dict[str, str]]]:
    """Delete multiple detection jobs.

    Returns (deleted_count, blocked_list) where blocked_list contains
    dicts with 'job_id' and 'detail' for jobs that could not be deleted.
    """
    count = 0
    blocked: list[dict[str, str]] = []
    for job_id in job_ids:
        try:
            if await delete_detection_job(session, job_id, storage_root):
                count += 1
        except DetectionJobDependencyError as exc:
            blocked.append({"job_id": exc.job_id, "detail": exc.message})
    return count, blocked


async def bulk_delete_classifier_models(
    session: AsyncSession, model_ids: list[str], storage_root: Path
) -> int:
    """Delete multiple classifier models. Returns count of deleted models."""
    count = 0
    for model_id in model_ids:
        if await delete_classifier_model(session, model_id, storage_root):
            count += 1
    return count


async def get_training_data_summary(
    session: AsyncSession, model_id: str
) -> Optional[dict[str, Any]]:
    """Build training data provenance summary for a classifier model."""
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        return None

    # Find the training job
    if not cm.training_job_id:
        return None
    result = await session.execute(
        select(ClassifierTrainingJob).where(
            ClassifierTrainingJob.id == cm.training_job_id
        )
    )
    tj = result.scalar_one_or_none()
    if tj is None:
        return None

    if tj.source_mode == "autoresearch_candidate":
        training_summary = (
            json.loads(cm.training_summary) if cm.training_summary else {}
        )
        total_pos = int(training_summary.get("n_positive") or 0)
        total_neg = int(training_summary.get("n_negative") or 0)
        balance = total_pos / total_neg if total_neg > 0 else float("inf")
        return {
            "model_id": cm.id,
            "model_name": cm.name,
            "positive_sources": [],
            "negative_sources": [],
            "total_positive": total_pos,
            "total_negative": total_neg,
            "balance_ratio": balance,
            "window_size_seconds": cm.window_size_seconds,
            "positive_duration_sec": total_pos * cm.window_size_seconds
            if total_pos
            else None,
            "negative_duration_sec": total_neg * cm.window_size_seconds
            if total_neg
            else None,
        }

    pos_ids = json.loads(tj.positive_embedding_set_ids)
    neg_ids = json.loads(tj.negative_embedding_set_ids)

    async def _resolve_sources(es_ids: list[str]) -> tuple[list[dict], int]:
        if not es_ids:
            return [], 0
        result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(es_ids))
        )
        sets = list(result.scalars().all())

        # Batch-load audio files for folder_path + filename
        audio_ids = [es.audio_file_id for es in sets if es.audio_file_id]
        audio_map: dict[str, AudioFile] = {}
        if audio_ids:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id.in_(audio_ids))
            )
            audio_map = {af.id: af for af in af_result.scalars().all()}

        sources = []
        total = 0
        for es in sets:
            n_vectors = 0
            try:
                meta = pq.read_metadata(es.parquet_path)
                n_vectors = meta.num_rows
            except Exception:
                pass
            total += n_vectors
            duration = n_vectors * cm.window_size_seconds if n_vectors else None
            af = audio_map.get(es.audio_file_id) if es.audio_file_id else None
            sources.append(
                {
                    "embedding_set_id": es.id,
                    "audio_file_id": es.audio_file_id,
                    "filename": af.filename if af else None,
                    "folder_path": af.folder_path if af else None,
                    "n_vectors": n_vectors,
                    "duration_represented_sec": duration,
                }
            )
        return sources, total

    pos_sources, total_pos = await _resolve_sources(pos_ids)
    neg_sources, total_neg = await _resolve_sources(neg_ids)

    balance = total_pos / total_neg if total_neg > 0 else float("inf")

    return {
        "model_id": cm.id,
        "model_name": cm.name,
        "positive_sources": pos_sources,
        "negative_sources": neg_sources,
        "total_positive": total_pos,
        "total_negative": total_neg,
        "balance_ratio": balance,
        "window_size_seconds": cm.window_size_seconds,
        "positive_duration_sec": total_pos * cm.window_size_seconds
        if total_pos
        else None,
        "negative_duration_sec": total_neg * cm.window_size_seconds
        if total_neg
        else None,
    }


# ---- Retrain Workflows ----


async def trace_folder_roots(
    session: AsyncSession, training_job: ClassifierTrainingJob
) -> dict[str, list[str]]:
    """Trace back from training job's embedding sets to import folder roots."""
    pos_ids = json.loads(training_job.positive_embedding_set_ids)
    neg_ids = json.loads(training_job.negative_embedding_set_ids)

    async def _resolve_roots(es_ids: list[str]) -> list[str]:
        if not es_ids:
            return []
        result = await session.execute(
            select(EmbeddingSet.audio_file_id).where(EmbeddingSet.id.in_(es_ids))
        )
        audio_file_ids = list(set(result.scalars().all()))
        if not audio_file_ids:
            return []

        result = await session.execute(
            select(AudioFile.source_folder, AudioFile.folder_path).where(
                AudioFile.id.in_(audio_file_ids),
                AudioFile.source_folder.isnot(None),
            )
        )
        rows = result.all()

        roots = set()
        for source_folder, folder_path in rows:
            parts = folder_path.split("/")
            import_root = Path(source_folder)
            for _ in range(len(parts) - 1):
                import_root = import_root.parent
            roots.add(str(import_root))
        return sorted(roots)

    return {
        "positive_folder_roots": await _resolve_roots(pos_ids),
        "negative_folder_roots": await _resolve_roots(neg_ids),
    }


async def collect_embedding_sets_for_folders(
    session: AsyncSession,
    folder_roots: list[str],
    model_version: str,
) -> list[str]:
    """Find all embedding set IDs for audio files under the given import roots."""
    all_ids = []
    for root in folder_roots:
        base_name = Path(root).name
        result = await session.execute(
            select(AudioFile.id).where(
                AudioFile.source_folder.isnot(None),
                (AudioFile.folder_path == base_name)
                | AudioFile.folder_path.startswith(f"{base_name}/"),
            )
        )
        audio_ids = list(result.scalars().all())
        if not audio_ids:
            continue

        result = await session.execute(
            select(EmbeddingSet.id).where(
                EmbeddingSet.audio_file_id.in_(audio_ids),
                EmbeddingSet.model_version == model_version,
            )
        )
        all_ids.extend(result.scalars().all())
    return sorted(set(all_ids))


async def get_retrain_info(
    session: AsyncSession, model_id: str
) -> Optional[dict[str, Any]]:
    """Pre-flight info for retrain: folder roots and parameters."""
    cm = await get_classifier_model(session, model_id)
    if cm is None:
        return None

    if not cm.training_job_id:
        return None

    tj = await get_training_job(session, cm.training_job_id)
    if tj is None:
        return None
    if tj.source_mode == "autoresearch_candidate":
        return None

    roots = await trace_folder_roots(session, tj)

    parameters = json.loads(tj.parameters) if tj.parameters else {}
    parameters.pop("_config_mismatch_warning", None)

    return {
        "model_id": cm.id,
        "model_name": cm.name,
        "model_version": cm.model_version,
        "window_size_seconds": cm.window_size_seconds,
        "target_sample_rate": cm.target_sample_rate,
        "feature_config": json.loads(cm.feature_config) if cm.feature_config else None,
        "positive_folder_roots": roots["positive_folder_roots"],
        "negative_folder_roots": roots["negative_folder_roots"],
        "parameters": parameters,
    }


async def create_retrain_workflow(
    session: AsyncSession,
    source_model_id: str,
    new_model_name: str,
    parameter_overrides: Optional[dict[str, Any]] = None,
) -> RetrainWorkflow:
    """Create a retrain workflow from an existing classifier model."""
    cm = await get_classifier_model(session, source_model_id)
    if cm is None:
        raise ValueError(f"Source classifier model not found: {source_model_id}")

    if not cm.training_job_id:
        raise ValueError("Source model has no associated training job")

    tj = await get_training_job(session, cm.training_job_id)
    if tj is None:
        raise ValueError("Source model's training job not found")
    if tj.source_mode == "autoresearch_candidate":
        raise ValueError("Candidate-backed models do not support folder-root retrain")

    roots = await trace_folder_roots(session, tj)
    if not roots["positive_folder_roots"]:
        raise ValueError("Cannot trace positive folder roots from training data")
    if not roots["negative_folder_roots"]:
        raise ValueError("Cannot trace negative folder roots from training data")

    base_params = json.loads(tj.parameters) if tj.parameters else {}
    base_params.pop("_config_mismatch_warning", None)
    if parameter_overrides:
        base_params.update(parameter_overrides)

    workflow = RetrainWorkflow(
        source_model_id=source_model_id,
        new_model_name=new_model_name,
        model_version=cm.model_version,
        window_size_seconds=cm.window_size_seconds,
        target_sample_rate=cm.target_sample_rate,
        feature_config=cm.feature_config,
        parameters=json.dumps(base_params) if base_params else None,
        positive_folder_roots=json.dumps(roots["positive_folder_roots"]),
        negative_folder_roots=json.dumps(roots["negative_folder_roots"]),
    )
    session.add(workflow)
    await session.commit()
    return workflow


async def list_retrain_workflows(session: AsyncSession) -> list[RetrainWorkflow]:
    result = await session.execute(
        select(RetrainWorkflow).order_by(RetrainWorkflow.created_at.desc())
    )
    return list(result.scalars().all())


async def get_retrain_workflow(
    session: AsyncSession, workflow_id: str
) -> Optional[RetrainWorkflow]:
    result = await session.execute(
        select(RetrainWorkflow).where(RetrainWorkflow.id == workflow_id)
    )
    return result.scalar_one_or_none()
