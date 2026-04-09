"""Autoresearch candidate import and promotion."""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.trainer import map_autoresearch_config_to_training_parameters
from humpback.models.classifier import (
    AutoresearchCandidate,
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.models.processing import EmbeddingSet

AUTORESEARCH_CANDIDATE_DIRNAME = "autoresearch_candidates"
AUTORESEARCH_CANDIDATE_STATUS_PROMOTABLE = "promotable"
AUTORESEARCH_CANDIDATE_STATUS_BLOCKED = "blocked"


# ---- Validation / extraction helpers ----


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


# ---- Public API ----


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
