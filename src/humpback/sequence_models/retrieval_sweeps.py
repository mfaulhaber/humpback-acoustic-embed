"""Sweep planning and comparison helpers for retrieval-aware transformers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from humpback.services.masked_transformer_service import (
    default_related_label_policy_json,
)

LABEL_SEMANTICS = "authoritative_single_human_label"
INITIAL_SWEEP_PRESET = "initial-retrieval-aware-sweep"
DEFAULT_LAMBDAS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)
DEFAULT_K_VALUES: tuple[int, ...] = (150,)
REQUIRED_RETRIEVAL_MODE = "exclude_same_event_and_region"
PRIMARY_VARIANT = "raw_l2"
OUTPUT_VARIANTS: tuple[str, ...] = (
    "raw_l2",
    "centered_l2",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "remove_pc10",
    "whiten_pca",
)

BASELINE_STAGE0_JOB_ID = "9fd95e63-9f06-4cfb-8242-63a03dbbedd0"
PRE_SAMPLER_CONTRASTIVE_JOB_ID = "63b72897-fb98-44d3-ac2d-2354a4d3f515"
COMPLETED_100MS_JOB_ID = "5e160936-2f5a-4a10-9311-452d818d8ac9"
COMPLETED_100MS_CEJ_ID = "42900b68-d830-40e0-af4b-e8f0a20456e7"
FAILED_100MS_OOM_JOB_ID = "56d5700a-0f1b-45c2-9c30-a8151757c6fa"

ALLOWED_POLICY_FIELDS = {
    "related_label_policy_json",
    "require_cross_region_positive",
    "contrastive_sampler_enabled",
    "contrastive_labels_per_batch",
    "contrastive_events_per_label",
    "contrastive_max_unlabeled_fraction",
    "contrastive_region_balance",
}


def _metadata(**values: Any) -> dict[str, Any]:
    return {"label_semantics": LABEL_SEMANTICS, **values}


@dataclass(frozen=True)
class SweepRun:
    """One planned comparison or submit action in a retrieval sweep."""

    run_name: str
    action: Literal["compare_existing", "submit"]
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    job_id: str | None = None
    embedding_spaces: tuple[str, ...] = ("retrieval",)
    create_payload: dict[str, Any] = field(default_factory=dict)
    known_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    blocked_reason: str | None = None

    @property
    def runnable(self) -> bool:
        return self.blocked_reason is None

    def to_manifest_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["runnable"] = self.runnable
        if not self.runnable:
            payload["planned_action"] = payload["action"]
            payload["action"] = "blocked"
        return payload


@dataclass(frozen=True)
class ComparisonRow:
    """One ranked comparison row for a job/config/k/embedding-space report."""

    run_name: str
    job_id: str
    k: int | None
    embedding_space: str
    status: str = "complete"
    error: str | None = None
    primary_metric: float | None = None
    exact_label_set: float | None = None
    event_level_primary_metric: float | None = None
    same_event_rate: float | None = None
    same_region_rate: float | None = None
    similar_duration_rate: float | None = None
    good_queries: int | None = None
    mixed_queries: int | None = None
    bad_queries: int | None = None
    human_labeled_effective_events: int | None = None
    unlabeled_effective_events: int | None = None
    single_label_effective_events: int | None = None
    multi_label_effective_events: int | None = None
    skipped_contrastive_batches: float | None = None
    retrieval_raw_geometry_p50: float | None = None
    retrieval_raw_geometry_p75: float | None = None
    retrieval_raw_geometry_p95: float | None = None
    retrieval_raw_mean_vector_norm: float | None = None
    retrieval_raw_effective_rank: float | None = None
    retrieval_raw_pc1: float | None = None
    retrieval_raw_pc1_5: float | None = None
    retrieval_raw_pc1_10: float | None = None
    lambda_sweeps_blocked: bool | None = None
    variant_same_human_label: dict[str, float] = field(default_factory=dict)
    random_pair_percentiles: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for variant in OUTPUT_VARIANTS:
            row[f"{variant}_same_human_label"] = self.variant_same_human_label.get(
                variant
            )
        row.pop("variant_same_human_label")
        return _json_safe(row)


@dataclass(frozen=True)
class SweepOutputPaths:
    csv_path: Path
    markdown_path: Path
    json_path: Path


def validate_policy_variant(variant: dict[str, Any]) -> dict[str, Any]:
    """Validate that a policy variant only uses supported persisted fields."""
    unsupported = sorted(set(variant).difference(ALLOWED_POLICY_FIELDS))
    if unsupported:
        raise ValueError(
            "unsupported policy fields for Phase 5 sweep: " + ", ".join(unsupported)
        )
    return dict(variant)


def _run_name(prefix: str, **parts: Any) -> str:
    suffix = "-".join(
        f"{key}-{str(value).replace('.', 'p')}"
        for key, value in sorted(parts.items())
        if value is not None
    )
    return f"{prefix}-{suffix}" if suffix else prefix


def _base_contrastive_payload(
    continuous_embedding_job_id: str,
    *,
    event_classification_job_id: str | None,
    k_values: tuple[int, ...],
    contrastive_loss_weight: float,
    batch_size: int,
    labels_per_batch: int,
    events_per_label: int,
    pre_context: float = 2.0,
    post_context: float = 2.0,
    require_cross_region_positive: bool = True,
    related_label_policy_json: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "continuous_embedding_job_id": continuous_embedding_job_id,
        "event_classification_job_id": event_classification_job_id,
        "k_values": list(k_values),
        "retrieval_head_enabled": True,
        "retrieval_dim": 128,
        "retrieval_hidden_dim": 512,
        "retrieval_l2_normalize": True,
        "sequence_construction_mode": "mixed",
        "event_centered_fraction": 0.7,
        "pre_event_context_sec": pre_context,
        "post_event_context_sec": post_context,
        "contrastive_loss_weight": float(contrastive_loss_weight),
        "contrastive_temperature": 0.07,
        "contrastive_label_source": "human_corrections",
        "contrastive_min_events_per_label": 4,
        "contrastive_min_regions_per_label": 2,
        "require_cross_region_positive": require_cross_region_positive,
        "related_label_policy_json": related_label_policy_json
        or default_related_label_policy_json(),
        "contrastive_sampler_enabled": True,
        "contrastive_labels_per_batch": int(labels_per_batch),
        "contrastive_events_per_label": int(events_per_label),
        "contrastive_max_unlabeled_fraction": 0.25,
        "contrastive_region_balance": True,
        "batch_size": int(batch_size),
        "preset": "default",
        "seed": 42,
    }
    return payload


def build_initial_sweep_preset(
    *,
    continuous_embedding_job_id_250ms: str | None = None,
    continuous_embedding_job_id_100ms: str | None = None,
    event_classification_job_id: str | None = None,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
) -> list[SweepRun]:
    """Return the ordered first sweep plan from the Phase 5 implementation plan."""
    runs: list[SweepRun] = [
        SweepRun(
            run_name="baseline-250ms-stage0-contextual",
            action="compare_existing",
            job_id=BASELINE_STAGE0_JOB_ID,
            embedding_spaces=("contextual",),
            k_values=k_values,
            known_metrics={
                "raw_l2_same_human_label": 0.248,
                "remove_pc10_same_human_label": 0.318,
                "whiten_pca_same_human_label": 0.402,
            },
            metadata=_metadata(chunk_ms=250, baseline_kind="stage0"),
        ),
        SweepRun(
            run_name="baseline-250ms-pre-sampler-contrastive",
            action="compare_existing",
            job_id=PRE_SAMPLER_CONTRASTIVE_JOB_ID,
            embedding_spaces=("retrieval", "contextual"),
            k_values=k_values,
            metadata=_metadata(
                chunk_ms=250,
                baseline_kind="pre_sampler_contrastive",
                skipped_contrastive_batches=14.0,
            ),
        ),
        SweepRun(
            run_name="baseline-100ms-completed-contrastive",
            action="compare_existing",
            job_id=COMPLETED_100MS_JOB_ID,
            embedding_spaces=("retrieval", "contextual"),
            k_values=k_values,
            known_metrics={
                "retrieval_raw_l2_same_human_label": 0.184,
                "retrieval_whiten_pca_same_human_label": 0.342,
                "contextual_raw_l2_same_human_label": 0.248,
                "contextual_whiten_pca_same_human_label": 0.502,
            },
            metadata=_metadata(
                chunk_ms=100,
                continuous_embedding_job_id=COMPLETED_100MS_CEJ_ID,
                baseline_kind="completed_100ms",
            ),
        ),
    ]

    if continuous_embedding_job_id_250ms is None:
        blocked_250 = "requires --continuous-embedding-job-id-250ms"
    else:
        blocked_250 = None
    runs.append(
        SweepRun(
            run_name="250ms-projection-head-only-ablation",
            action="submit",
            k_values=k_values,
            create_payload=(
                {}
                if continuous_embedding_job_id_250ms is None
                else {
                    **_base_contrastive_payload(
                        continuous_embedding_job_id_250ms,
                        event_classification_job_id=event_classification_job_id,
                        k_values=k_values,
                        contrastive_loss_weight=1.0,
                        batch_size=16,
                        labels_per_batch=4,
                        events_per_label=4,
                    ),
                    "sequence_construction_mode": "region",
                    "event_centered_fraction": 0.0,
                    "pre_event_context_sec": None,
                    "post_event_context_sec": None,
                    "contrastive_temperature": 0.10,
                    "max_epochs": 10,
                    "early_stop_patience": 2,
                    "training_freeze_mode": "transformer_frozen_projection_head_only",
                    "source_masked_transformer_job_id": PRE_SAMPLER_CONTRASTIVE_JOB_ID,
                }
            ),
            metadata=_metadata(chunk_ms=250, sweep_stage="projection_head_ablation"),
            blocked_reason=blocked_250,
        )
    )
    runs.append(
        SweepRun(
            run_name="250ms-sampler-confirm-lambda-0p10",
            action="submit",
            k_values=k_values,
            create_payload=(
                {}
                if continuous_embedding_job_id_250ms is None
                else _base_contrastive_payload(
                    continuous_embedding_job_id_250ms,
                    event_classification_job_id=event_classification_job_id,
                    k_values=k_values,
                    contrastive_loss_weight=0.10,
                    batch_size=16,
                    labels_per_batch=4,
                    events_per_label=4,
                )
            ),
            metadata=_metadata(chunk_ms=250, sweep_stage="sampler_confirmation"),
            blocked_reason=blocked_250 or "awaits unsaturated projection-head geometry",
        )
    )

    for weight in DEFAULT_LAMBDAS:
        runs.append(
            SweepRun(
                run_name=_run_name("250ms-lambda", weight=f"{weight:.2f}"),
                action="submit",
                k_values=k_values,
                create_payload=(
                    {}
                    if continuous_embedding_job_id_250ms is None
                    else _base_contrastive_payload(
                        continuous_embedding_job_id_250ms,
                        event_classification_job_id=event_classification_job_id,
                        k_values=k_values,
                        contrastive_loss_weight=weight,
                        batch_size=16,
                        labels_per_batch=4,
                        events_per_label=4,
                    )
                ),
                metadata=_metadata(chunk_ms=250, sweep_stage="lambda"),
                blocked_reason=blocked_250
                or "awaits unsaturated projection-head geometry",
            )
        )

    for context_sec in (2.0, 4.0):
        runs.append(
            SweepRun(
                run_name=_run_name("250ms-context", seconds=f"{context_sec:.0f}"),
                action="submit",
                k_values=k_values,
                create_payload=(
                    {}
                    if continuous_embedding_job_id_250ms is None
                    else _base_contrastive_payload(
                        continuous_embedding_job_id_250ms,
                        event_classification_job_id=event_classification_job_id,
                        k_values=k_values,
                        contrastive_loss_weight=0.10,
                        batch_size=16,
                        labels_per_batch=4,
                        events_per_label=4,
                        pre_context=context_sec,
                        post_context=context_sec,
                    )
                ),
                metadata=_metadata(chunk_ms=250, sweep_stage="context_window"),
                blocked_reason=blocked_250 or "awaits best 250ms lambda",
            )
        )

    if continuous_embedding_job_id_100ms is None:
        blocked_100 = "requires --continuous-embedding-job-id-100ms"
    else:
        blocked_100 = None
    runs.append(
        SweepRun(
            run_name="100ms-memory-safe-confirm-lambda-0p10",
            action="submit",
            k_values=k_values,
            create_payload=(
                {}
                if continuous_embedding_job_id_100ms is None
                else _base_contrastive_payload(
                    continuous_embedding_job_id_100ms,
                    event_classification_job_id=event_classification_job_id,
                    k_values=k_values,
                    contrastive_loss_weight=0.10,
                    batch_size=4,
                    labels_per_batch=2,
                    events_per_label=2,
                )
            ),
            metadata=_metadata(
                chunk_ms=100,
                sweep_stage="memory_safe_confirmation",
                oom_reference_job_id=FAILED_100MS_OOM_JOB_ID,
            ),
            blocked_reason=blocked_100,
        )
    )

    for weight in (0.05, 0.10, 0.25):
        runs.append(
            SweepRun(
                run_name=_run_name("100ms-lambda", weight=f"{weight:.2f}"),
                action="submit",
                k_values=k_values,
                create_payload=(
                    {}
                    if continuous_embedding_job_id_100ms is None
                    else _base_contrastive_payload(
                        continuous_embedding_job_id_100ms,
                        event_classification_job_id=event_classification_job_id,
                        k_values=k_values,
                        contrastive_loss_weight=weight,
                        batch_size=4,
                        labels_per_batch=2,
                        events_per_label=2,
                    )
                ),
                metadata=_metadata(chunk_ms=100, sweep_stage="lambda"),
                blocked_reason=blocked_100 or "awaits 100ms memory-safe confirmation",
            )
        )

    for chunk_ms, cej in (
        (250, continuous_embedding_job_id_250ms),
        (100, continuous_embedding_job_id_100ms),
    ):
        for cross_region in (True, False):
            policy_json = default_related_label_policy_json()
            runs.append(
                SweepRun(
                    run_name=_run_name(
                        f"{chunk_ms}ms-policy", cross_region=str(cross_region).lower()
                    ),
                    action="submit",
                    k_values=k_values,
                    create_payload=(
                        {}
                        if cej is None
                        else _base_contrastive_payload(
                            cej,
                            event_classification_job_id=event_classification_job_id,
                            k_values=k_values,
                            contrastive_loss_weight=0.10,
                            batch_size=16 if chunk_ms == 250 else 4,
                            labels_per_batch=4 if chunk_ms == 250 else 2,
                            events_per_label=4 if chunk_ms == 250 else 2,
                            require_cross_region_positive=cross_region,
                            related_label_policy_json=policy_json,
                        )
                    ),
                    metadata=_metadata(
                        chunk_ms=chunk_ms, sweep_stage="policy_ablation"
                    ),
                    blocked_reason=(
                        f"requires --continuous-embedding-job-id-{chunk_ms}ms"
                        if cej is None
                        else f"awaits best {chunk_ms}ms lambda"
                    ),
                )
            )
        runs.append(
            SweepRun(
                run_name=f"{chunk_ms}ms-policy-empty-related-exclusions",
                action="submit",
                k_values=k_values,
                create_payload=(
                    {}
                    if cej is None
                    else _base_contrastive_payload(
                        cej,
                        event_classification_job_id=event_classification_job_id,
                        k_values=k_values,
                        contrastive_loss_weight=0.10,
                        batch_size=16 if chunk_ms == 250 else 4,
                        labels_per_batch=4 if chunk_ms == 250 else 2,
                        events_per_label=4 if chunk_ms == 250 else 2,
                        related_label_policy_json='{"exclude_pairs":[]}',
                    )
                ),
                metadata=_metadata(chunk_ms=chunk_ms, sweep_stage="policy_ablation"),
                blocked_reason=(
                    f"requires --continuous-embedding-job-id-{chunk_ms}ms"
                    if cej is None
                    else f"awaits best {chunk_ms}ms lambda"
                ),
            )
        )

    return runs


def build_lambda_sweep(
    *,
    continuous_embedding_job_id: str,
    event_classification_job_id: str | None = None,
    lambda_values: tuple[float, ...] = DEFAULT_LAMBDAS,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    batch_size: int = 16,
    labels_per_batch: int = 4,
    events_per_label: int = 4,
    policy_variant: dict[str, Any] | None = None,
) -> list[SweepRun]:
    """Build an ad hoc contrastive lambda sweep."""
    policy = validate_policy_variant(policy_variant or {})
    runs: list[SweepRun] = []
    for weight in lambda_values:
        payload = _base_contrastive_payload(
            continuous_embedding_job_id,
            event_classification_job_id=event_classification_job_id,
            k_values=k_values,
            contrastive_loss_weight=weight,
            batch_size=batch_size,
            labels_per_batch=labels_per_batch,
            events_per_label=events_per_label,
        )
        payload.update(policy)
        runs.append(
            SweepRun(
                run_name=_run_name("lambda", weight=f"{weight:.2f}"),
                action="submit",
                k_values=k_values,
                create_payload=payload,
                metadata=_metadata(),
            )
        )
    return runs


def comparison_row_from_report(
    run_name: str,
    report: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> ComparisonRow:
    """Flatten one nearest-neighbor report into a ranked comparison row."""
    mode_results = report.get("results", {}).get(REQUIRED_RETRIEVAL_MODE, {})
    raw_metrics = mode_results.get(PRIMARY_VARIANT, {})
    event_mode_results = (report.get("event_level_results") or {}).get(
        REQUIRED_RETRIEVAL_MODE, {}
    )
    event_raw_metrics = event_mode_results.get(PRIMARY_VARIANT, {})
    variant_same = {
        variant: float(metrics.get("same_human_label", 0.0))
        for variant, metrics in mode_results.items()
    }
    verdicts = raw_metrics.get("verdicts", {}) or {}
    good = int(verdicts.get("good", 0))
    mixed = sum(int(v) for k, v in verdicts.items() if str(k).startswith("mixed"))
    bad = sum(int(v) for k, v in verdicts.items() if str(k).startswith("bad"))
    label_coverage = report.get("label_coverage", {}) or {}
    job = report.get("job", {}) or {}
    options = report.get("options", {}) or {}
    geometry = report.get("geometry_report") or {}
    retrieval_raw_geometry = (geometry.get("spaces") or {}).get(
        "retrieval.raw_l2"
    ) or {}
    retrieval_raw_cosine = retrieval_raw_geometry.get("random_pair_percentiles") or {}
    retrieval_raw_pca = retrieval_raw_geometry.get("pca_explained_variance") or {}
    geometry_summary = geometry.get("summary") or {}
    meta = dict(metadata or {})
    return ComparisonRow(
        run_name=run_name,
        job_id=str(job.get("job_id", "")),
        k=int(job["k"]) if job.get("k") is not None else None,
        embedding_space=str(options.get("embedding_space", "")),
        primary_metric=variant_same.get(PRIMARY_VARIANT),
        exact_label_set=_optional_float(raw_metrics.get("exact_human_label_set")),
        event_level_primary_metric=_optional_float(
            event_raw_metrics.get("same_human_label")
        ),
        same_event_rate=_optional_float(raw_metrics.get("same_event")),
        same_region_rate=_optional_float(raw_metrics.get("same_region")),
        similar_duration_rate=_optional_float(raw_metrics.get("similar_duration")),
        good_queries=good,
        mixed_queries=mixed,
        bad_queries=bad,
        human_labeled_effective_events=_optional_int(
            label_coverage.get("human_labeled_effective_events")
        ),
        unlabeled_effective_events=_optional_int(
            label_coverage.get("unlabeled_effective_events")
        ),
        single_label_effective_events=_optional_int(
            label_coverage.get("single_label_effective_events")
        ),
        multi_label_effective_events=_optional_int(
            label_coverage.get("multi_label_effective_events")
        ),
        skipped_contrastive_batches=_optional_float(
            meta.get("skipped_contrastive_batches")
        ),
        retrieval_raw_geometry_p50=_optional_float(retrieval_raw_cosine.get("p50")),
        retrieval_raw_geometry_p75=_optional_float(retrieval_raw_cosine.get("p75")),
        retrieval_raw_geometry_p95=_optional_float(retrieval_raw_cosine.get("p95")),
        retrieval_raw_mean_vector_norm=_optional_float(
            retrieval_raw_geometry.get("mean_vector_norm")
        ),
        retrieval_raw_effective_rank=_optional_float(
            retrieval_raw_geometry.get("effective_rank")
        ),
        retrieval_raw_pc1=_optional_float(retrieval_raw_pca.get("pc1")),
        retrieval_raw_pc1_5=_optional_float(retrieval_raw_pca.get("pc1_5")),
        retrieval_raw_pc1_10=_optional_float(retrieval_raw_pca.get("pc1_10")),
        lambda_sweeps_blocked=(
            bool(geometry_summary["lambda_sweeps_blocked"])
            if "lambda_sweeps_blocked" in geometry_summary
            else None
        ),
        variant_same_human_label=variant_same,
        random_pair_percentiles={
            str(k): float(v)
            for k, v in (raw_metrics.get("random_pair_percentiles", {}) or {}).items()
        },
        metadata=meta,
    )


def failure_row(
    *,
    run_name: str,
    job_id: str,
    k: int | None,
    embedding_space: str,
    error: str,
    metadata: dict[str, Any] | None = None,
) -> ComparisonRow:
    return ComparisonRow(
        run_name=run_name,
        job_id=job_id,
        k=k,
        embedding_space=embedding_space,
        status="failed",
        error=error,
        metadata=dict(metadata or {}),
    )


def rank_comparison_rows(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    """Rank successful rows first by primary metric, then deterministic identity."""
    return sorted(
        rows,
        key=lambda row: (
            row.status != "complete",
            -(row.primary_metric if row.primary_metric is not None else -1.0),
            row.run_name,
            row.embedding_space,
            row.job_id,
        ),
    )


def evaluate_first_sweep_stop_rules(
    rows: list[ComparisonRow],
    *,
    skipped_batch_baseline: float = 14.0,
    contextual_whitened_margin: float = 0.05,
) -> dict[str, bool | None]:
    """Evaluate first-sweep go/no-go checks from comparison rows."""
    skipped_values = [
        row.skipped_contrastive_batches
        for row in rows
        if row.skipped_contrastive_batches is not None
    ]
    retrieval_raw = [
        row.variant_same_human_label.get("raw_l2", row.primary_metric)
        for row in rows
        if row.status == "complete" and row.embedding_space == "retrieval"
    ]
    contextual_raw = [
        row.variant_same_human_label.get("raw_l2", row.primary_metric)
        for row in rows
        if row.status == "complete" and row.embedding_space == "contextual"
    ]
    contextual_whitened = [
        row.variant_same_human_label.get("whiten_pca")
        for row in rows
        if row.status == "complete" and row.embedding_space == "contextual"
    ]
    retrieval_raw_values = [v for v in retrieval_raw if v is not None]
    contextual_raw_values = [v for v in contextual_raw if v is not None]
    contextual_whitened_values = [v for v in contextual_whitened if v is not None]
    geometry_blocked = [
        row.lambda_sweeps_blocked
        for row in rows
        if row.lambda_sweeps_blocked is not None
    ]

    return {
        "retrieval_raw_geometry_unsaturated": (
            None if not geometry_blocked else not any(geometry_blocked)
        ),
        "skipped_batches_below_baseline": (
            None
            if not skipped_values
            else min(skipped_values) < float(skipped_batch_baseline)
        ),
        "raw_retrieval_beats_contextual_raw": (
            None
            if not retrieval_raw_values or not contextual_raw_values
            else max(retrieval_raw_values) >= max(contextual_raw_values)
        ),
        "raw_retrieval_within_margin_of_contextual_whitened": (
            None
            if not retrieval_raw_values or not contextual_whitened_values
            else max(retrieval_raw_values)
            >= max(contextual_whitened_values) - float(contextual_whitened_margin)
        ),
        "event_level_retrieval_available": any(
            row.event_level_primary_metric is not None
            for row in rows
            if row.status == "complete" and row.embedding_space == "retrieval"
        ),
    }


def write_comparison_outputs(
    rows: list[ComparisonRow],
    output_dir: Path,
    *,
    timestamped: bool = False,
    diagnostic_options: dict[str, Any] | None = None,
) -> SweepOutputPaths:
    """Write stable CSV, Markdown, and JSON comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if timestamped:
        suffix = "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"comparison{suffix}.csv"
    md_path = output_dir / f"comparison{suffix}.md"
    json_path = output_dir / f"comparison{suffix}.json"
    ranked = rank_comparison_rows(rows)
    dict_rows = [row.to_dict() for row in ranked]

    fieldnames = _comparison_fieldnames(dict_rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)

    md_path.write_text(
        render_markdown_comparison(ranked, diagnostic_options=diagnostic_options),
        encoding="utf-8",
    )
    payload = {
        "diagnostic_options": diagnostic_options or {},
        "stop_rules": evaluate_first_sweep_stop_rules(ranked),
        "rows": dict_rows,
    }
    json_path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    return SweepOutputPaths(
        csv_path=csv_path, markdown_path=md_path, json_path=json_path
    )


def render_markdown_comparison(
    rows: list[ComparisonRow],
    *,
    diagnostic_options: dict[str, Any] | None = None,
) -> str:
    ranked = rank_comparison_rows(rows)
    lines = [
        "# Retrieval-Aware Transformer Sweep Comparison",
        "",
        "## Ranked Results",
        "",
        "| Run | Space | Status | Raw | Event Raw | Remove PC10 | Whiten | Retrieval p75 | Rank | Blocked | Error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.run_name,
                    row.embedding_space,
                    row.status,
                    _fmt_pct(row.variant_same_human_label.get("raw_l2")),
                    _fmt_pct(row.event_level_primary_metric),
                    _fmt_pct(row.variant_same_human_label.get("remove_pc10")),
                    _fmt_pct(row.variant_same_human_label.get("whiten_pca")),
                    _fmt_float(row.retrieval_raw_geometry_p75),
                    _fmt_float(row.retrieval_raw_effective_rank),
                    ""
                    if row.lambda_sweeps_blocked is None
                    else str(row.lambda_sweeps_blocked),
                    row.error or "",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Label Coverage", ""])
    lines.append(
        "| Run | Space | Human Events | Unlabeled | Single Label | Multi Label |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.run_name,
                    row.embedding_space,
                    _fmt_int(row.human_labeled_effective_events),
                    _fmt_int(row.unlabeled_effective_events),
                    _fmt_int(row.single_label_effective_events),
                    _fmt_int(row.multi_label_effective_events),
                ]
            )
            + " |"
        )
    if any((row.multi_label_effective_events or 0) > 0 for row in ranked):
        lines.extend(
            [
                "",
                "> Warning: at least one row observed multi-label human-corrected effective events; the Phase 5 primary ranking assumes authoritative single-label events.",
            ]
        )
    lines.extend(["", "## Stop Rules", ""])
    for key, value in evaluate_first_sweep_stop_rules(ranked).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            "- `Raw`: same authoritative human-label overlap for `exclude_same_event_and_region` / `raw_l2`.",
            "- `Event Raw`: event-level mean-pooled version of the same metric when requested.",
            "- `Remove PC10` and `Whiten`: the same overlap after PCA post-processing variants.",
            "- `Same Region`: fraction of retrieved neighbors from the query region; lower is better for cross-region retrieval.",
        ]
    )
    if diagnostic_options:
        lines.extend(["", "## Diagnostic Options", ""])
        lines.append("```json")
        lines.append(json.dumps(_json_safe(diagnostic_options), indent=2))
        lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _comparison_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "run_name",
        "job_id",
        "k",
        "embedding_space",
        "status",
        "error",
        "primary_metric",
        "raw_l2_same_human_label",
        "event_level_primary_metric",
        "remove_pc10_same_human_label",
        "whiten_pca_same_human_label",
        "same_event_rate",
        "same_region_rate",
        "similar_duration_rate",
        "good_queries",
        "mixed_queries",
        "bad_queries",
        "human_labeled_effective_events",
        "unlabeled_effective_events",
        "single_label_effective_events",
        "multi_label_effective_events",
        "skipped_contrastive_batches",
        "retrieval_raw_geometry_p50",
        "retrieval_raw_geometry_p75",
        "retrieval_raw_geometry_p95",
        "retrieval_raw_mean_vector_norm",
        "retrieval_raw_effective_rank",
        "retrieval_raw_pc1",
        "retrieval_raw_pc1_5",
        "retrieval_raw_pc1_10",
        "lambda_sweeps_blocked",
    ]
    seen = set(preferred)
    extras = sorted({key for row in rows for key in row if key not in seen})
    return [key for key in preferred if any(key in row for row in rows)] + extras


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{value * 100:.1f}%"


def _fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.3g}"


def _fmt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    return value


__all__ = [
    "DEFAULT_LAMBDAS",
    "DEFAULT_K_VALUES",
    "INITIAL_SWEEP_PRESET",
    "LABEL_SEMANTICS",
    "ComparisonRow",
    "SweepOutputPaths",
    "SweepRun",
    "build_initial_sweep_preset",
    "build_lambda_sweep",
    "comparison_row_from_report",
    "evaluate_first_sweep_stop_rules",
    "failure_row",
    "rank_comparison_rows",
    "render_markdown_comparison",
    "validate_policy_variant",
    "write_comparison_outputs",
]
