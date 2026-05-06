"""Tests for retrieval-aware transformer sweep helpers."""

from __future__ import annotations

import pytest

from humpback.schemas.sequence_models import MaskedTransformerJobCreate
from humpback.sequence_models import retrieval_sweeps as sweeps


def test_lambda_sweep_expands_deterministically() -> None:
    first = sweeps.build_lambda_sweep(
        continuous_embedding_job_id="cej-250",
        event_classification_job_id="cls-1",
    )
    second = sweeps.build_lambda_sweep(
        continuous_embedding_job_id="cej-250",
        event_classification_job_id="cls-1",
    )

    assert [run.run_name for run in first] == [
        "lambda-weight-0p05",
        "lambda-weight-0p10",
        "lambda-weight-0p25",
        "lambda-weight-0p50",
    ]
    assert [run.to_manifest_row() for run in first] == [
        run.to_manifest_row() for run in second
    ]


def test_lambda_sweep_allows_caller_overrides() -> None:
    runs = sweeps.build_lambda_sweep(
        continuous_embedding_job_id="cej-250",
        lambda_values=(0.2, 0.3),
        k_values=(150, 300),
        batch_size=12,
        labels_per_batch=3,
        events_per_label=5,
    )

    assert [run.run_name for run in runs] == [
        "lambda-weight-0p20",
        "lambda-weight-0p30",
    ]
    assert runs[0].create_payload["k_values"] == [150, 300]
    assert runs[0].create_payload["batch_size"] == 12
    assert runs[0].create_payload["contrastive_labels_per_batch"] == 3
    assert runs[0].create_payload["contrastive_events_per_label"] == 5
    assert runs[0].create_payload["retrieval_head_arch"] == "mlp"


def test_lambda_sweep_can_include_linear_head_variant() -> None:
    runs = sweeps.build_lambda_sweep(
        continuous_embedding_job_id="cej-250",
        event_classification_job_id="cls-1",
        lambda_values=(0.1,),
        include_linear_head=True,
    )

    assert [run.run_name for run in runs] == [
        "lambda-weight-0p10",
        "lambda-linear-head-weight-0p10",
    ]
    assert runs[0].create_payload["retrieval_head_arch"] == "mlp"
    assert runs[0].create_payload["retrieval_hidden_dim"] == 512
    assert runs[1].create_payload["retrieval_head_arch"] == "linear"
    assert runs[1].create_payload["retrieval_hidden_dim"] is None
    assert runs[1].metadata["failure_mode_probe"] == "linear_projection_head"
    assert runs[1].metadata["matched_mlp_run_name"] == "lambda-weight-0p10"

    parsed = MaskedTransformerJobCreate.model_validate(runs[1].create_payload)
    assert parsed.retrieval_head_arch == "linear"
    assert parsed.retrieval_hidden_dim is None


def test_unsupported_hard_negative_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported policy fields"):
        sweeps.validate_policy_variant({"hard_negative_policy": "semi_hard"})


def test_generated_contrastive_payload_validates_against_create_schema() -> None:
    run = sweeps.build_lambda_sweep(
        continuous_embedding_job_id="cej-250",
        event_classification_job_id="cls-1",
    )[0]

    parsed = MaskedTransformerJobCreate.model_validate(run.create_payload)

    assert parsed.retrieval_head_enabled is True
    assert parsed.retrieval_head_arch == "mlp"
    assert parsed.contrastive_loss_weight == 0.05
    assert parsed.contrastive_label_source == "human_corrections"
    assert parsed.sequence_construction_mode == "mixed"


def test_initial_sweep_preset_order_and_metadata() -> None:
    runs = sweeps.build_initial_sweep_preset(
        continuous_embedding_job_id_250ms="cej-250",
        continuous_embedding_job_id_100ms="cej-100",
        event_classification_job_id="cls-1",
    )

    assert [run.run_name for run in runs[:4]] == [
        "baseline-250ms-stage0-contextual",
        "baseline-250ms-pre-sampler-contrastive",
        "baseline-100ms-completed-contrastive",
        "250ms-projection-head-only-ablation",
    ]
    assert runs[0].job_id == sweeps.BASELINE_STAGE0_JOB_ID
    assert runs[1].job_id == sweeps.PRE_SAMPLER_CONTRASTIVE_JOB_ID
    assert runs[2].job_id == sweeps.COMPLETED_100MS_JOB_ID
    assert all(
        run.metadata["label_semantics"] == sweeps.LABEL_SEMANTICS for run in runs
    )
    assert runs[3].runnable is True
    assert runs[3].create_payload["training_freeze_mode"] == (
        "transformer_frozen_projection_head_only"
    )
    assert runs[3].create_payload["retrieval_head_arch"] == "mlp"
    assert (
        runs[3].create_payload["source_masked_transformer_job_id"]
        == sweeps.PRE_SAMPLER_CONTRASTIVE_JOB_ID
    )
    assert "negative_label_family_policy_json" in runs[3].create_payload
    assert (
        runs[3].metadata["failure_mode_probe"] == "projection_head_only_metric_learning"
    )
    assert runs[3].metadata["retrieval_head_arch"] == "mlp"
    assert "projection-head geometry" in str(runs[4].blocked_reason)


def test_initial_sweep_can_include_linear_head_probe() -> None:
    runs = sweeps.build_initial_sweep_preset(
        continuous_embedding_job_id_250ms="cej-250",
        event_classification_job_id="cls-1",
        include_linear_head=True,
    )

    linear = next(
        run for run in runs if run.run_name == "250ms-linear-head-confirm-lambda-0p10"
    )
    assert linear.create_payload["retrieval_head_arch"] == "linear"
    assert linear.create_payload["retrieval_hidden_dim"] is None
    assert linear.metadata["failure_mode_probe"] == "linear_projection_head"
    assert (
        linear.metadata["matched_mlp_run_name"] == "250ms-sampler-confirm-lambda-0p10"
    )


def test_initial_sweep_ablation_payload_validates_against_create_schema() -> None:
    run = sweeps.build_initial_sweep_preset(
        continuous_embedding_job_id_250ms="cej-250",
        event_classification_job_id="cls-1",
    )[3]

    parsed = MaskedTransformerJobCreate.model_validate(run.create_payload)

    assert parsed.training_freeze_mode == "transformer_frozen_projection_head_only"
    assert parsed.source_masked_transformer_job_id == (
        sweeps.PRE_SAMPLER_CONTRASTIVE_JOB_ID
    )
    assert parsed.negative_label_family_policy_json is not None


def test_rank_uses_cross_region_raw_same_human_label() -> None:
    low = sweeps.comparison_row_from_report(
        "low",
        {
            "job": {"job_id": "job-low", "k": 150},
            "options": {"embedding_space": "retrieval"},
            "label_coverage": {"single_label_effective_events": 4},
            "results": {
                sweeps.REQUIRED_RETRIEVAL_MODE: {
                    sweeps.PRIMARY_VARIANT: {"same_human_label": 0.2}
                }
            },
        },
    )
    high = sweeps.comparison_row_from_report(
        "high",
        {
            "job": {"job_id": "job-high", "k": 150},
            "options": {"embedding_space": "retrieval"},
            "label_coverage": {"single_label_effective_events": 4},
            "results": {
                sweeps.REQUIRED_RETRIEVAL_MODE: {
                    sweeps.PRIMARY_VARIANT: {"same_human_label": 0.6}
                }
            },
        },
    )

    assert [row.run_name for row in sweeps.rank_comparison_rows([low, high])] == [
        "high",
        "low",
    ]


def test_failure_rows_stay_in_ranked_output() -> None:
    complete = sweeps.ComparisonRow(
        run_name="complete",
        job_id="job-ok",
        k=150,
        embedding_space="retrieval",
        primary_metric=0.1,
    )
    failed = sweeps.failure_row(
        run_name="failed",
        job_id="job-fail",
        k=150,
        embedding_space="retrieval",
        error="missing retrieval artifact",
    )

    ranked = sweeps.rank_comparison_rows([failed, complete])

    assert [row.status for row in ranked] == ["complete", "failed"]
    assert ranked[1].error == "missing retrieval artifact"


def test_label_cardinality_flattens_from_report() -> None:
    row = sweeps.comparison_row_from_report(
        "mixed-labels",
        {
            "job": {"job_id": "job-1", "k": 150},
            "options": {"embedding_space": "retrieval"},
            "label_coverage": {
                "human_labeled_effective_events": 3,
                "unlabeled_effective_events": 2,
                "single_label_effective_events": 2,
                "multi_label_effective_events": 1,
            },
            "results": {
                sweeps.REQUIRED_RETRIEVAL_MODE: {
                    sweeps.PRIMARY_VARIANT: {"same_human_label": 0.5}
                }
            },
        },
    )

    assert row.human_labeled_effective_events == 3
    assert row.unlabeled_effective_events == 2
    assert row.single_label_effective_events == 2
    assert row.multi_label_effective_events == 1


def test_geometry_metrics_flatten_from_report() -> None:
    row = sweeps.comparison_row_from_report(
        "geometry",
        {
            "job": {"job_id": "job-1", "k": 150},
            "options": {"embedding_space": "retrieval"},
            "label_coverage": {},
            "results": {
                sweeps.REQUIRED_RETRIEVAL_MODE: {
                    sweeps.PRIMARY_VARIANT: {"same_human_label": 0.5}
                }
            },
            "geometry_report": {
                "spaces": {
                    "retrieval.raw_l2": {
                        "random_pair_percentiles": {
                            "p50": 0.2,
                            "p75": 0.8,
                            "p95": 0.97,
                        },
                        "mean_vector_norm": 0.4,
                        "effective_rank": 6.0,
                        "pca_explained_variance": {
                            "pc1": 0.4,
                            "pc1_5": 0.75,
                            "pc1_10": 0.9,
                        },
                    }
                },
                "summary": {"lambda_sweeps_blocked": True},
            },
        },
    )

    assert row.retrieval_raw_geometry_p75 == 0.8
    assert row.retrieval_raw_mean_vector_norm == 0.4
    assert row.retrieval_raw_effective_rank == 6.0
    assert row.retrieval_raw_pc1_5 == 0.75
    assert row.lambda_sweeps_blocked is True


def test_first_sweep_stop_rules() -> None:
    rows = [
        sweeps.ComparisonRow(
            run_name="retrieval",
            job_id="job-r",
            k=150,
            embedding_space="retrieval",
            skipped_contrastive_batches=2.0,
            variant_same_human_label={"raw_l2": 0.35},
            event_level_primary_metric=0.4,
        ),
        sweeps.ComparisonRow(
            run_name="contextual",
            job_id="job-c",
            k=150,
            embedding_space="contextual",
            variant_same_human_label={"raw_l2": 0.30, "whiten_pca": 0.39},
        ),
    ]

    checks = sweeps.evaluate_first_sweep_stop_rules(rows)

    assert checks["retrieval_raw_geometry_unsaturated"] is None
    assert checks["skipped_batches_below_baseline"] is True
    assert checks["raw_retrieval_beats_contextual_raw"] is True
    assert checks["raw_retrieval_within_margin_of_contextual_whitened"] is True
    assert checks["event_level_retrieval_available"] is True
