"""Tests for retrieval sweep comparison artifacts."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np

from humpback.sequence_models.retrieval_sweeps import (
    ComparisonRow,
    render_markdown_comparison,
    write_comparison_outputs,
)


def _rows() -> list[ComparisonRow]:
    return [
        ComparisonRow(
            run_name="run-b",
            job_id="job-b",
            k=150,
            embedding_space="retrieval",
            primary_metric=0.2,
            variant_same_human_label={"raw_l2": 0.2, "whiten_pca": 0.3},
            human_labeled_effective_events=4,
            single_label_effective_events=4,
        ),
        ComparisonRow(
            run_name="run-a",
            job_id="job-a",
            k=150,
            embedding_space="retrieval",
            primary_metric=0.4,
            variant_same_human_label={"raw_l2": 0.4, "remove_pc10": 0.5},
            random_pair_percentiles={"50": cast(Any, np.float32(0.12))},
            human_labeled_effective_events=5,
            single_label_effective_events=5,
        ),
    ]


def test_write_comparison_outputs_are_deterministic(tmp_path) -> None:
    first = write_comparison_outputs(
        _rows(),
        tmp_path,
        diagnostic_options={"seed": 7},
    )
    first_csv = first.csv_path.read_text()
    first_md = first.markdown_path.read_text()
    first_json = first.json_path.read_text()

    second = write_comparison_outputs(
        list(reversed(_rows())),
        tmp_path,
        diagnostic_options={"seed": 7},
    )

    assert second.csv_path.name == "comparison.csv"
    assert second.markdown_path.name == "comparison.md"
    assert second.json_path.name == "comparison.json"
    assert second.csv_path.read_text() == first_csv
    assert second.markdown_path.read_text() == first_md
    assert second.json_path.read_text() == first_json


def test_markdown_contains_ranked_table_and_coverage() -> None:
    markdown = render_markdown_comparison(_rows(), diagnostic_options={"k": 150})

    assert "# Retrieval-Aware Transformer Sweep Comparison" in markdown
    assert "| run-a | retrieval | complete | 40.0%" in markdown
    assert "## Label Coverage" in markdown
    assert "## Diagnostic Options" in markdown


def test_json_output_is_plain_serializable(tmp_path) -> None:
    paths = write_comparison_outputs(_rows(), tmp_path)

    payload = json.loads(paths.json_path.read_text())

    assert payload["rows"][0]["run_name"] == "run-a"
    assert isinstance(payload["rows"][0]["random_pair_percentiles"]["50"], float)
