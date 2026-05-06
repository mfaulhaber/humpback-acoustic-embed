"""Tests for the retrieval-aware masked-transformer sweep CLI."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from humpback.models.processing import JobStatus


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "masked_transformer_retrieval_sweep.py"
)
spec = importlib.util.spec_from_file_location(
    "masked_transformer_retrieval_sweep", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)


class _FakeEngine:
    async def dispose(self) -> None:
        return None


class _FakeFactory:
    def __call__(self):
        return self

    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeJob:
    def __init__(self, job_id: str, status: str = JobStatus.queued.value) -> None:
        self.id = job_id
        self.status = status


def test_submit_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["submit", "--dry-run"])

    assert args.command == "submit"
    assert args.preset == cli.INITIAL_SWEEP_PRESET
    assert args.k_values == cli.DEFAULT_K_VALUES
    assert args.lambda_values == cli.DEFAULT_LAMBDAS


def test_dry_run_initial_preset_writes_manifest(tmp_path, capsys) -> None:
    exit_code = cli.main(
        [
            "submit",
            "--dry-run",
            "--output-dir",
            str(tmp_path),
            "--continuous-embedding-job-id-250ms",
            "cej-250",
            "--continuous-embedding-job-id-100ms",
            "cej-100",
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "submit-manifest.json").read_text())
    assert manifest["dry_run"] is True
    assert manifest["runs"][0]["action"] == "compare_existing"
    assert manifest["runs"][3]["run_name"] == "250ms-projection-head-only-ablation"
    assert manifest["runs"][3]["action"] == "submit"
    assert manifest["runs"][3]["metadata"]["source_masked_transformer_job_id"]
    assert (
        manifest["runs"][3]["metadata"]["failure_mode_probe"]
        == "projection_head_only_metric_learning"
    )
    assert "negative_label_family_policy_json" in manifest["runs"][3]["create_payload"]
    assert manifest["runs"][4]["run_name"] == "250ms-sampler-confirm-lambda-0p10"
    assert manifest["runs"][4]["action"] == "blocked"
    assert "wrote" in capsys.readouterr().out


def test_submit_calls_services_with_normalized_payloads(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []

    async def fake_create(_session, **payload):
        calls.append(payload)
        return _FakeJob("job-1", status=JobStatus.complete.value), False

    async def fake_extend(_session, job_id, k_values):
        calls.append({"extend": job_id, "k_values": k_values})
        return _FakeJob(job_id)

    monkeypatch.setattr(cli, "create_engine", lambda _url: _FakeEngine())
    monkeypatch.setattr(cli, "create_session_factory", lambda _engine: _FakeFactory())
    monkeypatch.setattr(cli, "create_masked_transformer_job", fake_create)
    monkeypatch.setattr(cli, "extend_k_sweep_job", fake_extend)

    exit_code = cli.main(
        [
            "submit",
            "--preset",
            "lambda",
            "--continuous-embedding-job-id",
            "cej-250",
            "--event-classification-job-id",
            "cls-1",
            "--lambda-values",
            "0.10",
            "--k-values",
            "150,300",
            "--extend-k-sweep",
            "--geometry-gate-passed",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert calls[0]["continuous_embedding_job_id"] == "cej-250"
    assert calls[0]["contrastive_loss_weight"] == 0.10
    assert calls[0]["k_values"] == [150, 300]
    assert calls[1] == {"extend": "job-1", "k_values": [150, 300]}


def test_submit_records_service_error(tmp_path, monkeypatch) -> None:
    async def fake_create(_session, **_payload):
        raise ValueError("cannot resolve Classify binding")

    monkeypatch.setattr(cli, "create_engine", lambda _url: _FakeEngine())
    monkeypatch.setattr(cli, "create_session_factory", lambda _engine: _FakeFactory())
    monkeypatch.setattr(cli, "create_masked_transformer_job", fake_create)

    exit_code = cli.main(
        [
            "submit",
            "--preset",
            "lambda",
            "--continuous-embedding-job-id",
            "cej-250",
            "--lambda-values",
            "0.10",
            "--geometry-gate-passed",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "submit-manifest.json").read_text())
    assert manifest["runs"][0]["submission_status"] == "failed"
    assert "Classify binding" in manifest["runs"][0]["error"]


def test_submit_blocks_lambda_without_geometry_gate(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []

    async def fake_create(_session, **payload):
        calls.append(payload)
        return _FakeJob("job-1"), True

    monkeypatch.setattr(cli, "create_engine", lambda _url: _FakeEngine())
    monkeypatch.setattr(cli, "create_session_factory", lambda _engine: _FakeFactory())
    monkeypatch.setattr(cli, "create_masked_transformer_job", fake_create)

    exit_code = cli.main(
        [
            "submit",
            "--preset",
            "lambda",
            "--continuous-embedding-job-id",
            "cej-250",
            "--lambda-values",
            "0.10",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "submit-manifest.json").read_text())
    assert manifest["runs"][0]["submission_status"] == "blocked"
    assert "unsaturated retrieval raw geometry" in manifest["runs"][0]["blocked_reason"]
    assert calls == []


def test_compare_keeps_failed_jobs_in_outputs(tmp_path, monkeypatch) -> None:
    async def fake_report(*_args, **_kwargs):
        raise RuntimeError("artifact missing")

    monkeypatch.setattr(cli, "create_engine", lambda _url: _FakeEngine())
    monkeypatch.setattr(cli, "create_session_factory", lambda _engine: _FakeFactory())
    monkeypatch.setattr(cli, "build_nearest_neighbor_report", fake_report)

    exit_code = cli.main(
        [
            "compare",
            "--job",
            "trial=job-1:retrieval",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert comparison["rows"][0]["status"] == "failed"
    assert comparison["rows"][0]["error"] == "artifact missing"


def test_compare_manifest_uses_row_k_when_no_override(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_name": "trial",
                        "job_id": "job-1",
                        "embedding_spaces": ["retrieval"],
                        "k_values": [100],
                    }
                ]
            }
        )
    )
    seen: list[int | None] = []

    async def fake_report(_session, *, storage_root, job_id, options):
        seen.append(options.k)
        assert options.include_geometry_report is True
        return {
            "job": {"job_id": job_id, "k": options.k},
            "options": {"embedding_space": options.embedding_space},
            "label_coverage": {},
            "results": {
                cli.REQUIRED_RETRIEVAL_MODE: {
                    cli.PRIMARY_VARIANT: {"same_human_label": 0.5}
                }
            },
        }

    monkeypatch.setattr(cli, "create_engine", lambda _url: _FakeEngine())
    monkeypatch.setattr(cli, "create_session_factory", lambda _engine: _FakeFactory())
    monkeypatch.setattr(cli, "build_nearest_neighbor_report", fake_report)

    exit_code = cli.main(
        [
            "compare",
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    assert seen == [100]


def test_submit_async_function_can_be_called_directly(tmp_path) -> None:
    args = cli.build_parser().parse_args(
        ["submit", "--dry-run", "--output-dir", str(tmp_path)]
    )

    assert asyncio.run(cli.submit_runs(args)) == 0
