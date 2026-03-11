"""Unit tests for scripts/stage_s3_epoch_cache.py."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.stage_s3_epoch_cache as stage_cache


def _required_args(*extra: str) -> list[str]:
    return [
        "--bucket",
        "example-bucket",
        "--prefix",
        "site/hls",
        "--start",
        "2025-07-12T00:00:00Z",
        "--hours",
        "1",
        *extra,
    ]


def _selection(
    matches: list[stage_cache.TimestampPrefix],
) -> stage_cache.PrefixSelection:
    return stage_cache.PrefixSelection(
        coarse_matches=matches,
        overlap_candidates=matches,
        final_matches=matches,
        overlap_inspected=len(matches),
        overlap_filter_used=True,
    )


def test_parser_defaults_to_execution_mode() -> None:
    parser = stage_cache.build_parser()
    args = parser.parse_args(_required_args())
    assert args.dry_run is False
    assert args.pre_count is True


def test_parser_accepts_dry_run_and_rejects_legacy_run_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = stage_cache.build_parser()
    args = parser.parse_args(_required_args("--dry-run"))
    assert args.dry_run is True

    with pytest.raises(SystemExit):
        parser.parse_args(_required_args("--run"))
    assert "unrecognized arguments: --run" in capsys.readouterr().err


def test_parser_supports_disabling_pre_count() -> None:
    parser = stage_cache.build_parser()
    args = parser.parse_args(_required_args("--no-pre-count"))
    assert args.pre_count is False


def test_main_dry_run_skips_s5cmd_and_prints_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    required: list[str] = []
    matches = [stage_cache.TimestampPrefix(epoch=1752303617, name="1752303617")]

    monkeypatch.setattr(
        stage_cache, "require_binary", lambda name: required.append(name)
    )
    monkeypatch.setattr(
        stage_cache, "aws_ls_prefixes_optimized", lambda **_kwargs: matches
    )
    monkeypatch.setattr(
        stage_cache,
        "resolve_matching_prefixes",
        lambda **_kwargs: _selection(matches),
    )

    def _fail_if_called(**_kwargs):
        pytest.fail("run_s5cmd_copy_commands should not be called in dry-run mode")

    monkeypatch.setattr(stage_cache, "run_s5cmd_copy_commands", _fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_s3_epoch_cache.py",
            *_required_args("--local-root", str(tmp_path), "--dry-run"),
        ],
    )

    assert stage_cache.main() == 0
    assert required == ["aws"]
    stdout = capsys.readouterr().out
    assert "Planned prefixes to download (1):" in stdout
    assert "1752303617 ->" in stdout
    assert "Dry-run copy commands:" in stdout
    assert "cp " in stdout
    assert "s3://example-bucket/site/hls/1752303617/*" in stdout


def test_main_execution_runs_per_prefix_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required: list[str] = []
    called: dict[str, object] = {}
    matches = [stage_cache.TimestampPrefix(epoch=1752303617, name="1752303617")]
    log_file = tmp_path / "download.log"

    monkeypatch.setattr(
        stage_cache, "require_binary", lambda name: required.append(name)
    )
    monkeypatch.setattr(
        stage_cache, "aws_ls_prefixes_optimized", lambda **_kwargs: matches
    )
    monkeypatch.setattr(
        stage_cache,
        "resolve_matching_prefixes",
        lambda **_kwargs: _selection(matches),
    )
    monkeypatch.setattr(
        stage_cache,
        "estimate_transfer_totals_with_breakdown",
        lambda **_kwargs: (10, 1000, {"1752303617": (10, 1000)}),
    )

    def _fake_run(
        commands: list[str],
        numworkers: int,
        log_path: Path | None,
        expected_files_total: int | None,
        expected_bytes_total: int | None,
        expected_by_prefix: dict[str, tuple[int, int]] | None,
    ) -> int:
        called["commands"] = commands
        called["numworkers"] = numworkers
        called["log_path"] = log_path
        called["expected_files_total"] = expected_files_total
        called["expected_bytes_total"] = expected_bytes_total
        called["expected_by_prefix"] = expected_by_prefix
        return 0

    monkeypatch.setattr(stage_cache, "run_s5cmd_copy_commands", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_s3_epoch_cache.py",
            *_required_args(
                "--local-root",
                str(tmp_path),
                "--numworkers",
                "32",
                "--log-file",
                str(log_file),
            ),
        ],
    )

    assert stage_cache.main() == 0
    assert required == ["aws", "s5cmd"]
    assert called["numworkers"] == 32
    assert called["log_path"] == log_file.resolve()
    assert isinstance(called["commands"], list) and len(called["commands"]) == 1
    assert called["expected_files_total"] == 10
    assert called["expected_bytes_total"] == 1000
    assert called["expected_by_prefix"] == {"1752303617": (10, 1000)}


def test_main_execution_no_pre_count_skips_estimation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required: list[str] = []
    called: dict[str, object] = {}
    matches = [stage_cache.TimestampPrefix(epoch=1752303617, name="1752303617")]

    monkeypatch.setattr(
        stage_cache, "require_binary", lambda name: required.append(name)
    )
    monkeypatch.setattr(
        stage_cache, "aws_ls_prefixes_optimized", lambda **_kwargs: matches
    )
    monkeypatch.setattr(
        stage_cache,
        "resolve_matching_prefixes",
        lambda **_kwargs: _selection(matches),
    )
    monkeypatch.setattr(
        stage_cache,
        "estimate_transfer_totals_with_breakdown",
        lambda **_kwargs: pytest.fail(
            "estimate_transfer_totals_with_breakdown should not be called"
        ),
    )

    def _fake_run(
        commands: list[str],
        numworkers: int,
        log_path: Path | None,
        expected_files_total: int | None,
        expected_bytes_total: int | None,
        expected_by_prefix: dict[str, tuple[int, int]] | None,
    ) -> int:
        called["commands"] = commands
        called["numworkers"] = numworkers
        called["log_path"] = log_path
        called["expected_files_total"] = expected_files_total
        called["expected_bytes_total"] = expected_bytes_total
        called["expected_by_prefix"] = expected_by_prefix
        return 0

    monkeypatch.setattr(stage_cache, "run_s5cmd_copy_commands", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage_s3_epoch_cache.py",
            *_required_args("--local-root", str(tmp_path), "--no-pre-count"),
        ],
    )

    assert stage_cache.main() == 0
    assert required == ["aws", "s5cmd"]
    assert called["expected_files_total"] is None
    assert called["expected_bytes_total"] is None
    assert called["expected_by_prefix"] is None


def test_run_s5cmd_copy_commands_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    progress = {"total": 0, "updates": 0}
    postfix_values: list[str] = []

    class _FakeBar:
        def __init__(self, *, total: int, desc: str, unit: str, unit_scale: bool):
            assert desc == "Downloading"
            assert unit == "B"
            assert unit_scale is True
            progress["total"] = total

        def __enter__(self) -> "_FakeBar":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def update(self, value: int) -> None:
            progress["updates"] += value

        def set_postfix_str(self, value: str) -> None:
            postfix_values.append(value)

        @property
        def total(self) -> int:
            return progress["total"]

        @property
        def n(self) -> int:
            return progress["updates"]

    class _FakePopen:
        def __init__(
            self,
            cmd: list[str],
            stdout,
            stderr,
            text,
            encoding,
            errors,
            bufsize,
        ):
            assert stdout == stage_cache.subprocess.PIPE
            assert stderr == stage_cache.subprocess.STDOUT
            assert text is True
            assert encoding == "utf-8"
            assert errors == "replace"
            assert bufsize == 1
            calls.append(cmd)
            self.stdout = io.StringIO(
                (
                    '{"operation":"cp","success":true,"object":{"type":"file","size":1234}}\n'
                    '{"operation":"cp","success":true,"object":{"type":"file","size":5678}}\n'
                )
            )
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(stage_cache, "tqdm", _FakeBar)
    monkeypatch.setattr(stage_cache.subprocess, "Popen", _FakePopen)

    exit_code = stage_cache.run_s5cmd_copy_commands(
        commands=[
            "cp s3://bucket/site/hls/1752303617/* /tmp/cache/site/hls/1752303617/",
            "cp s3://bucket/site/hls/1752303677/* /tmp/cache/site/hls/1752303677/",
        ],
        numworkers=16,
        expected_bytes_total=2 * 1024 * 1024,
        expected_by_prefix={
            "1752303617": (10, 1024 * 1024),
            "1752303677": (10, 1024 * 1024),
        },
    )

    assert exit_code == 0
    assert len(calls) == 2
    assert calls[0][:6] == [
        "s5cmd",
        "--no-sign-request",
        "--numworkers",
        "16",
        "--json",
        "--log",
    ]
    assert calls[0][6] == "info"
    assert "cp" in calls[0]
    assert "--show-progress" not in calls[0]
    assert progress["updates"] > 0
    assert any("current=1752303617" in value for value in postfix_values)
    assert any("current=1752303677" in value for value in postfix_values)


def test_run_s5cmd_copy_commands_returns_failure_and_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _FakeBar:
        def __init__(self, *, total: int, desc: str, unit: str, unit_scale: bool):
            del total, desc, unit, unit_scale

        def __enter__(self) -> "_FakeBar":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def update(self, value: int) -> None:
            del value

        def set_postfix_str(self, value: str) -> None:
            del value

        @property
        def total(self) -> int:
            return 1

        @property
        def n(self) -> int:
            return 0

    class _FakePopen:
        def __init__(
            self,
            cmd: list[str],
            stdout,
            stderr,
            text,
            encoding,
            errors,
            bufsize,
        ):
            del cmd
            assert stdout == stage_cache.subprocess.PIPE
            assert stderr == stage_cache.subprocess.STDOUT
            assert text is True
            assert encoding == "utf-8"
            assert errors == "replace"
            assert bufsize == 1
            self.stdout = io.StringIO('{"operation":"cp","error":"network timeout"}\n')
            self.returncode = 9

        def wait(self) -> int:
            return self.returncode

    monkeypatch.setattr(stage_cache, "tqdm", _FakeBar)
    monkeypatch.setattr(stage_cache.subprocess, "Popen", _FakePopen)

    exit_code = stage_cache.run_s5cmd_copy_commands(
        commands=[
            "cp s3://bucket/site/hls/1752303617/* /tmp/cache/site/hls/1752303617/"
        ],
        numworkers=8,
    )

    assert exit_code == 9
    captured = capsys.readouterr()
    assert "s5cmd failed for prefix command 1/1" in captured.err
    assert "network timeout" in captured.err


def test_run_s5cmd_copy_commands_log_mode_tees_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    class _FakePopen:
        def __init__(
            self,
            cmd: list[str],
            stdout,
            stderr,
            text,
            encoding,
            errors,
            bufsize,
        ):
            assert stdout == stage_cache.subprocess.PIPE
            assert stderr == stage_cache.subprocess.STDOUT
            assert text is True
            assert encoding == "utf-8"
            assert errors == "replace"
            assert bufsize == 1
            calls.append(cmd)
            self.stdout = io.StringIO(
                '{"operation":"cp","success":true,"object":{"type":"file","size":42}}\n'
            )
            self.returncode = 0

        def wait(self) -> int:
            return self.returncode

    class _FakeBar:
        def __init__(self, *, total: int, desc: str, unit: str, unit_scale: bool):
            del total, desc, unit, unit_scale

        def __enter__(self) -> "_FakeBar":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def update(self, value: int) -> None:
            del value

        def set_postfix_str(self, value: str) -> None:
            del value

        @property
        def total(self) -> int:
            return 1

        @property
        def n(self) -> int:
            return 0

    monkeypatch.setattr(stage_cache, "tqdm", _FakeBar)
    monkeypatch.setattr(stage_cache.subprocess, "Popen", _FakePopen)

    log_path = tmp_path / "s5.log"
    exit_code = stage_cache.run_s5cmd_copy_commands(
        commands=[
            "cp s3://bucket/site/hls/1752303617/* /tmp/cache/site/hls/1752303617/"
        ],
        numworkers=8,
        log_path=log_path,
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert "--json" in calls[0]
    assert "--show-progress" not in calls[0]
    assert '"success":true' in log_path.read_text(encoding="utf-8")


def test_aws_ls_prefixes_optimized_uses_start_after_and_early_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    payload = {
        "CommonPrefixes": [
            {"Prefix": "site/hls/1752303617/"},
            {"Prefix": "site/hls/1752307217/"},
        ],
        "IsTruncated": True,
        "NextContinuationToken": "next-token",
    }

    def _fake_subprocess_run(
        cmd: list[str], text: bool, capture_output: bool, check: bool
    ):
        assert text is True
        assert capture_output is True
        assert check is True
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(stage_cache.subprocess, "run", _fake_subprocess_run)

    prefixes = stage_cache.aws_ls_prefixes_optimized(
        bucket="example-bucket",
        prefix="site/hls",
        region="us-west-2",
        start_epoch=1752303617,
        end_epoch=1752304000,
        start_after_lookback_seconds=0,
    )

    assert [p.name for p in prefixes] == ["1752303617", "1752307217"]
    assert len(calls) == 1
    first_call = calls[0]
    assert first_call[:8] == [
        "aws",
        "--region",
        "us-west-2",
        "--no-sign-request",
        "s3api",
        "list-objects-v2",
        "--bucket",
        "example-bucket",
    ]
    assert "--delimiter" in first_call
    assert "--start-after" in first_call
    assert first_call[first_call.index("--start-after") + 1] == "site/hls/1752303617/"


def test_aws_ls_prefixes_optimized_paginates_with_continuation_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []
    payloads = [
        {
            "CommonPrefixes": [{"Prefix": "site/hls/1752303617/"}],
            "IsTruncated": True,
            "NextContinuationToken": "token-2",
        },
        {
            "CommonPrefixes": [{"Prefix": "site/hls/1752303677/"}],
            "IsTruncated": False,
        },
    ]

    def _fake_subprocess_run(
        cmd: list[str], text: bool, capture_output: bool, check: bool
    ):
        assert text is True
        assert capture_output is True
        assert check is True
        calls.append(cmd)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps(payloads.pop(0)), stderr=""
        )

    monkeypatch.setattr(stage_cache.subprocess, "run", _fake_subprocess_run)

    prefixes = stage_cache.aws_ls_prefixes_optimized(
        bucket="example-bucket",
        prefix="site/hls",
        region="us-west-2",
        start_epoch=1752300000,
        end_epoch=1752310000,
        start_after_lookback_seconds=0,
    )

    assert [p.name for p in prefixes] == ["1752303617", "1752303677"]
    assert len(calls) == 2
    assert "--start-after" in calls[0]
    assert "--continuation-token" in calls[1]
    assert "--start-after" not in calls[1]
