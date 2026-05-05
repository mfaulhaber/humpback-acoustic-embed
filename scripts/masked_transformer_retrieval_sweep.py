"""Plan, submit, and compare retrieval-aware masked-transformer sweeps."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.processing import JobStatus
from humpback.sequence_models.retrieval_diagnostics import (
    EmbeddingSpace,
    EmbeddingVariant,
    RetrievalMode,
    RetrievalReportOptions,
    build_nearest_neighbor_report,
)
from humpback.sequence_models.retrieval_sweeps import (
    DEFAULT_K_VALUES,
    DEFAULT_LAMBDAS,
    INITIAL_SWEEP_PRESET,
    LABEL_SEMANTICS,
    OUTPUT_VARIANTS,
    REQUIRED_RETRIEVAL_MODE,
    ComparisonRow,
    SweepRun,
    build_initial_sweep_preset,
    build_lambda_sweep,
    comparison_row_from_report,
    failure_row,
    write_comparison_outputs,
)
from humpback.services.masked_transformer_service import (
    create_masked_transformer_job,
    extend_k_sweep_job,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _manifest_path(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / str(args.manifest_name)


def plan_submit_runs(args: argparse.Namespace) -> list[SweepRun]:
    """Expand submit arguments into deterministic sweep rows."""
    k_values = cast(tuple[int, ...], args.k_values)
    if args.preset == INITIAL_SWEEP_PRESET:
        return build_initial_sweep_preset(
            continuous_embedding_job_id_250ms=args.continuous_embedding_job_id_250ms,
            continuous_embedding_job_id_100ms=args.continuous_embedding_job_id_100ms,
            event_classification_job_id=args.event_classification_job_id,
            k_values=k_values,
        )
    if args.continuous_embedding_job_id is None:
        raise SystemExit("--continuous-embedding-job-id is required for lambda sweeps")

    policy_variant: dict[str, Any] = {}
    if args.related_label_policy_json is not None:
        policy_variant["related_label_policy_json"] = args.related_label_policy_json
    if args.require_cross_region_positive is not None:
        policy_variant["require_cross_region_positive"] = (
            args.require_cross_region_positive
        )

    runs = build_lambda_sweep(
        continuous_embedding_job_id=args.continuous_embedding_job_id,
        event_classification_job_id=args.event_classification_job_id,
        lambda_values=cast(tuple[float, ...], args.lambda_values),
        k_values=k_values,
        batch_size=args.batch_size,
        labels_per_batch=args.labels_per_batch,
        events_per_label=args.events_per_label,
        policy_variant=policy_variant,
    )
    if args.seed != 42:
        runs = [
            replace(run, create_payload={**run.create_payload, "seed": args.seed})
            for run in runs
        ]
    return runs


async def submit_runs(args: argparse.Namespace) -> int:
    """Submit runnable sweep rows through the masked-transformer service."""
    runs = plan_submit_runs(args)
    manifest_rows: list[dict[str, Any]] = []
    if args.dry_run:
        manifest_rows = [run.to_manifest_row() for run in runs]
    else:
        settings = Settings.from_repo_env()
        engine = create_engine(settings.database_url)
        factory = create_session_factory(engine)
        try:
            async with factory() as session:
                for run in runs:
                    row = run.to_manifest_row()
                    if not run.runnable:
                        row["submission_status"] = "blocked"
                    elif run.action != "submit":
                        row["submission_status"] = "comparison_only"
                    else:
                        try:
                            job, created = await create_masked_transformer_job(
                                session, **run.create_payload
                            )
                            row["job_id"] = job.id
                            row["created"] = bool(created)
                            row["submission_status"] = (
                                "created" if created else "reused"
                            )
                            if (
                                args.extend_k_sweep
                                and not created
                                and job.status == JobStatus.complete.value
                            ):
                                job = await extend_k_sweep_job(
                                    session, job.id, list(run.k_values)
                                )
                                row["job_id"] = job.id
                                row["submission_status"] = "extended_k_sweep"
                        except Exception as exc:  # noqa: BLE001
                            row["submission_status"] = "failed"
                            row["error"] = str(exc)
                    manifest_rows.append(row)
        finally:
            await engine.dispose()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "preset": args.preset,
        "label_semantics": LABEL_SEMANTICS,
        "runs": manifest_rows,
    }
    _write_json(_manifest_path(args), payload)
    print(f"wrote {_manifest_path(args)}")
    for row in manifest_rows:
        status = row.get("submission_status") or (
            "blocked" if row.get("blocked_reason") else row.get("action")
        )
        print(f"{row['run_name']}: {status} {row.get('job_id') or ''}".rstrip())
    return 0


def _known_metric_row(row: dict[str, Any], *, space: str) -> ComparisonRow | None:
    metrics = row.get("known_metrics") or {}
    if not metrics:
        return None
    variants: dict[str, float] = {}
    for variant in OUTPUT_VARIANTS:
        value = metrics.get(f"{space}_{variant}_same_human_label")
        if value is None:
            value = metrics.get(f"{variant}_same_human_label")
        if value is not None:
            variants[variant] = float(value)
    if not variants:
        return None
    k_values = row.get("k_values") or list(DEFAULT_K_VALUES)
    metadata = dict(row.get("metadata") or {})
    metadata["metric_source"] = "known_session_metric"
    return ComparisonRow(
        run_name=str(row["run_name"]),
        job_id=str(row.get("job_id") or ""),
        k=int(k_values[0]) if k_values else None,
        embedding_space=space,
        primary_metric=variants.get("raw_l2"),
        variant_same_human_label=variants,
        metadata=metadata,
    )


def _manifest_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.manifest is None:
        return []
    payload = json.loads(Path(args.manifest).read_text())
    return list(payload.get("runs", []))


def _explicit_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for value in args.job or []:
        if "=" in value:
            run_name, rest = value.split("=", 1)
        else:
            run_name, rest = value, value
        if ":" in rest:
            job_id, space = rest.rsplit(":", 1)
            spaces = [space]
        else:
            job_id = rest
            spaces = [args.embedding_space]
        targets.append(
            {
                "run_name": run_name,
                "job_id": job_id,
                "embedding_spaces": spaces,
                "k_values": [args.k if args.k is not None else DEFAULT_K_VALUES[0]],
                "metadata": {"label_semantics": LABEL_SEMANTICS},
            }
        )
    return targets


async def compare_runs(args: argparse.Namespace) -> int:
    """Build nearest-neighbor reports and write ranked comparison artifacts."""
    target_rows = _manifest_targets(args) + _explicit_targets(args)
    if not target_rows:
        raise SystemExit("compare requires --manifest or at least one --job")

    diagnostic_options = {
        "k": args.k,
        "samples": args.samples,
        "topn": args.topn,
        "seed": args.seed,
        "retrieval_modes": [REQUIRED_RETRIEVAL_MODE],
        "embedding_variants": list(OUTPUT_VARIANTS),
        "include_event_level": True,
    }
    rows: list[ComparisonRow] = []
    if args.include_known_metrics:
        for target in target_rows:
            for space in target.get("embedding_spaces") or [args.embedding_space]:
                known = _known_metric_row(target, space=str(space))
                if known is not None:
                    rows.append(known)

    settings = Settings.from_repo_env()
    engine = create_engine(settings.database_url)
    factory = create_session_factory(engine)
    try:
        async with factory() as session:
            for target in target_rows:
                job_id = target.get("job_id")
                if not job_id:
                    continue
                k_values = target.get("k_values") or [args.k]
                k = int(args.k if args.k is not None else k_values[0])
                for space in target.get("embedding_spaces") or [args.embedding_space]:
                    options = RetrievalReportOptions(
                        k=k,
                        embedding_space=cast(EmbeddingSpace, space),
                        samples=args.samples,
                        topn=args.topn,
                        seed=args.seed,
                        retrieval_modes=(cast(RetrievalMode, REQUIRED_RETRIEVAL_MODE),),
                        embedding_variants=cast(
                            tuple[EmbeddingVariant, ...], OUTPUT_VARIANTS
                        ),
                        include_event_level=True,
                    )
                    try:
                        report = await build_nearest_neighbor_report(
                            session,
                            storage_root=settings.storage_root,
                            job_id=str(job_id),
                            options=options,
                        )
                        rows.append(
                            comparison_row_from_report(
                                str(target.get("run_name") or job_id),
                                report,
                                metadata=dict(target.get("metadata") or {}),
                            )
                        )
                    except Exception as exc:  # noqa: BLE001
                        rows.append(
                            failure_row(
                                run_name=str(target.get("run_name") or job_id),
                                job_id=str(job_id),
                                k=k,
                                embedding_space=str(space),
                                error=str(exc),
                                metadata=dict(target.get("metadata") or {}),
                            )
                        )
    finally:
        await engine.dispose()

    paths = write_comparison_outputs(
        rows,
        Path(args.output_dir),
        timestamped=args.timestamped,
        diagnostic_options=diagnostic_options,
    )
    print(f"wrote {paths.csv_path}")
    print(f"wrote {paths.markdown_path}")
    print(f"wrote {paths.json_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="plan or submit sweep jobs")
    submit.add_argument(
        "--preset",
        choices=[INITIAL_SWEEP_PRESET, "lambda"],
        default=INITIAL_SWEEP_PRESET,
    )
    submit.add_argument("--continuous-embedding-job-id")
    submit.add_argument("--continuous-embedding-job-id-250ms")
    submit.add_argument("--continuous-embedding-job-id-100ms")
    submit.add_argument("--event-classification-job-id")
    submit.add_argument("--k-values", type=_parse_csv_ints, default=DEFAULT_K_VALUES)
    submit.add_argument(
        "--lambda-values", type=_parse_csv_floats, default=DEFAULT_LAMBDAS
    )
    submit.add_argument("--batch-size", type=int, default=16)
    submit.add_argument("--labels-per-batch", type=int, default=4)
    submit.add_argument("--events-per-label", type=int, default=4)
    submit.add_argument("--seed", type=int, default=42)
    submit.add_argument("--related-label-policy-json")
    submit.add_argument(
        "--require-cross-region-positive",
        dest="require_cross_region_positive",
        action="store_true",
        default=None,
    )
    submit.add_argument(
        "--allow-same-region-positive",
        dest="require_cross_region_positive",
        action="store_false",
    )
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--extend-k-sweep", action="store_true")
    submit.add_argument("--output-dir", default="data/retrieval_sweeps")
    submit.add_argument("--manifest-name", default="submit-manifest.json")

    compare = subparsers.add_parser("compare", help="write comparison artifacts")
    compare.add_argument("--manifest")
    compare.add_argument(
        "--job",
        action="append",
        help="JOB_ID, RUN_NAME=JOB_ID, or RUN_NAME=JOB_ID:retrieval",
    )
    compare.add_argument("--embedding-space", default="retrieval")
    compare.add_argument("--k", type=int)
    compare.add_argument("--samples", type=int, default=50)
    compare.add_argument("--topn", type=int, default=10)
    compare.add_argument("--seed", type=int, default=20260504)
    compare.add_argument("--include-known-metrics", action="store_true")
    compare.add_argument("--timestamped", action="store_true")
    compare.add_argument("--output-dir", default="data/retrieval_sweeps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "submit":
        return asyncio.run(submit_runs(args))
    if args.command == "compare":
        return asyncio.run(compare_runs(args))
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
