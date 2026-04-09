"""Unit tests for hyperparameter worker queue and job execution."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.models.hyperparameter import (
    HyperparameterManifest,
    HyperparameterSearchJob,
)
from humpback.workers.queue import (
    claim_hyperparameter_search_job,
    claim_manifest_job,
)


# ---------------------------------------------------------------------------
# Queue claim tests
# ---------------------------------------------------------------------------


async def test_claim_manifest_job(session) -> None:
    job = HyperparameterManifest(
        name="test manifest",
        status="queued",
        training_job_ids="[]",
        detection_job_ids="[]",
        split_ratio="[70, 15, 15]",
        seed=42,
    )
    session.add(job)
    await session.commit()

    claimed = await claim_manifest_job(session)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "running"


async def test_claim_manifest_job_no_jobs(session) -> None:
    claimed = await claim_manifest_job(session)
    assert claimed is None


async def test_claim_manifest_job_skips_running(session) -> None:
    job = HyperparameterManifest(
        name="running manifest",
        status="running",
        training_job_ids="[]",
        detection_job_ids="[]",
        split_ratio="[70, 15, 15]",
        seed=42,
    )
    session.add(job)
    await session.commit()

    claimed = await claim_manifest_job(session)
    assert claimed is None


async def test_claim_search_job(session) -> None:
    manifest = HyperparameterManifest(
        name="m",
        status="complete",
        training_job_ids="[]",
        detection_job_ids="[]",
        split_ratio="[70, 15, 15]",
        seed=42,
    )
    session.add(manifest)
    await session.flush()

    job = HyperparameterSearchJob(
        name="test search",
        status="queued",
        manifest_id=manifest.id,
        search_space=json.dumps({"classifier": ["logreg"]}),
        n_trials=5,
        seed=42,
    )
    session.add(job)
    await session.commit()

    claimed = await claim_hyperparameter_search_job(session)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.status == "running"


async def test_claim_search_job_no_jobs(session) -> None:
    claimed = await claim_hyperparameter_search_job(session)
    assert claimed is None


# ---------------------------------------------------------------------------
# Manifest worker execution test
# ---------------------------------------------------------------------------

VECTOR_DIM = 16


def _write_synthetic_parquet(path, n_rows: int) -> None:
    rng = np.random.RandomState(42)
    schema = pa.schema(
        [
            ("row_index", pa.int32()),
            ("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )
    table = pa.table(
        {
            "row_index": list(range(n_rows)),
            "embedding": [
                rng.randn(VECTOR_DIM).astype(np.float32).tolist() for _ in range(n_rows)
            ],
        },
        schema=schema,
    )
    pq.write_table(table, str(path))


async def test_run_manifest_job_failure_sets_status(session, settings) -> None:
    """A manifest job with invalid sources should fail gracefully."""
    from humpback.workers.hyperparameter_worker import run_manifest_job

    job = HyperparameterManifest(
        name="bad manifest",
        status="running",
        training_job_ids='["nonexistent-id"]',
        detection_job_ids="[]",
        split_ratio="[70, 15, 15]",
        seed=42,
    )
    session.add(job)
    await session.commit()

    await run_manifest_job(session, job, settings)

    await session.refresh(job)
    assert job.status == "failed"
    assert job.error_message is not None
    assert (
        "not found" in job.error_message.lower() or "Training job" in job.error_message
    )


async def test_run_search_job_failure_sets_status(session, settings) -> None:
    """A search job referencing a missing manifest should fail gracefully."""
    from humpback.workers.hyperparameter_worker import run_hyperparameter_search_job

    manifest = HyperparameterManifest(
        name="m",
        status="complete",
        training_job_ids="[]",
        detection_job_ids="[]",
        split_ratio="[70, 15, 15]",
        seed=42,
        manifest_path="/nonexistent/manifest.json",
    )
    session.add(manifest)
    await session.flush()

    job = HyperparameterSearchJob(
        name="bad search",
        status="running",
        manifest_id=manifest.id,
        search_space=json.dumps({"classifier": ["logreg"]}),
        n_trials=5,
        seed=42,
    )
    session.add(job)
    await session.commit()

    await run_hyperparameter_search_job(session, job, settings)

    await session.refresh(job)
    assert job.status == "failed"
    assert job.error_message is not None
