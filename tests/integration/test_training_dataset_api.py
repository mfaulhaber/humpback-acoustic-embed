"""Integration tests for detection-job-based training dataset APIs."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.labeling import VocalizationLabel
from humpback.services.training_dataset import create_training_dataset_snapshot


DIM = 8


def _label_names(row: dict) -> list[str]:
    return [lbl["label"] for lbl in row["labels"]]


def _make_detection_embeddings_parquet(
    storage_root: Path,
    det_job_id: str,
    row_ids: list[str],
    model_version: str = "v1",
) -> None:
    emb_dir = storage_root / "detections" / det_job_id / "embeddings" / model_version
    emb_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(
                [
                    np.random.default_rng(i).random(DIM).astype(np.float32).tolist()
                    for i in range(len(row_ids))
                ],
                type=pa.list_(pa.float32()),
            ),
            "confidence": pa.array(
                [0.9 - i * 0.1 for i in range(len(row_ids))], type=pa.float32()
            ),
        }
    )
    pq.write_table(table, str(emb_dir / "detection_embeddings.parquet"))


async def _seed_dataset(app_settings):
    """Create a detection-job-only training dataset with Whup x3 and Moan x3."""
    storage = app_settings.storage_root
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        # Two labeled detection jobs
        for det_job_id, labels in {
            "dj1": {
                "dj1-row-1": "Whup",
                "dj1-row-2": "Whup",
                "dj1-row-3": "Whup",
            },
            "dj2": {
                "dj2-row-1": "Moan",
                "dj2-row-2": "Moan",
                "dj2-row-3": "Moan",
            },
        }.items():
            _make_detection_embeddings_parquet(storage, det_job_id, list(labels.keys()))
            session.add(
                ClassifierModel(
                    id=f"cm-{det_job_id}",
                    name=f"cm-{det_job_id}",
                    model_path="/tmp/model",
                    model_version="v1",
                    vector_dim=DIM,
                    window_size_seconds=5.0,
                    target_sample_rate=32000,
                )
            )
            session.add(
                DetectionJob(
                    id=det_job_id,
                    status="complete",
                    classifier_model_id=f"cm-{det_job_id}",
                )
            )
            for row_id, label in labels.items():
                session.add(
                    VocalizationLabel(
                        detection_job_id=det_job_id,
                        row_id=row_id,
                        label=label,
                    )
                )

        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"detection_job_ids": ["dj1", "dj2"]},
            storage,
        )
        await session.commit()
        dataset_id = dataset.id

    await engine.dispose()
    return dataset_id


@pytest.mark.asyncio
async def test_list_training_datasets(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get("/vocalization/training-datasets")
    assert resp.status_code == 200
    assert any(d["id"] == dataset_id for d in resp.json())


@pytest.mark.asyncio
async def test_get_training_dataset(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(f"/vocalization/training-datasets/{dataset_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 6
    assert "Whup" in data["vocabulary"]
    assert "Moan" in data["vocabulary"]


@pytest.mark.asyncio
async def test_get_training_dataset_404(client):
    resp = await client.get("/vocalization/training-datasets/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_rows_unfiltered(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["rows"]) == 6
    assert all(row["source_type"] == "detection_job" for row in data["rows"])


@pytest.mark.asyncio
async def test_get_rows_filter_by_type_positive(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"type": "Whup", "group": "positive"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    for row in data["rows"]:
        assert "Whup" in _label_names(row)


@pytest.mark.asyncio
async def test_get_rows_filter_by_type_negative(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"type": "Whup", "group": "negative"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    for row in data["rows"]:
        assert "Whup" not in _label_names(row)


@pytest.mark.asyncio
async def test_get_rows_pagination(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"offset": 0, "limit": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["rows"]) == 2


@pytest.mark.asyncio
async def test_get_rows_filter_by_source_type(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)

    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"source_type": "detection_job"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert all(row["source_type"] == "detection_job" for row in data["rows"])


@pytest.mark.asyncio
async def test_get_rows_source_type_composes_with_type_filter(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)

    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={
            "source_type": "detection_job",
            "type": "Whup",
            "group": "positive",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    for row in data["rows"]:
        assert row["source_type"] == "detection_job"
        assert "Whup" in _label_names(row)


@pytest.mark.asyncio
async def test_create_and_delete_label(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)

    resp = await client.post(
        f"/vocalization/training-datasets/{dataset_id}/labels",
        json={"row_index": 0, "label": "Shriek"},
    )
    assert resp.status_code == 201
    label_id = resp.json()["id"]

    resp2 = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = [r for r in resp2.json()["rows"] if r["row_index"] == 0][0]
    assert "Shriek" in _label_names(row0)

    resp3 = await client.delete(
        f"/vocalization/training-datasets/{dataset_id}/labels/{label_id}"
    )
    assert resp3.status_code == 204


@pytest.mark.asyncio
async def test_duplicate_label_prevented(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)

    resp1 = await client.post(
        f"/vocalization/training-datasets/{dataset_id}/labels",
        json={"row_index": 0, "label": "Whup"},
    )
    assert resp1.status_code == 201

    resp2 = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = [r for r in resp2.json()["rows"] if r["row_index"] == 0][0]
    whup_labels = [lbl for lbl in row0["labels"] if lbl["label"] == "Whup"]
    assert len(whup_labels) == 1


@pytest.mark.asyncio
async def test_label_objects_have_ids(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = resp.json()["rows"][0]
    assert len(row0["labels"]) > 0
    for lbl in row0["labels"]:
        assert "id" in lbl
        assert "label" in lbl


@pytest.mark.asyncio
async def test_negative_label_mutual_exclusivity(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)

    resp = await client.post(
        f"/vocalization/training-datasets/{dataset_id}/labels",
        json={"row_index": 0, "label": "(Negative)"},
    )
    assert resp.status_code == 201

    resp2 = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = [r for r in resp2.json()["rows"] if r["row_index"] == 0][0]
    assert "(Negative)" in _label_names(row0)
    assert "Whup" not in _label_names(row0)


@pytest.mark.asyncio
async def test_training_job_requires_source_or_dataset(client):
    resp = await client.post("/vocalization/training-jobs", json={})
    assert resp.status_code == 422 or resp.status_code == 400


@pytest.mark.asyncio
async def test_training_job_rejects_both(client, app_settings):
    dataset_id = await _seed_dataset(app_settings)
    resp = await client.post(
        "/vocalization/training-jobs",
        json={
            "source_config": {"detection_job_ids": ["dj1"]},
            "training_dataset_id": dataset_id,
        },
    )
    assert resp.status_code == 400
