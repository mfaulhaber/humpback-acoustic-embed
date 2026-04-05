"""Integration tests for training dataset API endpoints."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.services.training_dataset import create_training_dataset_snapshot


DIM = 8


def _make_embedding_set_parquet(path: Path, n_rows: int) -> None:
    table = pa.table(
        {
            "row_index": pa.array(list(range(n_rows)), type=pa.int32()),
            "embedding": pa.array(
                [
                    np.random.default_rng(i).random(DIM).astype(np.float32).tolist()
                    for i in range(n_rows)
                ],
                type=pa.list_(pa.float32()),
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


async def _seed_dataset(client, app_settings):
    """Create a training dataset with 2 types (Whup x3, Moan x3) for test use."""

    storage = app_settings.storage_root
    es_path_a = storage / "embeddings" / "v1" / "af1" / "sig.parquet"
    _make_embedding_set_parquet(es_path_a, 3)
    es_path_b = storage / "embeddings" / "v1" / "af2" / "sig.parquet"
    _make_embedding_set_parquet(es_path_b, 3)

    # Need direct DB access to seed data
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        af1 = AudioFile(
            id="af1",
            filename="a.wav",
            folder_path="/data/Whup",
            checksum_sha256="a1",
        )
        af2 = AudioFile(
            id="af2",
            filename="b.wav",
            folder_path="/data/Moan",
            checksum_sha256="a2",
        )
        session.add_all([af1, af2])
        es1 = EmbeddingSet(
            id="es1",
            audio_file_id="af1",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_a),
        )
        es2 = EmbeddingSet(
            id="es2",
            audio_file_id="af2",
            encoding_signature="sig",
            model_version="v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=DIM,
            parquet_path=str(es_path_b),
        )
        session.add_all([es1, es2])
        await session.flush()

        dataset = await create_training_dataset_snapshot(
            session,
            {"embedding_set_ids": ["es1", "es2"], "detection_job_ids": []},
            storage,
        )
        await session.commit()
        dataset_id = dataset.id

    await engine.dispose()
    return dataset_id


# ---- List / Get ----


@pytest.mark.asyncio
async def test_list_training_datasets(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.get("/vocalization/training-datasets")
    assert resp.status_code == 200
    datasets = resp.json()
    assert len(datasets) >= 1
    assert any(d["id"] == dataset_id for d in datasets)


@pytest.mark.asyncio
async def test_get_training_dataset(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
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


# ---- Rows ----


@pytest.mark.asyncio
async def test_get_rows_unfiltered(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["rows"]) == 6


@pytest.mark.asyncio
async def test_get_rows_filter_by_type_positive(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"type": "Whup", "group": "positive"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    for row in data["rows"]:
        assert "Whup" in row["labels"]


@pytest.mark.asyncio
async def test_get_rows_filter_by_type_negative(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"type": "Whup", "group": "negative"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # 3 Moan rows should be negative for Whup
    assert data["total"] == 3
    for row in data["rows"]:
        assert "Whup" not in row["labels"]


@pytest.mark.asyncio
async def test_get_rows_pagination(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"offset": 0, "limit": 2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 6
    assert len(data["rows"]) == 2


# ---- Source type filter ----


@pytest.mark.asyncio
async def test_get_rows_filter_by_source_type(client, app_settings):
    """source_type param filters rows by their parquet source_type column."""
    dataset_id = await _seed_dataset(client, app_settings)

    # All rows in the seed are from embedding sets
    resp_emb = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"source_type": "embedding_set"},
    )
    assert resp_emb.status_code == 200
    data_emb = resp_emb.json()
    assert data_emb["total"] == 6
    for row in data_emb["rows"]:
        assert row["source_type"] == "embedding_set"

    # No detection rows in the seed
    resp_det = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={"source_type": "detection_job"},
    )
    assert resp_det.status_code == 200
    assert resp_det.json()["total"] == 0


@pytest.mark.asyncio
async def test_get_rows_source_type_composes_with_type_filter(client, app_settings):
    """source_type filter composes with type+group filters."""
    dataset_id = await _seed_dataset(client, app_settings)

    resp = await client.get(
        f"/vocalization/training-datasets/{dataset_id}/rows",
        params={
            "source_type": "embedding_set",
            "type": "Whup",
            "group": "positive",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 3
    for row in data["rows"]:
        assert row["source_type"] == "embedding_set"
        assert "Whup" in row["labels"]


# ---- Labels ----


@pytest.mark.asyncio
async def test_create_and_delete_label(client, app_settings):
    dataset_id = await _seed_dataset(client, app_settings)

    # Create a new label
    resp = await client.post(
        f"/vocalization/training-datasets/{dataset_id}/labels",
        json={"row_index": 0, "label": "Shriek"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["label"] == "Shriek"
    assert data["source"] == "manual"
    label_id = data["id"]

    # Verify it appears in rows
    resp2 = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = [r for r in resp2.json()["rows"] if r["row_index"] == 0][0]
    assert "Shriek" in row0["labels"]

    # Delete the label
    resp3 = await client.delete(
        f"/vocalization/training-datasets/{dataset_id}/labels/{label_id}"
    )
    assert resp3.status_code == 204


@pytest.mark.asyncio
async def test_negative_label_mutual_exclusivity(client, app_settings):
    """Adding (Negative) removes existing type labels on the same row."""
    dataset_id = await _seed_dataset(client, app_settings)

    # Row 0 starts with "Whup" label from snapshot
    # Adding (Negative) should remove it
    resp = await client.post(
        f"/vocalization/training-datasets/{dataset_id}/labels",
        json={"row_index": 0, "label": "(Negative)"},
    )
    assert resp.status_code == 201

    # Check that Whup is gone and (Negative) is present
    resp2 = await client.get(f"/vocalization/training-datasets/{dataset_id}/rows")
    row0 = [r for r in resp2.json()["rows"] if r["row_index"] == 0][0]
    assert "(Negative)" in row0["labels"]
    assert "Whup" not in row0["labels"]


# ---- Training job modes ----


@pytest.mark.asyncio
async def test_training_job_requires_source_or_dataset(client):
    """Must provide either source_config or training_dataset_id."""
    resp = await client.post("/vocalization/training-jobs", json={})
    assert resp.status_code == 422 or resp.status_code == 400


@pytest.mark.asyncio
async def test_training_job_rejects_both(client, app_settings):
    """Cannot provide both source_config and training_dataset_id."""
    dataset_id = await _seed_dataset(client, app_settings)
    resp = await client.post(
        "/vocalization/training-jobs",
        json={
            "source_config": {"embedding_set_ids": ["es1"], "detection_job_ids": []},
            "training_dataset_id": dataset_id,
        },
    )
    assert resp.status_code == 400
