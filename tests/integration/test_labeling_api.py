"""Integration tests for vocalization labeling API."""

import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.classifier.detection_rows import (
    write_detection_row_store,
)
from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.processing import EmbeddingSet
from humpback.processing.embeddings import IncrementalParquetWriter
from humpback.storage import (
    detection_embeddings_path,
    detection_row_store_path,
)

# Consistent base epoch: 2024-06-15T08:00:00Z
BASE_EPOCH = 1718438400.0

# Stable row IDs for seeded detection rows
ROW_ID_1 = "row-aaa-111"
ROW_ID_2 = "row-bbb-222"
HYDRO_ROW_ID = "row-hydro-111"


async def _seed_detection_job(app_settings, tmp_path):
    """Create a completed detection job with row store and embeddings.

    Returns (job_id, model_id).
    """
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    storage_root = app_settings.storage_root

    # Create a classifier model
    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    async with sf() as session:
        cm = ClassifierModel(
            id=model_id,
            name="test-classifier",
            model_path="/fake/model.pkl",
            model_version="test_v1",
            vector_dim=4,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(cm)

        dj = DetectionJob(
            id=job_id,
            status="complete",
            classifier_model_id=model_id,
            audio_folder="/test/audio",
            detection_mode="windowed",
        )
        session.add(dj)
        await session.commit()

    # Write detection row store with stable row_ids
    row_store = detection_row_store_path(storage_root, job_id)
    row_store.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "row_id": ROW_ID_1,
            "start_utc": str(BASE_EPOCH),
            "end_utc": str(BASE_EPOCH + 5.0),
            "raw_start_utc": str(BASE_EPOCH),
            "raw_end_utc": str(BASE_EPOCH + 5.0),
            "merged_event_count": "1",
            "avg_confidence": "0.85",
            "peak_confidence": "0.92",
            "n_windows": "1",
            "hydrophone_name": "",
            "humpback": "",
            "orca": "",
            "ship": "",
            "background": "",
        },
        {
            "row_id": ROW_ID_2,
            "start_utc": str(BASE_EPOCH + 5.0),
            "end_utc": str(BASE_EPOCH + 10.0),
            "raw_start_utc": str(BASE_EPOCH + 5.0),
            "raw_end_utc": str(BASE_EPOCH + 10.0),
            "merged_event_count": "1",
            "avg_confidence": "0.70",
            "peak_confidence": "0.75",
            "n_windows": "1",
            "hydrophone_name": "",
            "humpback": "1",
            "orca": "",
            "ship": "",
            "background": "",
        },
    ]
    write_detection_row_store(row_store, rows)

    # Write detection embeddings with row_id schema
    emb_path = detection_embeddings_path(storage_root, job_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), 4)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": [ROW_ID_1, ROW_ID_2],
            "embedding": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            "confidence": [0.85, 0.70],
        },
        schema=schema,
    )
    pq.write_table(table, emb_path)

    return job_id, model_id


async def _seed_hydrophone_detection_job(app_settings):
    """Create a completed hydrophone detection job with row store and embeddings."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    storage_root = app_settings.storage_root

    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    hydro_start_utc = 1751439678.0
    hydro_end_utc = hydro_start_utc + 5.0
    job_start_ts = 1751439600.0

    async with sf() as session:
        cm = ClassifierModel(
            id=model_id,
            name="test-hydrophone-classifier",
            model_path="/fake/model.pkl",
            model_version="test_v1",
            vector_dim=4,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(cm)

        dj = DetectionJob(
            id=job_id,
            status="complete",
            classifier_model_id=model_id,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=job_start_ts,
            end_timestamp=job_start_ts + 3600.0,
            detection_mode="windowed",
        )
        session.add(dj)
        await session.commit()

    row_store = detection_row_store_path(storage_root, job_id)
    row_store.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "row_id": HYDRO_ROW_ID,
            "start_utc": str(hydro_start_utc),
            "end_utc": str(hydro_end_utc),
            "raw_start_utc": str(hydro_start_utc),
            "raw_end_utc": str(hydro_end_utc),
            "merged_event_count": "1",
            "avg_confidence": "0.85",
            "peak_confidence": "0.92",
            "n_windows": "1",
            "hydrophone_name": "orcasound_lab",
            "humpback": "",
            "orca": "",
            "ship": "",
            "background": "",
        }
    ]
    write_detection_row_store(row_store, rows)

    # Embedding file with row_id schema
    emb_path = detection_embeddings_path(storage_root, job_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), 4)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": [HYDRO_ROW_ID],
            "embedding": [[1.0, 0.0, 0.0, 0.0]],
            "confidence": [0.85],
        },
        schema=schema,
    )
    pq.write_table(table, emb_path)

    return job_id, model_id


async def _seed_reference_embeddings(app_settings, tmp_path):
    """Create reference embedding sets with folder-path-based labels."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    ref_path = tmp_path / "ref.parquet"
    writer = IncrementalParquetWriter(ref_path, vector_dim=4)
    writer.add(np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32))
    writer.add(np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32))
    writer.close()

    async with sf() as session:
        af = AudioFile(
            filename="call_sample.flac",
            folder_path="emily-vierling/whup",
            checksum_sha256=f"ref_{uuid.uuid4().hex[:8]}",
        )
        session.add(af)
        await session.flush()

        es = EmbeddingSet(
            audio_file_id=af.id,
            encoding_signature="ref_sig",
            model_version="test_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=4,
            parquet_path=str(ref_path),
        )
        session.add(es)
        await session.commit()
        return es.id


# ---- CRUD tests ----


@pytest.mark.asyncio
async def test_create_and_list_vocalization_labels(client):
    """Test creating and listing vocalization labels."""
    row_id = "test-row-crud-1"

    # Create a label
    resp = await client.post(
        f"/labeling/vocalization-labels/fake-job?row_id={row_id}",
        json={"label": "whup", "source": "manual"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["label"] == "whup"
    assert data["source"] == "manual"
    assert data["detection_job_id"] == "fake-job"
    assert data["row_id"] == row_id
    label_id = data["id"]

    # Create another label
    resp2 = await client.post(
        f"/labeling/vocalization-labels/fake-job?row_id={row_id}",
        json={"label": "moan", "confidence": 0.85, "source": "search"},
    )
    assert resp2.status_code == 201

    # List labels for the row
    resp3 = await client.get(f"/labeling/vocalization-labels/fake-job?row_id={row_id}")
    assert resp3.status_code == 200
    labels = resp3.json()
    assert len(labels) == 2
    assert {lbl["label"] for lbl in labels} == {"whup", "moan"}

    # List labels for different row_id — empty
    resp4 = await client.get(
        "/labeling/vocalization-labels/fake-job?row_id=nonexistent-row"
    )
    assert resp4.status_code == 200
    assert resp4.json() == []

    return label_id


@pytest.mark.asyncio
async def test_update_vocalization_label(client):
    """Test updating a vocalization label."""
    row_id = "test-row-update-1"

    resp = await client.post(
        f"/labeling/vocalization-labels/job1?row_id={row_id}",
        json={"label": "whup"},
    )
    label_id = resp.json()["id"]

    resp2 = await client.put(
        f"/labeling/vocalization-labels/{label_id}",
        json={"label": "moan", "confidence": 0.9},
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["label"] == "moan"
    assert data["confidence"] == 0.9


@pytest.mark.asyncio
async def test_delete_vocalization_label(client):
    """Test deleting a vocalization label."""
    row_id = "test-row-delete-1"

    resp = await client.post(
        f"/labeling/vocalization-labels/job1?row_id={row_id}",
        json={"label": "shriek"},
    )
    label_id = resp.json()["id"]

    resp2 = await client.delete(f"/labeling/vocalization-labels/{label_id}")
    assert resp2.status_code == 204

    # Should be gone
    resp3 = await client.get(f"/labeling/vocalization-labels/job1?row_id={row_id}")
    labels = resp3.json()
    assert not any(lbl["id"] == label_id for lbl in labels)


@pytest.mark.asyncio
async def test_delete_nonexistent_label(client):
    """Deleting a non-existent label returns 404."""
    resp = await client.delete("/labeling/vocalization-labels/nonexistent-id")
    assert resp.status_code == 404


# ---- Vocabulary ----


@pytest.mark.asyncio
async def test_label_vocabulary(client):
    """Test that vocabulary returns distinct labels."""
    await client.post(
        "/labeling/vocalization-labels/j1?row_id=vocab-row-1",
        json={"label": "whup"},
    )
    await client.post(
        "/labeling/vocalization-labels/j1?row_id=vocab-row-2",
        json={"label": "moan"},
    )
    await client.post(
        "/labeling/vocalization-labels/j2?row_id=vocab-row-3",
        json={"label": "whup"},
    )

    resp = await client.get("/labeling/label-vocabulary")
    assert resp.status_code == 200
    vocab = resp.json()
    assert "whup" in vocab
    assert "moan" in vocab
    assert len(vocab) == len(set(vocab))  # all distinct


# ---- Summary ----


@pytest.mark.asyncio
async def test_labeling_summary(client, app_settings, tmp_path):
    """Test labeling summary with detection row store."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # No labels yet
    resp = await client.get(f"/labeling/summary/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_rows"] == 2
    assert data["labeled_rows"] == 0
    assert data["unlabeled_rows"] == 2
    assert data["label_distribution"] == {}

    # Add labels using row_ids of the seeded rows
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}",
        json={"label": "moan"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_2}",
        json={"label": "whup"},
    )

    resp2 = await client.get(f"/labeling/summary/{job_id}")
    data2 = resp2.json()
    assert data2["total_rows"] == 2
    assert data2["labeled_rows"] == 2
    assert data2["unlabeled_rows"] == 0
    assert data2["label_distribution"]["whup"] == 2
    assert data2["label_distribution"]["moan"] == 1


@pytest.mark.asyncio
async def test_labeling_summary_nonexistent_job(client):
    """Summary for non-existent job returns 404."""
    resp = await client.get("/labeling/summary/nonexistent")
    assert resp.status_code == 404


# ---- Detection Neighbors ----


@pytest.mark.asyncio
async def test_detection_neighbors(client, app_settings, tmp_path):
    """Test that detection neighbors returns hits with inferred labels."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)
    ref_es_id = await _seed_reference_embeddings(app_settings, tmp_path)

    resp = await client.post(
        f"/labeling/detection-neighbors/{job_id}",
        json={
            "row_id": ROW_ID_1,
            "top_k": 5,
            "embedding_set_ids": [ref_es_id],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hits"]) > 0
    assert data["total_candidates"] > 0

    # At least one hit should have inferred_label from folder path
    inferred_labels = [h["inferred_label"] for h in data["hits"] if h["inferred_label"]]
    assert "whup" in inferred_labels


@pytest.mark.asyncio
async def test_detection_neighbors_missing_embedding(client, app_settings, tmp_path):
    """Requesting neighbors for non-existent detection row returns 404."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    resp = await client.post(
        f"/labeling/detection-neighbors/{job_id}",
        json={"row_id": "nonexistent-row-id"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_neighbors_nonexistent_job(client):
    """Requesting neighbors for non-existent job returns 404."""
    resp = await client.post(
        "/labeling/detection-neighbors/nonexistent",
        json={"row_id": "some-row"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_neighbors_hydrophone(client, app_settings, tmp_path):
    """Hydrophone neighbor lookup works with row_id."""
    job_id, _model_id = await _seed_hydrophone_detection_job(app_settings)
    ref_es_id = await _seed_reference_embeddings(app_settings, tmp_path)

    resp = await client.post(
        f"/labeling/detection-neighbors/{job_id}",
        json={
            "row_id": HYDRO_ROW_ID,
            "top_k": 5,
            "embedding_set_ids": [ref_es_id],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hits"]) > 0
    assert data["total_candidates"] > 0


# ---- Negative Label Mutual Exclusivity ----


@pytest.mark.asyncio
async def test_negative_label_removes_type_labels(client):
    """Adding (Negative) removes existing type labels on the same window."""
    row_id = "neg-test-row-1"
    job_id = "neg-test-job"

    # Create type labels
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id}",
        json={"label": "moan"},
    )

    # Verify both exist
    resp = await client.get(f"/labeling/vocalization-labels/{job_id}?row_id={row_id}")
    assert len(resp.json()) == 2

    # Add (Negative) — should remove both type labels
    resp2 = await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id}",
        json={"label": "(Negative)"},
    )
    assert resp2.status_code == 201

    # Only (Negative) should remain
    resp3 = await client.get(f"/labeling/vocalization-labels/{job_id}?row_id={row_id}")
    labels = resp3.json()
    assert len(labels) == 1
    assert labels[0]["label"] == "(Negative)"


@pytest.mark.asyncio
async def test_type_label_removes_negative(client):
    """Adding a type label removes existing (Negative) on the same window."""
    row_id = "neg-test-row-2"
    job_id = "neg-test-job2"

    # Create (Negative) label
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id}",
        json={"label": "(Negative)"},
    )

    resp = await client.get(f"/labeling/vocalization-labels/{job_id}?row_id={row_id}")
    assert len(resp.json()) == 1
    assert resp.json()[0]["label"] == "(Negative)"

    # Add type label — should remove (Negative)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id}",
        json={"label": "whup"},
    )

    resp2 = await client.get(f"/labeling/vocalization-labels/{job_id}?row_id={row_id}")
    labels = resp2.json()
    assert len(labels) == 1
    assert labels[0]["label"] == "whup"


@pytest.mark.asyncio
async def test_negative_does_not_affect_other_windows(client):
    """Mutual exclusivity only applies to the same row_id."""
    job_id = "neg-test-job3"
    row_id_1 = "neg-test-row-3a"
    row_id_2 = "neg-test-row-3b"

    # Label window 1 with type, window 2 with (Negative)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id_1}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id_2}",
        json={"label": "(Negative)"},
    )

    # Both windows should retain their labels
    resp1 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id_1}"
    )
    assert len(resp1.json()) == 1
    assert resp1.json()[0]["label"] == "whup"

    resp2 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={row_id_2}"
    )
    assert len(resp2.json()) == 1
    assert resp2.json()[0]["label"] == "(Negative)"


# ---- Training Data Assembly ----


@pytest.mark.asyncio
async def test_training_assembly_skips_unlabeled_windows(
    client, app_settings, tmp_path
):
    """Verify that training only includes labeled windows, not unlabeled ones."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Row 1: label with "whup"
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}",
        json={"label": "whup"},
    )
    # Row 2: leave unlabeled (should be excluded from training)

    # Simulate training data assembly (same logic as vocalization_worker.py)
    from sqlalchemy import select

    from humpback.database import create_engine, create_session_factory
    from humpback.models.labeling import VocalizationLabel
    from humpback.storage import detection_embeddings_path

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        result = await session.execute(
            select(VocalizationLabel).where(
                VocalizationLabel.detection_job_id == job_id
            )
        )
        voc_labels = result.scalars().all()

        labels_by_row_id: dict[str, set[str]] = {}
        for vl in voc_labels:
            if vl.row_id not in labels_by_row_id:
                labels_by_row_id[vl.row_id] = set()
            labels_by_row_id[vl.row_id].add(vl.label)

        emb_path = detection_embeddings_path(app_settings.storage_root, job_id)
        table = pq.read_table(str(emb_path))
        row_ids = table.column("row_id").to_pylist()

        included_label_sets: list[set[str]] = []
        for rid in row_ids:
            if rid not in labels_by_row_id:
                continue
            label_set = labels_by_row_id[rid]
            if "(Negative)" in label_set:
                label_set = set()
            included_label_sets.append(label_set)

    # Only 1 of 2 rows should be included (the labeled one)
    assert len(included_label_sets) == 1
    assert included_label_sets[0] == {"whup"}


@pytest.mark.asyncio
async def test_training_assembly_negative_becomes_empty_set(
    client, app_settings, tmp_path
):
    """Verify that (Negative) labels become empty set in training assembly."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Row 1: label with "whup"
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}",
        json={"label": "whup"},
    )
    # Row 2: label with "(Negative)"
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_2}",
        json={"label": "(Negative)"},
    )

    from sqlalchemy import select

    from humpback.database import create_engine, create_session_factory
    from humpback.models.labeling import VocalizationLabel
    from humpback.storage import detection_embeddings_path

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        result = await session.execute(
            select(VocalizationLabel).where(
                VocalizationLabel.detection_job_id == job_id
            )
        )
        voc_labels = result.scalars().all()

        labels_by_row_id: dict[str, set[str]] = {}
        for vl in voc_labels:
            if vl.row_id not in labels_by_row_id:
                labels_by_row_id[vl.row_id] = set()
            labels_by_row_id[vl.row_id].add(vl.label)

        emb_path = detection_embeddings_path(app_settings.storage_root, job_id)
        table = pq.read_table(str(emb_path))
        row_ids = table.column("row_id").to_pylist()

        included_label_sets: list[set[str]] = []
        for rid in row_ids:
            if rid not in labels_by_row_id:
                continue
            label_set = labels_by_row_id[rid]
            if "(Negative)" in label_set:
                label_set = set()
            included_label_sets.append(label_set)

    # Both rows should be included
    assert len(included_label_sets) == 2
    # One should be {"whup"}, one should be empty set (from Negative)
    assert {"whup"} in included_label_sets
    assert set() in included_label_sets


# ---- Legacy Endpoint Removal ----


@pytest.mark.asyncio
async def test_annotation_endpoints_removed(client):
    """Annotation endpoints no longer exist after sub-window annotation removal."""
    resp = await client.get("/labeling/annotations/job-1?row_id=some-row")
    assert resp.status_code in (404, 405)


@pytest.mark.asyncio
async def test_legacy_training_endpoints_removed(client):
    """Legacy vocalization training endpoints moved to /vocalization/ router."""
    resp = await client.post(
        "/labeling/training-jobs",
        json={"name": "x", "source_detection_job_ids": []},
    )
    assert resp.status_code in (404, 405, 422)

    resp2 = await client.get("/labeling/vocalization-models")
    assert resp2.status_code in (404, 405)


@pytest.mark.asyncio
async def test_refresh_endpoints_removed(client, app_settings, tmp_path):
    """Refresh/reconciliation endpoints no longer exist."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    resp = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh")
    assert resp.status_code in (404, 405)

    resp2 = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh/apply")
    assert resp2.status_code in (404, 405)


# ---- Cascade Delete ----


@pytest.mark.asyncio
async def test_cascade_delete_labels_on_row_delete(client, app_settings, tmp_path):
    """Deleting a detection row via batch edit cascade-deletes vocalization labels."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Create vocalization labels on both rows
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_2}",
        json={"label": "moan"},
    )

    # Verify both exist
    resp1 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}"
    )
    assert len(resp1.json()) == 1
    resp2 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_2}"
    )
    assert len(resp2.json()) == 1

    # Delete ROW_ID_2 via batch edit
    resp3 = await client.patch(
        f"/classifier/detection-jobs/{job_id}/labels",
        json={"edits": [{"action": "delete", "row_id": ROW_ID_2}]},
    )
    assert resp3.status_code == 200

    # ROW_ID_1 label should survive
    resp4 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_1}"
    )
    assert len(resp4.json()) == 1
    assert resp4.json()[0]["label"] == "whup"

    # ROW_ID_2 label should be gone
    resp5 = await client.get(
        f"/labeling/vocalization-labels/{job_id}?row_id={ROW_ID_2}"
    )
    assert len(resp5.json()) == 0
