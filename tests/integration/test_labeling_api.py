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


async def _seed_detection_job(app_settings, tmp_path):
    """Create a completed detection job with row store and embeddings."""
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

    # Write detection row store (UTC identity schema)
    row_store = detection_row_store_path(storage_root, job_id)
    row_store.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
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

    # Write detection embeddings (still uses filename/start_sec/end_sec schema)
    emb_path = detection_embeddings_path(storage_root, job_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    # The embedding file uses a filename with an embedded UTC timestamp so that
    # parse_recording_timestamp can derive BASE_EPOCH from it.
    emb_filename = "20240615T080000Z.wav"
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("start_sec", pa.float32()),
            ("end_sec", pa.float32()),
            ("embedding", pa.list_(pa.float32(), 4)),
        ]
    )
    table = pa.table(
        {
            "filename": [emb_filename, emb_filename],
            "start_sec": [0.0, 5.0],
            "end_sec": [5.0, 10.0],
            "embedding": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
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
    # Hydrophone job: row UTC = 1751439678.0 (2025-07-02T00:01:18Z)
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

    # Embedding file uses timestamp-based filename for UTC resolution.
    # parse_recording_timestamp("20250702T070118Z.wav") => 1751439678.0
    emb_filename = "20250702T070118Z.wav"
    emb_path = detection_embeddings_path(storage_root, job_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("start_sec", pa.float32()),
            ("end_sec", pa.float32()),
            ("embedding", pa.list_(pa.float32(), 4)),
        ]
    )
    table = pa.table(
        {
            "filename": [emb_filename],
            "start_sec": [0.0],
            "end_sec": [5.0],
            "embedding": [[1.0, 0.0, 0.0, 0.0]],
        },
        schema=schema,
    )
    pq.write_table(table, emb_path)

    return job_id, model_id, hydro_start_utc, hydro_end_utc


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


# Helper to build query string for UTC-keyed label/annotation endpoints
def _utc_qs(start_utc: float, end_utc: float) -> str:
    return f"?start_utc={start_utc}&end_utc={end_utc}"


# ---- CRUD tests ----


@pytest.mark.asyncio
async def test_create_and_list_vocalization_labels(client):
    """Test creating and listing vocalization labels."""
    s_utc = BASE_EPOCH
    e_utc = BASE_EPOCH + 5.0
    qs = _utc_qs(s_utc, e_utc)

    # Create a label (no detection job needed for basic CRUD)
    resp = await client.post(
        f"/labeling/vocalization-labels/fake-job{qs}",
        json={"label": "whup", "source": "manual"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["label"] == "whup"
    assert data["source"] == "manual"
    assert data["detection_job_id"] == "fake-job"
    assert data["start_utc"] == s_utc
    assert data["end_utc"] == e_utc
    label_id = data["id"]

    # Create another label
    resp2 = await client.post(
        f"/labeling/vocalization-labels/fake-job{qs}",
        json={"label": "moan", "confidence": 0.85, "source": "search"},
    )
    assert resp2.status_code == 201

    # List labels for the row
    resp3 = await client.get(f"/labeling/vocalization-labels/fake-job{qs}")
    assert resp3.status_code == 200
    labels = resp3.json()
    assert len(labels) == 2
    assert {lbl["label"] for lbl in labels} == {"whup", "moan"}

    # List labels for different UTC range — empty
    other_qs = _utc_qs(BASE_EPOCH + 100.0, BASE_EPOCH + 105.0)
    resp4 = await client.get(f"/labeling/vocalization-labels/fake-job{other_qs}")
    assert resp4.status_code == 200
    assert resp4.json() == []

    return label_id


@pytest.mark.asyncio
async def test_update_vocalization_label(client):
    """Test updating a vocalization label."""
    s_utc = BASE_EPOCH + 200.0
    e_utc = BASE_EPOCH + 205.0
    qs = _utc_qs(s_utc, e_utc)

    resp = await client.post(
        f"/labeling/vocalization-labels/job1{qs}",
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
    s_utc = BASE_EPOCH + 300.0
    e_utc = BASE_EPOCH + 305.0
    qs = _utc_qs(s_utc, e_utc)

    resp = await client.post(
        f"/labeling/vocalization-labels/job1{qs}",
        json={"label": "shriek"},
    )
    label_id = resp.json()["id"]

    resp2 = await client.delete(f"/labeling/vocalization-labels/{label_id}")
    assert resp2.status_code == 204

    # Should be gone
    resp3 = await client.get(f"/labeling/vocalization-labels/job1{qs}")
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
    qs1 = _utc_qs(BASE_EPOCH + 400.0, BASE_EPOCH + 405.0)
    qs2 = _utc_qs(BASE_EPOCH + 410.0, BASE_EPOCH + 415.0)
    qs3 = _utc_qs(BASE_EPOCH + 420.0, BASE_EPOCH + 425.0)

    await client.post(
        f"/labeling/vocalization-labels/j1{qs1}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/j1{qs2}",
        json={"label": "moan"},
    )
    await client.post(
        f"/labeling/vocalization-labels/j2{qs3}",
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

    # Add labels using UTC identity of the seeded rows
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "moan"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row2_qs}",
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
            "start_utc": BASE_EPOCH,
            "end_utc": BASE_EPOCH + 5.0,
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
        json={
            "start_utc": 9999999999.0,
            "end_utc": 9999999999.0 + 5.0,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_neighbors_nonexistent_job(client):
    """Requesting neighbors for non-existent job returns 404."""
    resp = await client.post(
        "/labeling/detection-neighbors/nonexistent",
        json={
            "start_utc": BASE_EPOCH,
            "end_utc": BASE_EPOCH + 5.0,
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_detection_neighbors_hydrophone_utc(client, app_settings, tmp_path):
    """Hydrophone neighbor lookup works with UTC identity params."""
    (
        job_id,
        _model_id,
        hydro_start_utc,
        hydro_end_utc,
    ) = await _seed_hydrophone_detection_job(app_settings)
    ref_es_id = await _seed_reference_embeddings(app_settings, tmp_path)

    resp = await client.post(
        f"/labeling/detection-neighbors/{job_id}",
        json={
            "start_utc": hydro_start_utc,
            "end_utc": hydro_end_utc,
            "top_k": 5,
            "embedding_set_ids": [ref_es_id],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["hits"]) > 0
    assert data["total_candidates"] > 0


# ---- Training Job Creation ----


@pytest.mark.asyncio
async def test_create_vocalization_training_job(client, app_settings, tmp_path):
    """Test creating a vocalization training job from labeled detection data."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Add vocalization labels (need at least 2 distinct labels)
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row2_qs}",
        json={"label": "moan"},
    )

    resp = await client.post(
        "/labeling/training-jobs",
        json={
            "name": "test-voc-classifier",
            "source_detection_job_ids": [job_id],
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-voc-classifier"
    assert data["job_purpose"] == "vocalization"
    assert data["status"] == "queued"
    assert data["source_detection_job_ids"] == [job_id]


@pytest.mark.asyncio
async def test_get_vocalization_training_job(client, app_settings, tmp_path):
    """Test fetching a vocalization training job by ID."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Add vocalization labels (need at least 2 distinct labels)
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row2_qs}",
        json={"label": "moan"},
    )

    # Create a training job
    create_resp = await client.post(
        "/labeling/training-jobs",
        json={
            "name": "test-voc-get",
            "source_detection_job_ids": [job_id],
        },
    )
    assert create_resp.status_code == 201
    training_job_id = create_resp.json()["id"]

    # Fetch it
    resp = await client.get(f"/labeling/training-jobs/{training_job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == training_job_id
    assert data["name"] == "test-voc-get"
    assert data["status"] == "queued"
    assert data["job_purpose"] == "vocalization"
    assert data["source_detection_job_ids"] == [job_id]


@pytest.mark.asyncio
async def test_get_vocalization_training_job_not_found(client):
    """Fetching a nonexistent training job returns 404."""
    resp = await client.get("/labeling/training-jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_vocalization_training_job_too_few_labels(
    client,
    app_settings,
    tmp_path,
):
    """Reject training when fewer than 2 distinct labels exist."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Only one label
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )

    resp = await client.post(
        "/labeling/training-jobs",
        json={
            "name": "should-fail",
            "source_detection_job_ids": [job_id],
        },
    )
    assert resp.status_code == 400


# ---- Vocalization Models ----


@pytest.mark.asyncio
async def test_list_vocalization_models_empty(client):
    """Initially no vocalization models exist."""
    resp = await client.get("/labeling/vocalization-models")
    assert resp.status_code == 200
    assert resp.json() == []


# ---- Prediction ----


@pytest.mark.asyncio
async def test_predict_nonexistent_model(client, app_settings, tmp_path):
    """Prediction with nonexistent model returns 404."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    resp = await client.post(
        f"/labeling/predict/{job_id}",
        json={"vocalization_model_id": "nonexistent"},
    )
    assert resp.status_code == 404


# ---- Annotations CRUD ----


@pytest.mark.asyncio
async def test_annotation_crud(client):
    """Test create, list, update, and delete annotations."""
    s_utc = BASE_EPOCH + 500.0
    e_utc = BASE_EPOCH + 505.0
    qs = _utc_qs(s_utc, e_utc)

    # Create
    resp = await client.post(
        f"/labeling/annotations/job-1{qs}",
        json={
            "start_offset_sec": 1.0,
            "end_offset_sec": 3.5,
            "label": "whup",
            "notes": "clear call",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["start_offset_sec"] == 1.0
    assert data["end_offset_sec"] == 3.5
    assert data["label"] == "whup"
    assert data["notes"] == "clear call"
    assert data["start_utc"] == s_utc
    assert data["end_utc"] == e_utc
    ann_id = data["id"]

    # List
    resp2 = await client.get(f"/labeling/annotations/job-1{qs}")
    assert resp2.status_code == 200
    anns = resp2.json()
    assert len(anns) == 1
    assert anns[0]["id"] == ann_id

    # Update
    resp3 = await client.put(
        f"/labeling/annotations/{ann_id}",
        json={"label": "moan", "end_offset_sec": 4.0},
    )
    assert resp3.status_code == 200
    assert resp3.json()["label"] == "moan"
    assert resp3.json()["end_offset_sec"] == 4.0

    # Delete
    resp4 = await client.delete(f"/labeling/annotations/{ann_id}")
    assert resp4.status_code == 204

    # List again — empty
    resp5 = await client.get(f"/labeling/annotations/job-1{qs}")
    assert resp5.json() == []


@pytest.mark.asyncio
async def test_annotation_invalid_bounds(client):
    """Reject annotation where end <= start."""
    qs = _utc_qs(BASE_EPOCH + 600.0, BASE_EPOCH + 605.0)
    resp = await client.post(
        f"/labeling/annotations/job-1{qs}",
        json={
            "start_offset_sec": 3.0,
            "end_offset_sec": 2.0,
            "label": "whup",
        },
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_annotation_list_empty(client):
    """List annotations for row with none returns empty list."""
    qs = _utc_qs(BASE_EPOCH + 700.0, BASE_EPOCH + 705.0)
    resp = await client.get(f"/labeling/annotations/job-1{qs}")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_annotation_delete_nonexistent(client):
    """Delete nonexistent annotation returns 404."""
    resp = await client.delete("/labeling/annotations/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_multiple_annotations_per_row(client):
    """Multiple annotations can exist on the same detection row."""
    s_utc = BASE_EPOCH + 800.0
    e_utc = BASE_EPOCH + 805.0
    qs = _utc_qs(s_utc, e_utc)

    await client.post(
        f"/labeling/annotations/job-2{qs}",
        json={"start_offset_sec": 0.0, "end_offset_sec": 1.5, "label": "whup"},
    )
    await client.post(
        f"/labeling/annotations/job-2{qs}",
        json={"start_offset_sec": 2.0, "end_offset_sec": 4.0, "label": "moan"},
    )

    resp = await client.get(f"/labeling/annotations/job-2{qs}")
    anns = resp.json()
    assert len(anns) == 2
    # Should be sorted by start_offset_sec
    assert anns[0]["start_offset_sec"] < anns[1]["start_offset_sec"]


@pytest.mark.asyncio
async def test_summary_includes_annotation_labels(client, app_settings, tmp_path):
    """Annotations contribute to labeling summary label_distribution."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # No vocalization labels — only annotations
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/annotations/{job_id}{row1_qs}",
        json={"start_offset_sec": 0.5, "end_offset_sec": 3.0, "label": "whup"},
    )
    await client.post(
        f"/labeling/annotations/{job_id}{row2_qs}",
        json={"start_offset_sec": 1.0, "end_offset_sec": 4.0, "label": "moan"},
    )

    resp = await client.get(f"/labeling/summary/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["labeled_rows"] == 2
    assert "whup" in data["label_distribution"]
    assert "moan" in data["label_distribution"]


@pytest.mark.asyncio
async def test_training_summary_aggregates_across_jobs(client, app_settings, tmp_path):
    """Training summary aggregates labels from all detection jobs."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Add annotations to this job
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/annotations/{job_id}{row1_qs}",
        json={"start_offset_sec": 0.5, "end_offset_sec": 3.0, "label": "whup"},
    )
    await client.post(
        f"/labeling/annotations/{job_id}{row2_qs}",
        json={"start_offset_sec": 1.0, "end_offset_sec": 4.0, "label": "moan"},
    )

    resp = await client.get("/labeling/training-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert job_id in data["labeled_job_ids"]
    assert data["labeled_rows"] >= 2
    assert "whup" in data["label_distribution"]
    assert "moan" in data["label_distribution"]


@pytest.mark.asyncio
async def test_training_job_accepts_annotation_only_labels(
    client, app_settings, tmp_path
):
    """Training job creation succeeds when labels come from annotations only."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Only annotations, no vocalization labels
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)

    await client.post(
        f"/labeling/annotations/{job_id}{row1_qs}",
        json={"start_offset_sec": 0.5, "end_offset_sec": 3.0, "label": "whup"},
    )
    await client.post(
        f"/labeling/annotations/{job_id}{row2_qs}",
        json={"start_offset_sec": 1.0, "end_offset_sec": 4.0, "label": "moan"},
    )

    resp = await client.post(
        "/labeling/training-jobs",
        json={
            "name": "annotation-only-classifier",
            "source_detection_job_ids": [job_id],
        },
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "queued"


# ---- Active Learning ----


@pytest.mark.asyncio
async def test_active_learning_cycle_nonexistent_model(client, app_settings, tmp_path):
    """Cycle with nonexistent model returns 404."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)
    resp = await client.post(
        "/labeling/active-learning-cycle",
        json={
            "vocalization_model_id": "nonexistent",
            "detection_job_ids": [job_id],
            "name": "cycle-1",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_convergence_nonexistent_model(client):
    """Convergence for nonexistent model returns 404."""
    resp = await client.get("/labeling/convergence/nonexistent")
    assert resp.status_code == 404
