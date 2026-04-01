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


# ---- Negative Label Mutual Exclusivity ----


@pytest.mark.asyncio
async def test_negative_label_removes_type_labels(client):
    """Adding (Negative) removes existing type labels on the same window."""
    s_utc = BASE_EPOCH + 600.0
    e_utc = BASE_EPOCH + 605.0
    qs = _utc_qs(s_utc, e_utc)
    job_id = "neg-test-job"

    # Create type labels
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs}",
        json={"label": "moan"},
    )

    # Verify both exist
    resp = await client.get(f"/labeling/vocalization-labels/{job_id}{qs}")
    assert len(resp.json()) == 2

    # Add (Negative) — should remove both type labels
    resp2 = await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs}",
        json={"label": "(Negative)"},
    )
    assert resp2.status_code == 201

    # Only (Negative) should remain
    resp3 = await client.get(f"/labeling/vocalization-labels/{job_id}{qs}")
    labels = resp3.json()
    assert len(labels) == 1
    assert labels[0]["label"] == "(Negative)"


@pytest.mark.asyncio
async def test_type_label_removes_negative(client):
    """Adding a type label removes existing (Negative) on the same window."""
    s_utc = BASE_EPOCH + 700.0
    e_utc = BASE_EPOCH + 705.0
    qs = _utc_qs(s_utc, e_utc)
    job_id = "neg-test-job2"

    # Create (Negative) label
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs}",
        json={"label": "(Negative)"},
    )

    resp = await client.get(f"/labeling/vocalization-labels/{job_id}{qs}")
    assert len(resp.json()) == 1
    assert resp.json()[0]["label"] == "(Negative)"

    # Add type label — should remove (Negative)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs}",
        json={"label": "whup"},
    )

    resp2 = await client.get(f"/labeling/vocalization-labels/{job_id}{qs}")
    labels = resp2.json()
    assert len(labels) == 1
    assert labels[0]["label"] == "whup"


@pytest.mark.asyncio
async def test_negative_does_not_affect_other_windows(client):
    """Mutual exclusivity only applies to the same window."""
    job_id = "neg-test-job3"
    qs1 = _utc_qs(BASE_EPOCH + 800.0, BASE_EPOCH + 805.0)
    qs2 = _utc_qs(BASE_EPOCH + 810.0, BASE_EPOCH + 815.0)

    # Label window 1 with type, window 2 with (Negative)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs1}",
        json={"label": "whup"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{qs2}",
        json={"label": "(Negative)"},
    )

    # Both windows should retain their labels
    resp1 = await client.get(f"/labeling/vocalization-labels/{job_id}{qs1}")
    assert len(resp1.json()) == 1
    assert resp1.json()[0]["label"] == "whup"

    resp2 = await client.get(f"/labeling/vocalization-labels/{job_id}{qs2}")
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
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    # Row 2: leave unlabeled (should be excluded from training)

    # Simulate training data assembly (same logic as vocalization_worker.py)
    from sqlalchemy import select

    from humpback.classifier.detection_rows import parse_recording_timestamp
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

        labels_by_utc: dict[tuple[float, float], set[str]] = {}
        for vl in voc_labels:
            key = (vl.start_utc, vl.end_utc)
            if key not in labels_by_utc:
                labels_by_utc[key] = set()
            labels_by_utc[key].add(vl.label)

        emb_path = detection_embeddings_path(app_settings.storage_root, job_id)
        table = pq.read_table(str(emb_path))
        filenames = table.column("filename").to_pylist()
        start_secs = table.column("start_sec").to_pylist()
        end_secs = table.column("end_sec").to_pylist()

        included_label_sets: list[set[str]] = []
        for i in range(table.num_rows):
            fname = filenames[i]
            ts = parse_recording_timestamp(fname)
            base_epoch = ts.timestamp() if ts else 0.0
            utc_key = (
                base_epoch + float(start_secs[i]),
                base_epoch + float(end_secs[i]),
            )

            if utc_key not in labels_by_utc:
                continue

            label_set = labels_by_utc[utc_key]
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
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    # Row 2: label with "(Negative)"
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row2_qs}",
        json={"label": "(Negative)"},
    )

    from sqlalchemy import select

    from humpback.classifier.detection_rows import parse_recording_timestamp
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

        labels_by_utc: dict[tuple[float, float], set[str]] = {}
        for vl in voc_labels:
            key = (vl.start_utc, vl.end_utc)
            if key not in labels_by_utc:
                labels_by_utc[key] = set()
            labels_by_utc[key].add(vl.label)

        emb_path = detection_embeddings_path(app_settings.storage_root, job_id)
        table = pq.read_table(str(emb_path))
        filenames = table.column("filename").to_pylist()
        start_secs = table.column("start_sec").to_pylist()
        end_secs = table.column("end_sec").to_pylist()

        included_label_sets: list[set[str]] = []
        for i in range(table.num_rows):
            fname = filenames[i]
            ts = parse_recording_timestamp(fname)
            base_epoch = ts.timestamp() if ts else 0.0
            utc_key = (
                base_epoch + float(start_secs[i]),
                base_epoch + float(end_secs[i]),
            )

            if utc_key not in labels_by_utc:
                continue

            label_set = labels_by_utc[utc_key]
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
    qs = _utc_qs(BASE_EPOCH + 500.0, BASE_EPOCH + 505.0)
    resp = await client.get(f"/labeling/annotations/job-1{qs}")
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


# ---- Refresh / Reconciliation ----


@pytest.mark.asyncio
async def test_refresh_preview_all_matched(client, app_settings, tmp_path):
    """Preview with no changes shows all labels matched."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )

    resp = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched_count"] == 1
    assert data["orphaned_count"] == 0
    assert data["orphaned_labels"] == []


@pytest.mark.asyncio
async def test_refresh_preview_detects_orphans(client, app_settings, tmp_path):
    """Preview detects orphaned labels after row deletion."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Label both rows
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

    # Delete row2 from the row store by rewriting with only row1
    row_store = detection_row_store_path(app_settings.storage_root, job_id)
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
        }
    ]
    write_detection_row_store(row_store, rows)

    resp = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh")
    assert resp.status_code == 200
    data = resp.json()
    assert data["matched_count"] == 1
    assert data["orphaned_count"] == 1
    assert len(data["orphaned_labels"]) == 1
    assert data["orphaned_labels"][0]["label"] == "moan"


@pytest.mark.asyncio
async def test_refresh_apply_deletes_orphans(client, app_settings, tmp_path):
    """Apply deletes orphaned labels and updates version on survivors."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

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

    # Delete row2 from row store
    row_store = detection_row_store_path(app_settings.storage_root, job_id)
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
        }
    ]
    write_detection_row_store(row_store, rows)

    resp = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh/apply")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted_count"] == 1
    assert data["surviving_count"] == 1

    # Verify orphaned label is gone
    resp2 = await client.get(f"/labeling/vocalization-labels/{job_id}{row2_qs}")
    assert resp2.json() == []

    # Surviving label still exists
    resp3 = await client.get(f"/labeling/vocalization-labels/{job_id}{row1_qs}")
    assert len(resp3.json()) == 1


@pytest.mark.asyncio
async def test_refresh_apply_idempotent(client, app_settings, tmp_path):
    """Calling apply when already in sync is a no-op."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )

    resp = await client.post(f"/labeling/vocalization-labels/{job_id}/refresh/apply")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted_count"] == 0
    assert data["surviving_count"] == 1


@pytest.mark.asyncio
async def test_refresh_preview_nonexistent_job(client):
    """Preview for non-existent job returns 404."""
    resp = await client.post("/labeling/vocalization-labels/nonexistent/refresh")
    assert resp.status_code == 404


# ---- Version Tracking ----


@pytest.mark.asyncio
async def test_label_records_row_store_version(client, app_settings, tmp_path):
    """Newly created labels record the detection job's row_store_version."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    resp = await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["row_store_version_at_import"] == 1


# ---- Deletion Guards ----


@pytest.mark.asyncio
async def test_delete_detection_job_blocked_by_labels(client, app_settings, tmp_path):
    """Cannot delete a detection job that has vocalization labels."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup"},
    )

    resp = await client.delete(f"/classifier/detection-jobs/{job_id}")
    assert resp.status_code == 409
    assert "vocalization label" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_delete_detection_job_succeeds_without_deps(
    client, app_settings, tmp_path
):
    """Deletion succeeds when no vocalization labels or training datasets."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    resp = await client.delete(f"/classifier/detection-jobs/{job_id}")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_bulk_delete_partial_block(client, app_settings, tmp_path):
    """Bulk delete: deletable jobs succeed, blocked ones return details."""
    job_id_1, _ = await _seed_detection_job(app_settings, tmp_path)

    # Seed a second detection job (re-use helper but with fresh IDs)
    job_id_2, _ = await _seed_detection_job(app_settings, tmp_path)

    # Add label to job_1 only
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id_1}{row1_qs}",
        json={"label": "whup"},
    )

    resp = await client.post(
        "/classifier/detection-jobs/bulk-delete",
        json={"ids": [job_id_1, job_id_2]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1  # only job_2 deleted
    assert len(data["blocked"]) == 1
    assert data["blocked"][0]["job_id"] == job_id_1


# ---- Bulk list (all labels) endpoint ----


@pytest.mark.asyncio
async def test_list_all_vocalization_labels(client, app_settings, tmp_path):
    """GET /all returns all vocalization labels for a detection job."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    # Create labels on two different windows
    row1_qs = _utc_qs(BASE_EPOCH, BASE_EPOCH + 5.0)
    row2_qs = _utc_qs(BASE_EPOCH + 5.0, BASE_EPOCH + 10.0)
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "whup", "source": "manual"},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row1_qs}",
        json={"label": "moan", "source": "inference", "confidence": 0.85},
    )
    await client.post(
        f"/labeling/vocalization-labels/{job_id}{row2_qs}",
        json={"label": "whup", "source": "inference", "confidence": 0.72},
    )

    resp = await client.get(f"/labeling/vocalization-labels/{job_id}/all")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    # Sorted by start_utc, then source, then label
    # Both "inference" < "manual" alphabetically
    assert data[0]["label"] == "moan"
    assert data[0]["source"] == "inference"
    assert data[0]["start_utc"] == BASE_EPOCH
    assert data[1]["label"] == "whup"
    assert data[1]["source"] == "manual"
    assert data[1]["start_utc"] == BASE_EPOCH
    assert data[2]["label"] == "whup"
    assert data[2]["source"] == "inference"
    assert data[2]["start_utc"] == BASE_EPOCH + 5.0
    # No id/created_at/updated_at in timeline response
    assert "id" not in data[0]


@pytest.mark.asyncio
async def test_list_all_vocalization_labels_empty(client, app_settings, tmp_path):
    """GET /all returns empty list when no labels exist."""
    job_id, _ = await _seed_detection_job(app_settings, tmp_path)

    resp = await client.get(f"/labeling/vocalization-labels/{job_id}/all")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_list_all_vocalization_labels_not_found(client):
    """GET /all returns 404 for nonexistent detection job."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    resp = await client.get(f"/labeling/vocalization-labels/{fake_id}/all")
    assert resp.status_code == 404
