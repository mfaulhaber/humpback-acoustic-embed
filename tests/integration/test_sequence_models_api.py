"""Integration tests for the Sequence Models API router."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.storage import (
    hmm_sequence_exemplars_path,
    hmm_sequence_overlay_path,
)


async def _seed_segmentation_job(app_settings, status: str) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1600.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=status,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.commit()
        await session.refresh(seg_job)
        return seg_job.id


async def test_create_returns_201_then_200_on_idempotent_resubmit(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    payload = {"event_segmentation_job_id": seg_job_id}

    first = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert first.status_code == 201, first.text
    body = first.json()
    job_id = body["id"]
    assert body["status"] == JobStatus.queued.value
    assert body["event_segmentation_job_id"] == seg_job_id

    second = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == job_id


async def test_list_filters_by_status(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )

    listed = await client.get(
        "/sequence-models/continuous-embeddings",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["status"] == JobStatus.queued.value

    listed_complete = await client.get(
        "/sequence-models/continuous-embeddings",
        params={"status": JobStatus.complete.value},
    )
    assert listed_complete.status_code == 200
    assert listed_complete.json() == []


async def test_get_detail_returns_404_on_missing(client):
    response = await client.get("/sequence-models/continuous-embeddings/missing-id")
    assert response.status_code == 404


async def test_get_detail_returns_job_without_manifest(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/continuous-embeddings/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["manifest"] is None


async def test_cancel_queued_returns_canceled(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    cancel_resp = await client.post(
        f"/sequence-models/continuous-embeddings/{job_id}/cancel"
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == JobStatus.canceled.value


async def test_cancel_returns_404_on_missing(client):
    response = await client.post(
        "/sequence-models/continuous-embeddings/missing-id/cancel"
    )
    assert response.status_code == 404


async def test_cancel_terminal_returns_409(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from humpback.models.sequence_models import ContinuousEmbeddingJob

        job = await session.get(ContinuousEmbeddingJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    cancel_resp = await client.post(
        f"/sequence-models/continuous-embeddings/{job_id}/cancel"
    )
    assert cancel_resp.status_code == 409


async def test_create_rejects_invalid_hop(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id, "hop_seconds": 0},
    )
    assert response.status_code == 422


async def test_create_rejects_missing_segmentation_job(client):
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": "nonexistent"},
    )
    assert response.status_code == 422


async def test_create_rejects_non_complete_segmentation_job(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.running.value)
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# HMM Sequence Job endpoints
# ---------------------------------------------------------------------------


async def _seed_complete_continuous_embedding_job(app_settings) -> str:
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1600.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.flush()
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature="test-sig-hmm",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


async def test_hmm_create_returns_201(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    response = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == JobStatus.queued.value
    assert body["n_states"] == 4
    assert body["continuous_embedding_job_id"] == cej_id


async def test_hmm_create_rejects_invalid_source(client):
    response = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": "nonexistent", "n_states": 3},
    )
    assert response.status_code == 422


async def test_hmm_list_filters_by_status(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )

    listed = await client.get(
        "/sequence-models/hmm-sequences",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    assert len(listed.json()) >= 1

    listed_complete = await client.get(
        "/sequence-models/hmm-sequences",
        params={"status": JobStatus.complete.value},
    )
    assert listed_complete.status_code == 200
    assert listed_complete.json() == []


async def test_hmm_get_detail_returns_404_on_missing(client):
    response = await client.get("/sequence-models/hmm-sequences/missing-id")
    assert response.status_code == 404


async def test_hmm_get_detail_returns_job_without_summary(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 5},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["summary"] is None


async def test_hmm_detail_includes_region_detection_job_id(client, app_settings):
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1600.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.flush()
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature="test-sig-hmm-region",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        cej_id = cej.id
        region_job_id = region_job.id

    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["region_detection_job_id"] == region_job_id
    assert body["region_start_timestamp"] == 1000.0
    assert body["region_end_timestamp"] == 1600.0


async def test_hmm_cancel_queued_returns_canceled(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    job_id = create_resp.json()["id"]

    cancel_resp = await client.post(f"/sequence-models/hmm-sequences/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == JobStatus.canceled.value


async def test_hmm_cancel_returns_404_on_missing(client):
    response = await client.post("/sequence-models/hmm-sequences/missing-id/cancel")
    assert response.status_code == 404


async def test_hmm_cancel_terminal_returns_409(client, app_settings):
    from humpback.models.sequence_models import HMMSequenceJob

    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    job_id = create_resp.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(HMMSequenceJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    cancel_resp = await client.post(f"/sequence-models/hmm-sequences/{job_id}/cancel")
    assert cancel_resp.status_code == 409


# ---------------------------------------------------------------------------
# Delete endpoints
# ---------------------------------------------------------------------------


async def test_delete_continuous_embedding_returns_204(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    delete_resp = await client.delete(
        f"/sequence-models/continuous-embeddings/{job_id}"
    )
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/sequence-models/continuous-embeddings/{job_id}")
    assert get_resp.status_code == 404


async def test_delete_continuous_embedding_missing_returns_404(client):
    response = await client.delete("/sequence-models/continuous-embeddings/nonexistent")
    assert response.status_code == 404


async def test_delete_hmm_sequence_returns_204(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    job_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/sequence-models/hmm-sequences/{job_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert get_resp.status_code == 404


async def test_delete_hmm_sequence_missing_returns_404(client):
    response = await client.delete("/sequence-models/hmm-sequences/nonexistent")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Overlay / exemplars endpoints — unified shape + legacy adapter (ADR-059)
# ---------------------------------------------------------------------------


async def _create_complete_hmm_job(client, app_settings) -> str:
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    job_id = create_resp.json()["id"]

    from humpback.models.sequence_models import HMMSequenceJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(HMMSequenceJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()
    return job_id


def _legacy_overlay_table() -> pa.Table:
    return pa.table(
        {
            "merged_span_id": pa.array([0, 0, 1], type=pa.int32()),
            "window_index_in_span": pa.array([0, 1, 0], type=pa.int32()),
            "start_timestamp": pa.array([10.0, 11.0, 20.0], type=pa.float64()),
            "end_timestamp": pa.array([15.0, 16.0, 25.0], type=pa.float64()),
            "pca_x": pa.array([0.1, 0.2, 0.3], type=pa.float32()),
            "pca_y": pa.array([0.4, 0.5, 0.6], type=pa.float32()),
            "umap_x": pa.array([0.7, 0.8, 0.9], type=pa.float32()),
            "umap_y": pa.array([1.0, 1.1, 1.2], type=pa.float32()),
            "viterbi_state": pa.array([0, 1, 2], type=pa.int16()),
            "max_state_probability": pa.array([0.9, 0.8, 0.7], type=pa.float32()),
        }
    )


def _unified_overlay_table() -> pa.Table:
    return pa.table(
        {
            "sequence_id": pa.array(["0", "0", "1"], type=pa.string()),
            "position_in_sequence": pa.array([0, 1, 0], type=pa.int32()),
            "start_timestamp": pa.array([10.0, 11.0, 20.0], type=pa.float64()),
            "end_timestamp": pa.array([15.0, 16.0, 25.0], type=pa.float64()),
            "pca_x": pa.array([0.1, 0.2, 0.3], type=pa.float32()),
            "pca_y": pa.array([0.4, 0.5, 0.6], type=pa.float32()),
            "umap_x": pa.array([0.7, 0.8, 0.9], type=pa.float32()),
            "umap_y": pa.array([1.0, 1.1, 1.2], type=pa.float32()),
            "viterbi_state": pa.array([0, 1, 2], type=pa.int16()),
            "max_state_probability": pa.array([0.9, 0.8, 0.7], type=pa.float32()),
        }
    )


async def test_overlay_endpoint_translates_legacy_columns(client, app_settings):
    job_id = await _create_complete_hmm_job(client, app_settings)
    overlay_path = hmm_sequence_overlay_path(app_settings.storage_root, job_id)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_legacy_overlay_table(), overlay_path)

    resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}/overlay")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 3
    item = body["items"][0]
    assert item["sequence_id"] == "0"
    assert item["position_in_sequence"] == 0
    assert "merged_span_id" not in item

    # On-disk file remains untouched (still has legacy columns).
    on_disk = pq.read_table(overlay_path)
    assert "merged_span_id" in on_disk.column_names


async def test_overlay_endpoint_unified_format_is_no_op(client, app_settings):
    job_id = await _create_complete_hmm_job(client, app_settings)
    overlay_path = hmm_sequence_overlay_path(app_settings.storage_root, job_id)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(_unified_overlay_table(), overlay_path)

    resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}/overlay")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    item = body["items"][2]
    assert item["sequence_id"] == "1"
    assert item["position_in_sequence"] == 0


async def test_exemplars_endpoint_translates_legacy_keys(client, app_settings):
    job_id = await _create_complete_hmm_job(client, app_settings)
    exemplars_path = hmm_sequence_exemplars_path(app_settings.storage_root, job_id)
    exemplars_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "n_states": 1,
        "states": {
            "0": [
                {
                    "merged_span_id": 5,
                    "window_index_in_span": 12,
                    "audio_file_id": 200,
                    "start_timestamp": 10.0,
                    "end_timestamp": 15.0,
                    "max_state_probability": 0.95,
                    "exemplar_type": "high_confidence",
                }
            ]
        },
    }
    exemplars_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}/exemplars")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    record = body["states"]["0"][0]
    assert record["sequence_id"] == "5"
    assert record["position_in_sequence"] == 12
    assert record["extras"] == {}
    assert "merged_span_id" not in record


async def test_exemplars_endpoint_unified_format_is_no_op(client, app_settings):
    job_id = await _create_complete_hmm_job(client, app_settings)
    exemplars_path = hmm_sequence_exemplars_path(app_settings.storage_root, job_id)
    exemplars_path.parent.mkdir(parents=True, exist_ok=True)
    unified_payload = {
        "n_states": 1,
        "states": {
            "0": [
                {
                    "sequence_id": "region-A",
                    "position_in_sequence": 17,
                    "audio_file_id": 200,
                    "start_timestamp": 10.0,
                    "end_timestamp": 10.25,
                    "max_state_probability": 0.93,
                    "exemplar_type": "high_confidence",
                    "extras": {"tier": "event_core"},
                }
            ]
        },
    }
    exemplars_path.write_text(json.dumps(unified_payload), encoding="utf-8")

    resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}/exemplars")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    record = body["states"]["0"][0]
    assert record["sequence_id"] == "region-A"
    assert record["position_in_sequence"] == 17
    assert record["extras"] == {"tier": "event_core"}
