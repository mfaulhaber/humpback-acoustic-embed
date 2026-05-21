"""Integration tests for retained Sequence Models / Continuous Embedding APIs."""

import math
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.piano_roll_notes import PianoRollNotesJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, EventEncoderJob
from humpback.sequence_models.event_encoder import DESCRIPTOR_ORDER


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


async def _seed_crnn_event_encoder_source(app_settings):
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
        continuous = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            event_source_mode="raw",
            region_detection_job_id=region_job.id,
            model_version="crnn-call-parsing-pytorch",
            target_sample_rate=16000,
            encoding_signature=f"cej-sig-{seg_job.id}",
            crnn_checkpoint_sha256="abc123",
            chunk_size_seconds=0.25,
            chunk_hop_seconds=0.25,
            projection_kind="identity",
            projection_dim=1024,
            total_regions=1,
            total_chunks=10,
        )
        session.add(continuous)
        await session.commit()
        await session.refresh(seg_job)
        await session.refresh(continuous)
        return seg_job.id, continuous.id


async def _seed_event_encoder_timeline_job(
    app_settings,
    *,
    status: str = JobStatus.complete.value,
    write_tokens: bool = True,
    write_vectors: bool = True,
    source_model_version: str = "crnn-call-parsing-pytorch",
) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=2000.0,
            end_timestamp=2600.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.flush()
        continuous = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            event_source_mode="raw",
            region_detection_job_id=(
                region_job.id
                if source_model_version == "crnn-call-parsing-pytorch"
                else None
            ),
            model_version=source_model_version,
            target_sample_rate=16000,
            encoding_signature=f"timeline-cej-sig-{seg_job.id}",
            crnn_checkpoint_sha256="abc123",
            chunk_size_seconds=0.25,
            chunk_hop_seconds=0.25,
            projection_kind="identity",
            projection_dim=1024,
            total_regions=1,
            total_chunks=10,
        )
        session.add(continuous)
        await session.flush()
        job = EventEncoderJob(
            status=status,
            event_segmentation_job_id=seg_job.id,
            event_source_mode="effective",
            continuous_embedding_job_id=continuous.id,
            continuous_embedding_signature=continuous.encoding_signature,
            tokenizer_version="crnn-event-encoder-v2",
            pooling_config_json="{}",
            descriptor_config_json="{}",
            preprocessing_config_json="{}",
            k_values_json="[2,3]",
            random_seed=0,
            tokenization_signature=f"timeline-eej-sig-{seg_job.id}",
            total_events=2,
            encoded_events=2,
            skipped_events=0,
        )
        session.add(job)
        await session.flush()

        token_path = (
            app_settings.storage_root
            / "event_encoders"
            / job.id
            / "event_tokens.parquet"
        )
        vector_path = (
            app_settings.storage_root
            / "event_encoders"
            / job.id
            / "event_vectors.parquet"
        )
        token_path.parent.mkdir(parents=True, exist_ok=True)
        job.event_tokens_path = str(token_path)
        job.event_vectors_path = str(vector_path)
        if write_tokens:
            rows = [
                {
                    "k": 2,
                    "event_id": "evt-b",
                    "region_id": "region-1",
                    "source_sequence_key": "hydrophone:rpi_orcasound_lab",
                    "sequence_index": 1,
                    "start_timestamp": 2125.0,
                    "end_timestamp": 2126.0,
                    "token_id": 1,
                    "token_label": "T01",
                    "distance_to_centroid": 0.4,
                    "second_centroid_distance": 0.8,
                    "token_confidence": 0.5,
                    **_descriptor_payload(offset=1.0),
                },
                {
                    "k": 3,
                    "event_id": "evt-a",
                    "region_id": "region-1",
                    "source_sequence_key": "hydrophone:rpi_orcasound_lab",
                    "sequence_index": 0,
                    "start_timestamp": 2123.5,
                    "end_timestamp": 2124.25,
                    "token_id": 2,
                    "token_label": "T02",
                    "distance_to_centroid": 0.2,
                    "second_centroid_distance": None,
                    "token_confidence": 0.75,
                    **_descriptor_payload(offset=0.0),
                },
                {
                    "k": 2,
                    "event_id": "evt-a",
                    "region_id": "region-1",
                    "source_sequence_key": "hydrophone:rpi_orcasound_lab",
                    "sequence_index": 0,
                    "start_timestamp": 2123.5,
                    "end_timestamp": 2124.25,
                    "token_id": 0,
                    "token_label": "T00",
                    "distance_to_centroid": 0.2,
                    "second_centroid_distance": None,
                    "token_confidence": 0.75,
                    **_descriptor_payload(offset=0.0),
                },
            ]
            pq.write_table(pa.Table.from_pylist(rows), token_path)
        if write_vectors:
            vector_rows = [
                {
                    "event_id": "evt-a",
                    "region_id": "region-1",
                    "source_sequence_key": "hydrophone:rpi_orcasound_lab",
                    "sequence_index": 0,
                    "descriptor_vector": [
                        0.1 * (i + 1) for i in range(len(DESCRIPTOR_ORDER))
                    ],
                    "event_vector": [0.0, 0.0, 0.0],
                },
                {
                    "event_id": "evt-b",
                    "region_id": "region-1",
                    "source_sequence_key": "hydrophone:rpi_orcasound_lab",
                    "sequence_index": 1,
                    "descriptor_vector": [
                        1.1 + 0.1 * i for i in range(len(DESCRIPTOR_ORDER))
                    ],
                    "event_vector": [2.0, 0.0, 0.0],
                },
            ]
            pq.write_table(pa.Table.from_pylist(vector_rows), vector_path)
        await session.commit()
        await session.refresh(job)
        return job.id


def _descriptor_payload(*, offset: float) -> dict[str, float]:
    return {name: offset + float(index) for index, name in enumerate(DESCRIPTOR_ORDER)}


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


async def test_retired_sequence_model_routes_are_unregistered(client):
    retired_paths = [
        "/sequence-models/hmm-sequences",
        "/sequence-models/masked-transformers",
        "/sequence-models/motif-extractions",
    ]

    for path in retired_paths:
        response = await client.get(path)
        assert response.status_code == 404


async def test_create_event_encoder_returns_201_then_200(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    payload = {
        "event_segmentation_job_id": seg_job_id,
        "continuous_embedding_job_id": continuous_id,
        "k_values": [50],
    }

    first = await client.post("/sequence-models/event-encoders", json=payload)
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["status"] == JobStatus.queued.value
    assert body["event_segmentation_job_id"] == seg_job_id
    assert body["continuous_embedding_job_id"] == continuous_id

    second = await client.post("/sequence-models/event-encoders", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == body["id"]


async def test_list_event_encoders_filters_by_status(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )

    listed = await client.get(
        "/sequence-models/event-encoders",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    complete = await client.get(
        "/sequence-models/event-encoders",
        params={"status": JobStatus.complete.value},
    )
    assert complete.status_code == 200
    assert complete.json() == []


async def test_get_event_encoder_detail_without_sidecars(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    detail = await client.get(f"/sequence-models/event-encoders/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["manifest"] is None
    assert body["report"] is None

    missing = await client.get("/sequence-models/event-encoders/missing-id")
    assert missing.status_code == 404


async def test_cancel_event_encoder(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    canceled = await client.post(f"/sequence-models/event-encoders/{job_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == JobStatus.canceled.value

    missing = await client.post("/sequence-models/event-encoders/missing-id/cancel")
    assert missing.status_code == 404


async def test_cancel_terminal_event_encoder_returns_409(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from humpback.models.sequence_models import EventEncoderJob

        job = await session.get(EventEncoderJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    response = await client.post(f"/sequence-models/event-encoders/{job_id}/cancel")
    assert response.status_code == 409


async def test_create_event_encoder_rejects_invalid_source(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    response = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": "missing",
            "k_values": [50],
        },
    )
    assert response.status_code == 422


async def test_delete_event_encoder(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    deleted = await client.delete(f"/sequence-models/event-encoders/{job_id}")
    assert deleted.status_code == 204

    missing = await client.delete("/sequence-models/event-encoders/missing-id")
    assert missing.status_code == 404


async def test_get_event_encoder_timeline_defaults_to_lowest_k(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job_id"] == job_id
    assert body["event_source_mode"] == "effective"
    assert body["selected_k"] == 2
    assert body["valid_k_values"] == [2, 3]
    assert body["descriptor_feature_names"] == DESCRIPTOR_ORDER
    assert body["descriptor_feature_units"]["ridge_log_frequency_slope"] == "octaves/s"
    assert body["job_start_timestamp"] == 2000.0
    assert body["job_end_timestamp"] == 2600.0
    assert [event["event_id"] for event in body["events"]] == ["evt-a", "evt-b"]
    assert body["events"][0]["start_timestamp"] == 2123.5
    assert body["events"][0]["token_label"] == "T00"
    assert body["events"][0]["second_centroid_distance"] is None
    assert body["events"][0]["descriptor_values"]["ridge_log_frequency_slope"] == 6.0
    assert (
        abs(
            body["events"][0]["descriptor_vector_values"]["ridge_log_frequency_slope"]
            - 0.7
        )
        < 1e-9
    )
    assert body["events"][1]["descriptor_values"]["ridge_log_frequency_slope"] == 7.0
    assert (
        abs(
            body["events"][1]["descriptor_vector_values"]["ridge_log_frequency_slope"]
            - 1.7
        )
        < 1e-9
    )


async def test_get_event_encoder_timeline_filters_requested_k(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/timeline",
        params={"k": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["selected_k"] == 3
    assert [event["token_label"] for event in body["events"]] == ["T02"]


async def test_get_event_encoder_timeline_rejects_invalid_k(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/timeline",
        params={"k": 99},
    )
    assert response.status_code == 422


async def test_get_event_encoder_timeline_requires_complete_job(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(
        app_settings, status=JobStatus.queued.value
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")
    assert response.status_code == 409


async def test_get_event_encoder_timeline_missing_job_or_artifact(client, app_settings):
    missing = await client.get("/sequence-models/event-encoders/missing-id/timeline")
    assert missing.status_code == 404

    job_id = await _seed_event_encoder_timeline_job(app_settings, write_tokens=False)
    artifact_missing = await client.get(
        f"/sequence-models/event-encoders/{job_id}/timeline"
    )
    assert artifact_missing.status_code == 404


async def test_get_event_encoder_timeline_tolerates_missing_vector_artifact(
    client,
    app_settings,
):
    job_id = await _seed_event_encoder_timeline_job(
        app_settings,
        write_vectors=False,
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["descriptor_feature_names"] == DESCRIPTOR_ORDER
    assert body["events"][0]["descriptor_values"]["ridge_log_frequency_slope"] == 6.0
    assert body["events"][0]["descriptor_vector_values"] == {}


async def test_get_event_encoder_timeline_rejects_non_region_crnn_provenance(
    client, app_settings
):
    job_id = await _seed_event_encoder_timeline_job(
        app_settings,
        source_model_version="surfperch-tensorflow2",
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")
    assert response.status_code == 409


async def test_get_event_encoder_projection_defaults_to_lowest_k_pca(
    client,
    app_settings,
):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/projection",
        params={"method": "pca"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job_id"] == job_id
    assert body["method"] == "pca"
    assert body["selected_k"] == 2
    assert body["valid_k_values"] == [2, 3]
    assert body["x_axis_label"] == "PC 1"
    assert body["y_axis_label"] == "PC 2"
    assert [point["event_id"] for point in body["points"]] == ["evt-a", "evt-b"]
    assert [point["token_label"] for point in body["points"]] == ["T00", "T01"]
    assert all(math.isfinite(point["x"]) for point in body["points"])
    assert all(math.isfinite(point["y"]) for point in body["points"])


async def test_get_event_encoder_projection_umap_tiny_fallback(
    client,
    app_settings,
):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/projection",
        params={"method": "umap"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"] == "umap"
    assert body["x_axis_label"] == "UMAP 1"
    assert body["y_axis_label"] == "UMAP 2"
    assert len(body["points"]) == 2
    assert all(math.isfinite(point["x"]) for point in body["points"])
    assert all(math.isfinite(point["y"]) for point in body["points"])


async def test_get_event_encoder_projection_rejects_invalid_k(
    client,
    app_settings,
):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/projection",
        params={"k": 99, "method": "pca"},
    )

    assert response.status_code == 422


async def test_get_event_encoder_projection_missing_vector_artifact(
    client,
    app_settings,
):
    job_id = await _seed_event_encoder_timeline_job(
        app_settings,
        write_vectors=False,
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/projection")

    assert response.status_code == 404


# ---------- Piano Roll Notes endpoints ----------


_NOTES_ROW_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string()),
        pa.field("event_token", pa.int32()),
        pa.field("partial_index", pa.int32()),
        pa.field("midi_pitch", pa.uint8()),
        pa.field("start_utc", pa.float64()),
        pa.field("start_offset_s", pa.float64()),
        pa.field("duration_s", pa.float64()),
        pa.field("velocity", pa.uint8()),
        pa.field("peak_magnitude", pa.float32()),
        pa.field("track_id", pa.uint32()),
    ]
)


async def _write_notes_sidecar(
    app_settings,
    encoder_job_id: str,
    rows: list[dict],
    *,
    extractor_version: str = "v2",
):
    path = (
        app_settings.storage_root
        / "event_encoders"
        / encoder_job_id
        / f"event_notes_{extractor_version}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_NOTES_ROW_SCHEMA), path)
    return path


async def _seed_notes_job(
    app_settings,
    encoder_job_id: str,
    *,
    status: str,
    notes_path: str | None = None,
    extractor_version: str = "v2",
    n_notes: int | None = None,
) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    now = datetime.now(timezone.utc)
    async with sf() as session:
        row = PianoRollNotesJob(
            event_encoder_job_id=encoder_job_id,
            extractor_version=extractor_version,
            status=status,
            notes_path=notes_path,
            n_events=2 if status == JobStatus.complete.value else None,
            n_notes=n_notes
            if n_notes is not None
            else (3 if status == JobStatus.complete.value else None),
            compute_seconds=1.5 if status == JobStatus.complete.value else None,
            finished_at=now if status == JobStatus.complete.value else None,
            params_json="{}",
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row.id


async def test_timeline_payload_includes_absent_notes_status(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")
    assert response.status_code == 200, response.text
    assert response.json()["notes_status"] == {"status": "absent"}


async def test_timeline_payload_includes_completed_notes_status(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    notes_path = await _write_notes_sidecar(app_settings, job_id, rows=[])
    await _seed_notes_job(
        app_settings,
        job_id,
        status=JobStatus.complete.value,
        notes_path=str(notes_path),
        n_notes=0,
    )

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/timeline")
    assert response.status_code == 200, response.text
    notes_status = response.json()["notes_status"]
    assert notes_status["status"] == "complete"
    assert notes_status["extractor_version"] == "v2"
    assert notes_status["n_notes"] == 0


async def test_get_notes_status_absent(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes-status"
    )
    assert response.status_code == 200
    assert response.json() == {"status": "absent"}


async def test_get_notes_status_returns_existing_row(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(app_settings, job_id, status=JobStatus.running.value)

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes-status"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "running"
    assert body["event_encoder_job_id"] == job_id
    assert body["extractor_version"] == "v2"


async def test_get_notes_status_missing_encoder_returns_404(client):
    response = await client.get(
        "/sequence-models/event-encoders/missing-id/notes-status"
    )
    assert response.status_code == 404


async def test_create_notes_job_enqueues_and_returns_201(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/notes-jobs",
        json={},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["event_encoder_job_id"] == job_id
    assert body["status"] == "queued"
    assert body["extractor_version"] == "v2"


async def test_create_notes_job_conflicts_with_running_row(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    await _seed_notes_job(app_settings, job_id, status=JobStatus.running.value)

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/notes-jobs",
        json={},
    )
    assert response.status_code == 409


async def test_create_notes_job_requires_complete_encoder(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(
        app_settings, status=JobStatus.queued.value
    )

    response = await client.post(
        f"/sequence-models/event-encoders/{job_id}/notes-jobs",
        json={},
    )
    assert response.status_code == 409


async def test_get_notes_returns_rows_filtered_by_viewport(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    notes_rows = [
        {
            "event_id": "evt-a",
            "event_token": 0,
            "partial_index": 0,
            "midi_pitch": 60,
            "start_utc": 2100.0,
            "start_offset_s": 0.0,
            "duration_s": 0.5,
            "velocity": 80,
            "peak_magnitude": -2.5,
            "track_id": 1,
        },
        {
            "event_id": "evt-b",
            "event_token": 1,
            "partial_index": 1,
            "midi_pitch": 72,
            "start_utc": 2200.0,
            "start_offset_s": 0.0,
            "duration_s": 0.25,
            "velocity": 60,
            "peak_magnitude": -3.0,
            "track_id": 2,
        },
        {
            "event_id": "evt-c",
            "event_token": -1,
            "partial_index": -1,
            "midi_pitch": 84,
            "start_utc": 2400.0,
            "start_offset_s": 0.0,
            "duration_s": 0.5,
            "velocity": 40,
            "peak_magnitude": -3.5,
            "track_id": 3,
        },
    ]
    notes_path = await _write_notes_sidecar(app_settings, job_id, notes_rows)
    await _seed_notes_job(
        app_settings,
        job_id,
        status=JobStatus.complete.value,
        notes_path=str(notes_path),
        n_notes=len(notes_rows),
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes",
        params={"start_utc": 2150.0, "end_utc": 2300.0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["extractor_version"] == "v2"
    assert body["n_notes"] == 1
    assert [row["event_id"] for row in body["notes"]] == ["evt-b"]
    assert body["notes"][0]["midi_pitch"] == 72


async def test_get_notes_filters_by_event_ids(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    notes_rows = [
        {
            "event_id": "evt-a",
            "event_token": 0,
            "partial_index": 0,
            "midi_pitch": 60,
            "start_utc": 2100.0,
            "start_offset_s": 0.0,
            "duration_s": 0.5,
            "velocity": 80,
            "peak_magnitude": -2.5,
            "track_id": 1,
        },
        {
            "event_id": "evt-b",
            "event_token": 1,
            "partial_index": 0,
            "midi_pitch": 72,
            "start_utc": 2200.0,
            "start_offset_s": 0.0,
            "duration_s": 0.25,
            "velocity": 60,
            "peak_magnitude": -3.0,
            "track_id": 2,
        },
    ]
    notes_path = await _write_notes_sidecar(app_settings, job_id, notes_rows)
    await _seed_notes_job(
        app_settings,
        job_id,
        status=JobStatus.complete.value,
        notes_path=str(notes_path),
        n_notes=len(notes_rows),
    )

    response = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes",
        params={"event_ids": ["evt-a"]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_notes"] == 1
    assert body["notes"][0]["event_id"] == "evt-a"


async def test_get_notes_returns_404_without_completed_row(client, app_settings):
    job_id = await _seed_event_encoder_timeline_job(app_settings)

    response = await client.get(f"/sequence-models/event-encoders/{job_id}/notes")
    assert response.status_code == 404


async def test_get_notes_pins_to_explicit_extractor_version(client, app_settings):
    """Older complete row should be reachable by pinning to its version,
    even when a newer version is the default ``latest_for_encoder_job`` pick."""
    job_id = await _seed_event_encoder_timeline_job(app_settings)
    base_rows = [
        {
            "event_id": "evt-a",
            "event_token": 0,
            "partial_index": 0,
            "midi_pitch": 60,
            "start_utc": 2100.0,
            "start_offset_s": 0.0,
            "duration_s": 0.5,
            "velocity": 80,
            "peak_magnitude": -2.5,
            "track_id": 1,
        }
    ]
    v1_path = await _write_notes_sidecar(
        app_settings, job_id, base_rows, extractor_version="v1"
    )
    await _seed_notes_job(
        app_settings,
        job_id,
        status=JobStatus.complete.value,
        notes_path=str(v1_path),
        n_notes=1,
        extractor_version="v1",
    )

    v2_path = v1_path.with_name("event_notes_v2-experimental.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [{**base_rows[0], "event_id": "evt-b", "midi_pitch": 72}],
            schema=_NOTES_ROW_SCHEMA,
        ),
        v2_path,
    )
    await _seed_notes_job(
        app_settings,
        job_id,
        status=JobStatus.complete.value,
        notes_path=str(v2_path),
        n_notes=1,
        extractor_version="v2-experimental",
    )

    response_v1 = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes",
        params={"extractor_version": "v1"},
    )
    assert response_v1.status_code == 200, response_v1.text
    body_v1 = response_v1.json()
    assert body_v1["extractor_version"] == "v1"
    assert body_v1["notes"][0]["event_id"] == "evt-a"

    response_v2 = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes",
        params={"extractor_version": "v2-experimental"},
    )
    assert response_v2.status_code == 200, response_v2.text
    body_v2 = response_v2.json()
    assert body_v2["extractor_version"] == "v2-experimental"
    assert body_v2["notes"][0]["event_id"] == "evt-b"

    response_missing = await client.get(
        f"/sequence-models/event-encoders/{job_id}/notes",
        params={"extractor_version": "v99"},
    )
    assert response_missing.status_code == 404
