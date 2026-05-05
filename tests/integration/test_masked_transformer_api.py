"""Integration tests for the masked-transformer API surface (ADR-061)."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_engine, create_session_factory
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    MaskedTransformerJob,
)
from humpback.services.masked_transformer_service import serialize_k_values
from humpback.storage import (
    masked_transformer_dir,
    masked_transformer_k_decoded_path,
    masked_transformer_k_dir,
    masked_transformer_k_run_lengths_path,
    masked_transformer_loss_curve_path,
    masked_transformer_reconstruction_error_path,
)


async def _seed_crnn_cej(
    app_settings, *, status: str = JobStatus.complete.value
) -> str:
    from humpback.models.call_parsing import (
        EventClassificationJob,
        EventSegmentationJob,
        RegionDetectionJob,
    )

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
        # Bind a completed Classify job so MT submit can resolve a default
        # event_classification_job_id.
        session.add(
            EventClassificationJob(
                status=JobStatus.complete.value,
                event_segmentation_job_id=seg_job.id,
            )
        )
        cej = ContinuousEmbeddingJob(
            status=status,
            event_segmentation_job_id=seg_job.id,
            region_detection_job_id=region_job.id,
            model_version="crnn-call-parsing-pytorch",
            target_sample_rate=32000,
            encoding_signature=f"enc-mt-{status}",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


async def _seed_surfperch_cej(app_settings) -> str:
    from humpback.models.call_parsing import (
        EventClassificationJob,
        EventSegmentationJob,
        RegionDetectionJob,
    )

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
        session.add(
            EventClassificationJob(
                status=JobStatus.complete.value,
                event_segmentation_job_id=seg_job.id,
            )
        )
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature="enc-mt-surf",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


async def _mark_completed(app_settings, job_id: str, k_values: list[int]) -> None:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        mt = await session.get(MaskedTransformerJob, job_id)
        assert mt is not None
        mt.status = JobStatus.complete.value
        mt.k_values = serialize_k_values(k_values)
        await session.commit()


async def test_create_happy_path(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_id,
            "preset": "small",
            "k_values": [50, 100],
            "sequence_construction_mode": "mixed",
            "event_centered_fraction": 0.4,
            "pre_event_context_sec": 1.5,
            "post_event_context_sec": 2.5,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["preset"] == "small"
    assert body["k_values"] == [50, 100]
    assert body["status"] == "queued"
    assert body["sequence_construction_mode"] == "mixed"
    assert body["event_centered_fraction"] == 0.4
    assert body["pre_event_context_sec"] == 1.5
    assert body["post_event_context_sec"] == 2.5
    assert body["contrastive_loss_weight"] == 0.0
    assert body["contrastive_label_source"] == "none"


async def test_create_contrastive_round_trips(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_id,
            "preset": "small",
            "retrieval_head_enabled": True,
            "contrastive_loss_weight": 0.1,
            "contrastive_temperature": 0.07,
            "contrastive_label_source": "human_corrections",
            "contrastive_min_events_per_label": 2,
            "contrastive_min_regions_per_label": 1,
            "require_cross_region_positive": False,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["retrieval_head_enabled"] is True
    assert body["contrastive_loss_weight"] == 0.1
    assert body["contrastive_temperature"] == 0.07
    assert body["contrastive_label_source"] == "human_corrections"
    assert body["contrastive_min_events_per_label"] == 2
    assert body["contrastive_min_regions_per_label"] == 1
    assert body["require_cross_region_positive"] is False
    assert body["related_label_policy_json"] is not None


async def test_create_idempotent_on_signature(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    payload = {"continuous_embedding_job_id": cej_id, "preset": "default"}

    first = await client.post("/sequence-models/masked-transformers", json=payload)
    second = await client.post("/sequence-models/masked-transformers", json=payload)

    assert first.status_code == 201
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]


async def test_create_rejects_non_crnn_upstream(client, app_settings):
    cej_id = await _seed_surfperch_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    assert response.status_code == 422


async def test_create_rejects_empty_k_values(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": []},
    )
    assert response.status_code == 422


async def test_create_rejects_invalid_preset(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "preset": "huge"},
    )
    assert response.status_code == 422


async def test_create_rejects_invalid_sequence_construction(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_id,
            "sequence_construction_mode": "mixed",
            "event_centered_fraction": 1.0,
        },
    )
    assert response.status_code == 422


async def test_create_rejects_contrastive_without_retrieval_head(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_id,
            "contrastive_loss_weight": 0.1,
            "contrastive_label_source": "human_corrections",
        },
    )
    assert response.status_code == 422


async def test_create_rejects_k_below_2(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    response = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [1, 5]},
    )
    assert response.status_code == 422


async def test_list_and_detail(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    listed = await client.get("/sequence-models/masked-transformers")
    assert listed.status_code == 200
    assert any(j["id"] == job_id for j in listed.json())

    detail = await client.get(f"/sequence-models/masked-transformers/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["job"]["id"] == job_id
    assert detail.json()["job"]["sequence_construction_mode"] == "region"
    assert detail.json()["job"]["event_centered_fraction"] == 0.0
    assert detail.json()["job"]["pre_event_context_sec"] is None
    assert detail.json()["job"]["post_event_context_sec"] is None
    assert detail.json()["source_kind"] == "region_crnn"


async def test_cancel_and_terminal_cancel_returns_409(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    canceled = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/cancel"
    )
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"

    second = await client.post(f"/sequence-models/masked-transformers/{job_id}/cancel")
    assert second.status_code == 409


async def test_extend_k_sweep_only_on_completed(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/extend-k-sweep",
        json={"additional_k": [50]},
    )
    assert response.status_code == 409


async def test_extend_k_sweep_dedupes(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]
    await _mark_completed(app_settings, job_id, [100])

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/extend-k-sweep",
        json={"additional_k": [50, 100, 50]},
    )
    assert response.status_code == 200
    assert response.json()["k_values"] == [100, 50]


async def test_delete(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    masked_transformer_dir(app_settings.storage_root, job_id).mkdir(parents=True)

    response = await client.delete(f"/sequence-models/masked-transformers/{job_id}")
    assert response.status_code == 204
    assert not masked_transformer_dir(app_settings.storage_root, job_id).exists()


async def test_loss_curve_404_when_missing(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/loss-curve"
    )
    assert response.status_code == 404


async def test_loss_curve_returns_payload(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    masked_transformer_dir(app_settings.storage_root, job_id).mkdir(parents=True)
    masked_transformer_loss_curve_path(app_settings.storage_root, job_id).write_text(
        json.dumps(
            {
                "epochs": [1, 2, 3],
                "train_loss": [0.5, 0.3, 0.2],
                "val_loss": [0.6, 0.4, 0.3],
                "val_metrics": {"final_val_loss": 0.3},
            }
        ),
        encoding="utf-8",
    )

    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/loss-curve"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["epochs"] == [1, 2, 3]
    assert body["val_metrics"]["final_val_loss"] == 0.3


async def test_tokens_default_k_and_unknown_k_404(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]

    # Unknown k → 404 even before files exist.
    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/tokens",
        params={"k": 50},
    )
    assert response.status_code == 404

    # Default k=100, but no decoded.parquet → 404.
    response = await client.get(f"/sequence-models/masked-transformers/{job_id}/tokens")
    assert response.status_code == 404


async def test_tokens_returns_rows_when_file_present(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]

    k_dir = masked_transformer_k_dir(app_settings.storage_root, job_id, 100)
    k_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sequence_id": "r1",
                    "position": 0,
                    "label": 5,
                    "confidence": 0.7,
                    "audio_file_id": 1,
                    "start_timestamp": 0.0,
                    "end_timestamp": 0.25,
                    "tier": "event_core",
                    "chunk_index_in_region": 0,
                    "region_id": "r1",
                },
                {
                    "sequence_id": "r1",
                    "position": 1,
                    "label": 7,
                    "confidence": 0.5,
                    "audio_file_id": 1,
                    "start_timestamp": 0.25,
                    "end_timestamp": 0.5,
                    "tier": "background",
                    "chunk_index_in_region": 1,
                    "region_id": "r1",
                },
            ]
        ),
        masked_transformer_k_decoded_path(app_settings.storage_root, job_id, 100),
    )

    response = await client.get(f"/sequence-models/masked-transformers/{job_id}/tokens")
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 2
    assert items[0]["label"] == 5
    assert items[0]["tier"] == "event_core"


async def test_run_lengths_returns_payload(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]

    k_dir = masked_transformer_k_dir(app_settings.storage_root, job_id, 100)
    k_dir.mkdir(parents=True)
    masked_transformer_k_run_lengths_path(
        app_settings.storage_root, job_id, 100
    ).write_text(
        json.dumps({"k": 100, "tau": 1.5, "run_lengths": {"5": [3, 4, 1]}}),
        encoding="utf-8",
    )

    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/run-lengths"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["k"] == 100
    assert body["run_lengths"]["5"] == [3, 4, 1]


async def test_reconstruction_error_returns_paginated(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    masked_transformer_dir(app_settings.storage_root, job_id).mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sequence_id": "r1",
                    "position": 0,
                    "score": 0.05,
                    "start_timestamp": 0.0,
                    "end_timestamp": 0.25,
                },
                {
                    "sequence_id": "r1",
                    "position": 1,
                    "score": 0.10,
                    "start_timestamp": 0.25,
                    "end_timestamp": 0.5,
                },
            ]
        ),
        masked_transformer_reconstruction_error_path(app_settings.storage_root, job_id),
    )

    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/reconstruction-error"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["items"][0]["score"] == 0.05


async def test_overlay_404_when_missing(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "k_values": [100]},
    )
    job_id = created.json()["id"]
    response = await client.get(
        f"/sequence-models/masked-transformers/{job_id}/overlay"
    )
    assert response.status_code == 404


async def test_generate_interpretations_requires_complete(client, app_settings):
    cej_id = await _seed_crnn_cej(app_settings)
    created = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id},
    )
    job_id = created.json()["id"]

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/generate-interpretations",
        json={"k_values": None},
    )
    assert response.status_code == 400


async def test_get_404_for_unknown_job(client, app_settings):
    response = await client.get("/sequence-models/masked-transformers/missing")
    assert response.status_code == 404
