"""Integration tests for masked-transformer retrieval diagnostics API."""

from __future__ import annotations

import math

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event
from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
    VocalizationCorrection,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, MaskedTransformerJob
from humpback.services.masked_transformer_service import serialize_k_values
from humpback.storage import (
    continuous_embedding_parquet_path,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_dir,
    masked_transformer_k_decoded_path,
    masked_transformer_k_dir,
    masked_transformer_retrieval_embeddings_path,
)


def _event(event_id: str, region_id: str, start: float, end: float) -> Event:
    return Event(
        event_id=event_id,
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        center_sec=(start + end) / 2.0,
        segmentation_confidence=0.9,
    )


def _metric_payload() -> dict:
    return {
        "same_human_label": 0.0,
        "exact_human_label_set": 0.0,
        "same_event": 0.0,
        "same_region": 0.0,
        "adjacent_1s": 0.0,
        "nearby_5s": 0.0,
        "same_token": 0.0,
        "similar_duration": 0.0,
        "without_human_label": 0.0,
        "low_event_overlap": 0.0,
        "avg_cosine": 0.0,
        "median_cosine": 0.0,
        "random_pair_percentiles": {"50": 0.0},
        "verdicts": {},
        "label_specific_same_human_label": {},
    }


def _fake_report(job_id: str = "fake-job") -> dict:
    return {
        "job": {
            "job_id": job_id,
            "status": JobStatus.complete.value,
            "continuous_embedding_job_id": "cej",
            "event_classification_job_id": None,
            "region_detection_job_id": "rdj",
            "k_values": [100],
            "k": 100,
            "total_sequences": None,
            "total_chunks": None,
            "final_train_loss": None,
            "final_val_loss": None,
            "total_epochs": None,
        },
        "options": {
            "embedding_space": "contextual",
            "samples": 50,
            "topn": 10,
            "seed": 20260504,
            "retrieval_modes": ["unrestricted"],
            "embedding_variants": ["raw_l2"],
            "include_query_rows": False,
            "include_neighbor_rows": False,
            "include_event_level": False,
        },
        "artifacts": {},
        "label_coverage": {
            "embedding_rows": 0,
            "sampled_queries": 0,
            "human_labeled_query_pool_rows": 0,
            "human_labeled_effective_events": 0,
            "vocalization_correction_rows": 0,
            "human_label_chunk_counts": {},
            "human_label_event_counts": {},
            "corrections_by_type": {},
        },
        "results": {"unrestricted": {"raw_l2": _metric_payload()}},
        "event_level_results": None,
        "representative_good_queries": [],
        "representative_risky_queries": [],
        "query_rows": [],
        "neighbor_rows": [],
    }


async def _seed_diagnostics_job(
    app_settings,
    *,
    status: str = JobStatus.complete.value,
    k_values: list[int] | None = None,
) -> str:
    k_values = k_values or [100]
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        rdj = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1_000.0,
            end_timestamp=1_100.0,
        )
        session.add(rdj)
        await session.flush()

        seg = EventSegmentationJob(
            status=JobStatus.complete.value,
            region_detection_job_id=rdj.id,
        )
        session.add(seg)
        await session.flush()

        cls = EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg.id,
        )
        session.add(cls)
        await session.flush()

        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg.id,
            region_detection_job_id=rdj.id,
            model_version="crnn-call-parsing-pytorch",
            target_sample_rate=32000,
            encoding_signature=f"retrieval-diag-{seg.id}",
            total_chunks=6,
            total_regions=3,
            vector_dim=3,
        )
        session.add(cej)
        await session.flush()

        mt = MaskedTransformerJob(
            status=status,
            continuous_embedding_job_id=cej.id,
            event_classification_job_id=cls.id,
            training_signature=f"retrieval-diag-mt-{seg.id}",
            k_values=serialize_k_values(k_values),
            total_sequences=3,
            total_chunks=6,
        )
        session.add(mt)

        session.add_all(
            [
                VocalizationCorrection(
                    region_detection_job_id=rdj.id,
                    start_sec=0.0,
                    end_sec=0.5,
                    type_name="Moan",
                    correction_type="add",
                ),
                VocalizationCorrection(
                    region_detection_job_id=rdj.id,
                    start_sec=2.0,
                    end_sec=2.5,
                    type_name="Moan",
                    correction_type="add",
                ),
                VocalizationCorrection(
                    region_detection_job_id=rdj.id,
                    start_sec=4.0,
                    end_sec=4.5,
                    type_name="Growl",
                    correction_type="add",
                ),
            ]
        )
        await session.commit()
        await session.refresh(mt)
        await session.refresh(cej)
        await session.refresh(seg)

        segmentation_job_dir(app_settings.storage_root, seg.id).mkdir(
            parents=True, exist_ok=True
        )
        write_events(
            segmentation_job_dir(app_settings.storage_root, seg.id) / "events.parquet",
            [
                _event("E1", "R1", 0.0, 0.5),
                _event("E2", "R2", 2.0, 2.5),
                _event("E3", "R3", 4.0, 4.5),
            ],
        )

        ce_rows = [
            {
                "region_id": "R1",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 0,
                "audio_file_id": 1,
                "start_timestamp": 1_000.0,
                "end_timestamp": 1_000.25,
                "is_in_pad": False,
                "call_probability": 0.95,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E1",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [1.0, 0.0, 0.0],
            },
            {
                "region_id": "R1",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 1,
                "audio_file_id": 1,
                "start_timestamp": 1_000.25,
                "end_timestamp": 1_000.5,
                "is_in_pad": False,
                "call_probability": 0.95,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E1",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [0.99, 0.01, 0.0],
            },
            {
                "region_id": "R2",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 0,
                "audio_file_id": 2,
                "start_timestamp": 1_002.0,
                "end_timestamp": 1_002.25,
                "is_in_pad": False,
                "call_probability": 0.93,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E2",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [0.98, 0.02, 0.0],
            },
            {
                "region_id": "R2",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 1,
                "audio_file_id": 2,
                "start_timestamp": 1_002.25,
                "end_timestamp": 1_002.5,
                "is_in_pad": False,
                "call_probability": 0.93,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E2",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [0.97, 0.03, 0.0],
            },
            {
                "region_id": "R3",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 0,
                "audio_file_id": 3,
                "start_timestamp": 1_004.0,
                "end_timestamp": 1_004.25,
                "is_in_pad": False,
                "call_probability": 0.9,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E3",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [0.0, 1.0, 0.0],
            },
            {
                "region_id": "R3",
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": 1,
                "audio_file_id": 3,
                "start_timestamp": 1_004.25,
                "end_timestamp": 1_004.5,
                "is_in_pad": False,
                "call_probability": 0.9,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": "E3",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [0.0, 0.99, 0.01],
            },
        ]

        continuous_embedding_parquet_path(
            app_settings.storage_root, cej.id
        ).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(ce_rows),
            continuous_embedding_parquet_path(app_settings.storage_root, cej.id),
        )

        mt_dir = masked_transformer_dir(app_settings.storage_root, mt.id)
        mt_dir.mkdir(parents=True, exist_ok=True)
        embedding_rows = [
            {
                "region_id": row["region_id"],
                "chunk_index_in_region": row["chunk_index_in_region"],
                "audio_file_id": row["audio_file_id"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "tier": row["tier"],
                "embedding": row["embedding"],
            }
            for row in ce_rows
        ]
        pq.write_table(
            pa.Table.from_pylist(embedding_rows),
            masked_transformer_contextual_embeddings_path(
                app_settings.storage_root, mt.id
            ),
        )

        k_dir = masked_transformer_k_dir(app_settings.storage_root, mt.id, k_values[0])
        k_dir.mkdir(parents=True, exist_ok=True)
        decoded_rows = [
            {
                "sequence_id": row["region_id"],
                "position": row["chunk_index_in_region"],
                "label": idx % 3,
                "confidence": 0.75,
                "audio_file_id": row["audio_file_id"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "tier": row["tier"],
                "chunk_index_in_region": row["chunk_index_in_region"],
                "region_id": row["region_id"],
            }
            for idx, row in enumerate(ce_rows)
        ]
        pq.write_table(
            pa.Table.from_pylist(decoded_rows),
            masked_transformer_k_decoded_path(
                app_settings.storage_root, mt.id, k_values[0]
            ),
        )

        return mt.id


async def test_nearest_neighbor_report_returns_aggregate_metrics(client, app_settings):
    job_id = await _seed_diagnostics_job(app_settings)

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={
            "k": 100,
            "samples": 1,
            "topn": 1,
            "retrieval_modes": ["exclude_same_event_and_region"],
            "embedding_variants": ["raw_l2"],
            "include_query_rows": True,
            "include_neighbor_rows": True,
            "include_event_level": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["job_id"] == job_id
    assert body["options"]["embedding_space"] == "contextual"
    metrics = body["results"]["exclude_same_event_and_region"]["raw_l2"]
    assert math.isfinite(metrics["same_human_label"])
    assert body["label_coverage"]["human_labeled_effective_events"] == 3
    assert body["query_rows"]
    assert body["neighbor_rows"]
    assert body["event_level_results"] is not None


async def test_nearest_neighbor_report_unknown_k_returns_404(client, app_settings):
    job_id = await _seed_diagnostics_job(app_settings, k_values=[100])

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={"k": 50},
    )

    assert response.status_code == 404


async def test_nearest_neighbor_report_incomplete_job_returns_409(client, app_settings):
    job_id = await _seed_diagnostics_job(app_settings, status=JobStatus.queued.value)

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={"k": 100},
    )

    assert response.status_code == 409


async def test_retrieval_space_without_artifact_returns_409(client, app_settings):
    job_id = await _seed_diagnostics_job(app_settings)

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={"embedding_space": "retrieval", "k": 100},
    )

    assert response.status_code == 409
    assert "retrieval embeddings artifact not found" in response.text


async def test_contextual_report_reads_but_does_not_write_decoded_artifact(
    client, app_settings
):
    job_id = await _seed_diagnostics_job(app_settings)
    decoded_path = masked_transformer_k_decoded_path(
        app_settings.storage_root, job_id, 100
    )
    before = decoded_path.stat().st_mtime_ns

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={"k": 100, "samples": 1, "topn": 1},
    )

    assert response.status_code == 200, response.text
    assert decoded_path.stat().st_mtime_ns == before


async def test_nearest_neighbor_report_route_invokes_source_module(client, monkeypatch):
    from humpback.api.routers import sequence_models as router_mod

    called = {}

    async def _fake_build_report(*_args, **kwargs):
        called["job_id"] = kwargs["job_id"]
        return _fake_report(kwargs["job_id"])

    monkeypatch.setattr(
        router_mod.retrieval_diagnostics,
        "build_nearest_neighbor_report",
        _fake_build_report,
    )

    response = await client.post(
        "/sequence-models/masked-transformers/anything/nearest-neighbor-report",
        json={"retrieval_modes": ["unrestricted"], "embedding_variants": ["raw_l2"]},
    )

    assert response.status_code == 200, response.text
    assert called["job_id"] == "anything"


async def test_retrieval_space_uses_retrieval_artifact_when_present(
    client, app_settings
):
    job_id = await _seed_diagnostics_job(app_settings)
    contextual_path = masked_transformer_contextual_embeddings_path(
        app_settings.storage_root, job_id
    )
    rows = pq.read_table(contextual_path).to_pylist()
    for row in rows:
        row["embedding"] = [float(v) * 0.5 for v in row["embedding"]]
    pq.write_table(
        pa.Table.from_pylist(rows),
        masked_transformer_retrieval_embeddings_path(app_settings.storage_root, job_id),
    )

    response = await client.post(
        f"/sequence-models/masked-transformers/{job_id}/nearest-neighbor-report",
        json={
            "embedding_space": "retrieval",
            "k": 100,
            "samples": 1,
            "topn": 1,
            "retrieval_modes": ["unrestricted"],
            "embedding_variants": ["raw_l2"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["options"]["embedding_space"] == "retrieval"
