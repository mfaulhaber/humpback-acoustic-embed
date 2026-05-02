import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_engine, create_session_factory
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MotifExtractionJob,
)
from humpback.storage import (
    motif_extraction_dir,
    motif_extraction_manifest_path,
    motif_extraction_motifs_path,
    motif_extraction_occurrences_path,
)


async def _seed_hmm(app_settings, *, status: str = JobStatus.complete.value) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id="seg-1",
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature=f"enc-{status}",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        hmm = HMMSequenceJob(
            status=status,
            continuous_embedding_job_id=cej.id,
            n_states=4,
            pca_dims=8,
        )
        session.add(hmm)
        await session.commit()
        await session.refresh(hmm)
        return hmm.id


def _write_artifacts(app_settings, job_id: str) -> None:
    out_dir = motif_extraction_dir(app_settings.storage_root, job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    motif_extraction_manifest_path(app_settings.storage_root, job_id).write_text(
        """
        {
          "schema_version": 1,
          "motif_extraction_job_id": "%s",
          "hmm_sequence_job_id": "hmm-1",
          "continuous_embedding_job_id": "ce-1",
          "source_kind": "surfperch",
          "config": {},
          "config_signature": "sig",
          "generated_at": "2026-04-30T00:00:00+00:00",
          "total_groups": 2,
          "total_collapsed_tokens": 6,
          "total_candidate_occurrences": 4,
          "total_motifs": 1,
          "event_source_key_strategy": "event_id"
        }
        """
        % job_id,
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "motif_key": "1-2",
                    "states": [1, 2],
                    "length": 2,
                    "occurrence_count": 5,
                    "event_source_count": 2,
                    "audio_source_count": 2,
                    "group_count": 2,
                    "event_core_fraction": 0.8,
                    "background_fraction": 0.1,
                    "mean_call_probability": None,
                    "mean_duration_seconds": 2.0,
                    "median_duration_seconds": 2.0,
                    "rank_score": 0.95,
                    "example_occurrence_ids": ["occ-1"],
                }
            ]
        ),
        motif_extraction_motifs_path(app_settings.storage_root, job_id),
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "occurrence_id": "occ-1",
                    "motif_key": "1-2",
                    "states": [1, 2],
                    "source_kind": "surfperch",
                    "group_key": "0",
                    "event_source_key": "e1",
                    "audio_source_key": "10",
                    "token_start_index": 0,
                    "token_end_index": 1,
                    "raw_start_index": 0,
                    "raw_end_index": 2,
                    "start_timestamp": 1.0,
                    "end_timestamp": 3.0,
                    "duration_seconds": 2.0,
                    "event_core_fraction": 1.0,
                    "background_fraction": 0.0,
                    "mean_call_probability": None,
                    "anchor_event_id": "e1",
                    "anchor_timestamp": 2.0,
                    "relative_start_seconds": -1.0,
                    "relative_end_seconds": 1.0,
                    "anchor_strategy": "event_midpoint",
                }
            ]
        ),
        motif_extraction_occurrences_path(app_settings.storage_root, job_id),
    )


async def test_create_list_detail_cancel_delete(client, app_settings):
    hmm_id = await _seed_hmm(app_settings)

    created = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id},
    )
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]

    duplicate = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id},
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["id"] == job_id

    listed = await client.get(
        "/sequence-models/motif-extractions",
        params={"status": "queued", "hmm_sequence_job_id": hmm_id},
    )
    assert listed.status_code == 200
    assert [j["id"] for j in listed.json()] == [job_id]

    detail = await client.get(f"/sequence-models/motif-extractions/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["job"]["id"] == job_id

    canceled = await client.post(f"/sequence-models/motif-extractions/{job_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == "canceled"

    deleted = await client.delete(f"/sequence-models/motif-extractions/{job_id}")
    assert deleted.status_code == 204


async def test_create_rejects_non_complete_hmm(client, app_settings):
    hmm_id = await _seed_hmm(app_settings, status=JobStatus.running.value)
    response = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id},
    )
    assert response.status_code == 422


async def test_completed_artifact_endpoints(client, app_settings):
    hmm_id = await _seed_hmm(app_settings)
    created = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id},
    )
    job_id = created.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(MotifExtractionJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        job.total_motifs = 1
        await session.commit()

    _write_artifacts(app_settings, job_id)

    detail = await client.get(f"/sequence-models/motif-extractions/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["manifest"]["motif_extraction_job_id"] == job_id

    motifs = await client.get(f"/sequence-models/motif-extractions/{job_id}/motifs")
    assert motifs.status_code == 200
    assert motifs.json()["items"][0]["motif_key"] == "1-2"

    occurrences = await client.get(
        f"/sequence-models/motif-extractions/{job_id}/motifs/1-2/occurrences"
    )
    assert occurrences.status_code == 200
    assert occurrences.json()["items"][0]["occurrence_id"] == "occ-1"


async def test_terminal_cancel_returns_409(client, app_settings):
    hmm_id = await _seed_hmm(app_settings)
    created = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id},
    )
    job_id = created.json()["id"]
    await client.post(f"/sequence-models/motif-extractions/{job_id}/cancel")
    second = await client.post(f"/sequence-models/motif-extractions/{job_id}/cancel")
    assert second.status_code == 409


async def _seed_masked_transformer(app_settings) -> str:
    from humpback.models.sequence_models import MaskedTransformerJob
    from humpback.services.masked_transformer_service import serialize_k_values

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id="seg-mt",
            model_version="crnn-call-parsing-pytorch",
            target_sample_rate=32000,
            encoding_signature="enc-mt-api",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        mt = MaskedTransformerJob(
            status=JobStatus.complete.value,
            continuous_embedding_job_id=cej.id,
            training_signature="sig-mt-api",
            k_values=serialize_k_values([100]),
        )
        session.add(mt)
        await session.commit()
        await session.refresh(mt)
        return mt.id


async def test_motif_create_with_masked_transformer_parent(client, app_settings):
    mt_id = await _seed_masked_transformer(app_settings)

    created = await client.post(
        "/sequence-models/motif-extractions",
        json={
            "parent_kind": "masked_transformer",
            "masked_transformer_job_id": mt_id,
            "k": 100,
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["parent_kind"] == "masked_transformer"
    assert body["masked_transformer_job_id"] == mt_id
    assert body["k"] == 100
    assert body["hmm_sequence_job_id"] is None
    assert body["source_kind"] == "region_crnn"


async def test_motif_create_xor_violation_returns_422(client, app_settings):
    hmm_id = await _seed_hmm(app_settings)
    mt_id = await _seed_masked_transformer(app_settings)

    response = await client.post(
        "/sequence-models/motif-extractions",
        json={
            "parent_kind": "masked_transformer",
            "hmm_sequence_job_id": hmm_id,
            "masked_transformer_job_id": mt_id,
            "k": 100,
        },
    )
    assert response.status_code == 422


async def test_motif_create_masked_transformer_requires_k(client, app_settings):
    mt_id = await _seed_masked_transformer(app_settings)
    response = await client.post(
        "/sequence-models/motif-extractions",
        json={
            "parent_kind": "masked_transformer",
            "masked_transformer_job_id": mt_id,
        },
    )
    assert response.status_code == 422


async def test_motif_create_hmm_with_k_returns_422(client, app_settings):
    hmm_id = await _seed_hmm(app_settings)
    response = await client.post(
        "/sequence-models/motif-extractions",
        json={"hmm_sequence_job_id": hmm_id, "k": 100},
    )
    assert response.status_code == 422
