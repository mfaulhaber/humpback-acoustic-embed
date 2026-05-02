from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.call_parsing.storage import segmentation_job_dir
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MaskedTransformerJob,
    MotifExtractionJob,
)
from humpback.services.masked_transformer_service import serialize_k_values
from humpback.storage import (
    continuous_embedding_dir,
    continuous_embedding_parquet_path,
    hmm_sequence_dir,
    hmm_sequence_states_path,
    masked_transformer_k_decoded_path,
    masked_transformer_k_dir,
    motif_extraction_dir,
    motif_extraction_manifest_path,
    motif_extraction_motifs_path,
    motif_extraction_occurrences_path,
)
from humpback.workers.motif_extraction_worker import run_motif_extraction_job


def _surf_states() -> pa.Table:
    states = [1, 1, 2, 3, 1, 2, 3]
    rows = []
    for i, state in enumerate(states):
        rows.append(
            {
                "merged_span_id": 0 if i < 3 else 1,
                "window_index_in_span": i if i < 3 else i - 3,
                "audio_file_id": 10 if i < 3 else 11,
                "start_timestamp": float(i),
                "end_timestamp": float(i + 1),
                "is_in_pad": False,
                "event_id": "e1" if i < 3 else "e2",
                "viterbi_state": state,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            }
        )
    return pa.Table.from_pylist(rows)


async def _jobs(session, settings, *, model_version: str = "surfperch-tensorflow2"):
    cej = ContinuousEmbeddingJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id="seg-1",
        model_version=model_version,
        window_size_seconds=5.0 if model_version == "surfperch-tensorflow2" else None,
        hop_seconds=1.0 if model_version == "surfperch-tensorflow2" else None,
        pad_seconds=2.0 if model_version == "surfperch-tensorflow2" else None,
        target_sample_rate=32000,
        encoding_signature=f"enc-{model_version}",
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)

    hmm = HMMSequenceJob(
        status=JobStatus.complete.value,
        continuous_embedding_job_id=cej.id,
        n_states=4,
        pca_dims=8,
    )
    session.add(hmm)
    await session.commit()
    await session.refresh(hmm)

    motif = MotifExtractionJob(
        status=JobStatus.running.value,
        hmm_sequence_job_id=hmm.id,
        source_kind="surfperch",
        min_ngram=2,
        max_ngram=2,
        minimum_occurrences=1,
        minimum_event_sources=1,
        config_signature=f"sig-{model_version}",
    )
    session.add(motif)
    await session.commit()
    await session.refresh(motif)

    hmm_sequence_dir(settings.storage_root, hmm.id).mkdir(parents=True)
    pq.write_table(
        _surf_states(), hmm_sequence_states_path(settings.storage_root, hmm.id)
    )

    seg_dir = segmentation_job_dir(settings.storage_root, "seg-1")
    seg_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_id": "e1",
                    "region_id": "r1",
                    "start_sec": 0.0,
                    "end_sec": 3.0,
                    "peak_score": 0.9,
                    "segmentation_confidence": 0.9,
                },
                {
                    "event_id": "e2",
                    "region_id": "r2",
                    "start_sec": 4.0,
                    "end_sec": 7.0,
                    "peak_score": 0.9,
                    "segmentation_confidence": 0.9,
                },
            ]
        ),
        seg_dir / "events.parquet",
    )
    return cej, hmm, motif


async def test_worker_writes_surfperch_artifacts(session, settings):
    _, _, motif = await _jobs(session, settings)

    await run_motif_extraction_job(session, motif, settings)
    await session.refresh(motif)

    assert motif.status == JobStatus.complete.value
    assert motif.total_motifs and motif.total_motifs > 0
    assert motif_extraction_manifest_path(settings.storage_root, motif.id).exists()
    assert motif_extraction_motifs_path(settings.storage_root, motif.id).exists()
    assert motif_extraction_occurrences_path(settings.storage_root, motif.id).exists()


async def test_worker_writes_crnn_artifacts(session, settings):
    cej, hmm, motif = await _jobs(
        session, settings, model_version="crnn-call-parsing-pytorch"
    )
    motif.source_kind = "region_crnn"
    await session.commit()

    states = pa.Table.from_pylist(
        [
            {
                "region_id": "r1",
                "chunk_index_in_region": 0,
                "audio_file_id": 1,
                "start_timestamp": 0.0,
                "end_timestamp": 0.25,
                "is_in_pad": False,
                "tier": "event_core",
                "viterbi_state": 1,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
            {
                "region_id": "r1",
                "chunk_index_in_region": 1,
                "audio_file_id": 1,
                "start_timestamp": 0.25,
                "end_timestamp": 0.5,
                "is_in_pad": False,
                "tier": "background",
                "viterbi_state": 2,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
        ]
    )
    pq.write_table(states, hmm_sequence_states_path(settings.storage_root, hmm.id))
    continuous_embedding_dir(settings.storage_root, cej.id).mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "region_id": "r1",
                    "chunk_index_in_region": 0,
                    "nearest_event_id": "e1",
                    "call_probability": 0.8,
                },
                {
                    "region_id": "r1",
                    "chunk_index_in_region": 1,
                    "nearest_event_id": "e1",
                    "call_probability": 0.2,
                },
            ]
        ),
        continuous_embedding_parquet_path(settings.storage_root, cej.id),
    )

    await run_motif_extraction_job(session, motif, settings)
    await session.refresh(motif)

    assert motif.status == JobStatus.complete.value
    assert motif.source_kind == "region_crnn"
    assert motif_extraction_motifs_path(settings.storage_root, motif.id).exists()


async def test_worker_cancellation_removes_partial_artifacts(session, settings):
    _, _, motif = await _jobs(session, settings)
    motif.status = JobStatus.canceled.value
    job_dir = motif_extraction_dir(settings.storage_root, motif.id)
    job_dir.mkdir(parents=True)
    (job_dir / "old.tmp").write_text("partial", encoding="utf-8")
    await session.commit()

    await run_motif_extraction_job(session, motif, settings)
    await session.refresh(motif)

    assert motif.status == JobStatus.canceled.value
    assert not job_dir.exists()


async def test_worker_failure_marks_job_failed(session, settings):
    _, hmm, motif = await _jobs(session, settings)
    Path(hmm_sequence_states_path(settings.storage_root, hmm.id)).unlink()

    await run_motif_extraction_job(session, motif, settings)
    await session.refresh(motif)

    assert motif.status == JobStatus.failed.value
    assert "decoded.parquet" in (motif.error_message or "")


async def test_worker_with_masked_transformer_parent(session, settings):
    cej = ContinuousEmbeddingJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id="seg-mt",
        model_version="crnn-call-parsing-pytorch",
        target_sample_rate=32000,
        encoding_signature="enc-mt-worker",
    )
    session.add(cej)
    await session.commit()
    await session.refresh(cej)

    mt = MaskedTransformerJob(
        status=JobStatus.complete.value,
        continuous_embedding_job_id=cej.id,
        training_signature="sig-mt-worker",
        k_values=serialize_k_values([100]),
    )
    session.add(mt)
    await session.commit()
    await session.refresh(mt)

    motif = MotifExtractionJob(
        status=JobStatus.running.value,
        parent_kind="masked_transformer",
        hmm_sequence_job_id=None,
        masked_transformer_job_id=mt.id,
        k=100,
        source_kind="region_crnn",
        min_ngram=2,
        max_ngram=2,
        minimum_occurrences=2,
        minimum_event_sources=1,
        config_signature="sig-motif-mt-worker",
    )
    session.add(motif)
    await session.commit()
    await session.refresh(motif)

    # Write the per-k decoded.parquet that the worker should read.
    k_dir = masked_transformer_k_dir(settings.storage_root, mt.id, 100)
    k_dir.mkdir(parents=True)
    decoded = pa.Table.from_pylist(
        [
            {
                "region_id": "r1",
                "chunk_index_in_region": 0,
                "audio_file_id": 1,
                "start_timestamp": 0.0,
                "end_timestamp": 0.25,
                "tier": "event_core",
                "label": 1,
            },
            {
                "region_id": "r1",
                "chunk_index_in_region": 1,
                "audio_file_id": 1,
                "start_timestamp": 0.25,
                "end_timestamp": 0.5,
                "tier": "background",
                "label": 2,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 0,
                "audio_file_id": 2,
                "start_timestamp": 10.0,
                "end_timestamp": 10.25,
                "tier": "event_core",
                "label": 1,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 1,
                "audio_file_id": 2,
                "start_timestamp": 10.25,
                "end_timestamp": 10.5,
                "tier": "background",
                "label": 2,
            },
        ]
    )
    pq.write_table(
        decoded, masked_transformer_k_decoded_path(settings.storage_root, mt.id, 100)
    )

    # Upstream embeddings parquet (CRNN source).
    continuous_embedding_dir(settings.storage_root, cej.id).mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "region_id": "r1",
                    "chunk_index_in_region": 0,
                    "nearest_event_id": "e1",
                },
                {
                    "region_id": "r1",
                    "chunk_index_in_region": 1,
                    "nearest_event_id": "e1",
                },
                {
                    "region_id": "r2",
                    "chunk_index_in_region": 0,
                    "nearest_event_id": "e2",
                },
                {
                    "region_id": "r2",
                    "chunk_index_in_region": 1,
                    "nearest_event_id": "e2",
                },
            ]
        ),
        continuous_embedding_parquet_path(settings.storage_root, cej.id),
    )

    await run_motif_extraction_job(session, motif, settings)
    await session.refresh(motif)

    assert motif.status == JobStatus.complete.value
    assert motif.source_kind == "region_crnn"
    assert motif_extraction_motifs_path(settings.storage_root, motif.id).exists()
