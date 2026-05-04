"""Submit-time Classify-binding validation for HMM and Masked Transformer.

Covers ADR (this PR) §6.8 and the listing endpoint at
``/call-parsing/classification-jobs/by-segmentation``.
"""

from __future__ import annotations

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.processing import JobStatus


async def _seed_segmentation_with_optional_classify(
    app_settings,
    *,
    classify_status: str | None = JobStatus.complete.value,
    extra_classify_status: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Seed RegionDetectionJob + EventSegmentationJob and up to two Classify jobs.

    Returns ``(seg_id, first_classify_id, second_classify_id)``. The
    second classify is None unless ``extra_classify_status`` is provided.
    """
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

        first_id: str | None = None
        if classify_status is not None:
            first = EventClassificationJob(
                status=classify_status,
                event_segmentation_job_id=seg_job.id,
            )
            session.add(first)
            await session.flush()
            first_id = first.id

        second_id: str | None = None
        if extra_classify_status is not None:
            second = EventClassificationJob(
                status=extra_classify_status,
                event_segmentation_job_id=seg_job.id,
            )
            session.add(second)
            await session.flush()
            second_id = second.id

        await session.commit()
        return seg_job.id, first_id, second_id


async def _seed_surfperch_cej(app_settings, seg_job_id: str) -> str:
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job_id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature=f"submit-test-{seg_job_id}",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


async def _seed_crnn_cej(app_settings, seg_job_id: str) -> str:
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        seg = await session.get(EventSegmentationJob, seg_job_id)
        assert seg is not None
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job_id,
            region_detection_job_id=seg.region_detection_job_id,
            model_version="crnn-call-parsing-pytorch",
            window_size_seconds=0.25,
            hop_seconds=0.25,
            pad_seconds=0.0,
            target_sample_rate=16000,
            encoding_signature=f"submit-test-crnn-{seg_job_id}",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


# ---------------------------------------------------------------------------
# HMM submit
# ---------------------------------------------------------------------------


async def test_hmm_submit_rejects_when_no_classification(client, app_settings):
    seg_id, _, _ = await _seed_segmentation_with_optional_classify(
        app_settings, classify_status=None
    )
    cej_id = await _seed_surfperch_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    assert resp.status_code == 422
    assert seg_id in resp.text

    listed = await client.get(
        "/sequence-models/hmm-sequences",
        params={"continuous_embedding_job_id": cej_id},
    )
    assert listed.json() == []


async def test_hmm_submit_defaults_to_latest_classification(client, app_settings):
    seg_id, older_id, newer_id = await _seed_segmentation_with_optional_classify(
        app_settings,
        classify_status=JobStatus.complete.value,
        extra_classify_status=JobStatus.complete.value,
    )
    cej_id = await _seed_surfperch_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    # newer_id was seeded second, so its created_at is later → it must be picked.
    assert body["event_classification_job_id"] == newer_id
    assert body["event_classification_job_id"] != older_id


async def test_hmm_submit_rejects_mismatched_classification(client, app_settings):
    seg_a_id, classify_a, _ = await _seed_segmentation_with_optional_classify(
        app_settings
    )
    seg_b_id, classify_b, _ = await _seed_segmentation_with_optional_classify(
        app_settings
    )
    cej_a = await _seed_surfperch_cej(app_settings, seg_a_id)

    resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={
            "continuous_embedding_job_id": cej_a,
            "n_states": 3,
            "event_classification_job_id": classify_b,
        },
    )
    assert resp.status_code == 422
    assert "does not match" in resp.text


async def test_hmm_submit_rejects_non_completed_classification(client, app_settings):
    seg_id, classify_id, _ = await _seed_segmentation_with_optional_classify(
        app_settings, classify_status=JobStatus.running.value
    )
    cej_id = await _seed_surfperch_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={
            "continuous_embedding_job_id": cej_id,
            "n_states": 3,
            "event_classification_job_id": classify_id,
        },
    )
    assert resp.status_code == 422
    assert "must be completed" in resp.text


async def test_hmm_submit_stores_explicit_classification(client, app_settings):
    seg_id, c0_id, c1_id = await _seed_segmentation_with_optional_classify(
        app_settings,
        classify_status=JobStatus.complete.value,
        extra_classify_status=JobStatus.complete.value,
    )
    cej_id = await _seed_surfperch_cej(app_settings, seg_id)

    # Explicitly bind to the older Classify even though newer is the default.
    resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={
            "continuous_embedding_job_id": cej_id,
            "n_states": 3,
            "event_classification_job_id": c0_id,
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["event_classification_job_id"] == c0_id
    assert c1_id != c0_id  # sanity


# ---------------------------------------------------------------------------
# Masked Transformer submit (mirrored)
# ---------------------------------------------------------------------------


async def test_mt_submit_rejects_when_no_classification(client, app_settings):
    seg_id, _, _ = await _seed_segmentation_with_optional_classify(
        app_settings, classify_status=None
    )
    cej_id = await _seed_crnn_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "preset": "small"},
    )
    assert resp.status_code == 422
    assert seg_id in resp.text


async def test_mt_submit_defaults_to_latest_classification(client, app_settings):
    seg_id, older_id, newer_id = await _seed_segmentation_with_optional_classify(
        app_settings,
        classify_status=JobStatus.complete.value,
        extra_classify_status=JobStatus.complete.value,
    )
    cej_id = await _seed_crnn_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/masked-transformers",
        json={"continuous_embedding_job_id": cej_id, "preset": "small"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["event_classification_job_id"] == newer_id
    assert body["event_classification_job_id"] != older_id


async def test_mt_submit_rejects_mismatched_classification(client, app_settings):
    seg_a_id, _, _ = await _seed_segmentation_with_optional_classify(app_settings)
    seg_b_id, classify_b, _ = await _seed_segmentation_with_optional_classify(
        app_settings
    )
    cej_a = await _seed_crnn_cej(app_settings, seg_a_id)

    resp = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_a,
            "preset": "small",
            "event_classification_job_id": classify_b,
        },
    )
    assert resp.status_code == 422
    assert "does not match" in resp.text


async def test_mt_submit_rejects_non_completed_classification(client, app_settings):
    seg_id, classify_id, _ = await _seed_segmentation_with_optional_classify(
        app_settings, classify_status=JobStatus.running.value
    )
    cej_id = await _seed_crnn_cej(app_settings, seg_id)

    resp = await client.post(
        "/sequence-models/masked-transformers",
        json={
            "continuous_embedding_job_id": cej_id,
            "preset": "small",
            "event_classification_job_id": classify_id,
        },
    )
    assert resp.status_code == 422
    assert "must be completed" in resp.text


# ---------------------------------------------------------------------------
# Listing endpoint
# ---------------------------------------------------------------------------


async def test_listing_returns_completed_jobs_newest_first(client, app_settings):
    seg_id, older_id, newer_id = await _seed_segmentation_with_optional_classify(
        app_settings,
        classify_status=JobStatus.complete.value,
        extra_classify_status=JobStatus.complete.value,
    )

    resp = await client.get(
        "/call-parsing/classification-jobs/by-segmentation",
        params={"event_segmentation_job_id": seg_id, "status": "complete"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [row["id"] for row in body] == [newer_id, older_id]
    assert all(row["status"] == "complete" for row in body)


async def test_listing_filters_by_status(client, app_settings):
    seg_id, complete_id, running_id = await _seed_segmentation_with_optional_classify(
        app_settings,
        classify_status=JobStatus.complete.value,
        extra_classify_status=JobStatus.running.value,
    )

    resp = await client.get(
        "/call-parsing/classification-jobs/by-segmentation",
        params={"event_segmentation_job_id": seg_id, "status": "complete"},
    )
    body = resp.json()
    assert [row["id"] for row in body] == [complete_id]
    assert running_id not in {row["id"] for row in body}


async def test_listing_returns_empty_when_no_classify_for_segmentation(
    client, app_settings
):
    seg_id, _, _ = await _seed_segmentation_with_optional_classify(
        app_settings, classify_status=None
    )

    resp = await client.get(
        "/call-parsing/classification-jobs/by-segmentation",
        params={"event_segmentation_job_id": seg_id, "status": "complete"},
    )
    assert resp.status_code == 200
    assert resp.json() == []
