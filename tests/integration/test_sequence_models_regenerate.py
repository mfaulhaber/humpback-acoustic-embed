"""Regenerate-label-distribution endpoints (HMM + Masked Transformer).

Covers the optional re-bind path: validate → write artifacts via
temp-then-rename → commit FK update. The heavy ``generate_interpretations``
generator is stubbed at the service layer so the tests focus on the
endpoint plumbing and the atomic-rebind ordering.
"""

from __future__ import annotations

import json

import humpback.services.hmm_sequence_service as hmm_service
import humpback.services.masked_transformer_service as mt_service
from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    HMMSequenceJob,
    MaskedTransformerJob,
)
from humpback.services.masked_transformer_service import serialize_k_values
from humpback.storage import (
    hmm_sequence_label_distribution_path,
    masked_transformer_k_label_distribution_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _seed_segmentation_with_two_classifies(
    app_settings,
) -> tuple[str, str, str]:
    """Return ``(seg_id, classify_a_id, classify_b_id)`` on the same segmentation."""
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
        classify_a = EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
        classify_b = EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
        session.add_all([classify_a, classify_b])
        await session.commit()
        await session.refresh(classify_a)
        await session.refresh(classify_b)
        return seg_job.id, classify_a.id, classify_b.id


async def _seed_other_segmentation_with_classify(app_settings) -> str:
    """Create an unrelated segmentation + Classify; return that classify id."""
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
        other = EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
        session.add(other)
        await session.commit()
        await session.refresh(other)
        return other.id


async def _seed_hmm_job_bound_to(app_settings, seg_id: str, classify_id: str) -> str:
    """Build a complete CEJ + complete HMM bound to ``classify_id``."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature=f"regen-hmm-{seg_id}",
        )
        session.add(cej)
        await session.flush()
        hmm = HMMSequenceJob(
            status=JobStatus.complete.value,
            continuous_embedding_job_id=cej.id,
            event_classification_job_id=classify_id,
            n_states=3,
            pca_dims=8,
        )
        session.add(hmm)
        await session.commit()
        await session.refresh(hmm)
        return hmm.id


async def _seed_mt_job_bound_to(
    app_settings, seg_id: str, classify_id: str, k_values: list[int]
) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        seg = await session.get(EventSegmentationJob, seg_id)
        assert seg is not None
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_id,
            region_detection_job_id=seg.region_detection_job_id,
            model_version="crnn-call-parsing-pytorch",
            window_size_seconds=0.25,
            hop_seconds=0.25,
            pad_seconds=0.0,
            target_sample_rate=16000,
            encoding_signature=f"regen-mt-{seg_id}",
        )
        session.add(cej)
        await session.flush()
        mt = MaskedTransformerJob(
            status=JobStatus.complete.value,
            continuous_embedding_job_id=cej.id,
            event_classification_job_id=classify_id,
            training_signature=f"regen-mt-sig-{cej.id}",
            preset="small",
            k_values=serialize_k_values(k_values),
        )
        session.add(mt)
        await session.commit()
        await session.refresh(mt)
        return mt.id


async def _read_hmm_classify_id(app_settings, hmm_id: str) -> str | None:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(HMMSequenceJob, hmm_id)
        assert job is not None
        return job.event_classification_job_id


async def _read_mt_classify_id(app_settings, mt_id: str) -> str | None:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(MaskedTransformerJob, mt_id)
        assert job is not None
        return job.event_classification_job_id


# ---------------------------------------------------------------------------
# HMM regenerate
# ---------------------------------------------------------------------------


async def test_regenerate_hmm_rebuilds_artifacts(client, app_settings, monkeypatch):
    seg_id, classify_a, _ = await _seed_segmentation_with_two_classifies(app_settings)
    hmm_id = await _seed_hmm_job_bound_to(app_settings, seg_id, classify_a)

    dist_path = hmm_sequence_label_distribution_path(app_settings.storage_root, hmm_id)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    stale_payload = {
        "n_states": 3,
        "total_windows": 99,
        "states": {"0": {"stale": 99}, "1": {}, "2": {}},
    }
    dist_path.write_text(json.dumps(stale_payload), encoding="utf-8")

    fresh = {
        "n_states": 3,
        "total_windows": 5,
        "states": {"0": {"song": 2}, "1": {"(background)": 3}, "2": {}},
    }

    async def _fake_generate(session, storage_root, job, _cej):
        # Real generate also writes the file via temp-then-rename; mimic that.
        target = hmm_sequence_label_distribution_path(storage_root, job.id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fresh), encoding="utf-8")
        return fresh

    monkeypatch.setattr(hmm_service, "generate_interpretations", _fake_generate)

    resp = await client.post(
        f"/sequence-models/hmm-sequences/{hmm_id}/regenerate-label-distribution",
        json={},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok"
    assert body["event_classification_job_id"] == classify_a
    assert body["label_distribution"] == fresh

    # On-disk file refreshed.
    assert json.loads(dist_path.read_text(encoding="utf-8")) == fresh


async def test_regenerate_hmm_rebinds_classification_job(
    client, app_settings, monkeypatch
):
    seg_id, classify_a, classify_b = await _seed_segmentation_with_two_classifies(
        app_settings
    )
    hmm_id = await _seed_hmm_job_bound_to(app_settings, seg_id, classify_a)

    seen_classify_ids: list[str | None] = []

    async def _fake_generate(session, storage_root, job, _cej):
        seen_classify_ids.append(job.event_classification_job_id)
        return {
            "n_states": 3,
            "total_windows": 0,
            "states": {"0": {}, "1": {}, "2": {}},
        }

    monkeypatch.setattr(hmm_service, "generate_interpretations", _fake_generate)

    resp = await client.post(
        f"/sequence-models/hmm-sequences/{hmm_id}/regenerate-label-distribution",
        json={"event_classification_job_id": classify_b},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["event_classification_job_id"] == classify_b

    # Generator saw the new FK.
    assert seen_classify_ids == [classify_b]
    # Persisted FK is updated.
    assert await _read_hmm_classify_id(app_settings, hmm_id) == classify_b


async def test_regenerate_hmm_rejects_mismatched_rebind(
    client, app_settings, monkeypatch
):
    seg_id, classify_a, _ = await _seed_segmentation_with_two_classifies(app_settings)
    hmm_id = await _seed_hmm_job_bound_to(app_settings, seg_id, classify_a)
    foreign_classify = await _seed_other_segmentation_with_classify(app_settings)

    called = False

    async def _fake_generate(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(hmm_service, "generate_interpretations", _fake_generate)

    resp = await client.post(
        f"/sequence-models/hmm-sequences/{hmm_id}/regenerate-label-distribution",
        json={"event_classification_job_id": foreign_classify},
    )
    assert resp.status_code == 400, resp.text
    assert "does not match" in resp.text
    # Validation rejected before any write happened.
    assert called is False
    # FK unchanged.
    assert await _read_hmm_classify_id(app_settings, hmm_id) == classify_a


async def test_regenerate_hmm_atomic_on_failure(client, app_settings, monkeypatch):
    """Step-2 (artifact write) failure leaves FK and prior artifacts intact.

    Exercise the service directly (rather than via the HTTP client) so the
    propagated ``RuntimeError`` from the stub is observable; the router
    surfaces this as a 500 in production but the relevant invariant —
    nothing changed on disk or in the DB — is what we lock here.
    """
    import pytest

    seg_id, classify_a, classify_b = await _seed_segmentation_with_two_classifies(
        app_settings
    )
    hmm_id = await _seed_hmm_job_bound_to(app_settings, seg_id, classify_a)

    dist_path = hmm_sequence_label_distribution_path(app_settings.storage_root, hmm_id)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    intact_payload = {
        "n_states": 3,
        "total_windows": 7,
        "states": {"0": {"prior": 7}, "1": {}, "2": {}},
    }
    dist_path.write_text(json.dumps(intact_payload), encoding="utf-8")

    async def _failing_generate(*_args, **_kwargs):
        raise RuntimeError("simulated artifact-write failure")

    monkeypatch.setattr(hmm_service, "generate_interpretations", _failing_generate)

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(HMMSequenceJob, hmm_id)
        assert job is not None
        with pytest.raises(RuntimeError, match="simulated artifact-write failure"):
            await hmm_service.regenerate_label_distribution(
                session,
                app_settings.storage_root,
                job,
                requested_classify_id=classify_b,
            )
        # In-memory revert: the service-level swap is rolled back when the
        # write step raises so the row is observable as classify_a still.
        assert job.event_classification_job_id == classify_a

    # FK unchanged on disk.
    assert await _read_hmm_classify_id(app_settings, hmm_id) == classify_a
    # Prior artifact unchanged.
    assert json.loads(dist_path.read_text(encoding="utf-8")) == intact_payload


# ---------------------------------------------------------------------------
# Masked Transformer regenerate
# ---------------------------------------------------------------------------


async def test_regenerate_mt_rebuilds_all_k(client, app_settings, monkeypatch):
    seg_id, classify_a, _ = await _seed_segmentation_with_two_classifies(app_settings)
    mt_id = await _seed_mt_job_bound_to(app_settings, seg_id, classify_a, [50, 100])

    # Pre-write stale payloads under each k.
    for k in (50, 100):
        path = masked_transformer_k_label_distribution_path(
            app_settings.storage_root, mt_id, k
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"stale": True, "k": k}), encoding="utf-8")

    load_calls = 0
    seen_ks: list[int] = []

    async def _fake_load_effective_event_labels(*_args, **_kwargs):
        nonlocal load_calls
        load_calls += 1
        return []

    async def _fake_generate(session, storage_root, job, k, *, events_cache=None):
        seen_ks.append(int(k))
        # Only the events_cache path should be exercised in the multi-k loop.
        assert events_cache is not None
        target = masked_transformer_k_label_distribution_path(
            storage_root, job.id, int(k)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_states": 3,
            "total_windows": 1,
            "states": {"0": {"x": 1}, "1": {}, "2": {}},
            "k": int(k),
        }
        target.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(
        mt_service, "load_effective_event_labels", _fake_load_effective_event_labels
    )
    monkeypatch.setattr(mt_service, "generate_interpretations", _fake_generate)

    resp = await client.post(
        f"/sequence-models/masked-transformers/{mt_id}/regenerate-label-distribution",
        params={"k": 100},
        json={},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["k"] == 100
    assert body["event_classification_job_id"] == classify_a
    assert body["label_distribution"]["k"] == 100

    # All k payloads were rebuilt.
    assert sorted(seen_ks) == [50, 100]
    # Effective events loaded exactly once.
    assert load_calls == 1
    # Each k file is fresh on disk.
    for k in (50, 100):
        path = masked_transformer_k_label_distribution_path(
            app_settings.storage_root, mt_id, k
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded.get("stale") is None
        assert loaded["k"] == k


async def test_regenerate_mt_rejects_mismatched_rebind(
    client, app_settings, monkeypatch
):
    seg_id, classify_a, _ = await _seed_segmentation_with_two_classifies(app_settings)
    mt_id = await _seed_mt_job_bound_to(app_settings, seg_id, classify_a, [50])
    foreign = await _seed_other_segmentation_with_classify(app_settings)

    called = False

    async def _fake_load(*_args, **_kwargs):
        nonlocal called
        called = True
        return []

    async def _fake_generate(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(mt_service, "load_effective_event_labels", _fake_load)
    monkeypatch.setattr(mt_service, "generate_interpretations", _fake_generate)

    resp = await client.post(
        f"/sequence-models/masked-transformers/{mt_id}/regenerate-label-distribution",
        json={"event_classification_job_id": foreign},
    )
    assert resp.status_code == 400, resp.text
    assert "does not match" in resp.text
    assert called is False
    assert await _read_mt_classify_id(app_settings, mt_id) == classify_a
