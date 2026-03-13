"""Unit tests for classifier worker extraction behavior."""

import asyncio
import json

from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.database import create_session_factory
from humpback.workers.classifier_worker import (
    run_extraction_job,
    run_hydrophone_detection_job,
)


async def test_hydrophone_extraction_uses_archive_playback_builder_with_default_cache_path(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """Orcasound extraction should use the archive playback provider builder."""
    tsv_path = tmp_path / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "20250615T080000Z.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    settings.s3_cache_path = str(tmp_path / "cache-root")
    capture: dict[str, object] = {}

    class DummyProvider:
        source_id = "rpi_orcasound_lab"

    def _fake_extract_hydrophone_labeled_samples(
        _tsv_path,
        provider,
        _positive_output_path,
        _negative_output_path,
        *_args,
        **_kwargs,
    ):
        capture["provider"] = provider
        return {"n_humpback": 1, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    def _fake_build_playback_provider(
        source_id: str,
        *,
        cache_path: str | None,
        noaa_cache_path: str | None = None,
    ):
        capture["source_id"] = source_id
        capture["cache_path"] = cache_path
        capture["noaa_cache_path"] = noaa_cache_path
        return DummyProvider()

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.build_archive_playback_provider",
        _fake_build_playback_provider,
    )
    monkeypatch.setattr(
        "humpback.classifier.extractor.extract_hydrophone_labeled_samples",
        _fake_extract_hydrophone_labeled_samples,
    )

    job = DetectionJob(
        status="complete",
        extract_status="running",
        classifier_model_id="missing-model-is-allowed",
        hydrophone_id="rpi_orcasound_lab",
        hydrophone_name="Orcasound Lab",
        start_timestamp=1000.0,
        end_timestamp=2000.0,
        output_tsv_path=str(tsv_path),
        extract_config=json.dumps(
            {
                "positive_output_path": str(tmp_path / "pos"),
                "negative_output_path": str(tmp_path / "neg"),
            }
        ),
        local_cache_path=None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["cache_path"] == settings.s3_cache_path
    assert capture["source_id"] == "rpi_orcasound_lab"
    assert isinstance(capture["provider"], DummyProvider)
    assert job.extract_status == "complete"
    assert job.extract_summary is not None


async def test_noaa_extraction_does_not_require_cache_path(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """NOAA extraction should use direct provider playback without cache config."""
    tsv_path = tmp_path / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "20150725T000009Z.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    settings.s3_cache_path = None
    capture: dict[str, object] = {}

    class DummyProvider:
        source_id = "noaa_glacier_bay"

    def _fake_extract_hydrophone_labeled_samples(
        _tsv_path,
        provider,
        _positive_output_path,
        _negative_output_path,
        *_args,
        **_kwargs,
    ):
        capture["provider"] = provider
        return {"n_humpback": 1, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    def _fake_build_playback_provider(
        source_id: str,
        *,
        cache_path: str | None,
        noaa_cache_path: str | None = None,
    ):
        capture["source_id"] = source_id
        capture["cache_path"] = cache_path
        capture["noaa_cache_path"] = noaa_cache_path
        return DummyProvider()

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.build_archive_playback_provider",
        _fake_build_playback_provider,
    )
    monkeypatch.setattr(
        "humpback.classifier.extractor.extract_hydrophone_labeled_samples",
        _fake_extract_hydrophone_labeled_samples,
    )

    job = DetectionJob(
        status="complete",
        extract_status="running",
        classifier_model_id="missing-model-is-allowed",
        hydrophone_id="noaa_glacier_bay",
        hydrophone_name="NOAA Glacier Bay (Bartlett Cove)",
        start_timestamp=1437782400.0,
        end_timestamp=1437786000.0,
        output_tsv_path=str(tsv_path),
        extract_config=json.dumps(
            {
                "positive_output_path": str(tmp_path / "pos"),
                "negative_output_path": str(tmp_path / "neg"),
            }
        ),
        local_cache_path=None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["source_id"] == "noaa_glacier_bay"
    assert capture["cache_path"] is None
    assert isinstance(capture["provider"], DummyProvider)
    assert job.extract_status == "complete"
    assert job.extract_summary is not None


async def test_hydrophone_detection_no_audio_marks_job_failed_with_range_message(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """No-overlap hydrophone ranges should fail with a clear UTC-range message."""
    model = ClassifierModel(
        name="hydro-test-model",
        model_path=str(tmp_path / "model.joblib"),
        model_version="test_v1",
        vector_dim=128,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(model)
    await session.flush()

    job = DetectionJob(
        status="running",
        classifier_model_id=model.id,
        hydrophone_id="rpi_north_sjc",
        hydrophone_name="North San Juan Channel",
        start_timestamp=1751461200.0,  # 2025-07-02T13:00:00Z
        end_timestamp=1751482800.0,  # 2025-07-02T19:00:00Z
        confidence_threshold=0.5,
        hop_seconds=1.0,
        high_threshold=0.7,
        low_threshold=0.45,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.joblib.load",
        lambda _path: object(),
    )

    async def _fake_get_model_by_version(_session, _model_version, _settings):
        class DummyModel:
            def embed(self, _batch):
                raise AssertionError(
                    "embed() should not be called when no timeline exists"
                )

        return DummyModel(), "waveform"

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.get_model_by_version",
        _fake_get_model_by_version,
    )

    def _raise_no_audio(*_args, **_kwargs):
        raise FileNotFoundError("No stream segments found in requested range")

    monkeypatch.setattr(
        "humpback.classifier.hydrophone_detector.run_hydrophone_detection",
        _raise_no_audio,
    )

    await run_hydrophone_detection_job(session, job, settings)
    await session.refresh(job)

    assert job.status == "failed"
    assert job.error_message is not None
    assert (
        "No hydrophone audio segments found for hydrophone 'rpi_north_sjc'"
        in job.error_message
    )
    assert "[2025-07-02T13:00:00Z, 2025-07-02T19:00:00Z]" in job.error_message


async def test_hydrophone_detection_success_updates_progress_and_completes(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """Hydrophone worker should persist progress callbacks and complete successfully."""
    model = ClassifierModel(
        name="hydro-test-model-progress",
        model_path=str(tmp_path / "model.joblib"),
        model_version="test_v1",
        vector_dim=128,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(model)
    await session.flush()

    job = DetectionJob(
        status="running",
        classifier_model_id=model.id,
        hydrophone_id="rpi_orcasound_lab",
        hydrophone_name="Orcasound Lab",
        start_timestamp=1751439600.0,
        end_timestamp=1751443200.0,
        confidence_threshold=0.5,
        hop_seconds=1.0,
        high_threshold=0.7,
        low_threshold=0.45,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.joblib.load",
        lambda _path: object(),
    )

    async def _fake_get_model_by_version(_session, _model_version, _settings):
        class DummyModel:
            def embed(self, _batch):
                raise AssertionError("embed() should not be called in this mocked test")

        return DummyModel(), "waveform"

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.get_model_by_version",
        _fake_get_model_by_version,
    )

    class DummyProvider:
        source_id = "rpi_orcasound_lab"

    capture: dict[str, object] = {}

    def _fake_build_detection_provider(
        source_id: str,
        *,
        local_cache_path: str | None,
        s3_cache_path: str | None,
        noaa_cache_path: str | None = None,
    ):
        capture["source_id"] = source_id
        capture["local_cache_path"] = local_cache_path
        capture["s3_cache_path"] = s3_cache_path
        return DummyProvider()

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.build_archive_detection_provider",
        _fake_build_detection_provider,
    )

    def _fake_run_hydrophone_detection(*args, **_kwargs):
        capture["provider"] = args[0]
        on_chunk_complete = args[13]
        det = {
            "filename": "20250702T070018Z.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
            "avg_confidence": 0.9,
            "peak_confidence": 0.95,
            "n_windows": 1,
            "detection_filename": "20250702T070018Z_20250702T070023Z.wav",
            "extract_filename": "20250702T070018Z_20250702T070023Z.wav",
        }
        on_chunk_complete([det], 1, 1, 60.0)
        return [det], {
            "n_windows": 1,
            "n_detections": 1,
            "n_spans": 1,
            "time_covered_sec": 60.0,
            "hydrophone_id": "rpi_orcasound_lab",
        }

    monkeypatch.setattr(
        "humpback.classifier.hydrophone_detector.run_hydrophone_detection",
        _fake_run_hydrophone_detection,
    )

    session_factory = create_session_factory(session.bind)
    await run_hydrophone_detection_job(
        session,
        job,
        settings,
        session_factory=session_factory,
    )
    await asyncio.sleep(0.05)
    await session.refresh(job)

    assert job.status == "complete"
    assert job.result_summary is not None
    assert job.segments_processed == 1
    assert job.segments_total == 1
    assert capture["source_id"] == "rpi_orcasound_lab"
    assert capture["provider"] is not None
