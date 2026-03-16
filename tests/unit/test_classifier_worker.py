"""Unit tests for classifier worker extraction behavior."""

import asyncio
import json
import queue
import threading
from types import SimpleNamespace

from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.database import create_session_factory
from humpback.workers.classifier_worker import (
    _avg_audio_x_realtime,
    _hydrophone_detection_subprocess_main,
    _hydrophone_provider_mode,
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
    settings.spectrogram_hop_length = 300
    settings.spectrogram_dynamic_range_db = 72.0
    settings.spectrogram_width_px = 777
    settings.spectrogram_height_px = 333
    capture: dict[str, object] = {}

    class DummyProvider:
        source_id = "rpi_orcasound_lab"

    def _fake_extract_hydrophone_labeled_samples(**kwargs):
        capture["provider"] = kwargs["provider"]
        capture["window_diagnostics_path"] = kwargs["window_diagnostics_path"]
        capture["positive_selection_smoothing_window"] = kwargs[
            "positive_selection_smoothing_window"
        ]
        capture["positive_selection_min_score"] = kwargs["positive_selection_min_score"]
        capture["positive_selection_extend_min_score"] = kwargs[
            "positive_selection_extend_min_score"
        ]
        capture["spectrogram_hop_length"] = kwargs["spectrogram_hop_length"]
        capture["spectrogram_dynamic_range_db"] = kwargs["spectrogram_dynamic_range_db"]
        capture["spectrogram_width_px"] = kwargs["spectrogram_width_px"]
        capture["spectrogram_height_px"] = kwargs["spectrogram_height_px"]
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
                "positive_selection_smoothing_window": 5,
                "positive_selection_min_score": 0.8,
                "positive_selection_extend_min_score": 0.6,
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
    assert (
        capture["window_diagnostics_path"]
        == tsv_path.parent / "window_diagnostics.parquet"
    )
    assert capture["positive_selection_smoothing_window"] == 5
    assert capture["positive_selection_min_score"] == 0.8
    assert capture["positive_selection_extend_min_score"] == 0.6
    assert capture["spectrogram_hop_length"] == 300
    assert capture["spectrogram_dynamic_range_db"] == 72.0
    assert capture["spectrogram_width_px"] == 777
    assert capture["spectrogram_height_px"] == 333
    assert job.extract_status == "complete"
    assert job.extract_summary is not None


async def test_local_extraction_forwards_spectrogram_settings(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """Local extraction should receive the same spectrogram settings as the UI endpoint."""
    tsv_path = tmp_path / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "test.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    settings.spectrogram_hop_length = 144
    settings.spectrogram_dynamic_range_db = 65.0
    settings.spectrogram_width_px = 512
    settings.spectrogram_height_px = 256
    capture: dict[str, object] = {}

    def _fake_extract_labeled_samples(**kwargs):
        capture["audio_folder"] = kwargs["audio_folder"]
        capture["window_diagnostics_path"] = kwargs["window_diagnostics_path"]
        capture["spectrogram_hop_length"] = kwargs["spectrogram_hop_length"]
        capture["spectrogram_dynamic_range_db"] = kwargs["spectrogram_dynamic_range_db"]
        capture["spectrogram_width_px"] = kwargs["spectrogram_width_px"]
        capture["spectrogram_height_px"] = kwargs["spectrogram_height_px"]
        return {"n_humpback": 1, "n_ship": 0, "n_background": 0, "n_skipped": 0}

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.extract_labeled_samples",
        _fake_extract_labeled_samples,
    )

    job = DetectionJob(
        status="complete",
        extract_status="running",
        classifier_model_id="missing-model-is-allowed",
        audio_folder=str(tmp_path / "audio"),
        output_tsv_path=str(tsv_path),
        extract_config=json.dumps(
            {
                "positive_output_path": str(tmp_path / "pos"),
                "negative_output_path": str(tmp_path / "neg"),
            }
        ),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["audio_folder"] == str(tmp_path / "audio")
    assert (
        capture["window_diagnostics_path"]
        == tsv_path.parent / "window_diagnostics.parquet"
    )
    assert capture["spectrogram_hop_length"] == 144
    assert capture["spectrogram_dynamic_range_db"] == 65.0
    assert capture["spectrogram_width_px"] == 512
    assert capture["spectrogram_height_px"] == 256
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

    def _fake_extract_hydrophone_labeled_samples(**kwargs):
        capture["provider"] = kwargs["provider"]
        capture["window_diagnostics_path"] = kwargs["window_diagnostics_path"]
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
    assert (
        capture["window_diagnostics_path"]
        == tsv_path.parent / "window_diagnostics.parquet"
    )
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

    def _fake_run_hydrophone_detection(**kwargs):
        capture["provider"] = kwargs["provider"]
        on_chunk_complete = kwargs["on_chunk_complete"]
        on_chunk_diagnostics = kwargs["on_chunk_diagnostics"]
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
        on_chunk_diagnostics(
            [
                {
                    "filename": "20250702T070018Z.wav",
                    "window_index": 0,
                    "offset_sec": 0.0,
                    "end_sec": 5.0,
                    "confidence": 0.95,
                    "is_overlapped": False,
                    "overlap_sec": 0.0,
                }
            ],
            1,
        )
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
    assert job.result_summary is not None
    assert json.loads(job.result_summary)["has_diagnostics"] is True


def test_hydrophone_provider_mode_reports_cache_strategy() -> None:
    """Provider-mode tagging should distinguish Orcasound and NOAA cache modes."""
    assert (
        _hydrophone_provider_mode(
            "rpi_orcasound_lab",
            local_cache_path="/cache",
            s3_cache_path="/s3-cache",
            noaa_cache_path="/noaa-cache",
        )
        == "local_cache_only"
    )
    assert (
        _hydrophone_provider_mode(
            "rpi_orcasound_lab",
            local_cache_path=None,
            s3_cache_path="/s3-cache",
            noaa_cache_path=None,
        )
        == "s3_write_through_cache"
    )
    assert (
        _hydrophone_provider_mode(
            "rpi_orcasound_lab",
            local_cache_path=None,
            s3_cache_path=None,
            noaa_cache_path=None,
        )
        == "direct_s3"
    )
    assert (
        _hydrophone_provider_mode(
            "noaa_glacier_bay",
            local_cache_path=None,
            s3_cache_path=None,
            noaa_cache_path="/noaa-cache",
        )
        == "noaa_cache"
    )
    assert (
        _hydrophone_provider_mode(
            "noaa_glacier_bay",
            local_cache_path=None,
            s3_cache_path=None,
            noaa_cache_path=None,
        )
        == "direct_gcs"
    )


def test_avg_audio_x_realtime_uses_end_to_end_measured_time() -> None:
    """Throughput should include fetch and decode, not only pipeline time."""
    assert (
        _avg_audio_x_realtime(
            {
                "time_covered_sec": 60.0,
                "fetch_sec": 1.0,
                "decode_sec": 2.0,
                "pipeline_total_sec": 3.0,
            }
        )
        == 10.0
    )


def test_hydrophone_detection_subprocess_main_emits_expected_events(
    monkeypatch,
) -> None:
    """The child worker should proxy progress, diagnostics, alerts, and results."""
    event_queue: queue.Queue[dict] = queue.Queue()
    cancel_event = threading.Event()
    pause_gate = threading.Event()
    pause_gate.set()

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
    diag = {
        "filename": "20250702T070018Z.wav",
        "window_index": 0,
        "offset_sec": 0.0,
        "end_sec": 5.0,
        "confidence": 0.95,
        "is_overlapped": False,
        "overlap_sec": 0.0,
    }

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.joblib.load",
        lambda _path: object(),
    )

    class DummyProvider:
        source_id = "rpi_orcasound_lab"

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.build_archive_detection_provider",
        lambda *_args, **_kwargs: DummyProvider(),
    )

    def _fake_run_hydrophone_detection(**kwargs):
        kwargs["on_resume_invalidation"]()
        kwargs["on_chunk_complete"]([det], 1, 2, 60.0)
        kwargs["on_chunk_diagnostics"]([diag], 1)
        kwargs["on_alert"](
            {
                "type": "warning",
                "message": "decode hiccup",
                "timestamp": "2026-03-16T15:35:00Z",
            }
        )
        return [det], {
            "n_windows": 1,
            "n_detections": 1,
            "n_spans": 1,
            "time_covered_sec": 60.0,
            "pipeline_total_sec": 2.0,
        }

    monkeypatch.setattr(
        "humpback.classifier.hydrophone_detector.run_hydrophone_detection",
        _fake_run_hydrophone_detection,
    )

    _hydrophone_detection_subprocess_main(
        event_queue=event_queue,
        cancel_event=cancel_event,
        pause_gate=pause_gate,
        runtime={
            "classifier_model_path": "/tmp/classifier.joblib",
            "model_runtime": {
                "model_version": "surfperch-tensorflow2",
                "model_type": "tf2_saved_model",
                "input_format": "waveform",
                "model_path": "models/surfperch-tensorflow2",
                "vector_dim": 1280,
            },
            "settings": {"use_real_model": False, "tf_force_cpu": False},
            "hydrophone_id": "rpi_orcasound_lab",
            "local_cache_path": None,
            "s3_cache_path": "/tmp/s3-cache",
            "noaa_cache_path": None,
            "start_timestamp": 1751439600.0,
            "end_timestamp": 1751443200.0,
            "window_size_seconds": 5.0,
            "target_sample_rate": 32000,
            "confidence_threshold": 0.9,
            "feature_config": None,
            "hop_seconds": 1.0,
            "high_threshold": 0.8,
            "low_threshold": 0.7,
            "skip_segments": 0,
            "prior_detections": [],
            "prefetch_enabled": True,
            "prefetch_workers": 4,
            "prefetch_inflight_segments": 16,
        },
    )

    messages: list[dict] = []
    while not event_queue.empty():
        messages.append(event_queue.get())

    assert [m["type"] for m in messages] == [
        "resume_invalidated",
        "progress",
        "diagnostics",
        "alert",
        "result",
    ]
    result = messages[-1]
    assert result["summary"]["n_windows"] == 1
    assert result["child_pid"] is not None
    assert result["peak_worker_rss_mb"] is not None


async def test_tf2_hydrophone_detection_uses_subprocess_path(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """TF2 hydrophone jobs should avoid in-process model loading across runs."""
    model = ClassifierModel(
        name="hydro-test-model-subprocess",
        model_path=str(tmp_path / "model.joblib"),
        model_version="surfperch-tensorflow2",
        vector_dim=1280,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(model)
    await session.flush()

    jobs = [
        DetectionJob(
            status="running",
            classifier_model_id=model.id,
            hydrophone_id="rpi_orcasound_lab",
            hydrophone_name="Orcasound Lab",
            start_timestamp=1751439600.0,
            end_timestamp=1751443200.0,
            confidence_threshold=0.9,
            hop_seconds=1.0,
            high_threshold=0.8,
            low_threshold=0.7,
        ),
        DetectionJob(
            status="running",
            classifier_model_id=model.id,
            hydrophone_id="rpi_orcasound_lab",
            hydrophone_name="Orcasound Lab",
            start_timestamp=1751443200.0,
            end_timestamp=1751446800.0,
            confidence_threshold=0.9,
            hop_seconds=1.0,
            high_threshold=0.8,
            low_threshold=0.7,
        ),
    ]
    session.add_all(jobs)
    await session.commit()
    for job in jobs:
        await session.refresh(job)

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.joblib.load",
        lambda _path: (_ for _ in ()).throw(
            AssertionError(
                "parent worker should not load classifier pipeline for TF2 subprocess jobs"
            )
        ),
    )

    async def _fake_get_model_by_name(_session, _model_version):
        return SimpleNamespace(
            model_type="tf2_saved_model",
            input_format="waveform",
            path="models/surfperch-tensorflow2",
            vector_dim=1280,
        )

    async def _fail_get_model_by_version(*_args, **_kwargs):
        raise AssertionError(
            "parent worker should not load the embedding model for TF2 subprocess jobs"
        )

    class DummyProvider:
        source_id = "rpi_orcasound_lab"

    monkeypatch.setattr(
        "humpback.workers.classifier_worker.get_model_by_name",
        _fake_get_model_by_name,
    )
    monkeypatch.setattr(
        "humpback.workers.classifier_worker.get_model_by_version",
        _fail_get_model_by_version,
    )
    monkeypatch.setattr(
        "humpback.workers.classifier_worker.build_archive_detection_provider",
        lambda *_args, **_kwargs: DummyProvider(),
    )

    subprocess_calls: list[str] = []

    async def _fake_subprocess_runner(**kwargs):
        subprocess_calls.append(str(kwargs["runtime"]["job_id"]))
        on_chunk_complete = kwargs["on_chunk_complete"]
        on_chunk_diagnostics = kwargs["on_chunk_diagnostics"]
        on_alert = kwargs["on_alert"]
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
        on_chunk_diagnostics(
            [
                {
                    "filename": "20250702T070018Z.wav",
                    "window_index": 0,
                    "offset_sec": 0.0,
                    "end_sec": 5.0,
                    "confidence": 0.95,
                    "is_overlapped": False,
                    "overlap_sec": 0.0,
                }
            ],
            1,
        )
        on_alert(
            {
                "type": "warning",
                "message": "prefetch hiccup",
                "timestamp": "2026-03-16T15:35:00Z",
            }
        )
        return [det], {
            "n_windows": 1,
            "n_detections": 1,
            "n_spans": 1,
            "time_covered_sec": 60.0,
            "fetch_sec": 1.0,
            "decode_sec": 1.0,
            "pipeline_total_sec": 2.0,
            "hydrophone_id": "rpi_orcasound_lab",
            "peak_worker_rss_mb": 256.0,
            "child_pid": 424242,
        }

    monkeypatch.setattr(
        "humpback.workers.classifier_worker._run_hydrophone_detection_in_subprocess",
        _fake_subprocess_runner,
    )

    session_factory = create_session_factory(session.bind)
    for job in jobs:
        await run_hydrophone_detection_job(
            session,
            job,
            settings,
            session_factory=session_factory,
        )
        await asyncio.sleep(0.05)
        await session.refresh(job)

        summary = json.loads(job.result_summary or "{}")
        assert job.status == "complete"
        assert summary["execution_mode"] == "subprocess"
        assert summary["provider_mode"] == "s3_write_through_cache"
        assert summary["avg_audio_x_realtime"] == 15.0
        assert summary["peak_worker_rss_mb"] == 256.0
        assert summary["child_pid"] == 424242

    assert subprocess_calls == [jobs[0].id, jobs[1].id]
