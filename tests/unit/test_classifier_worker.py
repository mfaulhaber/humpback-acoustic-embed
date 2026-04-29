"""Unit tests for classifier worker extraction behavior."""

import asyncio
import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_session_factory
from humpback.models.classifier import (
    AutoresearchCandidate,
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)
from humpback.workers.classifier_worker import (
    _avg_audio_x_realtime,
    _hydrophone_detection_subprocess_main,
    _hydrophone_provider_mode,
    run_extraction_job,
    run_hydrophone_detection_job,
    run_training_job,
)


def _write_embedding_set_parquet(path: Path, rows: list[list[float]]) -> None:
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(rows))), type=pa.int32()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _write_detection_embeddings_parquet(
    path: Path,
    row_ids: list[str],
    rows: list[list[float]],
) -> None:
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


async def test_hydrophone_extraction_uses_archive_playback_builder_with_default_cache_path(
    session,
    settings,
    tmp_path,
    monkeypatch,
):
    """Orcasound extraction should use the archive playback provider builder."""
    # Job will be created below; we need to pre-create files in the derived path
    # First, set up settings and capture dict
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
        detection_mode="windowed",
        hydrophone_id="rpi_orcasound_lab",
        hydrophone_name="Orcasound Lab",
        start_timestamp=1000.0,
        end_timestamp=2000.0,
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

    # Write TSV in the derived detection directory
    from humpback.storage import detection_dir

    ddir = detection_dir(settings.storage_root, job.id)
    ddir.mkdir(parents=True, exist_ok=True)
    tsv_path = ddir / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "20250615T080000Z.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["cache_path"] == settings.s3_cache_path
    assert capture["source_id"] == "rpi_orcasound_lab"
    assert isinstance(capture["provider"], DummyProvider)
    assert capture["window_diagnostics_path"] == ddir / "window_diagnostics.parquet"
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
        detection_mode="windowed",
        audio_folder=str(tmp_path / "audio"),
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

    # Write TSV in the derived detection directory
    from humpback.storage import detection_dir

    ddir = detection_dir(settings.storage_root, job.id)
    ddir.mkdir(parents=True, exist_ok=True)
    tsv_path = ddir / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "test.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

    await run_extraction_job(session, job, settings)
    await session.refresh(job)

    assert capture["audio_folder"] == str(tmp_path / "audio")
    assert capture["window_diagnostics_path"] == ddir / "window_diagnostics.parquet"
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
        detection_mode="windowed",
        hydrophone_id="noaa_glacier_bay",
        hydrophone_name="NOAA Glacier Bay (Bartlett Cove)",
        start_timestamp=1437782400.0,
        end_timestamp=1437786000.0,
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

    # Write TSV in the derived detection directory
    from humpback.storage import detection_dir

    ddir = detection_dir(settings.storage_root, job.id)
    ddir.mkdir(parents=True, exist_ok=True)
    tsv_path = ddir / "detections.tsv"
    tsv_path.write_text(
        "filename\tstart_sec\tend_sec\tavg_confidence\tpeak_confidence\thumpback\tship\tbackground\n"
        "20150725T000009Z.wav\t0.0\t5.0\t0.9\t0.95\t1\t\t\n"
    )

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
    assert job.timeline_tiles_ready is False


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
        assert job.timeline_tiles_ready is False

    assert subprocess_calls == [jobs[0].id, jobs[1].id]


async def test_run_training_job_from_autoresearch_candidate_manifest(
    session,
    settings,
    tmp_path,
) -> None:
    """Candidate-backed training jobs should train from manifest artifacts."""
    pos_path = tmp_path / "embeddings" / "pos.parquet"
    det_job_id = "det-job-1"
    det_path = (
        settings.storage_root
        / "detections"
        / det_job_id
        / "detection_embeddings.parquet"
    )
    _write_embedding_set_parquet(
        pos_path,
        rows=[[2.0, 2.0], [2.5, 2.5]],
    )
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=["neg-1", "neg-2"],
        rows=[[-2.0, -2.0], [-2.5, -2.5]],
    )

    source_model = ClassifierModel(
        id="source-model",
        name="source-model",
        model_path="/tmp/source.joblib",
        model_version="perch_v1",
        vector_dim=2,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(source_model)
    await session.flush()

    session.add(
        DetectionJob(
            id=det_job_id,
            status="complete",
            classifier_model_id=source_model.id,
            audio_folder=str(tmp_path / "audio"),
            confidence_threshold=0.5,
            detection_mode="windowed",
        )
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "examples": [
                    {
                        "id": "pos-0",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(pos_path),
                        "row_index": 0,
                    },
                    {
                        "id": "pos-1",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(pos_path),
                        "row_index": 1,
                    },
                    {
                        "id": "neg-1",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-1",
                    },
                    {
                        "id": "neg-2",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-2",
                    },
                ],
            }
        )
    )

    candidate = AutoresearchCandidate(
        id="cand-1",
        name="Candidate 1",
        status="training",
        manifest_path=str(manifest_path),
        best_run_path=str(tmp_path / "best_run.json"),
        promoted_config=json.dumps(
            {
                "classifier": "logreg",
                "feature_norm": "standard",
                "class_weight_pos": 1.0,
                "class_weight_neg": 1.0,
                "hard_negative_fraction": 0.0,
                "prob_calibration": "none",
                "threshold": 0.5,
                "context_pooling": "center",
                "seed": 42,
            }
        ),
        source_model_id=source_model.id,
        source_model_name=source_model.name,
        training_job_id="train-job-1",
        is_reproducible_exact=True,
    )
    session.add(candidate)
    session.add(
        ClassifierTrainingJob(
            id="train-job-1",
            status="running",
            name="candidate-model",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            parameters=json.dumps(
                {
                    "classifier_type": "logistic_regression",
                    "feature_norm": "standard",
                    "class_weight": {"0": 1.0, "1": 1.0},
                }
            ),
            source_mode="autoresearch_candidate",
            source_candidate_id=candidate.id,
            source_model_id=source_model.id,
            manifest_path=str(manifest_path),
            training_split_name="train",
            promoted_config=candidate.promoted_config,
            source_comparison_context=json.dumps(
                {
                    "candidate_id": candidate.id,
                    "candidate_name": candidate.name,
                    "source_model_id": source_model.id,
                    "source_model_name": source_model.name,
                }
            ),
        )
    )
    await session.commit()

    job = await session.get(ClassifierTrainingJob, "train-job-1")
    assert job is not None

    await run_training_job(session, job, settings)

    await session.refresh(candidate)
    await session.refresh(job)

    assert job.status == "complete"
    assert job.classifier_model_id is not None
    assert candidate.status == "complete"
    assert candidate.new_model_id == job.classifier_model_id

    model = await session.get(ClassifierModel, job.classifier_model_id)
    assert model is not None
    assert model.training_source_mode == "autoresearch_candidate"
    assert model.source_candidate_id == candidate.id
    assert model.source_model_id == source_model.id
    provenance = json.loads(model.promotion_provenance or "{}")
    assert provenance["candidate_id"] == candidate.id

    # Verify replay fields in training summary
    summary = json.loads(model.training_summary or "{}")
    assert "replay_effective_config" in summary
    assert summary["replay_effective_config"]["feature_norm"] == "standard"
    assert summary["replay_effective_config"]["context_pooling"] == "center"
    assert summary["replay_effective_config"]["prob_calibration"] == "none"
    assert "replay_pooling_report" in summary
    assert "promoted_config" in summary

    # Verify the model can predict
    import joblib

    loaded_pipeline = joblib.load(model.model_path)
    import numpy as np

    probs = loaded_pipeline.predict_proba(np.array([[1.0, 1.0]], dtype=np.float32))
    assert probs.shape == (1, 2)


async def test_run_training_job_candidate_replay_with_pca_and_calibration(
    session,
    settings,
    tmp_path,
) -> None:
    """Candidate-backed training with PCA + platt calibration uses replay pipeline."""
    import numpy as np

    rng = np.random.RandomState(42)
    dim = 16
    n_pos = 30
    n_neg = 30

    # Write positive embedding-set parquet
    pos_path = tmp_path / "embeddings" / "pos.parquet"
    pos_embeddings = rng.randn(n_pos, dim).astype(np.float32)
    pos_table = pa.table(
        {
            "row_index": pa.array(list(range(n_pos)), type=pa.int32()),
            "embedding": pa.array(
                [row.tolist() for row in pos_embeddings],
                type=pa.list_(pa.float32(), dim),
            ),
        }
    )
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pos_table, str(pos_path))

    # Write negative detection parquet (row_id format)
    det_job_id = "det-job-pca"
    det_path = (
        settings.storage_root
        / "detections"
        / det_job_id
        / "detection_embeddings.parquet"
    )
    neg_embeddings = rng.randn(n_neg, dim).astype(np.float32) - 2.0
    neg_row_ids = [f"neg-{i}" for i in range(n_neg)]
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=neg_row_ids,
        rows=[row.tolist() for row in neg_embeddings],
    )

    # Build manifest
    examples = []
    for i in range(n_pos):
        examples.append(
            {
                "id": f"pos-{i}",
                "split": "train",
                "label": 1,
                "parquet_path": str(pos_path),
                "row_index": i,
            }
        )
    for i in range(n_neg):
        examples.append(
            {
                "id": f"neg-{i}",
                "split": "train",
                "label": 0,
                "parquet_path": str(det_path),
                "row_id": f"neg-{i}",
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {}, "examples": examples}))

    source_model = ClassifierModel(
        id="source-model-pca",
        name="source-model-pca",
        model_path="/tmp/source.joblib",
        model_version="perch_v1",
        vector_dim=dim,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(source_model)

    session.add(
        DetectionJob(
            id=det_job_id,
            status="complete",
            classifier_model_id=source_model.id,
            audio_folder=str(tmp_path / "audio"),
            confidence_threshold=0.5,
            detection_mode="windowed",
        )
    )

    candidate = AutoresearchCandidate(
        id="cand-pca",
        name="PCA Candidate",
        status="training",
        manifest_path=str(manifest_path),
        best_run_path=str(tmp_path / "best_run.json"),
        promoted_config=json.dumps(
            {
                "classifier": "logreg",
                "feature_norm": "l2",
                "pca_dim": 8,
                "prob_calibration": "platt",
                "context_pooling": "center",
                "class_weight_pos": 2.0,
                "class_weight_neg": 1.0,
                "hard_negative_fraction": 0.0,
                "threshold": 0.5,
                "seed": 42,
            }
        ),
        source_model_id=source_model.id,
        source_model_name=source_model.name,
        training_job_id="train-job-pca",
        is_reproducible_exact=True,
    )
    session.add(candidate)
    session.add(
        ClassifierTrainingJob(
            id="train-job-pca",
            status="running",
            name="pca-candidate-model",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            parameters=json.dumps({}),
            source_mode="autoresearch_candidate",
            source_candidate_id=candidate.id,
            source_model_id=source_model.id,
            manifest_path=str(manifest_path),
            training_split_name="train",
            promoted_config=candidate.promoted_config,
            source_comparison_context=json.dumps(
                {
                    "candidate_id": candidate.id,
                    "candidate_name": candidate.name,
                }
            ),
        )
    )
    await session.commit()

    job = await session.get(ClassifierTrainingJob, "train-job-pca")
    assert job is not None

    await run_training_job(session, job, settings)

    await session.refresh(candidate)
    await session.refresh(job)

    assert job.status == "complete"
    assert job.classifier_model_id is not None
    assert candidate.status == "complete"

    model = await session.get(ClassifierModel, job.classifier_model_id)
    assert model is not None

    summary = json.loads(model.training_summary or "{}")
    eff = summary["replay_effective_config"]
    assert eff["feature_norm"] == "l2"
    assert eff["pca_dim"] == 8
    assert eff["pca_components_actual"] == 8
    assert eff["prob_calibration"] == "platt"
    assert eff["class_weight"] == {"0": 1.0, "1": 2.0}

    # Verify the model can predict (calibrated pipeline)
    import joblib

    loaded_pipeline = joblib.load(model.model_path)
    test_input = rng.randn(3, dim).astype(np.float32)
    probs = loaded_pipeline.predict_proba(test_input)
    assert probs.shape == (3, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


async def test_run_training_job_candidate_replay_linear_svm_verified(
    session,
    settings,
    tmp_path,
) -> None:
    """Linear SVM candidates train end-to-end via replay and pass verification."""
    import numpy as np

    rng = np.random.RandomState(7)
    dim = 16
    n_pos = 30
    n_neg = 30

    # Well-separated clusters so both the trained model and the "expected"
    # metrics converge to perfect classification on the train split.
    pos_path = tmp_path / "embeddings" / "pos.parquet"
    pos_embeddings = rng.randn(n_pos, dim).astype(np.float32) + 3.0
    pos_table = pa.table(
        {
            "row_index": pa.array(list(range(n_pos)), type=pa.int32()),
            "embedding": pa.array(
                [row.tolist() for row in pos_embeddings],
                type=pa.list_(pa.float32(), dim),
            ),
        }
    )
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pos_table, str(pos_path))

    det_job_id = "det-job-svm"
    det_path = (
        settings.storage_root
        / "detections"
        / det_job_id
        / "detection_embeddings.parquet"
    )
    neg_embeddings = rng.randn(n_neg, dim).astype(np.float32) - 3.0
    neg_row_ids = [f"neg-{i}" for i in range(n_neg)]
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=neg_row_ids,
        rows=[row.tolist() for row in neg_embeddings],
    )

    examples = []
    for i in range(n_pos):
        examples.append(
            {
                "id": f"pos-{i}",
                "split": "train",
                "label": 1,
                "parquet_path": str(pos_path),
                "row_index": i,
            }
        )
    for i in range(n_neg):
        examples.append(
            {
                "id": f"neg-{i}",
                "split": "train",
                "label": 0,
                "parquet_path": str(det_path),
                "row_id": f"neg-{i}",
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {}, "examples": examples}))

    source_model = ClassifierModel(
        id="source-model-svm",
        name="source-model-svm",
        model_path="/tmp/source.joblib",
        model_version="perch_v1",
        vector_dim=dim,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(source_model)

    session.add(
        DetectionJob(
            id=det_job_id,
            status="complete",
            classifier_model_id=source_model.id,
            audio_folder=str(tmp_path / "audio"),
            confidence_threshold=0.5,
            detection_mode="windowed",
        )
    )

    # Expected metrics the candidate would have reported on the train split
    # during autoresearch. With well-separated clusters these should match
    # the replayed results exactly.
    expected_train_metrics = {
        "threshold": 0.5,
        "precision": 1.0,
        "recall": 1.0,
        "fp_rate": 0.0,
        "high_conf_fp_rate": 0.0,
        "tp": n_pos,
        "fp": 0,
        "fn": 0,
        "tn": n_neg,
    }

    candidate = AutoresearchCandidate(
        id="cand-svm",
        name="SVM Candidate",
        status="training",
        manifest_path=str(manifest_path),
        best_run_path=str(tmp_path / "best_run.json"),
        promoted_config=json.dumps(
            {
                "classifier": "linear_svm",
                "feature_norm": "standard",
                "prob_calibration": "none",
                "context_pooling": "center",
                "class_weight_pos": 1.0,
                "class_weight_neg": 1.0,
                "hard_negative_fraction": 0.0,
                "threshold": 0.5,
                "seed": 42,
            }
        ),
        source_model_id=source_model.id,
        source_model_name=source_model.name,
        training_job_id="train-job-svm",
        is_reproducible_exact=True,
    )
    session.add(candidate)
    session.add(
        ClassifierTrainingJob(
            id="train-job-svm",
            status="running",
            name="svm-candidate-model",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            parameters=json.dumps(
                {
                    "classifier_type": "linear_svm",
                    "feature_norm": "standard",
                    "class_weight": {"0": 1.0, "1": 1.0},
                }
            ),
            source_mode="autoresearch_candidate",
            source_candidate_id=candidate.id,
            source_model_id=source_model.id,
            manifest_path=str(manifest_path),
            training_split_name="train",
            promoted_config=candidate.promoted_config,
            source_comparison_context=json.dumps(
                {
                    "candidate_id": candidate.id,
                    "candidate_name": candidate.name,
                    "split_metrics": {"train": expected_train_metrics},
                }
            ),
        )
    )
    await session.commit()

    job = await session.get(ClassifierTrainingJob, "train-job-svm")
    assert job is not None

    await run_training_job(session, job, settings)

    await session.refresh(candidate)
    await session.refresh(job)

    assert job.status == "complete"
    assert job.classifier_model_id is not None

    model = await session.get(ClassifierModel, job.classifier_model_id)
    assert model is not None

    summary = json.loads(model.training_summary or "{}")
    eff = summary["replay_effective_config"]
    assert eff["classifier_type"] == "linear_svm"
    assert eff["feature_norm"] == "standard"
    assert eff["prob_calibration"] == "none"

    verification = summary["replay_verification"]
    assert verification["status"] == "verified"
    assert verification["splits"]["train"]["pass"] is True

    # The saved pipeline must expose predict_proba for the detection path.
    import joblib

    loaded_pipeline = joblib.load(model.model_path)
    test_input = rng.randn(5, dim).astype(np.float32)
    probs = loaded_pipeline.predict_proba(test_input)
    assert probs.shape == (5, 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


# ---- Detection-manifest training path ----


async def test_run_training_job_detection_manifest(
    session,
    settings,
    monkeypatch,
) -> None:
    """Detection-manifest training builds a manifest, trains, and produces a model."""
    from humpback.classifier.detection_rows import write_detection_row_store
    from humpback.config import Settings
    from humpback.models.model_registry import ModelConfig
    from humpback.storage import detection_embeddings_path, detection_row_store_path

    # Patch Settings.from_repo_env so generate_manifest finds the test storage.
    monkeypatch.setattr(
        Settings,
        "from_repo_env",
        classmethod(lambda cls: settings),
    )

    model_version = "perch_v2"
    dim = 4

    # Seed model config (needed by generate_manifest's detection path).
    session.add(
        ModelConfig(
            name=model_version,
            display_name="Perch v2",
            path="models/perch_v2.tflite",
            model_type="tflite",
            input_format="waveform",
            vector_dim=dim,
            is_default=False,
        )
    )
    cm = ClassifierModel(
        name="source-classifier",
        model_path="/tmp/fake.joblib",
        model_version=model_version,
        vector_dim=dim,
        window_size_seconds=5.0,
        target_sample_rate=32000,
    )
    session.add(cm)
    await session.flush()

    dj = DetectionJob(
        status="complete",
        classifier_model_id=cm.id,
        audio_folder="/tmp/fake",
        confidence_threshold=0.5,
        detection_mode="windowed",
        has_positive_labels=True,
    )
    session.add(dj)
    await session.flush()

    # Write row store with labeled rows.
    rs_path = detection_row_store_path(settings.storage_root, dj.id)
    rs_path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(
        rs_path,
        [
            {
                "row_id": "r1",
                "start_utc": "1000.0",
                "end_utc": "1005.0",
                "avg_confidence": "0.9",
                "peak_confidence": "0.95",
                "n_windows": "1",
                "humpback": "1",
                "orca": "",
                "ship": "",
                "background": "",
            },
            {
                "row_id": "r2",
                "start_utc": "1005.0",
                "end_utc": "1010.0",
                "avg_confidence": "0.9",
                "peak_confidence": "0.95",
                "n_windows": "1",
                "humpback": "1",
                "orca": "",
                "ship": "",
                "background": "",
            },
            {
                "row_id": "r3",
                "start_utc": "2000.0",
                "end_utc": "2005.0",
                "avg_confidence": "0.3",
                "peak_confidence": "0.4",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "",
                "background": "1",
            },
            {
                "row_id": "r4",
                "start_utc": "2005.0",
                "end_utc": "2010.0",
                "avg_confidence": "0.2",
                "peak_confidence": "0.3",
                "n_windows": "1",
                "humpback": "",
                "orca": "",
                "ship": "1",
                "background": "",
            },
        ],
    )

    # Write detection embeddings at the model-versioned path.
    emb_path = detection_embeddings_path(settings.storage_root, dj.id, model_version)
    _write_detection_embeddings_parquet(
        emb_path,
        row_ids=["r1", "r2", "r3", "r4"],
        rows=[
            [1.0, 1.0, 0.0, 0.0],
            [0.9, 1.1, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.9, 1.1],
        ],
    )

    # Create a detection_manifest training job.
    job = ClassifierTrainingJob(
        status="running",
        name="detection-manifest-test",
        model_version=model_version,
        window_size_seconds=5.0,
        target_sample_rate=32000,
        source_mode="detection_manifest",
        source_detection_job_ids=json.dumps([dj.id]),
    )
    session.add(job)
    await session.commit()

    await run_training_job(session, job, settings)

    await session.refresh(job)
    assert job.status == "complete"
    assert job.classifier_model_id is not None
    assert job.manifest_path is not None

    model = await session.get(ClassifierModel, job.classifier_model_id)
    assert model is not None
    assert model.model_version == model_version
    assert model.vector_dim == dim
    assert model.training_source_mode == "detection_manifest"
    assert model.window_size_seconds == 5.0
    assert model.target_sample_rate == 32000

    # Verify training summary describes the detection source.
    summary = json.loads(model.training_summary or "{}")
    assert summary["training_source_mode"] == "detection_manifest"
    assert dj.id in summary["detection_job_ids"]
    assert "manifest_path" in summary

    # Verify the model artifact is loadable and can predict.
    import joblib
    import numpy as np

    loaded_pipeline = joblib.load(model.model_path)
    probs = loaded_pipeline.predict_proba(
        np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert probs.shape == (1, 2)


async def test_run_training_job_legacy_embedding_sets_fail_fast(
    session,
    settings,
) -> None:
    """Legacy embedding-set training jobs fail with the retirement message."""
    job = ClassifierTrainingJob(
        status="running",
        name="legacy-embedding-sets",
        model_version="perch_v2",
        window_size_seconds=5.0,
        target_sample_rate=32000,
        source_mode="embedding_sets",
    )
    session.add(job)
    await session.commit()

    await run_training_job(session, job, settings)

    await session.refresh(job)
    assert job.status == "failed"
    assert job.error_message is not None
    assert "Embedding-set classifier training jobs are retired" in job.error_message
    assert job.classifier_model_id is None


# ---------------------------------------------------------------------------
# _merge_candidate_standard_metrics
# ---------------------------------------------------------------------------


class TestMergeCandidateStandardMetrics:
    def test_merges_sample_counts(self):
        from humpback.workers.classifier_worker.training import (
            _merge_candidate_standard_metrics,
        )

        summary: dict = {
            "training_data_source": {"positive_count": 248, "negative_count": 216},
        }
        _merge_candidate_standard_metrics(summary, None)
        assert summary["n_positive"] == 248
        assert summary["n_negative"] == 216
        assert summary["balance_ratio"] == round(248 / 216, 4)

    def test_merges_metrics_from_split_metrics(self):
        from humpback.workers.classifier_worker.training import (
            _merge_candidate_standard_metrics,
        )

        summary: dict = {
            "training_data_source": {"positive_count": 100, "negative_count": 50},
        }
        provenance = {
            "split_metrics": {
                "test": {
                    "autoresearch": {
                        "precision": 0.95,
                        "recall": 0.90,
                        "tp": 90,
                        "fp": 5,
                        "fn": 10,
                        "tn": 45,
                    },
                },
            },
            "trainer_parameters": {
                "classifier_type": "logistic_regression",
                "class_weight": {"0": 1.0, "1": 3.0},
            },
        }
        _merge_candidate_standard_metrics(summary, provenance)

        assert summary["cv_precision"] == 0.95
        assert summary["cv_recall"] == 0.90
        expected_f1 = round(2 * 0.95 * 0.90 / (0.95 + 0.90), 6)
        assert summary["cv_f1"] == expected_f1
        expected_acc = round((90 + 45) / (90 + 5 + 10 + 45), 6)
        assert summary["cv_accuracy"] == expected_acc
        assert summary["train_confusion"] == {"tp": 90, "fp": 5, "fn": 10, "tn": 45}
        assert summary["classifier_type"] == "logistic_regression"
        assert summary["effective_class_weights"] == {"0": 1.0, "1": 3.0}

    def test_no_std_or_roc_fields(self):
        from humpback.workers.classifier_worker.training import (
            _merge_candidate_standard_metrics,
        )

        summary: dict = {
            "training_data_source": {"positive_count": 10, "negative_count": 10},
        }
        provenance = {
            "split_metrics": {
                "test": {
                    "autoresearch": {
                        "precision": 0.8,
                        "recall": 0.7,
                        "tp": 7,
                        "fp": 2,
                        "fn": 3,
                        "tn": 8,
                    }
                },
            },
            "trainer_parameters": {},
        }
        _merge_candidate_standard_metrics(summary, provenance)

        assert "cv_roc_auc" not in summary
        assert "cv_accuracy_std" not in summary
        assert "score_separation" not in summary
        assert "n_cv_folds" not in summary
