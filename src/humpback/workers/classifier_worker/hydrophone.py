"""Hydrophone detection job execution with subprocess support."""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import queue
import shutil
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, cast

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.classifier.detection_rows import (
    append_detection_row_store,
    ensure_detection_row_store,
    read_detection_row_store,
)
from humpback.classifier.detector import (
    read_window_diagnostics_table,
    write_window_diagnostics_shard,
)
from humpback.config import Settings, get_archive_source
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.storage import (
    detection_dir,
    detection_row_store_path,
    ensure_dir,
)
from humpback.workers.classifier_worker.detection import _detection_dicts_to_store_rows
from humpback.workers.queue import complete_detection_job, fail_detection_job

logger = logging.getLogger(__name__)

_PKG = "humpback.workers.classifier_worker"


def _pkg():
    return sys.modules[_PKG]


# ---- Helpers ----


def _peak_rss_mb() -> float | None:
    """Return the current process peak RSS in MiB when available."""
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None

    if rss <= 0:
        return None

    # macOS reports bytes; Linux reports KiB.
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return round(rss / divisor, 2)


def _avg_audio_x_realtime(summary: dict[str, Any]) -> float | None:
    """Compute end-to-end audio-throughput multiple from a hydrophone summary."""
    time_covered = float(summary.get("time_covered_sec") or 0.0)
    total_measured = (
        float(summary.get("fetch_sec") or 0.0)
        + float(summary.get("decode_sec") or 0.0)
        + float(summary.get("pipeline_total_sec") or 0.0)
    )
    if time_covered <= 0 or total_measured <= 0:
        return None
    return round(time_covered / total_measured, 2)


def _hydrophone_provider_mode(
    source_id: str,
    *,
    local_cache_path: str | None,
    s3_cache_path: str | None,
    noaa_cache_path: str | None,
) -> str:
    """Describe the archive-provider mode used for a hydrophone job."""
    source = get_archive_source(source_id)
    if source is None:
        raise ValueError(f"Unknown archive source: {source_id}")

    provider_kind = source["provider_kind"]
    if provider_kind == "orcasound_hls":
        if local_cache_path:
            return "local_cache_only"
        if s3_cache_path:
            return "s3_write_through_cache"
        return "direct_s3"

    if provider_kind == "noaa_gcs":
        if noaa_cache_path:
            return "noaa_cache"
        return "direct_gcs"

    return provider_kind


def _augment_hydrophone_summary(
    summary: dict[str, Any],
    *,
    provider_mode: str,
    execution_mode: str,
    peak_worker_rss_mb: float | None = None,
    child_pid: int | None = None,
) -> dict[str, Any]:
    """Attach runtime metadata used for hydrophone diagnostics."""
    summary_data = dict(summary)
    summary_data["provider_mode"] = provider_mode
    summary_data["execution_mode"] = execution_mode

    avg_audio_x_realtime = _avg_audio_x_realtime(summary_data)
    if avg_audio_x_realtime is not None:
        summary_data["avg_audio_x_realtime"] = avg_audio_x_realtime

    if peak_worker_rss_mb is not None:
        summary_data["peak_worker_rss_mb"] = peak_worker_rss_mb
    if child_pid is not None:
        summary_data["child_pid"] = child_pid

    return summary_data


def _resolve_model_runtime(
    model_version: str,
    model_config: Any,
    settings: Settings,
) -> dict[str, Any]:
    """Build a serializable runtime spec for the embedding model."""
    if model_config is not None:
        return {
            "model_version": model_version,
            "model_type": model_config.model_type,
            "input_format": model_config.input_format,
            "model_path": model_config.path,
            "vector_dim": int(model_config.vector_dim),
        }

    return {
        "model_version": model_version,
        "model_type": "tflite",
        "input_format": "spectrogram",
        "model_path": settings.model_path,
        "vector_dim": int(settings.vector_dim),
    }


def _load_embedding_model_from_runtime(
    runtime: dict[str, Any], settings: dict[str, Any]
):
    """Load an embedding model from a serializable runtime spec."""
    input_format = str(runtime["input_format"])
    vector_dim = int(runtime["vector_dim"])

    if not bool(settings["use_real_model"]):
        from humpback.processing.inference import FakeTF2Model, FakeTFLiteModel

        model = (
            FakeTF2Model(vector_dim)
            if input_format == "waveform"
            else FakeTFLiteModel(vector_dim)
        )
        return model, input_format

    model_type = str(runtime["model_type"])
    model_path = str(runtime["model_path"])
    if model_type == "tf2_saved_model":
        from humpback.processing.inference import TF2SavedModel

        model = TF2SavedModel(
            model_path,
            vector_dim,
            force_cpu=bool(settings["tf_force_cpu"]),
        )
    else:
        from humpback.processing.inference import TFLiteModel

        model = TFLiteModel(model_path, vector_dim)

    return model, input_format


# ---- Subprocess ----


def _hydrophone_detection_subprocess_main(
    *,
    event_queue: Any,
    cancel_event: Any,
    pause_gate: Any,
    runtime: dict[str, Any],
) -> None:
    """Run hydrophone detection in a short-lived child process."""
    try:
        import joblib

        from humpback.classifier.hydrophone_detector import run_hydrophone_detection
        from humpback.classifier.providers import build_archive_detection_provider

        pipeline = joblib.load(str(runtime["classifier_model_path"]))
        model, input_format = _load_embedding_model_from_runtime(
            runtime["model_runtime"],
            cast(dict[str, Any], runtime["settings"]),
        )
        provider = build_archive_detection_provider(
            str(runtime["hydrophone_id"]),
            local_cache_path=cast(str | None, runtime["local_cache_path"]),
            s3_cache_path=cast(str | None, runtime["s3_cache_path"]),
            noaa_cache_path=cast(str | None, runtime["noaa_cache_path"]),
        )

        def on_chunk_complete(
            chunk_detections: list[dict],
            segments_done: int,
            segments_total: int,
            time_covered_sec: float,
        ) -> None:
            event_queue.put(
                {
                    "type": "progress",
                    "chunk_detections": chunk_detections,
                    "segments_done": segments_done,
                    "segments_total": segments_total,
                    "time_covered_sec": time_covered_sec,
                }
            )

        def on_chunk_diagnostics(chunk_records: list[dict], segments_done: int) -> None:
            event_queue.put(
                {
                    "type": "diagnostics",
                    "chunk_records": chunk_records,
                    "segments_done": segments_done,
                }
            )

        def on_chunk_embeddings(chunk_records: list[dict]) -> None:
            event_queue.put({"type": "embeddings", "chunk_records": chunk_records})

        def on_alert(alert: dict) -> None:
            event_queue.put({"type": "alert", "alert": alert})

        def on_resume_invalidation() -> None:
            event_queue.put({"type": "resume_invalidated"})

        detections, summary = run_hydrophone_detection(
            provider=provider,
            start_timestamp=float(runtime["start_timestamp"]),
            end_timestamp=float(runtime["end_timestamp"]),
            pipeline=pipeline,
            model=model,
            window_size_seconds=float(runtime["window_size_seconds"]),
            target_sample_rate=int(runtime["target_sample_rate"]),
            confidence_threshold=float(runtime["confidence_threshold"]),
            input_format=input_format,
            feature_config=cast(dict[str, Any] | None, runtime["feature_config"]),
            hop_seconds=float(runtime["hop_seconds"]),
            high_threshold=float(runtime["high_threshold"]),
            low_threshold=float(runtime["low_threshold"]),
            on_chunk_complete=on_chunk_complete,
            on_chunk_diagnostics=on_chunk_diagnostics,
            on_chunk_embeddings=on_chunk_embeddings,
            on_alert=on_alert,
            cancel_check=cancel_event.is_set,
            pause_gate=pause_gate,
            skip_segments=int(runtime["skip_segments"]),
            prior_detections=cast(list[dict], runtime["prior_detections"]),
            on_resume_invalidation=on_resume_invalidation,
            prefetch_enabled=bool(runtime["prefetch_enabled"]),
            prefetch_workers=int(runtime["prefetch_workers"]),
            prefetch_inflight_segments=int(runtime["prefetch_inflight_segments"]),
            detection_mode=runtime.get("detection_mode"),
            window_selection=runtime.get("window_selection"),
            min_prominence=runtime.get("min_prominence"),
        )
        event_queue.put(
            {
                "type": "result",
                "detections": detections,
                "summary": summary,
                "peak_worker_rss_mb": _peak_rss_mb(),
                "child_pid": os.getpid(),
            }
        )
    except BaseException as exc:  # pragma: no cover - exercised via parent path
        event_queue.put(
            {
                "type": "error",
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


async def _run_hydrophone_detection_in_subprocess(
    *,
    runtime: dict[str, Any],
    cancel_event: Any,
    pause_gate: Any,
    on_chunk_complete,
    on_chunk_diagnostics,
    on_chunk_embeddings,
    on_alert,
    on_resume_invalidation,
) -> tuple[list[dict], dict]:
    """Execute hydrophone detection in a spawned child and proxy its events."""
    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    child_cancel_event = ctx.Event()
    child_pause_gate = ctx.Event()
    if cancel_event.is_set():
        child_cancel_event.set()
    if pause_gate.is_set():
        child_pause_gate.set()

    proc = ctx.Process(
        target=_hydrophone_detection_subprocess_main,
        kwargs={
            "event_queue": event_queue,
            "cancel_event": child_cancel_event,
            "pause_gate": child_pause_gate,
            "runtime": runtime,
        },
        name=f"hydrophone-detect-{runtime['job_id']}",
    )
    proc.start()

    async def _sync_control_state() -> None:
        last_pause_state: bool | None = None
        while proc.is_alive():
            if cancel_event.is_set() and not child_cancel_event.is_set():
                child_cancel_event.set()
            pause_state = bool(pause_gate.is_set())
            if pause_state != last_pause_state:
                if pause_state:
                    child_pause_gate.set()
                else:
                    child_pause_gate.clear()
                last_pause_state = pause_state
            await asyncio.sleep(0.1)

    control_task = asyncio.create_task(_sync_control_state())
    detections: list[dict] | None = None
    summary: dict[str, Any] | None = None
    error_message: dict[str, Any] | None = None

    try:
        while True:
            try:
                message = await asyncio.to_thread(event_queue.get, True, 0.5)
            except queue.Empty:
                if not proc.is_alive():
                    break
                continue

            msg_type = str(message.get("type", ""))
            if msg_type == "progress":
                on_chunk_complete(
                    cast(list[dict], message["chunk_detections"]),
                    int(message["segments_done"]),
                    int(message["segments_total"]),
                    float(message["time_covered_sec"]),
                )
                continue

            if msg_type == "diagnostics":
                on_chunk_diagnostics(
                    cast(list[dict], message["chunk_records"]),
                    int(message["segments_done"]),
                )
                continue

            if msg_type == "embeddings":
                on_chunk_embeddings(cast(list[dict], message["chunk_records"]))
                continue

            if msg_type == "alert":
                on_alert(cast(dict[str, Any], message["alert"]))
                continue

            if msg_type == "resume_invalidated":
                on_resume_invalidation()
                continue

            if msg_type == "result":
                detections = cast(list[dict], message["detections"])
                summary = cast(dict[str, Any], message["summary"])
                peak_worker_rss_mb = message.get("peak_worker_rss_mb")
                child_pid = message.get("child_pid")
                if peak_worker_rss_mb is not None:
                    summary["peak_worker_rss_mb"] = float(peak_worker_rss_mb)
                if child_pid is not None:
                    summary["child_pid"] = int(child_pid)
                break

            if msg_type == "error":
                error_message = cast(dict[str, Any], message)
                break

        if error_message is not None:
            error_type = str(error_message.get("error_type") or "RuntimeError")
            error_text = str(
                error_message.get("error") or "Hydrophone detection subprocess failed"
            )
            if error_type == "FileNotFoundError":
                raise FileNotFoundError(error_text)

            traceback_text = str(error_message.get("traceback") or "").strip()
            if traceback_text:
                raise RuntimeError(f"{error_text}\n{traceback_text}")
            raise RuntimeError(error_text)

        if detections is None or summary is None:
            raise RuntimeError(
                "Hydrophone detection subprocess exited without a result "
                f"(exitcode={proc.exitcode})"
            )

        return detections, summary
    finally:
        control_task.cancel()
        try:
            await control_task
        except asyncio.CancelledError:
            pass

        await asyncio.to_thread(proc.join, 5.0)
        if proc.is_alive():
            proc.terminate()
            await asyncio.to_thread(proc.join, 5.0)

        event_queue.close()
        event_queue.join_thread()


# ---- Main job runner ----


async def run_hydrophone_detection_job(
    session: AsyncSession,
    job: DetectionJob,
    settings: Settings,
    session_factory=None,
) -> None:
    """Execute a hydrophone detection job end-to-end."""
    import threading

    try:
        from sqlalchemy import select

        # Load classifier model
        result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
        )
        cm = result.scalar_one()
        registry_model = await _pkg().get_model_by_name(session, cm.model_version)
        model_runtime = _resolve_model_runtime(
            cm.model_version, registry_model, settings
        )
        input_format = str(model_runtime["input_format"])
        use_subprocess = (
            str(model_runtime["model_type"]) == "tf2_saved_model"
            and input_format == "waveform"
        )
        pipeline = None
        model = None
        if not use_subprocess:
            pipeline = _pkg().joblib.load(cm.model_path)
            model, input_format = await _pkg().get_model_by_version(
                session, cm.model_version, settings
            )

        feature_config = json.loads(cm.feature_config) if cm.feature_config else None

        # Set up output directory and row store path
        ddir = ensure_dir(detection_dir(settings.storage_root, job.id))
        rs_path = detection_row_store_path(settings.storage_root, job.id)
        diag_path = ddir / "window_diagnostics.parquet"

        await session.commit()

        # Resume support: if segments were already processed, read prior detections
        skip_segments = 0
        prior_detections: list[dict] = []
        if job.segments_processed and job.segments_processed > 0 and rs_path.is_file():
            _, prior_rows = read_detection_row_store(rs_path)
            # Row-store dicts (all-string, ROW_STORE_FIELDNAMES) join
            # all_detections alongside raw detector dicts (mixed types,
            # fewer fields).  Safe because all_detections is only used for
            # counting (summary) and is no longer persisted from the return
            # value — the incremental Parquet writes are the source of truth.
            prior_detections = prior_rows  # type: ignore[assignment]
            skip_segments = job.segments_processed
            logger.info(
                "Resuming hydrophone job %s from segment %d with %d prior detections",
                job.id,
                skip_segments,
                len(prior_detections),
            )
        elif diag_path.exists():
            if diag_path.is_dir():
                shutil.rmtree(diag_path)
            else:
                diag_path.unlink()

        # Cancel and pause support
        cancel_event = threading.Event()
        pause_gate = threading.Event()
        pause_gate.set()  # Initially not paused (set = open gate)
        loop = asyncio.get_event_loop()

        def _fmt_utc(ts: float | None) -> str:
            if ts is None:
                return "unknown"
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

        if (
            job.hydrophone_id is None
            or job.start_timestamp is None
            or job.end_timestamp is None
        ):
            raise ValueError(
                "Hydrophone detection job missing hydrophone_id or time bounds"
            )
        hydrophone_id = job.hydrophone_id
        start_timestamp = job.start_timestamp
        end_timestamp = job.end_timestamp
        hydrophone_provider = _pkg().build_archive_detection_provider(
            hydrophone_id,
            local_cache_path=job.local_cache_path,
            s3_cache_path=settings.s3_cache_path,
            noaa_cache_path=settings.noaa_cache_path,
        )
        provider_mode = _hydrophone_provider_mode(
            hydrophone_id,
            local_cache_path=job.local_cache_path,
            s3_cache_path=settings.s3_cache_path,
            noaa_cache_path=settings.noaa_cache_path,
        )
        execution_mode = "subprocess" if use_subprocess else "in_process"
        logger.info(
            "Hydrophone detection job %s using provider_mode=%s execution_mode=%s",
            job.id,
            provider_mode,
            execution_mode,
        )

        # Progress callback
        def on_chunk_complete(
            chunk_detections: list[dict],
            segments_done: int,
            segments_total: int,
            time_covered_sec: float,
        ):
            if chunk_detections:
                store_rows = _detection_dicts_to_store_rows(chunk_detections)
                append_detection_row_store(rs_path, store_rows)

            if session_factory is not None:

                async def _update_progress():
                    try:
                        async with cast(Any, session_factory)() as progress_session:
                            await progress_session.execute(
                                update(DetectionJob)
                                .where(DetectionJob.id == job.id)
                                .values(
                                    segments_processed=segments_done,
                                    segments_total=segments_total,
                                    time_covered_sec=time_covered_sec,
                                )
                            )
                            await progress_session.commit()
                    except Exception:
                        logger.debug(
                            "Failed to update hydrophone progress", exc_info=True
                        )

                loop.call_soon_threadsafe(asyncio.ensure_future, _update_progress())

        def on_chunk_diagnostics(chunk_records: list[dict], segments_done: int):
            if not chunk_records:
                return
            write_window_diagnostics_shard(
                chunk_records,
                diag_path,
                f"part-{segments_done:06d}.parquet",
            )

        accumulated_embedding_records: list[dict] = []

        def on_chunk_embeddings(chunk_emb_records: list[dict]) -> None:
            accumulated_embedding_records.extend(chunk_emb_records)

        def on_resume_invalidation():
            if not diag_path.exists():
                return
            if diag_path.is_dir():
                shutil.rmtree(diag_path)
            else:
                diag_path.unlink()

        # Alert callback
        alerts_list: list[dict] = []

        def on_alert(alert: dict):
            alerts_list.append(alert)
            if session_factory is not None:

                async def _update_alerts():
                    try:
                        async with cast(Any, session_factory)() as alert_session:
                            await alert_session.execute(
                                update(DetectionJob)
                                .where(DetectionJob.id == job.id)
                                .values(alerts=json.dumps(alerts_list))
                            )
                            await alert_session.commit()
                    except Exception:
                        logger.debug("Failed to update alerts", exc_info=True)

                loop.call_soon_threadsafe(asyncio.ensure_future, _update_alerts())

        # Poll for cancellation/pause in background
        async def _poll_cancel():
            while not cancel_event.is_set():
                await asyncio.sleep(2)
                try:
                    if session_factory is not None:
                        async with cast(Any, session_factory)() as poll_session:
                            result = await poll_session.execute(
                                select(DetectionJob.status).where(
                                    DetectionJob.id == job.id
                                )
                            )
                            status = result.scalar_one_or_none()
                            if status == "canceled":
                                cancel_event.set()
                                pause_gate.set()  # Unblock if paused so thread can exit
                                return
                            elif status == "paused":
                                pause_gate.clear()  # Block the detection thread
                            elif status == "running":
                                pause_gate.set()  # Unblock the detection thread
                except Exception:
                    pass

        cancel_task = asyncio.ensure_future(_poll_cancel())
        try:
            if use_subprocess:
                (
                    detections,
                    summary,
                ) = await _pkg()._run_hydrophone_detection_in_subprocess(
                    runtime={
                        "job_id": job.id,
                        "classifier_model_path": cm.model_path,
                        "model_runtime": model_runtime,
                        "settings": {
                            "use_real_model": settings.use_real_model,
                            "tf_force_cpu": settings.tf_force_cpu,
                        },
                        "hydrophone_id": hydrophone_id,
                        "local_cache_path": job.local_cache_path,
                        "s3_cache_path": settings.s3_cache_path,
                        "noaa_cache_path": settings.noaa_cache_path,
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "window_size_seconds": cm.window_size_seconds,
                        "target_sample_rate": cm.target_sample_rate,
                        "confidence_threshold": job.confidence_threshold,
                        "feature_config": feature_config,
                        "hop_seconds": job.hop_seconds,
                        "high_threshold": job.high_threshold,
                        "low_threshold": job.low_threshold,
                        "skip_segments": skip_segments,
                        "prior_detections": prior_detections,
                        "prefetch_enabled": settings.hydrophone_prefetch_enabled,
                        "prefetch_workers": settings.hydrophone_prefetch_workers,
                        "prefetch_inflight_segments": settings.hydrophone_prefetch_inflight_segments,
                        "detection_mode": job.detection_mode,
                        "window_selection": job.window_selection,
                        "min_prominence": job.min_prominence,
                    },
                    cancel_event=cancel_event,
                    pause_gate=pause_gate,
                    on_chunk_complete=on_chunk_complete,
                    on_chunk_diagnostics=on_chunk_diagnostics,
                    on_chunk_embeddings=on_chunk_embeddings,
                    on_alert=on_alert,
                    on_resume_invalidation=on_resume_invalidation,
                )
            else:
                from humpback.classifier.hydrophone_detector import (
                    run_hydrophone_detection,
                )

                assert pipeline is not None
                assert model is not None
                detections, summary = await asyncio.to_thread(
                    run_hydrophone_detection,
                    provider=hydrophone_provider,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    pipeline=pipeline,
                    model=model,
                    window_size_seconds=cm.window_size_seconds,
                    target_sample_rate=cm.target_sample_rate,
                    confidence_threshold=job.confidence_threshold,
                    input_format=input_format,
                    feature_config=feature_config,
                    hop_seconds=job.hop_seconds,
                    high_threshold=job.high_threshold,
                    low_threshold=job.low_threshold,
                    on_chunk_complete=on_chunk_complete,
                    on_chunk_diagnostics=on_chunk_diagnostics,
                    on_chunk_embeddings=on_chunk_embeddings,
                    on_alert=on_alert,
                    cancel_check=cancel_event.is_set,
                    pause_gate=pause_gate,
                    skip_segments=skip_segments,
                    prior_detections=prior_detections,
                    on_resume_invalidation=on_resume_invalidation,
                    prefetch_enabled=settings.hydrophone_prefetch_enabled,
                    prefetch_workers=settings.hydrophone_prefetch_workers,
                    prefetch_inflight_segments=settings.hydrophone_prefetch_inflight_segments,
                    detection_mode=job.detection_mode,
                    window_selection=job.window_selection,
                    min_prominence=job.min_prominence,
                )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No hydrophone audio segments found for hydrophone "
                f"'{hydrophone_id}' in requested UTC range "
                f"[{_fmt_utc(start_timestamp)}, {_fmt_utc(end_timestamp)}]"
            ) from exc
        finally:
            cancel_task.cancel()

        def _mark_has_diagnostics(summary_data: dict) -> None:
            if not diag_path.exists():
                return
            try:
                read_window_diagnostics_table(diag_path)
                summary_data["has_diagnostics"] = True
            except Exception:
                logger.debug(
                    "Failed to read persisted hydrophone diagnostics", exc_info=True
                )

        peak_worker_rss_mb = (
            float(summary["peak_worker_rss_mb"])
            if "peak_worker_rss_mb" in summary
            else _peak_rss_mb()
        )
        child_pid = int(summary["child_pid"]) if "child_pid" in summary else None

        if cancel_event.is_set():
            # Finalize row store with diagnostics enrichment
            ensure_detection_row_store(
                row_store_path=rs_path,
                diagnostics_path=diag_path if diag_path.exists() else None,
                window_size_seconds=cm.window_size_seconds,
                refresh_existing=True,
                detection_mode=job.detection_mode,
            )
            # Write partial detection embeddings on cancel (after row store)
            if accumulated_embedding_records:
                from humpback.classifier.detector import (
                    match_embedding_records_to_row_store,
                    write_detection_embeddings,
                )

                _, rs_rows = read_detection_row_store(rs_path)
                accumulated_embedding_records = match_embedding_records_to_row_store(
                    accumulated_embedding_records, rs_rows
                )
                emb_path = ddir / "detection_embeddings.parquet"
                write_detection_embeddings(accumulated_embedding_records, emb_path)
                summary["has_detection_embeddings"] = True
            _mark_has_diagnostics(summary)
            summary = _augment_hydrophone_summary(
                summary,
                provider_mode=provider_mode,
                execution_mode=execution_mode,
                peak_worker_rss_mb=peak_worker_rss_mb,
                child_pid=child_pid,
            )
            await session.execute(
                update(DetectionJob)
                .where(DetectionJob.id == job.id)
                .values(
                    status="canceled",
                    result_summary=json.dumps(summary),
                    alerts=json.dumps(alerts_list) if alerts_list else None,
                    updated_at=__import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ),
                )
            )
            await session.commit()
            return

        # Finalize row store with diagnostics enrichment
        ensure_detection_row_store(
            row_store_path=rs_path,
            diagnostics_path=diag_path if diag_path.exists() else None,
            window_size_seconds=cm.window_size_seconds,
            refresh_existing=True,
            detection_mode=job.detection_mode,
        )

        # Write detection embeddings (after row store so row_ids are available)
        if accumulated_embedding_records:
            from humpback.classifier.detector import (
                match_embedding_records_to_row_store,
                write_detection_embeddings,
            )

            _, rs_rows = read_detection_row_store(rs_path)
            accumulated_embedding_records = match_embedding_records_to_row_store(
                accumulated_embedding_records, rs_rows
            )
            emb_path = ddir / "detection_embeddings.parquet"
            write_detection_embeddings(accumulated_embedding_records, emb_path)
            summary["has_detection_embeddings"] = True

        _mark_has_diagnostics(summary)
        summary = _augment_hydrophone_summary(
            summary,
            provider_mode=provider_mode,
            execution_mode=execution_mode,
            peak_worker_rss_mb=peak_worker_rss_mb,
            child_pid=child_pid,
        )
        summary_path = ddir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        await session.execute(
            update(DetectionJob)
            .where(DetectionJob.id == job.id)
            .values(
                result_summary=json.dumps(summary),
                alerts=json.dumps(alerts_list) if alerts_list else None,
            )
        )
        await complete_detection_job(session, job.id)

    except Exception as e:
        logger.exception("Hydrophone detection job %s failed", job.id)
        try:
            await session.rollback()
        except Exception:
            pass
        try:
            await fail_detection_job(session, job.id, str(e))
        except Exception:
            logger.exception("Failed to mark hydrophone detection job as failed")
