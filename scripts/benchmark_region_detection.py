"""Benchmark Perch inference speed on real hydrophone audio.

Times audio fetch + scoring on a short slice of hydrophone data and
extrapolates to a 24-hour estimate.

Usage::

    uv run python scripts/benchmark_region_detection.py \\
        --hydrophone-id orcasound_lab \\
        --start 2021-10-31T00:00:00Z \\
        --duration-minutes 10 \\
        --model-config-id MODEL_CONFIG_ID \\
        --classifier-model-id CLASSIFIER_MODEL_ID

"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import select

from humpback.classifier.detector import score_audio_windows
from humpback.classifier.providers import build_archive_detection_provider
from humpback.classifier.s3_stream import iter_audio_chunks
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel
from humpback.schemas.call_parsing import RegionDetectionConfig
from humpback.workers.model_cache import get_model_by_version

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark region detection inference speed"
    )
    p.add_argument("--hydrophone-id", required=True, help="Hydrophone source ID")
    p.add_argument(
        "--start",
        required=True,
        help="Start timestamp in ISO 8601 UTC (e.g. 2021-10-31T00:00:00Z)",
    )
    p.add_argument(
        "--duration-minutes",
        type=float,
        default=10.0,
        help="Duration of audio to benchmark in minutes (default: 10)",
    )
    p.add_argument("--model-config-id", required=True, help="Perch model config ID")
    p.add_argument("--classifier-model-id", required=True, help="Classifier model ID")
    return p.parse_args()


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = Settings()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    start_ts = start_dt.timestamp()
    duration_sec = args.duration_minutes * 60.0
    end_ts = start_ts + duration_sec

    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            cm_result = await session.execute(
                select(ClassifierModel).where(
                    ClassifierModel.id == args.classifier_model_id
                )
            )
            cm = cm_result.scalar_one_or_none()
            if cm is None:
                raise SystemExit(
                    f"ClassifierModel {args.classifier_model_id} not found"
                )

            target_sample_rate = cm.target_sample_rate
            pipeline = joblib.load(cm.model_path)
            perch_model, input_format = await get_model_by_version(
                session, cm.model_version, settings
            )
    finally:
        await engine.dispose()

    config = RegionDetectionConfig()
    det_config = {
        "window_size_seconds": config.window_size_seconds,
        "hop_seconds": config.hop_seconds,
        "input_format": input_format,
    }

    provider = build_archive_detection_provider(
        args.hydrophone_id,
        local_cache_path=None,
        s3_cache_path=settings.s3_cache_path,
        noaa_cache_path=settings.noaa_cache_path,
    )

    print(
        f"\nBenchmark: {args.hydrophone_id} | "
        f"{start_dt.strftime('%Y-%m-%dT%H:%MZ')} - "
        f"{(start_dt.replace(second=0)).strftime('%H:%M')}+{args.duration_minutes:.0f}m "
        f"({duration_sec:.0f}s)"
    )
    print("-" * 60)

    # Fetch audio
    t_fetch_start = time.monotonic()
    audio_chunks: list[np.ndarray] = []
    for audio_buf, _seg_start_utc, _segs_done, _segs_total in iter_audio_chunks(
        provider,
        start_ts,
        end_ts,
        chunk_seconds=duration_sec,
        target_sr=target_sample_rate,
    ):
        audio_chunks.append(audio_buf)
    t_fetch = time.monotonic() - t_fetch_start

    if not audio_chunks:
        raise SystemExit("No audio segments found for the given time range")

    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
    actual_duration = float(len(audio)) / float(target_sample_rate)
    print(f"  Audio fetch:    {t_fetch:.1f}s ({actual_duration:.1f}s of audio)")

    # Score
    t_score_start = time.monotonic()
    records = score_audio_windows(
        audio=audio,
        sample_rate=target_sample_rate,
        perch_model=perch_model,
        classifier=pipeline,
        config=det_config,
    )
    t_score = time.monotonic() - t_score_start

    t_total = t_fetch + t_score
    rate = t_total / (actual_duration / 60.0) if actual_duration > 0 else 0.0

    print(f"  Scoring:        {t_score:.1f}s  ({len(records)} windows)")
    print(f"  Total:          {t_total:.1f}s  ({rate:.2f}s per minute of audio)")

    # Extrapolate
    day_sec = 86400.0
    day_minutes = day_sec / 60.0
    est_wall_sec = rate * day_minutes
    est_chunks = int(day_sec / config.stream_chunk_sec)
    est_per_chunk = est_wall_sec / est_chunks if est_chunks > 0 else 0.0

    print("\nExtrapolated 24h estimate:")
    print(f"  Wall time:     ~{est_wall_sec / 60.0:.1f} min")
    print(f"  Chunks ({config.stream_chunk_sec / 60:.0f}m):  {est_chunks}")
    print(f"  Per chunk:     ~{est_per_chunk:.1f}s")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
