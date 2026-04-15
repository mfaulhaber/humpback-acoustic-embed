"""Bootstrap an event classifier from segmentation training data with random type labels.

Breaks the cold-start circular dependency in the Pass 3 feedback
training loop: the classifier needs labeled data from human corrections,
but corrections require a trained model to produce predictions.

This script reads human-corrected event boundaries from a segmentation
training dataset, assigns random vocalization type labels, trains an
``EventClassifierCNN``, and registers the resulting model in the
database.  The model is intentionally bad — its purpose is to produce
predictions that humans can correct in the Classify Review workspace,
bootstrapping the feedback loop.

Usage::

    uv run python scripts/bootstrap_classifier.py DATASET_ID

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.audio_loader import build_multi_source_event_audio_loader
from humpback.call_parsing.event_classifier.trainer import (
    EventClassifierTrainingConfig,
    EventClassifierTrainingResult,
    train_event_classifier,
)
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.ml.device import select_device
from humpback.models.segmentation_training import (
    SegmentationTrainingDataset,
    SegmentationTrainingSample,
)
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationType,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class _BootstrapSample:
    """Sample compatible with the event classifier trainer interface."""

    start_sec: float
    end_sec: float
    type_index: int
    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float


def flatten_events(
    samples: list[SegmentationTrainingSample],
) -> list[tuple[float, float, str, float, float]]:
    """Parse events_json from training samples into flat event tuples.

    Event times in ``events_json`` are already job-relative (offsets from
    ``start_timestamp``), the same coordinate system as ``crop_start_sec``.
    No offset adjustment is needed.

    Returns list of (start_sec, end_sec, hydrophone_id, start_timestamp, end_timestamp).
    """
    events: list[tuple[float, float, str, float, float]] = []
    for sample in samples:
        if (
            not sample.hydrophone_id
            or sample.start_timestamp is None
            or sample.end_timestamp is None
        ):
            continue
        parsed = json.loads(sample.events_json)
        for ev in parsed:
            events.append(
                (
                    ev["start_sec"],
                    ev["end_sec"],
                    sample.hydrophone_id,
                    sample.start_timestamp,
                    sample.end_timestamp,
                )
            )
    return events


def assign_random_types(
    n_events: int,
    n_types: int,
    min_per_type: int = 10,
    seed: int = 42,
) -> list[int]:
    """Assign random type indices guaranteeing minimum coverage per type.

    Uses round-robin assignment first to ensure every type has at least
    ``min_per_type`` events (or as many as possible if n_events is too
    small), then assigns remaining events uniformly at random.
    """
    rng = random.Random(seed)

    if n_events == 0 or n_types == 0:
        return []

    assignments: list[int] = []

    # Round-robin phase: guarantee minimum coverage
    round_robin_total = min(n_events, min_per_type * n_types)
    type_cycle = list(range(n_types)) * min_per_type
    rng.shuffle(type_cycle)
    assignments.extend(type_cycle[:round_robin_total])

    # Random phase: fill remaining slots
    remaining = n_events - len(assignments)
    for _ in range(remaining):
        assignments.append(rng.randint(0, n_types - 1))

    # Shuffle all assignments so round-robin samples aren't clustered
    rng.shuffle(assignments)

    return assignments


async def run_bootstrap(
    session: AsyncSession,
    *,
    dataset_id: str,
    settings: Settings,
) -> str:
    """Core bootstrap logic. Returns the registered model ID."""

    # Validate dataset
    dataset = await session.get(SegmentationTrainingDataset, dataset_id)
    if dataset is None:
        raise SystemExit(f"ERROR: dataset {dataset_id!r} not found")

    # Load samples
    result = await session.execute(
        select(SegmentationTrainingSample).where(
            SegmentationTrainingSample.training_dataset_id == dataset_id
        )
    )
    db_samples = list(result.scalars().all())
    if not db_samples:
        raise SystemExit(f"ERROR: dataset {dataset_id!r} has no samples")

    # Load vocabulary
    type_result = await session.execute(
        select(VocalizationType).order_by(VocalizationType.name)
    )
    vocab_types = list(type_result.scalars().all())
    if not vocab_types:
        raise SystemExit("ERROR: no vocalization types defined")
    vocabulary = [vt.name for vt in vocab_types]

    # Flatten events from samples
    flat_events = flatten_events(db_samples)
    if not flat_events:
        raise SystemExit("ERROR: no events found in dataset samples")

    logger.info(
        "Found %d events across %d samples, %d vocalization types",
        len(flat_events),
        len(db_samples),
        len(vocabulary),
    )

    # Assign random types
    type_assignments = assign_random_types(
        n_events=len(flat_events),
        n_types=len(vocabulary),
    )

    # Build trainer samples
    samples = [
        _BootstrapSample(
            start_sec=ev[0],
            end_sec=ev[1],
            type_index=type_idx,
            hydrophone_id=ev[2],
            start_timestamp=ev[3],
            end_timestamp=ev[4],
        )
        for ev, type_idx in zip(flat_events, type_assignments)
    ]

    # Prepare model directory
    model_id = str(uuid.uuid4())
    model_dir = settings.storage_root / "vocalization_models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train
    config = EventClassifierTrainingConfig()
    feature_config = SegmentationFeatureConfig()
    audio_loader = build_multi_source_event_audio_loader(
        target_sr=16000,
        settings=settings,
        samples=samples,
    )
    device = select_device()

    logger.info("Starting training with %d samples...", len(samples))

    training_result: EventClassifierTrainingResult = await asyncio.to_thread(
        train_event_classifier,
        samples=samples,
        vocabulary=vocabulary,
        feature_config=feature_config,
        audio_loader=audio_loader,
        config=config,
        model_dir=model_dir,
        device=device,
    )

    # Register model
    model = VocalizationClassifierModel(
        id=model_id,
        name=f"bootstrap-classifier-{model_id[:8]}",
        model_dir_path=str(model_dir),
        vocabulary_snapshot=json.dumps(training_result.vocabulary),
        per_class_thresholds=json.dumps(training_result.per_type_thresholds),
        per_class_metrics=json.dumps(training_result.per_type_metrics),
        training_summary=json.dumps(training_result.to_summary()),
        model_family="pytorch_event_cnn",
        input_mode="segmented_event",
    )
    session.add(model)
    await session.commit()

    return model_id


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap an event classifier from segmentation training "
        "data with random type labels.",
    )
    parser.add_argument(
        "dataset_id",
        help="UUID of a SegmentationTrainingDataset to source events from.",
    )
    return parser.parse_args(argv)


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv()
    settings = Settings()

    engine = create_engine(settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            model_id = await run_bootstrap(
                session,
                dataset_id=args.dataset_id,
                settings=settings,
            )
        print(f"\nBootstrap complete. Model ID: {model_id}")
    finally:
        await engine.dispose()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
