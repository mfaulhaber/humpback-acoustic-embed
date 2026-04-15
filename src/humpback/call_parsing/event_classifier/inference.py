"""Pass 3 event classifier inference.

Loads a trained ``EventClassifierCNN`` checkpoint and classifies a list
of ``Event`` dataclasses into ``TypedEvent`` rows by cropping audio at
each event's exact bounds, extracting log-mel features, running batch
inference, and applying per-type thresholds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from humpback.call_parsing.event_classifier.dataset import collate_fn
from humpback.call_parsing.event_classifier.model import EventClassifierCNN
from humpback.call_parsing.segmentation.features import (
    extract_logmel,
    normalize_per_region_zscore,
)
from humpback.call_parsing.types import Event, TypedEvent
from humpback.ml.checkpointing import load_checkpoint
from humpback.schemas.call_parsing import SegmentationFeatureConfig

logger = logging.getLogger(__name__)

EventAudioLoader = Callable[[Event], tuple[np.ndarray, float]]
"""Callable returning ``(audio, audio_start_sec)`` for one event.

``audio`` is a 1-D float array at the feature config's sample rate.
``audio_start_sec`` is the time offset of the first sample in the
source's coordinate system.  For file-based sources this is ``0.0``;
for hydrophone sources it is the start of the loaded audio chunk.
``_extract_event_features`` subtracts this offset so that
``event.start_sec`` maps to the correct sample index.
"""


def load_event_classifier(
    model_dir: Path,
) -> tuple[EventClassifierCNN, list[str], dict[str, float], SegmentationFeatureConfig]:
    """Load a trained event classifier from ``model_dir``.

    Returns ``(model, vocabulary, thresholds, feature_config)``.
    """
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    config = json.loads(config_path.read_text())
    vocabulary: list[str] = config["vocabulary"]
    n_types = config["n_types"]
    n_mels = config.get("n_mels", 64)
    conv_channels = tuple(config.get("conv_channels", (32, 64, 128, 256)))

    model = EventClassifierCNN(
        n_types=n_types,
        n_mels=n_mels,
        conv_channels=conv_channels,
    )
    load_checkpoint(path=model_dir / "model.pt", model=model)

    thresholds_path = model_dir / "thresholds.json"
    thresholds: dict[str, float] = (
        json.loads(thresholds_path.read_text())
        if thresholds_path.exists()
        else {t: 0.5 for t in vocabulary}
    )

    feature_config = SegmentationFeatureConfig(**config.get("feature_config", {}))

    return model, vocabulary, thresholds, feature_config


def _extract_event_features(
    event: Event,
    audio: np.ndarray,
    feature_config: SegmentationFeatureConfig,
    audio_start_sec: float = 0.0,
) -> torch.Tensor:
    """Crop audio at event bounds, extract log-mel, return ``(1, n_mels, T)``."""
    sr = feature_config.sample_rate
    start_sample = max(0, int(round((event.start_sec - audio_start_sec) * sr)))
    end_sample = min(len(audio), int(round((event.end_sec - audio_start_sec) * sr)))
    if end_sample <= start_sample:
        end_sample = min(len(audio), start_sample + sr)

    crop = audio[start_sample:end_sample]
    logmel = extract_logmel(crop, feature_config)
    logmel = normalize_per_region_zscore(logmel)
    return torch.from_numpy(logmel).unsqueeze(0).float()


def classify_events(
    model: EventClassifierCNN,
    events: Sequence[Event],
    audio_loader: EventAudioLoader,
    feature_config: SegmentationFeatureConfig,
    vocabulary: list[str],
    thresholds: dict[str, float],
    device: torch.device | None = None,
    batch_size: int = 32,
) -> list[TypedEvent]:
    """Classify events and return ``TypedEvent`` rows.

    Produces one ``TypedEvent`` per (event, type) combination where
    ``above_threshold`` is ``True``. Events where no type exceeds the
    threshold still produce one row for the highest-scoring type.
    """
    if not events:
        return []

    resolved_device = device if device is not None else torch.device("cpu")
    model.to(resolved_device)
    model.eval()

    feature_list: list[torch.Tensor] = []
    for event in events:
        audio, audio_start = audio_loader(event)
        feat = _extract_event_features(event, audio, feature_config, audio_start)
        feature_list.append(feat)

    all_scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(feature_list), batch_size):
            batch_feats = feature_list[i : i + batch_size]
            dummy_labels = [torch.zeros(len(vocabulary)) for _ in batch_feats]
            batch_items = list(zip(batch_feats, dummy_labels))
            inputs, _ = collate_fn(batch_items)
            inputs = inputs.to(resolved_device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)

    scores = np.concatenate(all_scores, axis=0)

    typed_events: list[TypedEvent] = []
    for idx, event in enumerate(events):
        event_scores = scores[idx]
        any_above = False
        best_type_idx = int(np.argmax(event_scores))

        for type_idx, type_name in enumerate(vocabulary):
            thresh = thresholds.get(type_name, 0.5)
            above = bool(event_scores[type_idx] >= thresh)
            if above:
                any_above = True
                typed_events.append(
                    TypedEvent(
                        event_id=event.event_id,
                        start_sec=event.start_sec,
                        end_sec=event.end_sec,
                        type_name=type_name,
                        score=round(float(event_scores[type_idx]), 4),
                        above_threshold=True,
                    )
                )

        if not any_above:
            typed_events.append(
                TypedEvent(
                    event_id=event.event_id,
                    start_sec=event.start_sec,
                    end_sec=event.end_sec,
                    type_name=vocabulary[best_type_idx],
                    score=round(float(event_scores[best_type_idx]), 4),
                    above_threshold=False,
                )
            )

    return typed_events
