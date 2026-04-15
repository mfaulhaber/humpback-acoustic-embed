"""Pass 3 event classifier training driver.

Orchestrates training of an ``EventClassifierCNN`` on variable-length
event crops with multi-label BCE loss and per-type ``pos_weight``.
After training, sweeps per-type classification thresholds on the
validation set to maximize F1 and persists the model checkpoint,
config, thresholds, and metrics to a model directory.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from humpback.call_parsing.event_classifier.dataset import (
    AudioLoader,
    EventCropDataset,
    collate_fn,
)
from humpback.call_parsing.event_classifier.model import EventClassifierCNN
from humpback.call_parsing.segmentation.trainer import split_by_audio_source
from humpback.ml.checkpointing import save_checkpoint
from humpback.ml.training_loop import Callback, TrainingResult, fit
from humpback.schemas.call_parsing import SegmentationFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class EventClassifierTrainingConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    val_fraction: float = 0.2
    min_examples_per_type: int = 10
    corrections_only: bool = True
    conv_channels: tuple[int, ...] = (32, 64, 128, 256)


@dataclass
class EventClassifierTrainingResult:
    train_losses: list[float]
    val_losses: list[float]
    per_type_metrics: dict[str, dict[str, float]]
    per_type_thresholds: dict[str, float]
    vocabulary: list[str]
    pos_weights: dict[str, float]
    n_train_samples: int
    n_val_samples: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "per_type_metrics": dict(self.per_type_metrics),
            "per_type_thresholds": dict(self.per_type_thresholds),
            "vocabulary": list(self.vocabulary),
            "pos_weights": dict(self.pos_weights),
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
        }


def compute_per_type_pos_weight(
    samples: Sequence[Any],
    n_types: int,
) -> np.ndarray:
    """Return ``(n_types,)`` pos_weight where ``pw[i] = n_neg / n_pos`` for type i."""
    counts = np.zeros(n_types, dtype=np.int64)
    for sample in samples:
        counts[sample.type_index] += 1
    total = len(samples)
    pos_weight = np.ones(n_types, dtype=np.float32)
    for i in range(n_types):
        if counts[i] > 0:
            pos_weight[i] = float(total - counts[i]) / float(counts[i])
    return pos_weight


def _filter_by_min_examples(
    samples: Sequence[Any],
    vocabulary: list[str],
    min_examples: int,
) -> tuple[list[Any], list[str], dict[int, int]]:
    """Drop types with fewer than ``min_examples`` and remap type indices."""
    counts: dict[int, int] = {}
    for s in samples:
        counts[s.type_index] = counts.get(s.type_index, 0) + 1

    kept_indices = [
        i for i in range(len(vocabulary)) if counts.get(i, 0) >= min_examples
    ]
    if not kept_indices:
        raise ValueError(f"No types have >= {min_examples} examples; cannot train")

    old_to_new = {old: new for new, old in enumerate(kept_indices)}
    new_vocab = [vocabulary[i] for i in kept_indices]

    filtered: list[Any] = []
    for s in samples:
        if s.type_index in old_to_new:
            filtered.append(_remap_sample(s, old_to_new[s.type_index]))

    return filtered, new_vocab, old_to_new


@dataclass
class _RemappedSample:
    start_sec: float
    end_sec: float
    type_index: int
    audio_file_id: str | None = None
    hydrophone_id: str | None = None
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0


def _remap_sample(sample: Any, new_index: int) -> _RemappedSample:
    return _RemappedSample(
        start_sec=sample.start_sec,
        end_sec=sample.end_sec,
        type_index=new_index,
        audio_file_id=getattr(sample, "audio_file_id", None),
        hydrophone_id=getattr(sample, "hydrophone_id", None),
        start_timestamp=getattr(sample, "start_timestamp", 0.0),
        end_timestamp=getattr(sample, "end_timestamp", 0.0),
    )


def _optimize_threshold_for_type(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """Sweep thresholds to maximize F1 for one type; return (threshold, best_f1)."""
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.96, 0.05):
        preds = (scores >= thresh).astype(np.float32)
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        if f1 >= best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
    return best_thresh, best_f1


def _evaluate_val(
    model: nn.Module,
    val_loader: DataLoader[Any],
    vocabulary: list[str],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Run val inference, optimize per-type thresholds, compute metrics."""
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            labels = batch[1]
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(labels.numpy())

    if not all_scores:
        return (
            {t: 0.5 for t in vocabulary},
            {t: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for t in vocabulary},
        )

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    thresholds: dict[str, float] = {}
    metrics: dict[str, dict[str, float]] = {}
    for i, type_name in enumerate(vocabulary):
        thresh, _ = _optimize_threshold_for_type(scores[:, i], labels[:, i])
        thresholds[type_name] = thresh

        preds = (scores[:, i] >= thresh).astype(np.float32)
        tp = float(np.sum((preds == 1) & (labels[:, i] == 1)))
        fp = float(np.sum((preds == 1) & (labels[:, i] == 0)))
        fn = float(np.sum((preds == 0) & (labels[:, i] == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        metrics[type_name] = {"precision": precision, "recall": recall, "f1": f1}

    return thresholds, metrics


def _make_early_stop_callback(patience: int) -> Callback:
    state: dict[str, Any] = {"best": float("inf"), "bad": 0}

    def cb(epoch: int, result: TrainingResult, should_stop: list[bool]) -> None:
        if not result.val_losses:
            return
        current = float(result.val_losses[-1])
        if current < state["best"]:
            state["best"] = current
            state["bad"] = 0
        else:
            state["bad"] += 1
            if state["bad"] >= patience:
                should_stop[0] = True

    return cb


def _make_val_f1_callback(
    model: nn.Module,
    val_loader: DataLoader[Any] | None,
    n_types: int,
    device: torch.device,
) -> Callback:
    def cb(epoch: int, result: TrainingResult, should_stop: list[bool]) -> None:
        if val_loader is None:
            return
        model.eval()
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = batch[1]
                logits = model(inputs)
                preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.numpy())
        if not all_preds:
            return
        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)

        f1s: list[float] = []
        for i in range(n_types):
            tp = float(np.sum((preds_arr[:, i] == 1) & (labels_arr[:, i] == 1)))
            fp = float(np.sum((preds_arr[:, i] == 1) & (labels_arr[:, i] == 0)))
            fn = float(np.sum((preds_arr[:, i] == 0) & (labels_arr[:, i] == 1)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        history = result.callback_outputs.setdefault("val_macro_f1", [])
        assert isinstance(history, list)
        history.append(macro_f1)

    return cb


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_event_classifier(
    samples: Sequence[Any],
    vocabulary: list[str],
    feature_config: SegmentationFeatureConfig,
    audio_loader: AudioLoader,
    config: EventClassifierTrainingConfig,
    model_dir: Path,
    device: torch.device | None = None,
) -> EventClassifierTrainingResult:
    """Train an ``EventClassifierCNN`` and save artifacts to ``model_dir``."""
    _seed_everything(config.seed)

    filtered, filtered_vocab, _ = _filter_by_min_examples(
        samples, vocabulary, config.min_examples_per_type
    )
    if not filtered:
        raise ValueError("No training samples after min_examples filtering")

    n_types = len(filtered_vocab)
    logger.info(
        "Training event classifier with %d types (%d samples)",
        n_types,
        len(filtered),
    )

    train_samples, val_samples = split_by_audio_source(
        samples=filtered,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    if not train_samples:
        raise ValueError("Training split is empty")

    train_dataset = EventCropDataset(
        train_samples, feature_config, audio_loader, n_types
    )
    val_dataset = (
        EventCropDataset(val_samples, feature_config, audio_loader, n_types)
        if val_samples
        else None
    )

    pos_weight_arr = compute_per_type_pos_weight(train_samples, n_types)
    pos_weight_tensor = torch.from_numpy(pos_weight_arr).float()

    train_loader: DataLoader[Any] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader: DataLoader[Any] | None = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if val_dataset is not None
        else None
    )

    model = EventClassifierCNN(
        n_types=n_types,
        n_mels=feature_config.n_mels,
        conv_channels=list(config.conv_channels),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    resolved_device = device if device is not None else torch.device("cpu")
    model.to(resolved_device)
    loss_fn.to(resolved_device)

    callbacks: list[Callback] = [
        _make_val_f1_callback(model, val_loader, n_types, resolved_device),
        _make_early_stop_callback(config.early_stopping_patience),
    ]

    fit_result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        epochs=config.epochs,
        val_loader=val_loader,
        callbacks=callbacks,
        device=resolved_device,
        grad_clip=config.grad_clip,
    )

    if fit_result.best_model_state is not None:
        model.load_state_dict(fit_result.best_model_state)

    thresholds: dict[str, float] = {t: 0.5 for t in filtered_vocab}
    per_type_metrics: dict[str, dict[str, float]] = {
        t: {"precision": 0.0, "recall": 0.0, "f1": 0.0} for t in filtered_vocab
    }
    if val_loader is not None:
        thresholds, per_type_metrics = _evaluate_val(
            model, val_loader, filtered_vocab, resolved_device
        )

    model_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        path=model_dir / "model.pt",
        model=model,
        optimizer=None,
        config={
            "model_type": "EventClassifierCNN",
            "n_types": n_types,
            "n_mels": feature_config.n_mels,
            "conv_channels": list(config.conv_channels),
        },
    )

    config_data = {
        "model_type": "EventClassifierCNN",
        "n_types": n_types,
        "n_mels": feature_config.n_mels,
        "conv_channels": list(config.conv_channels),
        "vocabulary": filtered_vocab,
        "feature_config": feature_config.model_dump(),
        "training_config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "min_examples_per_type": config.min_examples_per_type,
            "seed": config.seed,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config_data, indent=2))
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(per_type_metrics, indent=2))

    pw_dict = {filtered_vocab[i]: float(pos_weight_arr[i]) for i in range(n_types)}

    return EventClassifierTrainingResult(
        train_losses=list(fit_result.train_losses),
        val_losses=list(fit_result.val_losses),
        per_type_metrics=per_type_metrics,
        per_type_thresholds=thresholds,
        vocabulary=filtered_vocab,
        pos_weights=pw_dict,
        n_train_samples=len(train_samples),
        n_val_samples=len(val_samples),
    )
