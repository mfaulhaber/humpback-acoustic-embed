"""Pass 2 segmentation trainer.

Drives ``ml.training_loop.fit`` for the ``SegmentationCRNN`` model. Owns
the audio-source-disjoint train/val split, the masked BCE loss, the
auto-computed ``pos_weight``, per-epoch validation metrics, early
stopping, and a final decoder-based event-level evaluation over the
validation set. Persists the best checkpoint via
``ml.checkpointing.save_checkpoint``.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from humpback.call_parsing.segmentation.dataset import (
    AudioLoader,
    SegmentationSampleDataset,
    build_framewise_target,
    collate_fn,
    compute_pos_weight,
)
from humpback.call_parsing.segmentation.decoder import decode_events
from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.call_parsing.types import Event
from humpback.ml.checkpointing import load_checkpoint, save_checkpoint
from humpback.ml.training_loop import Callback, TrainingResult, fit
from humpback.schemas.call_parsing import (
    SegmentationDecoderConfig,
    SegmentationFeatureConfig,
    SegmentationTrainingConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class EventMatchResult:
    hits: list[tuple[Event, Event]] = field(default_factory=list)
    misses: list[Event] = field(default_factory=list)
    extras: list[Event] = field(default_factory=list)
    onset_errors: list[float] = field(default_factory=list)
    offset_errors: list[float] = field(default_factory=list)


@dataclass
class SegmentationTrainingResult:
    """Output of ``train_model`` — combines fit history with final eval."""

    train_losses: list[float]
    val_losses: list[float]
    val_framewise_f1_history: list[float]
    framewise_precision: float
    framewise_recall: float
    framewise_f1: float
    event_precision: float
    event_recall: float
    event_f1: float
    onset_mae_sec: float
    offset_mae_sec: float
    pos_weight: float
    n_train_samples: int
    n_val_samples: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "val_framewise_f1_history": list(self.val_framewise_f1_history),
            "framewise": {
                "precision": self.framewise_precision,
                "recall": self.framewise_recall,
                "f1": self.framewise_f1,
            },
            "event": {
                "precision": self.event_precision,
                "recall": self.event_recall,
                "f1": self.event_f1,
                "iou_threshold": 0.3,
            },
            "onset_mae_sec": self.onset_mae_sec,
            "offset_mae_sec": self.offset_mae_sec,
            "pos_weight": self.pos_weight,
            "n_train_samples": self.n_train_samples,
            "n_val_samples": self.n_val_samples,
        }


def _source_key(sample: Any) -> str:
    if getattr(sample, "audio_file_id", None):
        return f"audio:{sample.audio_file_id}"
    if getattr(sample, "hydrophone_id", None):
        return f"hydro:{sample.hydrophone_id}"
    raise ValueError(
        "sample must have audio_file_id or hydrophone_id for train/val split"
    )


def _temporal_sort_key(sample: Any) -> float:
    """Extract a timestamp for temporal ordering."""
    ts = getattr(sample, "start_timestamp", None)
    if ts is not None:
        return float(ts)
    return float(getattr(sample, "crop_start_sec", 0.0))


def split_by_audio_source(
    samples: Sequence[Any],
    val_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    """Return ``(train_samples, val_samples)`` split by audio source.

    Samples are grouped by ``audio_file_id`` (or ``hydrophone_id`` as the
    fallback key).  When there are enough groups for group-level splitting
    to produce at least one val group, whole groups are assigned to val.
    Otherwise, falls back to a **per-group temporal split**: within each
    group, the last ``val_fraction`` of samples (sorted by timestamp) go
    to val, giving temporal separation without sacrificing entire sources.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    groups: dict[str, list[Any]] = {}
    for sample in samples:
        key = _source_key(sample)
        groups.setdefault(key, []).append(sample)

    group_keys = sorted(groups.keys())

    if val_fraction == 0.0:
        return list(samples), []

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_val_groups = int(round(val_fraction * n_groups))
    n_val_groups = max(0, min(n_groups - 1, n_val_groups))

    if n_val_groups >= 1:
        val_keys = set(group_keys[:n_val_groups])
        train_samples: list[Any] = []
        val_samples: list[Any] = []
        for key in group_keys:
            bucket = val_samples if key in val_keys else train_samples
            bucket.extend(groups[key])
        return train_samples, val_samples

    # Too few groups for group-level split — split temporally within
    # each group so every source contributes to both train and val.
    train_samples = []
    val_samples = []
    for key in group_keys:
        group = groups[key]
        group.sort(key=_temporal_sort_key)
        n_val = max(1, int(round(val_fraction * len(group))))
        split_idx = len(group) - n_val
        train_samples.extend(group[:split_idx])
        val_samples.extend(group[split_idx:])
    return train_samples, val_samples


def _interval_iou(a: Event, b: Event) -> float:
    overlap_start = max(a.start_sec, b.start_sec)
    overlap_end = min(a.end_sec, b.end_sec)
    intersection = max(0.0, overlap_end - overlap_start)
    if intersection == 0.0:
        return 0.0
    dur_a = max(0.0, a.end_sec - a.start_sec)
    dur_b = max(0.0, b.end_sec - b.start_sec)
    union = dur_a + dur_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def match_events_by_iou(
    pred_events: Sequence[Event],
    gt_events: Sequence[Event],
    iou_threshold: float,
) -> EventMatchResult:
    """Greedy one-to-one matching of predictions to ground truth by IoU.

    Pairs with the highest IoU ``>= iou_threshold`` are matched first;
    each prediction and each ground-truth event is matched at most once.
    Returns hits, misses (unmatched gt), extras (unmatched preds), and
    parallel lists of absolute onset/offset errors for the matched
    pairs.
    """
    result = EventMatchResult()
    if not pred_events and not gt_events:
        return result
    if not pred_events:
        result.misses = list(gt_events)
        return result
    if not gt_events:
        result.extras = list(pred_events)
        return result

    candidates: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_events):
        for g_idx, gt in enumerate(gt_events):
            iou = _interval_iou(pred, gt)
            if iou >= iou_threshold:
                candidates.append((iou, p_idx, g_idx))
    # Highest IoU first; ties break by pred index then gt index for
    # deterministic output.
    candidates.sort(key=lambda triple: (-triple[0], triple[1], triple[2]))

    used_preds: set[int] = set()
    used_gts: set[int] = set()
    for _, p_idx, g_idx in candidates:
        if p_idx in used_preds or g_idx in used_gts:
            continue
        used_preds.add(p_idx)
        used_gts.add(g_idx)
        pred = pred_events[p_idx]
        gt = gt_events[g_idx]
        result.hits.append((pred, gt))
        result.onset_errors.append(abs(pred.start_sec - gt.start_sec))
        result.offset_errors.append(abs(pred.end_sec - gt.end_sec))

    result.extras = [
        pred for idx, pred in enumerate(pred_events) if idx not in used_preds
    ]
    result.misses = [gt for idx, gt in enumerate(gt_events) if idx not in used_gts]
    return result


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE-with-logits averaged over real (unmasked) frames only.

    ``pos_weight`` is applied elementwise via the standard
    ``F.binary_cross_entropy_with_logits`` contract. Padded frames
    contribute zero loss and are excluded from the mean.
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        self.pos_weight_value = float(pos_weight)

    def forward(
        self,
        outputs: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pos_weight_tensor = torch.tensor(
            self.pos_weight_value,
            device=outputs.device,
            dtype=outputs.dtype,
        )
        loss = F.binary_cross_entropy_with_logits(
            outputs,
            target,
            reduction="none",
            pos_weight=pos_weight_tensor,
        )
        mask_f = mask.float()
        loss = loss * mask_f
        n_real = mask_f.sum().clamp(min=1.0)
        return loss.sum() / n_real


def _framewise_metrics_from_preds(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> tuple[float, float, float]:
    real = masks.astype(bool)
    p = preds[real]
    t = targets[real]
    tp = float(np.sum((p == 1) & (t == 1)))
    fp = float(np.sum((p == 1) & (t == 0)))
    fn = float(np.sum((p == 0) & (t == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _collect_val_frame_probs(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one no-grad val pass and return probs, targets, masks arrays.

    Batches are flattened so that shape is ``(total_frames,)`` — caller
    uses the mask to identify real frames when computing metrics.
    """
    model.eval()
    prob_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            target = batch[1].to(device)
            mask = batch[2].to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            prob_chunks.append(probs.cpu().numpy().reshape(-1))
            target_chunks.append(target.cpu().numpy().reshape(-1))
            mask_chunks.append(mask.cpu().numpy().reshape(-1))
    if not prob_chunks:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty.astype(bool)
    return (
        np.concatenate(prob_chunks),
        np.concatenate(target_chunks),
        np.concatenate(mask_chunks).astype(bool),
    )


def _make_val_f1_callback(
    model: nn.Module,
    val_loader: DataLoader[Any] | None,
    device: torch.device,
) -> Callback:
    import copy

    best_f1: list[float] = [0.0]

    def cb(epoch: int, result: TrainingResult, should_stop: list[bool]) -> None:
        if val_loader is None:
            return
        probs, targets, masks = _collect_val_frame_probs(model, val_loader, device)
        preds = (probs >= 0.5).astype(np.float32)
        _, _, f1 = _framewise_metrics_from_preds(preds, targets, masks)
        history = result.callback_outputs.setdefault("val_framewise_f1", [])
        assert isinstance(history, list)
        history.append(f1)
        if f1 > best_f1[0]:
            best_f1[0] = f1
            result.callback_outputs["best_f1_model_state"] = copy.deepcopy(
                model.state_dict()
            )

    return cb


def _make_early_stop_callback(patience: int) -> Callback:
    state = {"best": float("inf"), "bad": 0}

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


def _gt_events_for_sample(sample: Any, region_id: str) -> list[Event]:
    """Build ground-truth ``Event`` rows from a sample's ``events_json``."""
    import json

    raw = json.loads(sample.events_json) if sample.events_json else []
    events: list[Event] = []
    for entry in raw:
        start = float(entry["start_sec"])
        end = float(entry["end_sec"])
        events.append(
            Event(
                event_id=f"gt-{len(events)}",
                region_id=region_id,
                start_sec=start,
                end_sec=end,
                center_sec=(start + end) / 2.0,
                segmentation_confidence=1.0,
            )
        )
    return events


def _final_eval(
    model: nn.Module,
    val_samples: Sequence[Any],
    feature_config: SegmentationFeatureConfig,
    decoder_config: SegmentationDecoderConfig,
    audio_loader: AudioLoader,
    device: torch.device,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
]:
    """Run the full decoder on the val set and compute eval metrics."""
    from humpback.call_parsing.segmentation.features import (
        extract_logmel,
        normalize_per_region_zscore,
    )

    model.eval()
    all_framewise_preds: list[np.ndarray] = []
    all_framewise_targets: list[np.ndarray] = []
    all_framewise_masks: list[np.ndarray] = []
    hits = 0
    misses = 0
    extras = 0
    all_onset_errors: list[float] = []
    all_offset_errors: list[float] = []
    hop_sec = feature_config.hop_length / feature_config.sample_rate

    for sample_idx, sample in enumerate(val_samples):
        audio = audio_loader(sample)
        logmel = normalize_per_region_zscore(extract_logmel(audio, feature_config))
        features = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            logits = model(features)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        target = build_framewise_target(
            sample.events_json,
            sample.crop_start_sec,
            sample.crop_end_sec,
            feature_config,
        )
        n_feat = probs.shape[0]
        if target.shape[0] > n_feat:
            target = target[:n_feat]
        elif target.shape[0] < n_feat:
            target = np.pad(target, (0, n_feat - target.shape[0]))

        all_framewise_preds.append((probs >= 0.5).astype(np.float32))
        all_framewise_targets.append(target)
        all_framewise_masks.append(np.ones_like(target, dtype=bool))

        region_id = f"val-{sample_idx}"
        pred_events = decode_events(
            frame_probs=probs,
            region_id=region_id,
            region_start_sec=sample.crop_start_sec,
            hop_sec=hop_sec,
            config=decoder_config,
        )
        gt_events = _gt_events_for_sample(sample, region_id)
        match = match_events_by_iou(
            pred_events=pred_events,
            gt_events=gt_events,
            iou_threshold=0.3,
        )
        hits += len(match.hits)
        misses += len(match.misses)
        extras += len(match.extras)
        all_onset_errors.extend(match.onset_errors)
        all_offset_errors.extend(match.offset_errors)

    if all_framewise_preds:
        framewise_metrics = _framewise_metrics_from_preds(
            np.concatenate(all_framewise_preds),
            np.concatenate(all_framewise_targets),
            np.concatenate(all_framewise_masks),
        )
    else:
        framewise_metrics = (0.0, 0.0, 0.0)

    denom_p = hits + extras
    denom_r = hits + misses
    event_precision = hits / denom_p if denom_p > 0 else 0.0
    event_recall = hits / denom_r if denom_r > 0 else 0.0
    event_f1 = (
        (2 * event_precision * event_recall) / (event_precision + event_recall)
        if (event_precision + event_recall) > 0
        else 0.0
    )
    onset_mae = (
        float(sum(all_onset_errors) / len(all_onset_errors))
        if all_onset_errors
        else 0.0
    )
    offset_mae = (
        float(sum(all_offset_errors) / len(all_offset_errors))
        if all_offset_errors
        else 0.0
    )
    return (
        framewise_metrics,
        (event_precision, event_recall, event_f1),
        onset_mae,
        offset_mae,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    samples: Sequence[Any],
    feature_config: SegmentationFeatureConfig,
    decoder_config: SegmentationDecoderConfig,
    audio_loader: AudioLoader,
    config: SegmentationTrainingConfig,
    checkpoint_path: Path,
    device: torch.device | None = None,
    pretrained_checkpoint: Path | None = None,
) -> SegmentationTrainingResult:
    """Train a ``SegmentationCRNN`` on ``samples`` and save a checkpoint.

    Splits samples by audio source, builds train/val ``DataLoader``s,
    auto-computes ``pos_weight`` on the train split, calls
    ``ml.training_loop.fit`` with masked BCE loss, records per-epoch
    framewise F1 as a callback, applies early stopping on val loss,
    runs the full decoder + event-matching final eval, and writes the
    checkpoint atomically.

    When ``pretrained_checkpoint`` is provided, model weights are
    initialized from that checkpoint before training (fine-tuning).
    """
    _seed_everything(config.seed)

    train_samples, val_samples = split_by_audio_source(
        samples=samples,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    if not train_samples:
        raise ValueError("training split is empty — provide samples with sources")

    train_dataset = SegmentationSampleDataset(
        samples=train_samples,
        feature_config=feature_config,
        audio_loader=audio_loader,
    )
    val_dataset = (
        SegmentationSampleDataset(
            samples=val_samples,
            feature_config=feature_config,
            audio_loader=audio_loader,
        )
        if val_samples
        else None
    )

    pos_weight = compute_pos_weight(train_dataset)

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

    model = SegmentationCRNN(
        n_mels=config.n_mels,
        conv_channels=list(config.conv_channels),
        gru_hidden=config.gru_hidden,
        gru_layers=config.gru_layers,
    )
    if pretrained_checkpoint is not None:
        load_checkpoint(pretrained_checkpoint, model)
        logger.info(
            "Initialized model from pretrained checkpoint %s", pretrained_checkpoint
        )
        for param in model.conv.parameters():
            param.requires_grad = False
        logger.info("Froze conv layers for fine-tuning")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)

    resolved_device = device if device is not None else torch.device("cpu")
    model.to(resolved_device)
    loss_fn.to(resolved_device)

    callbacks: list[Callback] = [
        _make_val_f1_callback(model, val_loader, resolved_device),
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
    )

    best_f1_state = fit_result.callback_outputs.get("best_f1_model_state")
    if best_f1_state is not None:
        model.load_state_dict(best_f1_state)
    elif fit_result.best_model_state is not None:
        model.load_state_dict(fit_result.best_model_state)

    framewise_metrics = (0.0, 0.0, 0.0)
    event_metrics = (0.0, 0.0, 0.0)
    onset_mae = 0.0
    offset_mae = 0.0
    if val_samples:
        framewise_metrics, event_metrics, onset_mae, offset_mae = _final_eval(
            model=model,
            val_samples=val_samples,
            feature_config=feature_config,
            decoder_config=decoder_config,
            audio_loader=audio_loader,
            device=resolved_device,
        )

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        config={
            "model_type": "SegmentationCRNN",
            "n_mels": config.n_mels,
            "conv_channels": list(config.conv_channels),
            "gru_hidden": config.gru_hidden,
            "gru_layers": config.gru_layers,
            "feature_config": feature_config.model_dump(),
        },
    )

    val_f1_history_raw = fit_result.callback_outputs.get("val_framewise_f1", [])
    val_f1_history: list[float] = (
        list(val_f1_history_raw) if isinstance(val_f1_history_raw, Iterable) else []
    )

    return SegmentationTrainingResult(
        train_losses=list(fit_result.train_losses),
        val_losses=list(fit_result.val_losses),
        val_framewise_f1_history=val_f1_history,
        framewise_precision=framewise_metrics[0],
        framewise_recall=framewise_metrics[1],
        framewise_f1=framewise_metrics[2],
        event_precision=event_metrics[0],
        event_recall=event_metrics[1],
        event_f1=event_metrics[2],
        onset_mae_sec=onset_mae,
        offset_mae_sec=offset_mae,
        pos_weight=pos_weight,
        n_train_samples=len(train_samples),
        n_val_samples=len(val_samples),
    )
