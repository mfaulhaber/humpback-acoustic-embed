"""Minimal PyTorch training loop used by Pass 2 and Pass 3 trainers.

Handles the mechanical parts — device placement, train/eval mode
toggling, optimizer steps, optional validation, and a callback hook for
things like early stopping or metric collection — so each pass's trainer
focuses on model + data + loss only.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from humpback.ml.device import select_device

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Outputs collected across an entire ``fit()`` call."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    callback_outputs: dict[str, Any] = field(default_factory=dict)
    best_model_state: dict[str, Any] | None = field(default=None, repr=False)


class Callback(Protocol):
    """Callable invoked after each epoch.

    A callback can mutate ``result`` (e.g. record a metric under
    ``callback_outputs``) and flip ``should_stop[0] = True`` to end
    training early. ``epoch`` is zero-indexed.
    """

    def __call__(
        self,
        epoch: int,
        result: TrainingResult,
        should_stop: list[bool],
    ) -> None: ...


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    loss_fn: nn.Module | None,
    optimizer: Optimizer | None,
    grad_clip: float | None = None,
) -> float:
    """Run one pass over ``loader``. Training if ``optimizer`` is given."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_batches = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for batch in loader:
            # Support ``(inputs, targets)`` and longer tuples like
            # ``(inputs, targets, mask)``; any element after the first is
            # forwarded positionally to ``loss_fn``. This lets Pass 2's
            # segmentation trainer supply a framewise mask without the
            # harness having to know about it.
            inputs = batch[0].to(device)
            extras = [t.to(device) for t in batch[1:]]

            outputs = model(inputs)
            if loss_fn is None:
                raise ValueError("loss_fn is required")
            loss = loss_fn(outputs, *extras)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

    if n_batches == 0:
        return 0.0
    return total_loss / n_batches


def fit(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    train_loader: DataLoader[Any],
    epochs: int,
    val_loader: DataLoader[Any] | None = None,
    scheduler: LRScheduler | None = None,
    callbacks: list[Callback] | None = None,
    device: torch.device | None = None,
    grad_clip: float | None = None,
) -> TrainingResult:
    """Train ``model`` for up to ``epochs`` epochs.

    Each epoch runs training over ``train_loader`` and — if provided —
    one validation pass over ``val_loader``. Callbacks run after each
    epoch and may set ``should_stop[0] = True`` to terminate early.
    """
    if device is None:
        device = select_device()
    model.to(device)

    result = TrainingResult()
    should_stop = [False]
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
        result.train_losses.append(train_loss)

        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, device, loss_fn=loss_fn, optimizer=None
            )
            result.val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_model_state = copy.deepcopy(model.state_dict())

        if val_loader is not None:
            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                result.val_losses[-1],
            )
        else:
            logger.info(
                "Epoch %d/%d — train_loss=%.4f",
                epoch + 1,
                epochs,
                train_loss,
            )

        if scheduler is not None:
            scheduler.step()

        if callbacks:
            for cb in callbacks:
                cb(epoch, result, should_stop)

        if should_stop[0]:
            break

    return result
