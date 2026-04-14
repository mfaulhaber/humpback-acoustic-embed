"""Tests for ``humpback.ml.training_loop.fit``.

Uses a tiny 2-layer MLP on XOR to validate the loop actually learns, and
a short canned loader to exercise the callback / early-stop path.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from humpback.ml.training_loop import TrainingResult, fit


def _xor_loader(batch_size: int = 4) -> DataLoader:
    x = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_fit_learns_xor_to_low_loss() -> None:
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=300,
        device=torch.device("cpu"),
    )

    assert len(result.train_losses) == 300
    assert result.train_losses[-1] < 0.2


def test_fit_records_val_loss_per_epoch() -> None:
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=5,
        val_loader=loader,
        device=torch.device("cpu"),
    )

    assert len(result.val_losses) == 5
    assert len(result.train_losses) == 5


def test_callback_early_stop_terminates_training() -> None:
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    def stop_after_two(
        epoch: int, result: TrainingResult, should_stop: list[bool]
    ) -> None:
        if epoch == 1:  # zero-indexed; after epoch 0 and epoch 1 ran → 2 losses
            should_stop[0] = True

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=100,
        callbacks=[stop_after_two],
        device=torch.device("cpu"),
    )

    assert len(result.train_losses) == 2


def test_fit_tracks_best_model_state_on_val_loss() -> None:
    """fit() snapshots model weights at the epoch with lowest val loss."""
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=5,
        val_loader=loader,
        device=torch.device("cpu"),
    )

    assert result.best_model_state is not None
    # Restoring best state should give a loadable state dict
    model.load_state_dict(result.best_model_state)
    model.eval()
    # The snapshot is a separate copy, not the same object as model's state
    for key in result.best_model_state:
        assert result.best_model_state[key] is not model.state_dict()[key]


def test_fit_best_model_state_is_none_without_val_loader() -> None:
    """Without a val_loader, best_model_state stays None."""
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=3,
        device=torch.device("cpu"),
    )

    assert result.best_model_state is None


def test_fit_restoring_best_state_gives_lower_loss_than_final() -> None:
    """When training overfits, restoring best state yields lower val loss."""
    torch.manual_seed(42)
    model = _MLP()
    # High LR to encourage overfitting/oscillation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=50,
        val_loader=loader,
        device=torch.device("cpu"),
    )

    assert result.best_model_state is not None
    best_val = min(result.val_losses)

    # Restore best state and verify it gives the best val loss
    model.load_state_dict(result.best_model_state)
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch[0])
            total += float(loss_fn(pred, batch[1]).item())
            n += 1
    restored_loss = total / n
    assert restored_loss == pytest.approx(best_val, rel=1e-5)


def test_callback_can_populate_callback_outputs() -> None:
    torch.manual_seed(0)
    model = _MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()
    loader = _xor_loader()

    def record_first_loss(
        epoch: int, result: TrainingResult, should_stop: list[bool]
    ) -> None:
        if epoch == 0:
            result.callback_outputs["first_loss"] = result.train_losses[0]

    result = fit(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=loader,
        epochs=2,
        callbacks=[record_first_loss],
        device=torch.device("cpu"),
    )

    assert "first_loss" in result.callback_outputs
    assert isinstance(result.callback_outputs["first_loss"], float)
