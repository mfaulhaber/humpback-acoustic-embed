"""Atomic checkpoint save/load for PyTorch models.

Writes via a temp file + ``os.replace`` so a crash mid-save can't
corrupt an existing checkpoint. Stores ``model_state_dict``,
``optimizer_state_dict`` (optional), and a ``config`` dict the caller
uses to record hyperparameters or metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None,
    config: dict[str, Any],
) -> None:
    """Atomically save model + optional optimizer + config to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
) -> dict[str, Any]:
    """Load ``path`` into ``model`` (+ optional ``optimizer``); return config."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    config = payload.get("config", {})
    assert isinstance(config, dict)
    return config
