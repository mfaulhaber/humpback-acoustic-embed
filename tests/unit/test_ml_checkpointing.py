"""Tests for ``humpback.ml.checkpointing`` save/load roundtrip."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from humpback.ml.checkpointing import load_checkpoint, save_checkpoint


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_checkpoint_roundtrip_preserves_weights(tmp_path: Path) -> None:
    torch.manual_seed(7)
    a = _Tiny()
    opt = torch.optim.Adam(a.parameters(), lr=0.001)
    path = tmp_path / "ckpt.pt"

    config = {"lr": 0.001, "note": "unit-test"}
    save_checkpoint(path, a, opt, config)

    b = _Tiny()  # random init; will be replaced
    restored = load_checkpoint(path, b)
    assert restored == config

    x = torch.randn(3, 4)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_checkpoint_load_without_optimizer_returns_config(tmp_path: Path) -> None:
    torch.manual_seed(1)
    a = _Tiny()
    opt = torch.optim.Adam(a.parameters(), lr=0.001)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, a, opt, {"epoch": 3})

    b = _Tiny()
    config = load_checkpoint(path, b, optimizer=None)
    assert config == {"epoch": 3}


def test_checkpoint_save_without_optimizer(tmp_path: Path) -> None:
    torch.manual_seed(2)
    a = _Tiny()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, a, optimizer=None, config={"ok": True})

    b = _Tiny()
    config = load_checkpoint(path, b)
    assert config == {"ok": True}


def test_atomic_write_leaves_no_tmp_file(tmp_path: Path) -> None:
    a = _Tiny()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, a, optimizer=None, config={})
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
