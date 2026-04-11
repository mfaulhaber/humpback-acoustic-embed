"""Shared PyTorch training harness for the call parsing pipeline.

Pass 2 (segmentation CRNN) and Pass 3 (per-event CNN) both train PyTorch
models. This package centralizes device selection, the training loop,
and checkpoint I/O so each pass's trainer only has to define its model,
optimizer, and data loaders.
"""

from humpback.ml.checkpointing import load_checkpoint, save_checkpoint
from humpback.ml.device import select_device
from humpback.ml.training_loop import Callback, TrainingResult, fit

__all__ = [
    "Callback",
    "TrainingResult",
    "fit",
    "load_checkpoint",
    "save_checkpoint",
    "select_device",
]
