"""Declarative hyperparameter search space for classifier head tuning."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

DEFAULT_SEARCH_SPACE: dict[str, list[Any]] = {
    "feature_norm": ["none", "l2", "standard"],
    "pca_dim": [None, 32, 64, 128, 256],
    "classifier": ["logreg", "linear_svm", "mlp"],
    "class_weight_pos": [1.0, 1.5, 2.0, 3.0],
    "class_weight_neg": [1.0, 1.5, 2.0, 3.0],
    "prob_calibration": ["none", "platt", "isotonic"],
    "threshold": [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97],
    "context_pooling": ["center", "mean3", "max3"],
}


def sample_config(
    rng: random.Random, search_space: dict[str, list[Any]]
) -> dict[str, Any]:
    """Sample a random configuration from the given search space."""
    return {key: rng.choice(values) for key, values in search_space.items()}


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic hash of a config dict for deduplication."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
