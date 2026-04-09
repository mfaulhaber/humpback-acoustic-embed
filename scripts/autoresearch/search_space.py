"""Declarative hyperparameter search space for autoresearch.

Thin wrapper — canonical definitions live in
``humpback.services.hyperparameter_service.search_space``.
"""

from __future__ import annotations

from humpback.services.hyperparameter_service.search_space import (
    DEFAULT_SEARCH_SPACE,
    config_hash,
    sample_config as _sample_config,
)

__all__ = ["SEARCH_SPACE", "config_hash", "sample_config"]

# Legacy alias — the original script included hard_negative_fraction.
SEARCH_SPACE = {
    **DEFAULT_SEARCH_SPACE,
    "hard_negative_fraction": [0.0, 0.1, 0.2, 0.4],
}


def sample_config(rng):  # noqa: ANN001, ANN201
    """Sample from the full legacy search space (including hard_negative_fraction)."""
    return _sample_config(rng, SEARCH_SPACE)
