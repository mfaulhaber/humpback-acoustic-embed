"""Pluggable objective functions for autoresearch model selection.

Thin wrapper — the default objective lives in
``humpback.services.hyperparameter_service.search``.
"""

from __future__ import annotations

from typing import Any

from humpback.services.hyperparameter_service.search import default_objective

OBJECTIVES: dict[str, type[object] | object] = {
    "default": default_objective,
}


def get_objective(name: str) -> Any:
    """Look up an objective function by name."""
    if name not in OBJECTIVES:
        available = ", ".join(sorted(OBJECTIVES.keys()))
        msg = f"Unknown objective {name!r}. Available: {available}"
        raise ValueError(msg)
    return OBJECTIVES[name]
