"""Pluggable objective functions for autoresearch model selection."""

from __future__ import annotations

from typing import Any


def default_objective(metrics: dict[str, Any]) -> float:
    """Conservative FP-penalizing objective.

    Heavily penalizes high-confidence false positives (15x) and
    general false positives (3x) relative to recall.
    """
    return (
        metrics["recall"]
        - 15.0 * metrics["high_conf_fp_rate"]
        - 3.0 * metrics["fp_rate"]
    )


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
