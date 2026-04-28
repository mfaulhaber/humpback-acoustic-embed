"""State-to-label distribution for HMM interpretation.

Joins HMM window timestamps with detection-window extents and
vocalization labels using center-time-in-window semantics (spec §5.4).
Pure function — callers supply the label data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DetectionWindow:
    row_id: str
    start_utc: float
    end_utc: float


@dataclass
class LabelRecord:
    row_id: str
    label: str


def compute_label_distribution(
    states: list[dict[str, Any]],
    detection_windows: list[DetectionWindow],
    labels: list[LabelRecord],
    n_states: int,
) -> dict[str, Any]:
    """Compute per-state label distributions via center-time join.

    Parameters
    ----------
    states
        HMM window rows, each with ``start_timestamp``, ``end_timestamp``,
        and ``viterbi_state``.
    detection_windows
        Detection windows with ``row_id``, ``start_utc``, ``end_utc``.
    labels
        Vocalization labels with ``row_id`` and ``label``.
    n_states
        Number of HMM states.

    Returns
    -------
    dict
        ``{ "n_states": int, "total_windows": int,
            "states": { "0": { "label_a": count, ... }, ... } }``
    """
    labels_by_row: dict[str, list[str]] = {}
    for rec in labels:
        labels_by_row.setdefault(rec.row_id, []).append(rec.label)

    per_state: dict[str, dict[str, int]] = {str(s): {} for s in range(n_states)}

    for window in states:
        center = (window["start_timestamp"] + window["end_timestamp"]) / 2.0
        state_key = str(int(window["viterbi_state"]))

        matched_labels: set[str] = set()
        for dw in detection_windows:
            if dw.start_utc <= center < dw.end_utc:
                for lbl in labels_by_row.get(dw.row_id, []):
                    matched_labels.add(lbl)

        if matched_labels:
            for lbl in matched_labels:
                per_state[state_key][lbl] = per_state[state_key].get(lbl, 0) + 1
        else:
            per_state[state_key]["unlabeled"] = (
                per_state[state_key].get("unlabeled", 0) + 1
            )

    return {
        "n_states": n_states,
        "total_windows": len(states),
        "states": per_state,
    }
