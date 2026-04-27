"""Summary statistics for decoded HMM sequences.

Computes per-state occupancy, dwell-time histograms, and the
observed transition matrix from Viterbi state assignments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StateSummary:
    """Per-state summary entry."""

    state: int
    occupancy: float
    mean_dwell_frames: float
    dwell_histogram: list[int] = field(default_factory=list)


@dataclass
class SequenceSummary:
    """Full summary across all decoded sequences."""

    states: list[StateSummary]
    transition_matrix: np.ndarray


def _compute_dwell_runs(
    state_sequences: list[np.ndarray], n_states: int
) -> dict[int, list[int]]:
    """Extract per-state run lengths from Viterbi output."""
    runs: dict[int, list[int]] = {s: [] for s in range(n_states)}
    for seq in state_sequences:
        if len(seq) == 0:
            continue
        current_state = int(seq[0])
        run_len = 1
        for t in range(1, len(seq)):
            if int(seq[t]) == current_state:
                run_len += 1
            else:
                runs[current_state].append(run_len)
                current_state = int(seq[t])
                run_len = 1
        runs[current_state].append(run_len)
    return runs


def _compute_transition_matrix(
    state_sequences: list[np.ndarray], n_states: int
) -> np.ndarray:
    """Compute row-normalized transition counts from Viterbi sequences."""
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for seq in state_sequences:
        for t in range(len(seq) - 1):
            counts[int(seq[t]), int(seq[t + 1])] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def compute_summary(
    state_sequences: list[np.ndarray],
    n_states: int,
) -> SequenceSummary:
    """Compute summary statistics from Viterbi state assignments.

    Parameters
    ----------
    state_sequences
        Per-sequence Viterbi state arrays (int-valued).
    n_states
        Number of HMM states.

    Returns
    -------
    SequenceSummary
        Per-state occupancy, dwell histograms, and transition matrix.
    """
    all_states = np.concatenate(state_sequences) if state_sequences else np.array([])
    total_frames = len(all_states)

    runs = _compute_dwell_runs(state_sequences, n_states)
    transition_matrix = _compute_transition_matrix(state_sequences, n_states)

    summaries: list[StateSummary] = []
    for s in range(n_states):
        count = int(np.sum(all_states == s)) if total_frames > 0 else 0
        occupancy = count / total_frames if total_frames > 0 else 0.0
        dwell_list = runs[s]
        mean_dwell = float(np.mean(dwell_list)) if dwell_list else 0.0
        summaries.append(
            StateSummary(
                state=s,
                occupancy=occupancy,
                mean_dwell_frames=mean_dwell,
                dwell_histogram=dwell_list,
            )
        )

    return SequenceSummary(states=summaries, transition_matrix=transition_matrix)
