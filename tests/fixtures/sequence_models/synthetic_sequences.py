"""Synthetic sequence generator for HMM unit tests.

Generates embedding sequences with planted state structure: a Markov
chain drives discrete state transitions, and each state emits from a
distinct Gaussian cluster. This lets tests verify that the HMM recovers
known structure.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_sequences(
    *,
    n_states: int = 3,
    n_sequences: int = 10,
    min_length: int = 30,
    max_length: int = 80,
    vector_dim: int = 16,
    transition_matrix: np.ndarray | None = None,
    cluster_separation: float = 5.0,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate synthetic embedding sequences with planted HMM structure.

    Returns
    -------
    embeddings : list[ndarray[T_i, vector_dim]]
        Per-sequence embedding matrices.
    ground_truth_states : list[ndarray[T_i]]
        Per-sequence integer state labels (0-indexed).
    """
    rng = np.random.default_rng(seed)

    if transition_matrix is None:
        raw = rng.dirichlet(np.ones(n_states) * 0.3, size=n_states)
        # Strengthen self-transitions for temporal coherence
        for i in range(n_states):
            raw[i, i] += 2.0
        transition_matrix = raw / raw.sum(axis=1, keepdims=True)

    assert transition_matrix is not None
    tm = transition_matrix
    assert tm.shape == (n_states, n_states)

    # Per-state emission centers: well-separated in embedding space
    centers = rng.standard_normal((n_states, vector_dim)) * cluster_separation

    embeddings: list[np.ndarray] = []
    ground_truth_states: list[np.ndarray] = []

    for _ in range(n_sequences):
        length = rng.integers(min_length, max_length + 1)
        states = np.empty(length, dtype=np.int32)
        states[0] = rng.integers(0, n_states)
        for t in range(1, length):
            states[t] = rng.choice(n_states, p=tm[states[t - 1]])

        noise = rng.standard_normal((length, vector_dim)).astype(np.float32) * 0.5
        seq = centers[states].astype(np.float32) + noise

        embeddings.append(seq)
        ground_truth_states.append(states)

    return embeddings, ground_truth_states
