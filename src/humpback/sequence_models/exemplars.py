"""Per-state exemplar selection for HMM interpretation.

Picks representative windows for each HMM state:
- high_confidence: highest max_state_probability
- mean_nearest: closest to the PCA centroid (L2)
- boundary: lowest max_state_probability (state-ambiguous)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WindowMeta:
    sequence_id: str
    position_in_sequence: int
    audio_file_id: int | None
    start_timestamp: float
    end_timestamp: float
    max_state_probability: float
    extras: dict[str, str | int | float | None] = field(default_factory=dict)


def select_exemplars(
    pca_embeddings: np.ndarray,
    viterbi_states: np.ndarray,
    metadata: list[WindowMeta],
    n_states: int,
    *,
    n_exemplars_per_type: int = 3,
) -> dict[str, Any]:
    """Select exemplar windows for each HMM state.

    Parameters
    ----------
    pca_embeddings
        PCA-reduced embeddings, shape ``(N, pca_dims)``.
    viterbi_states
        Viterbi state per window, shape ``(N,)``.
    metadata
        Per-window metadata in the same order as embeddings.
    n_states
        Number of HMM states.
    n_exemplars_per_type
        How many exemplars to pick per category per state.
    """
    per_state: dict[str, list[dict[str, Any]]] = {}

    for s in range(n_states):
        mask = viterbi_states == s
        indices = np.where(mask)[0]
        state_key = str(s)

        if len(indices) == 0:
            per_state[state_key] = []
            continue

        probs = np.array([metadata[i].max_state_probability for i in indices])
        embeddings_s = pca_embeddings[indices]

        k = min(n_exemplars_per_type, len(indices))

        high_conf_order = np.argsort(-probs)[:k]
        boundary_order = np.argsort(probs)[:k]

        centroid = embeddings_s.mean(axis=0)
        dists = np.linalg.norm(embeddings_s - centroid, axis=1)
        nearest_order = np.argsort(dists)[:k]

        records: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()

        for picks, etype in [
            (high_conf_order, "high_confidence"),
            (nearest_order, "mean_nearest"),
            (boundary_order, "boundary"),
        ]:
            for local_idx in picks:
                global_idx = int(indices[local_idx])
                key = (etype, global_idx)
                if key in seen:
                    continue
                seen.add(key)
                m = metadata[global_idx]
                records.append(
                    {
                        "sequence_id": m.sequence_id,
                        "position_in_sequence": m.position_in_sequence,
                        "audio_file_id": m.audio_file_id,
                        "start_timestamp": m.start_timestamp,
                        "end_timestamp": m.end_timestamp,
                        "max_state_probability": float(m.max_state_probability),
                        "exemplar_type": etype,
                        "extras": dict(m.extras),
                    }
                )

        per_state[state_key] = records

    return {"n_states": n_states, "states": per_state}
