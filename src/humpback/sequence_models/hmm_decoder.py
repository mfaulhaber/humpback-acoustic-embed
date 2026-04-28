"""Viterbi decoding + posterior computation for fitted HMM models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class DecodedSequence:
    """Decode output for a single sequence."""

    viterbi_states: np.ndarray
    posteriors: np.ndarray
    max_state_probability: np.ndarray


def decode_sequence(model: GaussianHMM, sequence: np.ndarray) -> DecodedSequence:
    """Decode a single sequence via Viterbi and compute posteriors."""
    _, states = model.decode(sequence, algorithm="viterbi")
    posteriors = model.predict_proba(sequence)
    max_prob = posteriors.max(axis=1).astype(np.float32)

    return DecodedSequence(
        viterbi_states=states.astype(np.int16),
        posteriors=posteriors.astype(np.float32),
        max_state_probability=max_prob,
    )


def decode_sequences(
    model: GaussianHMM, sequences: list[np.ndarray]
) -> list[DecodedSequence]:
    """Decode all sequences (including those below training threshold)."""
    return [decode_sequence(model, seq) for seq in sequences]
