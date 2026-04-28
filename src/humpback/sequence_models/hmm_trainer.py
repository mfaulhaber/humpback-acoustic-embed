"""HMM training on PCA-reduced embedding sequences.

Wraps ``hmmlearn.GaussianHMM`` with the multi-sequence API
(concatenated array + lengths vector).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from hmmlearn.hmm import GaussianHMM


@dataclass
class HMMTrainResult:
    """Outputs from a single HMM fit."""

    model: GaussianHMM
    train_log_likelihood: float
    n_train_sequences: int
    n_train_frames: int
    training_mask: list[bool]


def fit_hmm(
    sequences: list[np.ndarray],
    *,
    n_states: int,
    covariance_type: str = "diag",
    n_iter: int = 100,
    tol: float = 1e-4,
    random_state: int = 42,
    min_sequence_length_frames: int = 10,
) -> HMMTrainResult:
    """Fit a GaussianHMM on the training-eligible subset of sequences.

    Parameters
    ----------
    sequences
        PCA-reduced sequences, each shape ``(T_i, D)``.
    min_sequence_length_frames
        Sequences shorter than this are excluded from training but are
        still available for decoding (marked via ``training_mask``).

    Returns
    -------
    HMMTrainResult
        Fitted model + training metadata.
    """
    training_mask = [len(seq) >= min_sequence_length_frames for seq in sequences]
    train_seqs = [s for s, m in zip(sequences, training_mask) if m]

    if not train_seqs:
        raise ValueError(
            f"No sequences meet min_sequence_length_frames={min_sequence_length_frames} "
            f"(longest sequence has {max(len(s) for s in sequences)} frames)"
        )

    concatenated = np.concatenate(train_seqs, axis=0)
    lengths = [len(s) for s in train_seqs]

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=False,
    )
    model.fit(concatenated, lengths)

    log_lik: Any = model.score(concatenated, lengths)

    return HMMTrainResult(
        model=model,
        train_log_likelihood=float(log_lik),
        n_train_sequences=len(train_seqs),
        n_train_frames=int(concatenated.shape[0]),
        training_mask=training_mask,
    )
