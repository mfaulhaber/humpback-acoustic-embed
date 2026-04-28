"""PCA / UMAP 2-D overlay for HMM state visualization.

Given a fitted PCA model and raw embedding sequences, projects embeddings
to 2-D via PCA (first two components) and UMAP (on PCA-reduced space),
then joins with Viterbi state assignments for scatter-plot rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from umap import UMAP


@dataclass
class OverlayMetadata:
    merged_span_ids: list[int]
    window_indices: list[int]
    start_times: list[float]
    end_times: list[float]


def compute_overlay(
    pca_model: PCA,
    raw_sequences: list[np.ndarray],
    viterbi_states: list[np.ndarray],
    max_state_probs: list[np.ndarray],
    metadata: OverlayMetadata,
    *,
    l2_normalize: bool = True,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int = 42,
) -> tuple[pa.Table, np.ndarray]:
    """Compute PCA + UMAP 2-D projections colored by HMM state.

    Returns the overlay table and the full PCA-reduced embeddings so
    callers can reuse them (e.g. for exemplar centroid distances).

    Parameters
    ----------
    pca_model
        Fitted sklearn PCA from the HMM training pipeline.
    raw_sequences
        Per-span raw embedding arrays, same order as Viterbi output.
    viterbi_states
        Per-span Viterbi state arrays.
    max_state_probs
        Per-span max-state-probability arrays.
    metadata
        Window-level identifiers (span ids, indices, timestamps).
    l2_normalize
        Whether to L2-normalize embeddings before PCA (must match the
        HMM job's setting).
    umap_n_neighbors, umap_min_dist, random_state
        UMAP hyperparameters.
    """
    concatenated = np.concatenate(raw_sequences, axis=0)

    if l2_normalize:
        concatenated = np.asarray(normalize(concatenated, norm="l2"), dtype=np.float32)

    pca_full = pca_model.transform(concatenated).astype(np.float32)
    pca_x = pca_full[:, 0]
    pca_y = pca_full[:, 1]

    n_samples = len(concatenated)
    if n_samples >= 3:
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(umap_n_neighbors, n_samples - 1),
            min_dist=umap_min_dist,
            random_state=random_state,
        )
        umap_2d = np.asarray(reducer.fit_transform(pca_full), dtype=np.float32)
        umap_x = umap_2d[:, 0]
        umap_y = umap_2d[:, 1]
    else:
        umap_x = np.full(n_samples, np.nan, dtype=np.float32)
        umap_y = np.full(n_samples, np.nan, dtype=np.float32)

    all_states = np.concatenate(viterbi_states).astype(np.int16)
    all_probs = np.concatenate(max_state_probs).astype(np.float32)

    table = pa.table(
        {
            "merged_span_id": pa.array(metadata.merged_span_ids, type=pa.int32()),
            "window_index_in_span": pa.array(metadata.window_indices, type=pa.int32()),
            "start_time_sec": pa.array(metadata.start_times, type=pa.float64()),
            "end_time_sec": pa.array(metadata.end_times, type=pa.float64()),
            "pca_x": pa.array(pca_x, type=pa.float32()),
            "pca_y": pa.array(pca_y, type=pa.float32()),
            "umap_x": pa.array(umap_x, type=pa.float32()),
            "umap_y": pa.array(umap_y, type=pa.float32()),
            "viterbi_state": pa.array(all_states, type=pa.int16()),
            "max_state_probability": pa.array(all_probs, type=pa.float32()),
        }
    )
    return table, pca_full
