"""Deterministic SurfPerch embedder stub used by the continuous-embedding
worker tests.

The stub satisfies the worker's ``EmbedderProtocol``: same audio inputs
produce the same fixed-shape embeddings. It does not load any audio —
the producer worker is plumbing-only, so a per-window deterministic
synthesis from ``(merged_span_id, window_index_in_span)`` is sufficient
for end-to-end coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humpback.processing.region_windowing import MergedSpan, iter_windows

DEFAULT_VECTOR_DIM = 16


@dataclass
class SurfPerchStub:
    """Deterministic stub matching the worker's EmbedderProtocol signature.

    Each window's vector is derived from a stable hash of
    ``(merged_span_id, window_index_in_span)`` so that the same job
    parameters produce the same parquet on repeat runs. The vector
    dimension is configurable so tests can confirm the worker reads
    ``vector_dim`` from the embedder's output rather than hard-coding.
    """

    vector_dim: int = DEFAULT_VECTOR_DIM
    fail_on_span: int | None = None
    call_count: int = 0

    def __call__(
        self,
        *,
        span: MergedSpan,
        region_job,
        model_version: str,
        hop_seconds: float,
        window_size_seconds: float,
        target_sample_rate: int,
        settings,
    ) -> list[np.ndarray]:
        self.call_count += 1
        if self.fail_on_span is not None and span.merged_span_id == self.fail_on_span:
            raise RuntimeError(
                f"surfperch stub configured to fail on span {span.merged_span_id}"
            )

        windows = list(
            iter_windows(
                span,
                hop_seconds=hop_seconds,
                window_size_seconds=window_size_seconds,
            )
        )
        return [
            self._make_vector(span.merged_span_id, w.window_index_in_span)
            for w in windows
        ]

    def _make_vector(self, span_id: int, window_index: int) -> np.ndarray:
        rng = np.random.default_rng(seed=(span_id * 1_000_003 + window_index))
        return rng.standard_normal(self.vector_dim).astype(np.float32)
