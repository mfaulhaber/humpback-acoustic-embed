"""Source-agnostic loaders for HMM interpretation artifacts (ADR-059).

Each embedding source (SurfPerch event-padded, CRNN region-based) has a
single loader module that knows the parquet column conventions for that
source. ``generate_interpretations()`` resolves the right loader at
runtime via :func:`get_loader` and consumes a generic
:class:`OverlayInputs` shape; downstream pure functions
(``compute_overlay``, ``select_exemplars``) never branch on source kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.exemplars import WindowMeta
from humpback.sequence_models.overlay import OverlayMetadata


@dataclass
class OverlayInputs:
    """Source-agnostic bundle consumed by overlay/exemplar computation."""

    pca_model: Any
    raw_sequences: list[np.ndarray]
    viterbi_states: list[np.ndarray]
    max_probs: list[np.ndarray]
    metadata: OverlayMetadata
    window_metas: list[WindowMeta]


class SequenceArtifactLoader(Protocol):
    """Loader Protocol — one impl per embedding source family."""

    def load(
        self,
        storage_root: Path,
        hmm_job: HMMSequenceJob,
        cej: ContinuousEmbeddingJob,
    ) -> OverlayInputs: ...


from humpback.sequence_models.loaders.crnn_region import CrnnRegionLoader  # noqa: E402
from humpback.sequence_models.loaders.surfperch import SurfPerchLoader  # noqa: E402
from humpback.services.continuous_embedding_service import (  # noqa: E402
    SOURCE_KIND_REGION_CRNN,
    SOURCE_KIND_SURFPERCH,
)

_LOADERS: dict[str, SequenceArtifactLoader] = {
    SOURCE_KIND_SURFPERCH: SurfPerchLoader(),
    SOURCE_KIND_REGION_CRNN: CrnnRegionLoader(),
}


def get_loader(source_kind: str) -> SequenceArtifactLoader:
    """Return the registered loader for ``source_kind``.

    Raises ``ValueError`` if the source kind is not registered.
    """
    loader = _LOADERS.get(source_kind)
    if loader is None:
        known = ", ".join(sorted(_LOADERS.keys()))
        raise ValueError(
            f"unknown sequence-models source kind {source_kind!r}; registered: {known}"
        )
    return loader


__all__ = [
    "CrnnRegionLoader",
    "OverlayInputs",
    "SequenceArtifactLoader",
    "SurfPerchLoader",
    "get_loader",
]
