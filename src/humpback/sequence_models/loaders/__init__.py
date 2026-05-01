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
from sqlalchemy.ext.asyncio import AsyncSession

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


@dataclass
class LabelDistributionInputs:
    """Source-agnostic bundle consumed by label-distribution computation.

    ``hydrophone_id`` resolves the DetectionJob/VocalizationLabel SQL fetch
    that lives in the service. ``state_rows`` carries the same shape across
    sources (``start_timestamp``, ``end_timestamp``, ``viterbi_state``);
    ``tier_per_row`` is parallel to ``state_rows`` and ``None`` for sources
    without a tier dimension (SurfPerch).
    """

    hydrophone_id: str | None
    state_rows: list[dict[str, Any]]
    tier_per_row: list[str] | None


class SequenceArtifactLoader(Protocol):
    """Loader Protocol — one impl per embedding source family."""

    def load(
        self,
        storage_root: Path,
        hmm_job: HMMSequenceJob,
        cej: ContinuousEmbeddingJob,
    ) -> OverlayInputs: ...

    async def load_label_distribution_inputs(
        self,
        session: AsyncSession,
        storage_root: Path,
        hmm_job: HMMSequenceJob,
        cej: ContinuousEmbeddingJob,
    ) -> LabelDistributionInputs: ...


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
    "LabelDistributionInputs",
    "OverlayInputs",
    "SequenceArtifactLoader",
    "SurfPerchLoader",
    "get_loader",
]
