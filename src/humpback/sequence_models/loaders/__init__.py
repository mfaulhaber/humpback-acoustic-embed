"""Source-agnostic loaders for HMM interpretation artifacts (ADR-059).

Each embedding source (SurfPerch event-padded, CRNN region-based) has a
single loader module that knows the parquet column conventions for that
source. ``generate_interpretations()`` resolves the right loader at
runtime via :func:`get_loader` and consumes a generic
:class:`OverlayInputs` shape; downstream pure functions
(``compute_overlay``, ``select_exemplars``) never branch on source kind.

ADR-061 generalized the on-disk schema: the decoded sequence file is
``decoded.parquet`` (legacy: ``states.parquet``) with a ``label`` column
(legacy: ``viterbi_state``). Loaders accept an explicit
``decoded_artifact_path`` so masked-transformer per-k bundles can share
the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.exemplars import WindowMeta
from humpback.sequence_models.overlay import OverlayMetadata


def read_decoded_parquet(decoded_artifact_path: str | Path) -> pa.Table:
    """Read a decoded-sequence parquet, applying the legacy backwards shim.

    Looks at ``decoded_artifact_path`` first; if absent and the file lives
    in an HMM job directory whose legacy ``states.parquet`` exists, reads
    that and renames the ``viterbi_state`` column to ``label`` in memory.
    Always returns a table whose label column is named ``label``.
    """
    path = Path(decoded_artifact_path)
    if path.exists():
        table = pq.read_table(path)
    else:
        legacy = path.with_name("states.parquet")
        if not legacy.exists():
            raise FileNotFoundError(
                f"decoded artifact not found: {path} "
                f"(also no legacy states.parquet sibling)"
            )
        table = pq.read_table(legacy)
    if "label" not in table.column_names and "viterbi_state" in table.column_names:
        renamed = ["label" if c == "viterbi_state" else c for c in table.column_names]
        table = table.rename_columns(renamed)
    return table


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
    the in-memory key is preserved as ``viterbi_state`` regardless of
    on-disk column rename so downstream consumers stay unchanged.
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

_LOADER_FACTORIES: dict[str, type] = {
    SOURCE_KIND_SURFPERCH: SurfPerchLoader,
    SOURCE_KIND_REGION_CRNN: CrnnRegionLoader,
}


def get_loader(
    source_kind: str, decoded_artifact_path: str | Path
) -> SequenceArtifactLoader:
    """Construct the registered loader for ``source_kind`` bound to a
    decoded-sequence parquet path.

    Raises ``ValueError`` if the source kind is not registered.
    """
    factory = _LOADER_FACTORIES.get(source_kind)
    if factory is None:
        known = ", ".join(sorted(_LOADER_FACTORIES.keys()))
        raise ValueError(
            f"unknown sequence-models source kind {source_kind!r}; registered: {known}"
        )
    return factory(decoded_artifact_path=str(decoded_artifact_path))


__all__ = [
    "CrnnRegionLoader",
    "LabelDistributionInputs",
    "OverlayInputs",
    "SequenceArtifactLoader",
    "SurfPerchLoader",
    "get_loader",
    "read_decoded_parquet",
]
