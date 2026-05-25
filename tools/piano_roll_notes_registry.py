"""Algorithm registry for the Piano Roll Notes debug test-bed.

Maps short variant names to thin callables with the shared signature
``(audio, sample_rate, *, job_id, event_id, event_start_utc,
ridge_sidecar_rows=None) -> NotesV3Result``. The wrapper constructs the
right ``ExtractNotesV*Params`` for each version so the CLI can swap
variants by name.

This module is for the test-bed only — the production worker dispatches
on ``extractor_version`` strings directly and does not import this
registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from humpback.processing.note_extractor_v3 import (
    ExtractNotesV3Params,
    NotesV3Result,
    extract_notes_v3,
)
from humpback.processing.note_extractor_v4 import (
    ExtractNotesV4Params,
    extract_notes_v4,
)
from humpback.processing.note_extractor_v5 import (
    ExtractNotesV5Params,
    extract_notes_v5,
)

__all__ = ["EXTRACTORS", "ExtractorFn"]


ExtractorFn = Callable[..., NotesV3Result]


def _run_v3(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    return extract_notes_v3(
        audio,
        sample_rate,
        params=ExtractNotesV3Params(
            job_id=job_id,
            event_id=event_id,
            event_start_utc=event_start_utc,
        ),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


def _run_v4(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    return extract_notes_v4(
        audio,
        sample_rate,
        params=ExtractNotesV4Params(
            job_id=job_id,
            event_id=event_id,
            event_start_utc=event_start_utc,
        ),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


def _run_v5(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    return extract_notes_v5(
        audio,
        sample_rate,
        params=ExtractNotesV5Params(
            job_id=job_id,
            event_id=event_id,
            event_start_utc=event_start_utc,
        ),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


EXTRACTORS: dict[str, ExtractorFn] = {
    "v3": _run_v3,
    "v4": _run_v4,
    "v5": _run_v5,
}
