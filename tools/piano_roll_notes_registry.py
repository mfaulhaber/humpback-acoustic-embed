"""Algorithm registry for the Piano Roll Notes debug test-bed.

Maps short variant names to thin callables with the shared signature
``(audio, sample_rate, *, job_id, event_id, event_start_utc,
pad_seconds=None, ridge_sidecar_rows=None) -> NotesV3Result``. The
wrapper constructs the right ``ExtractNotesV*Params`` for each version
so the CLI can swap variants by name. When ``pad_seconds`` is provided
it is forwarded to the extractor's params dataclass so the test-bed can
reproduce the worker's per-variant pad (v3/v4 default 0.05 s, v5
default 0.25 s — v5's background subtraction silently no-ops when the
pad provides fewer than ``background_min_pad_frames``).

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
from humpback.processing.note_extractor_v6 import (
    ExtractNotesV6Params,
    extract_notes_v6,
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
    pad_seconds: float | None = None,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    kwargs: dict[str, Any] = {
        "job_id": job_id,
        "event_id": event_id,
        "event_start_utc": event_start_utc,
    }
    if pad_seconds is not None:
        kwargs["pad_seconds"] = pad_seconds
    return extract_notes_v3(
        audio,
        sample_rate,
        params=ExtractNotesV3Params(**kwargs),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


def _run_v4(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    pad_seconds: float | None = None,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    kwargs: dict[str, Any] = {
        "job_id": job_id,
        "event_id": event_id,
        "event_start_utc": event_start_utc,
    }
    if pad_seconds is not None:
        kwargs["pad_seconds"] = pad_seconds
    return extract_notes_v4(
        audio,
        sample_rate,
        params=ExtractNotesV4Params(**kwargs),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


def _run_v5(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    pad_seconds: float | None = None,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    kwargs: dict[str, Any] = {
        "job_id": job_id,
        "event_id": event_id,
        "event_start_utc": event_start_utc,
    }
    if pad_seconds is not None:
        kwargs["pad_seconds"] = pad_seconds
    return extract_notes_v5(
        audio,
        sample_rate,
        params=ExtractNotesV5Params(**kwargs),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


def _run_v6(
    audio: np.ndarray,
    sample_rate: int,
    *,
    job_id: str,
    event_id: str,
    event_start_utc: float,
    pad_seconds: float | None = None,
    ridge_sidecar_rows: Sequence[Mapping[str, Any]] | None = None,
) -> NotesV3Result:
    kwargs: dict[str, Any] = {
        "job_id": job_id,
        "event_id": event_id,
        "event_start_utc": event_start_utc,
    }
    if pad_seconds is not None:
        kwargs["pad_seconds"] = pad_seconds
    return extract_notes_v6(
        audio,
        sample_rate,
        params=ExtractNotesV6Params(**kwargs),
        ridge_sidecar_rows=ridge_sidecar_rows,
    )


EXTRACTORS: dict[str, ExtractorFn] = {
    "v3": _run_v3,
    "v4": _run_v4,
    "v5": _run_v5,
    "v6": _run_v6,
}
