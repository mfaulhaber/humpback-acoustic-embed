"""Shared types and rejection reason codes for the 5-second sample builder."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from humpback.classifier.raven_parser import RavenAnnotation

# ---------------------------------------------------------------------------
# Rejection reason codes
# ---------------------------------------------------------------------------

REASON_INSUFFICIENT_BACKGROUND = "insufficient_background"
REASON_CONTAMINATION_DETECTED = "contamination_detected"
REASON_ANNOTATION_CLUSTER = "annotation_cluster"
REASON_ACOUSTIC_MISMATCH = "acoustic_mismatch"
REASON_ASSEMBLY_FAILURE = "assembly_failure"
REASON_INVALID_ANNOTATION = "invalid_annotation"
REASON_VALIDATION_FAILED = "validation_failed"

# ---------------------------------------------------------------------------
# Stage 1 — Annotation normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizedAnnotation:
    """An annotation with computed midpoint and validity status."""

    original: RavenAnnotation
    midpoint_sec: float
    duration_sec: float
    valid: bool
    rejection_reason: str | None = None


# ---------------------------------------------------------------------------
# Stage 2 — Exclusion map
# ---------------------------------------------------------------------------


@dataclass
class ProtectedInterval:
    """A time interval that must not be used as background audio."""

    start_sec: float
    end_sec: float
    annotation_index: int


@dataclass
class ExclusionMap:
    """Collection of merged protected intervals covering all annotations."""

    protected_intervals: list[ProtectedInterval]

    def overlaps(self, start: float, end: float) -> bool:
        """Return True if [start, end) overlaps any protected interval."""
        for iv in self.protected_intervals:
            if start < iv.end_sec and end > iv.start_sec:
                return True
        return False


# ---------------------------------------------------------------------------
# Stage 3 — Background fragment discovery
# ---------------------------------------------------------------------------


@dataclass
class BackgroundFragment:
    """A candidate background audio fragment outside protected intervals."""

    start_sec: float
    end_sec: float
    duration_sec: float
    distance_from_target: float  # seconds from annotation midpoint
    audio: NDArray[np.floating] | None = None  # populated when extracted


# ---------------------------------------------------------------------------
# Sample result (used in later phases, defined here for type stability)
# ---------------------------------------------------------------------------


@dataclass
class SampleMetadata:
    """Provenance metadata for an assembled sample."""

    fragment_starts: list[float] = field(default_factory=list)
    fragment_ends: list[float] = field(default_factory=list)
    fragment_durations: list[float] = field(default_factory=list)
    similarity_scores: list[float] = field(default_factory=list)
    splice_points: list[int] = field(default_factory=list)
    target_start_sec: float = 0.0
    target_end_sec: float = 0.0
    target_duration_sec: float = 0.0
    window_size_sec: float = 5.0


@dataclass
class SampleResult:
    """The outcome of sample building for a single annotation."""

    accepted: bool
    audio: NDArray[np.floating] | None
    sr: int
    call_type: str
    source_filename: str
    annotation: RavenAnnotation | None
    metadata: SampleMetadata | None = None
    rejection_reason: str | None = None
