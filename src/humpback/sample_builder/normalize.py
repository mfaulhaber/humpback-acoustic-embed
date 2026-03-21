"""Stage 1: Validate annotations and compute midpoints."""

from __future__ import annotations

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.types import (
    REASON_INVALID_ANNOTATION,
    NormalizedAnnotation,
)


def normalize_annotations(
    annotations: list[RavenAnnotation],
    *,
    min_duration: float = 0.3,
    max_duration: float = 4.0,
) -> list[NormalizedAnnotation]:
    """Validate annotations and compute midpoint/duration.

    Parameters
    ----------
    annotations:
        Raw Raven annotations for a single recording.
    min_duration:
        Minimum annotation duration in seconds. Shorter annotations are rejected.
    max_duration:
        Maximum annotation duration in seconds. Longer annotations are rejected.

    Returns
    -------
    List of NormalizedAnnotation with validity flags and computed fields.
    """
    results: list[NormalizedAnnotation] = []
    for ann in annotations:
        duration = ann.end_time - ann.begin_time
        midpoint = ann.begin_time + duration / 2.0

        if duration <= 0:
            results.append(
                NormalizedAnnotation(
                    original=ann,
                    midpoint_sec=midpoint,
                    duration_sec=duration,
                    valid=False,
                    rejection_reason=REASON_INVALID_ANNOTATION,
                )
            )
        elif duration < min_duration:
            results.append(
                NormalizedAnnotation(
                    original=ann,
                    midpoint_sec=midpoint,
                    duration_sec=duration,
                    valid=False,
                    rejection_reason=REASON_INVALID_ANNOTATION,
                )
            )
        elif duration > max_duration:
            results.append(
                NormalizedAnnotation(
                    original=ann,
                    midpoint_sec=midpoint,
                    duration_sec=duration,
                    valid=False,
                    rejection_reason=REASON_INVALID_ANNOTATION,
                )
            )
        else:
            results.append(
                NormalizedAnnotation(
                    original=ann,
                    midpoint_sec=midpoint,
                    duration_sec=duration,
                    valid=True,
                )
            )
    return results
