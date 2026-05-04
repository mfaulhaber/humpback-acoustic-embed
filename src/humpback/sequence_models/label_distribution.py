"""Event-scoped state-to-label distribution for HMM/Masked-Transformer.

Distributes types from Call Parsing Pass 3 (``EventClassificationJob``)
to sequence windows via event-scoped inversion. Events are non-overlapping
by construction (Pass 2 segmentation, ADR-062); each sequence window
whose center time falls inside an effective event inherits that event's
above-threshold types (with ``VocalizationCorrection`` overlay applied).
Windows outside every event — and windows inside an event whose
corrections wiped all types — fall into the reserved
``BACKGROUND_LABEL`` bucket.

Replaces the previous detection-window center-time match against
``vocalization_labels``. The Vocalization Labeling workspace and its
table are unaffected.

See: ``docs/specs/2026-05-04-sequence-models-classify-label-source-design.md``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.call_parsing.storage import (
    classification_job_dir,
    read_typed_events,
)
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
    VocalizationCorrection,
)

BACKGROUND_LABEL = "(background)"


@dataclass(frozen=True)
class EffectiveEventLabels:
    """One effective event with its corrected, above-threshold type set.

    ``start_utc`` / ``end_utc`` are absolute UTC epoch seconds — the same
    domain as ``decoded.parquet``'s ``start_timestamp`` / ``end_timestamp``,
    bridged from the source-audio-relative ``Event.start_sec`` /
    ``end_sec`` via the upstream ``RegionDetectionJob.start_timestamp``.

    ``types`` is the union of (model types where ``above_threshold``) and
    user-added ``VocalizationCorrection`` rows, minus user-removed rows.
    May be empty when corrections wipe every surviving type.
    """

    event_id: str
    start_utc: float
    end_utc: float
    types: frozenset[str]
    confidences: dict[str, float] = field(default_factory=dict)


async def load_effective_event_labels(
    session: AsyncSession,
    *,
    event_classification_job_id: str,
    storage_root: Path,
) -> list[EffectiveEventLabels]:
    """Load effective events with their corrected, above-threshold types.

    Reads ``typed_events.parquet`` for the bound Classify job, applies
    ``VocalizationCorrection`` overlay (region-scoped, keyed by
    ``(region_detection_job_id, start_sec, end_sec, type_name)``), and
    bridges seconds-from-region-start to absolute UTC epoch via the
    upstream ``RegionDetectionJob.start_timestamp``.

    Effective type set per event ::

        (model types where above_threshold == True)
        ∪ (VocalizationCorrection rows overlapping event with
           correction_type == "add")
        − (VocalizationCorrection rows overlapping event with
           correction_type == "remove")

    Events whose corrected type set is empty are still returned (the
    interval is real); ``assign_labels_to_windows`` treats them the same
    as "no event" — `event_id=None`, `event_types=[]` — so chart and
    exemplar invariants stay consistent (`event_id` is set iff at least
    one surviving label exists).

    Returns events sorted by ``start_utc``.
    """
    cls_job = await session.get(EventClassificationJob, event_classification_job_id)
    if cls_job is None:
        raise ValueError(
            f"EventClassificationJob {event_classification_job_id} not found"
        )

    seg_job = await session.get(EventSegmentationJob, cls_job.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"EventSegmentationJob {cls_job.event_segmentation_job_id} not found"
        )

    rdj = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if rdj is None:
        raise ValueError(
            f"RegionDetectionJob {seg_job.region_detection_job_id} not found"
        )
    timestamp_offset = float(rdj.start_timestamp or 0.0)

    # Effective events (boundary-corrected, segmentation-scoped).
    effective_events = await load_effective_events(
        session,
        event_segmentation_job_id=cls_job.event_segmentation_job_id,
        storage_root=storage_root,
    )

    # Per-type model output for each event_id (above-threshold only).
    typed_path = (
        classification_job_dir(storage_root, event_classification_job_id)
        / "typed_events.parquet"
    )
    types_by_event: dict[str, dict[str, float]] = defaultdict(dict)
    if typed_path.exists():
        for te in read_typed_events(typed_path):
            if te.above_threshold:
                types_by_event[te.event_id][te.type_name] = float(te.score)

    # VocalizationCorrection overlay — region-scoped, time-range-keyed,
    # correction_type ∈ {"add", "remove"}.
    corr_result = await session.execute(
        select(VocalizationCorrection).where(
            VocalizationCorrection.region_detection_job_id
            == seg_job.region_detection_job_id
        )
    )
    corrections = list(corr_result.scalars().all())

    out: list[EffectiveEventLabels] = []
    for event in effective_events:
        types = set(types_by_event.get(event.event_id, {}).keys())
        confidences = dict(types_by_event.get(event.event_id, {}))

        for vc in corrections:
            if vc.start_sec < event.end_sec and vc.end_sec > event.start_sec:
                if vc.correction_type == "add":
                    types.add(vc.type_name)
                elif vc.correction_type == "remove":
                    types.discard(vc.type_name)
                    confidences.pop(vc.type_name, None)

        # Drop confidences for any type no longer in the surviving set.
        confidences = {t: c for t, c in confidences.items() if t in types}

        out.append(
            EffectiveEventLabels(
                event_id=event.event_id,
                start_utc=float(event.start_sec) + timestamp_offset,
                end_utc=float(event.end_sec) + timestamp_offset,
                types=frozenset(types),
                confidences=confidences,
            )
        )

    out.sort(key=lambda e: (e.start_utc, e.end_utc, e.event_id))
    return out


@dataclass(frozen=True)
class WindowAnnotation:
    """Per-window annotation produced by ``assign_labels_to_windows``.

    Parallel to the input ``decoded`` rows. ``event_id`` is ``None`` for
    windows in the background bucket — either because their center time
    falls outside every effective event, or because the event they fall
    inside has an empty corrected type set.
    """

    event_id: str | None
    event_types: tuple[str, ...]
    event_confidence: dict[str, float]


def assign_labels_to_windows(
    rows: list[dict[str, Any]],
    events: list[EffectiveEventLabels],
) -> list[WindowAnnotation]:
    """Tag each sequence window with the surrounding event's types.

    Two-pointer event-scoped inversion in O(n_windows + n_events):

      1. The events list is already sorted by ``start_utc``
         (``load_effective_event_labels`` guarantees this).
      2. Compute a stable index ordering of ``rows`` by center time.
      3. For each event in order, walk row indices whose center is in
         ``[event.start_utc, event.end_utc)`` and tag them with the
         event's id and types.
      4. Untouched rows — and rows tagged to an event with an empty
         type set — produce ``WindowAnnotation(event_id=None,
         event_types=(), event_confidence={})``.

    Events are non-overlapping by construction (ADR-062), so each row
    matches at most one event.

    The returned list is parallel to the input ``rows`` list.
    """
    n = len(rows)
    annotations: list[WindowAnnotation | None] = [None] * n

    if n == 0:
        return []

    # Stable order by center time. Ties broken by original index so
    # the assignment is deterministic.
    order = sorted(
        range(n),
        key=lambda i: (
            (float(rows[i]["start_timestamp"]) + float(rows[i]["end_timestamp"])) / 2.0,
            i,
        ),
    )

    centers = [
        (float(rows[i]["start_timestamp"]) + float(rows[i]["end_timestamp"])) / 2.0
        for i in order
    ]

    cursor = 0
    for event in events:
        # Skip rows entirely before this event.
        while cursor < n and centers[cursor] < event.start_utc:
            cursor += 1
        # Tag rows whose center is inside this event.
        i = cursor
        while i < n and centers[i] < event.end_utc:
            row_idx = order[i]
            if event.types:
                annotations[row_idx] = WindowAnnotation(
                    event_id=event.event_id,
                    event_types=tuple(sorted(event.types)),
                    event_confidence=dict(event.confidences),
                )
            else:
                annotations[row_idx] = WindowAnnotation(
                    event_id=None,
                    event_types=(),
                    event_confidence={},
                )
            i += 1
        cursor = i

    background = WindowAnnotation(event_id=None, event_types=(), event_confidence={})
    return [a if a is not None else background for a in annotations]


def compute_label_distribution(
    rows: list[dict[str, Any]],
    annotations: list[WindowAnnotation],
    n_states: int,
) -> dict[str, Any]:
    """Bucket per-window annotations into the simplified per-state shape.

    Output ::

        {
          "n_states": int,
          "total_windows": int,
          "states": { "<state_idx>": { "<label>": count, ... }, ... },
        }

    No tier dimension. Each state is ``{label: int}``. ``BACKGROUND_LABEL``
    appears for windows with empty ``event_types``. Multi-label events
    contribute ``+1`` to each label's bucket per overlapping window
    (matching the union semantics of the previous loader). The per-state
    total may exceed ``total_windows`` because of multi-label events; the
    chart reads each label bar independently.
    """
    if len(rows) != len(annotations):
        raise ValueError(
            f"rows length {len(rows)} does not match annotations length "
            f"{len(annotations)}"
        )

    per_state: dict[str, dict[str, int]] = {str(s): {} for s in range(n_states)}

    for row, ann in zip(rows, annotations):
        state_key = str(int(row["viterbi_state"]))
        bucket = per_state.setdefault(state_key, {})
        if ann.event_types:
            for label in ann.event_types:
                bucket[label] = bucket.get(label, 0) + 1
        else:
            bucket[BACKGROUND_LABEL] = bucket.get(BACKGROUND_LABEL, 0) + 1

    return {
        "n_states": n_states,
        "total_windows": len(rows),
        "states": per_state,
    }


__all__ = [
    "BACKGROUND_LABEL",
    "EffectiveEventLabels",
    "WindowAnnotation",
    "assign_labels_to_windows",
    "compute_label_distribution",
    "load_effective_event_labels",
]
