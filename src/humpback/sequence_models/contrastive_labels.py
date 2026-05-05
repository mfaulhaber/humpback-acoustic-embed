"""Human-correction labels for retrieval-aware contrastive training."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.models.call_parsing import VocalizationCorrection


class EffectiveEventLike(Protocol):
    @property
    def event_id(self) -> str: ...

    @property
    def region_id(self) -> str: ...

    @property
    def start_sec(self) -> float: ...

    @property
    def end_sec(self) -> float: ...


@dataclass(frozen=True)
class ContrastiveEventLabel:
    """One effective event annotated only with human correction labels."""

    event_id: str
    region_id: str
    start_timestamp: float
    end_timestamp: float
    human_types: tuple[str, ...]


def apply_human_correction_labels(
    *,
    effective_events: Sequence[EffectiveEventLike],
    corrections: list[VocalizationCorrection],
    region_start_timestamp: float | None,
) -> tuple[list[ContrastiveEventLabel], dict[str, Any]]:
    """Overlay region-scoped human correction rows onto effective events."""
    offset = float(region_start_timestamp or 0.0)
    corrections_by_type: Counter[str] = Counter()
    out: list[ContrastiveEventLabel] = []
    for event in effective_events:
        added: set[str] = set()
        removed: set[str] = set()
        for correction in corrections:
            if float(correction.start_sec) < float(event.end_sec) and float(
                correction.end_sec
            ) > float(event.start_sec):
                key = f"{correction.correction_type}:{correction.type_name}"
                corrections_by_type[key] += 1
                if correction.correction_type == "add":
                    added.add(correction.type_name)
                elif correction.correction_type == "remove":
                    removed.add(correction.type_name)
        labels = tuple(sorted(added - removed))
        out.append(
            ContrastiveEventLabel(
                event_id=event.event_id,
                region_id=event.region_id,
                start_timestamp=float(event.start_sec) + offset,
                end_timestamp=float(event.end_sec) + offset,
                human_types=labels,
            )
        )

    label_counter: Counter[str] = Counter()
    for event in out:
        for label in event.human_types:
            label_counter[label] += 1
    return out, {
        "total_correction_rows": len(corrections),
        "events_with_human_labels": sum(1 for event in out if event.human_types),
        "corrections_by_type": dict(corrections_by_type.most_common()),
        "event_label_counts": dict(label_counter.most_common()),
    }


async def load_contrastive_event_labels(
    session: AsyncSession,
    *,
    storage_root: Path,
    event_segmentation_job_id: str,
    region_detection_job_id: str,
    region_start_timestamp: float | None,
) -> tuple[list[ContrastiveEventLabel], dict[str, Any]]:
    """Load effective events with human-correction-only label sets."""
    effective_events = await load_effective_events(
        session,
        event_segmentation_job_id=event_segmentation_job_id,
        storage_root=storage_root,
    )
    correction_result = await session.execute(
        select(VocalizationCorrection).where(
            VocalizationCorrection.region_detection_job_id == region_detection_job_id
        )
    )
    corrections = list(correction_result.scalars().all())
    return apply_human_correction_labels(
        effective_events=effective_events,
        corrections=corrections,
        region_start_timestamp=region_start_timestamp,
    )


__all__ = [
    "ContrastiveEventLabel",
    "apply_human_correction_labels",
    "load_contrastive_event_labels",
]
