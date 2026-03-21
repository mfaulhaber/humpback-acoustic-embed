"""Stage 6: Plan multi-fragment background assembly around a target annotation."""

from __future__ import annotations

from dataclasses import dataclass, field


from humpback.sample_builder.types import (
    REASON_INSUFFICIENT_BACKGROUND,
    BackgroundFragment,
    NormalizedAnnotation,
)


@dataclass
class AssemblyPlan:
    """Blueprint for assembling a 5-second sample."""

    left_fragments: list[BackgroundFragment] = field(default_factory=list)
    target_start_sec: float = 0.0
    target_end_sec: float = 0.0
    right_fragments: list[BackgroundFragment] = field(default_factory=list)
    splice_points_sec: list[float] = field(default_factory=list)
    can_assemble: bool = False
    rejection_reason: str | None = None
    window_size: float = 5.0
    left_needed: float = 0.0
    right_needed: float = 0.0


def plan_assembly(
    target: NormalizedAnnotation,
    scored_candidates: list[BackgroundFragment],
    *,
    window_size: float = 5.0,
    min_fill_fraction: float = 0.9,
) -> AssemblyPlan:
    """Plan how to assemble background fragments around a target annotation.

    Centers the vocalization midpoint at the window center (~2.5s in a 5s
    window). Greedily fills left and right sides from ranked candidates
    (assumed pre-sorted by proximity/similarity score).

    Parameters
    ----------
    target:
        The annotation to center in the sample window.
    scored_candidates:
        Background fragments sorted by preference (closest / best first).
    window_size:
        Total sample duration in seconds (default 5.0).
    min_fill_fraction:
        Minimum fraction of needed background that must be filled (default 0.9).
        Below this, assembly is rejected.

    Returns
    -------
    AssemblyPlan with fragment assignments, splice points, and assembly status.
    """
    half_window = window_size / 2.0
    call_half = target.duration_sec / 2.0

    # Time needed for background on each side
    left_needed = half_window - call_half
    right_needed = half_window - call_half

    # Clamp to non-negative (annotation longer than half window)
    left_needed = max(0.0, left_needed)
    right_needed = max(0.0, right_needed)

    plan = AssemblyPlan(
        target_start_sec=target.original.begin_time,
        target_end_sec=target.original.end_time,
        window_size=window_size,
        left_needed=left_needed,
        right_needed=right_needed,
    )

    if not scored_candidates and (left_needed > 0 or right_needed > 0):
        plan.rejection_reason = REASON_INSUFFICIENT_BACKGROUND
        return plan

    # Separate candidates by position relative to target midpoint
    left_candidates: list[BackgroundFragment] = []
    right_candidates: list[BackgroundFragment] = []
    for frag in scored_candidates:
        frag_mid = (frag.start_sec + frag.end_sec) / 2.0
        if frag_mid < target.midpoint_sec:
            left_candidates.append(frag)
        else:
            right_candidates.append(frag)

    # Greedily fill left side
    left_filled = 0.0
    for frag in left_candidates:
        if left_filled >= left_needed:
            break
        remaining = left_needed - left_filled
        usable = min(frag.duration_sec, remaining)
        # Take from the end of the fragment (closest to target)
        trimmed = BackgroundFragment(
            start_sec=frag.end_sec - usable,
            end_sec=frag.end_sec,
            duration_sec=usable,
            distance_from_target=frag.distance_from_target,
            audio=frag.audio,
        )
        plan.left_fragments.append(trimmed)
        left_filled += usable

    # Greedily fill right side
    right_filled = 0.0
    for frag in right_candidates:
        if right_filled >= right_needed:
            break
        remaining = right_needed - right_filled
        usable = min(frag.duration_sec, remaining)
        # Take from the start of the fragment (closest to target)
        trimmed = BackgroundFragment(
            start_sec=frag.start_sec,
            end_sec=frag.start_sec + usable,
            duration_sec=usable,
            distance_from_target=frag.distance_from_target,
            audio=frag.audio,
        )
        plan.right_fragments.append(trimmed)
        right_filled += usable

    # Check fill adequacy
    total_needed = left_needed + right_needed
    total_filled = left_filled + right_filled
    if total_needed > 0 and (total_filled / total_needed) < min_fill_fraction:
        plan.rejection_reason = REASON_INSUFFICIENT_BACKGROUND
        return plan

    # Build splice points (seconds relative to window start)
    # Layout: [left_bg...] [target] [right_bg...]
    cursor = 0.0
    for frag in plan.left_fragments:
        cursor += frag.duration_sec
        plan.splice_points_sec.append(cursor)

    # Splice after target (before right background)
    cursor += target.duration_sec
    if plan.right_fragments:
        plan.splice_points_sec.append(cursor)

    plan.can_assemble = True
    return plan
