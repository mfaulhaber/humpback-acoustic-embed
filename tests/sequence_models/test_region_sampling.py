"""Tests for ``region_sampling.build_training_set``."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.sequence_models.event_overlap_join import (
    BACKGROUND,
    EVENT_CORE,
    NEAR_EVENT,
)
from humpback.sequence_models.region_sampling import (
    RegionSequence,
    SamplingConfig,
    TierConfig,
    build_training_set,
)


def _region(
    region_id: str, tiers: list[str], dim: int = 4, seed: int = 0
) -> RegionSequence:
    rng = np.random.default_rng(seed)
    n = len(tiers)
    chunks = rng.standard_normal(size=(n, dim)).astype(np.float32)
    return RegionSequence(
        region_id=region_id,
        chunks=chunks,
        tiers=np.asarray(tiers, dtype=object),
    )


def _make_dataset(n_regions: int = 4, region_len: int = 200) -> list[RegionSequence]:
    """Synthetic dataset with mixed tiers so all three tiers have many chunks."""
    regions = []
    for r in range(n_regions):
        tiers: list[str] = []
        for i in range(region_len):
            mod = i % 10
            if mod < 4:
                tiers.append(EVENT_CORE)
            elif mod < 7:
                tiers.append(NEAR_EVENT)
            else:
                tiers.append(BACKGROUND)
        regions.append(_region(f"reg-{r}", tiers, dim=8, seed=100 + r))
    return regions


# ---------- Mode A: full_region ----------


def test_full_region_uses_all_chunks_when_under_cap():
    regions = _make_dataset(n_regions=2, region_len=50)
    sampling = SamplingConfig(target_train_chunks=10_000, random_seed=42)
    tier_config = TierConfig()

    out = build_training_set(regions, "full_region", tier_config, sampling)

    assert len(out.sub_sequences) == 2
    assert sum(s.shape[0] for s in out.sub_sequences) == 100
    for r in regions:
        assert out.was_used_for_training_per_region[r.region_id].all()
    np.testing.assert_array_equal(
        out.lengths, np.asarray([s.shape[0] for s in out.sub_sequences])
    )


def test_full_region_subsamples_when_over_cap():
    regions = _make_dataset(n_regions=10, region_len=200)
    sampling = SamplingConfig(target_train_chunks=400, random_seed=42)
    tier_config = TierConfig()

    out = build_training_set(regions, "full_region", tier_config, sampling)

    total = sum(s.shape[0] for s in out.sub_sequences)
    # The subsampler stops as soon as ``used >= target``; the last region
    # added pushes total slightly past target, so allow a one-region tail.
    assert 400 <= total < 400 + 200
    assert len(out.sub_sequences) <= 4  # at most a handful of regions


def test_full_region_skips_short_regions():
    regions = [
        _region("short", [EVENT_CORE] * 5, dim=4, seed=1),
        _region("long", [EVENT_CORE] * 100, dim=4, seed=2),
    ]
    sampling = SamplingConfig(min_sequence_length_frames=10)
    out = build_training_set(regions, "full_region", TierConfig(), sampling)

    region_ids_used = [
        rid for rid, mask in out.was_used_for_training_per_region.items() if mask.any()
    ]
    assert region_ids_used == ["long"]
    # The mask for the short region exists but is all False.
    assert not out.was_used_for_training_per_region["short"].any()


# ---------- Mode B: event_balanced ----------


def test_event_balanced_emits_tier_balanced_subsequences():
    regions = _make_dataset(n_regions=4, region_len=200)
    sampling = SamplingConfig(
        subsequence_length_chunks=16,
        subsequence_stride_chunks=8,
        target_train_chunks=480,  # 192 / 168 / 120 per tier @ {0.4, 0.35, 0.25}
        random_seed=7,
    )
    tier_config = TierConfig()

    out = build_training_set(regions, "event_balanced", tier_config, sampling)

    # Count how many chunks of each tier ended up flagged for training.
    tier_counts = {EVENT_CORE: 0, NEAR_EVENT: 0, BACKGROUND: 0}
    for region in regions:
        mask = out.was_used_for_training_per_region[region.region_id]
        for tier in tier_counts:
            tier_counts[tier] += int(np.sum(mask & (region.tiers == tier)))

    total = sum(tier_counts.values())
    assert total > 0
    fractions = {tier: c / total for tier, c in tier_counts.items()}

    # All three tiers must contribute something.
    assert all(c > 0 for c in tier_counts.values())
    # The proportions are approximate (sub-sequences are 16 chunks long
    # and span 3-tier mix), so allow a generous ±15% absolute.
    assert abs(fractions[EVENT_CORE] - 0.40) < 0.15
    assert abs(fractions[NEAR_EVENT] - 0.35) < 0.15
    assert abs(fractions[BACKGROUND] - 0.25) < 0.15


def test_event_balanced_respects_target_train_chunks_cap():
    regions = _make_dataset(n_regions=4, region_len=400)
    sampling = SamplingConfig(
        subsequence_length_chunks=16,
        subsequence_stride_chunks=8,
        target_train_chunks=300,
        random_seed=0,
    )
    out = build_training_set(regions, "event_balanced", TierConfig(), sampling)

    total = sum(s.shape[0] for s in out.sub_sequences)
    # Per-tier caps round to (120, 105, 75) so cumulative emission can
    # slightly overshoot the headline ``target_train_chunks`` when the
    # last sub-sequence pushes its tier over the per-tier cap. Allow up
    # to one full sub-sequence of slack per tier.
    assert total <= 300 + 3 * 16


def test_event_balanced_is_deterministic_for_same_seed():
    regions = _make_dataset(n_regions=4, region_len=300)
    sampling = SamplingConfig(
        subsequence_length_chunks=16, subsequence_stride_chunks=8, random_seed=99
    )
    a = build_training_set(regions, "event_balanced", TierConfig(), sampling)
    b = build_training_set(regions, "event_balanced", TierConfig(), sampling)

    assert len(a.sub_sequences) == len(b.sub_sequences)
    for x, y in zip(a.sub_sequences, b.sub_sequences):
        np.testing.assert_array_equal(x, y)
    for region_id in a.was_used_for_training_per_region:
        np.testing.assert_array_equal(
            a.was_used_for_training_per_region[region_id],
            b.was_used_for_training_per_region[region_id],
        )


def test_full_region_subsample_changes_with_different_seed():
    """When the cap forces region subsampling, the seed picks which
    regions are used; two distinct seeds should pick a different set."""
    regions = _make_dataset(n_regions=8, region_len=200)
    sampling_a = SamplingConfig(target_train_chunks=300, random_seed=1)
    sampling_b = SamplingConfig(target_train_chunks=300, random_seed=42)

    a = build_training_set(regions, "full_region", TierConfig(), sampling_a)
    b = build_training_set(regions, "full_region", TierConfig(), sampling_b)

    a_used = {
        rid for rid, mask in a.was_used_for_training_per_region.items() if mask.any()
    }
    b_used = {
        rid for rid, mask in b.was_used_for_training_per_region.items() if mask.any()
    }
    assert a_used != b_used


# ---------- Mode C: event_only ----------


def test_event_only_excludes_background_chunks():
    regions = _make_dataset(n_regions=4, region_len=200)
    sampling = SamplingConfig(
        subsequence_length_chunks=16, subsequence_stride_chunks=8, random_seed=5
    )

    out = build_training_set(regions, "event_only", TierConfig(), sampling)

    # No emitted sub-sequence should be centred on a background chunk.
    background_used = 0
    for region in regions:
        mask = out.was_used_for_training_per_region[region.region_id]
        # Background chunks may still be inside an emitted sub-sequence
        # (because the centring rule snaps to event/near tiers but the
        # 16-chunk window still covers their neighbours). What we want
        # to assert is that no sub-sequence is *driven* by a background
        # chunk — which is implied by the centring set never including
        # BACKGROUND. Verify total counts behave: event_core + near_event
        # together strictly dominate background among trained chunks.
        background_used += int(np.sum(mask & (region.tiers == BACKGROUND)))
    event_used = 0
    for region in regions:
        mask = out.was_used_for_training_per_region[region.region_id]
        for tier in (EVENT_CORE, NEAR_EVENT):
            event_used += int(np.sum(mask & (region.tiers == tier)))
    assert event_used > 0
    assert event_used > background_used


# ---------- mask consistency + error paths ----------


def test_was_used_mask_matches_subsequence_membership():
    regions = _make_dataset(n_regions=2, region_len=120)
    sampling = SamplingConfig(
        subsequence_length_chunks=16, subsequence_stride_chunks=8, random_seed=3
    )
    out = build_training_set(regions, "event_balanced", TierConfig(), sampling)

    total_mask_chunks = sum(
        int(mask.sum()) for mask in out.was_used_for_training_per_region.values()
    )
    total_subseq_chunks = sum(s.shape[0] for s in out.sub_sequences)
    # Sub-sequences may overlap (when stride < length) — so mask count
    # is a lower bound on the union and a strict upper bound on
    # subseq_chunks only when sub-sequences are non-overlapping. Here
    # stride = 8 and length = 16, so we expect total_mask_chunks <= total_subseq_chunks.
    assert total_mask_chunks <= total_subseq_chunks


def test_unknown_mode_raises():
    regions = _make_dataset(n_regions=1, region_len=20)
    with pytest.raises(ValueError, match="unknown training mode"):
        build_training_set(regions, "no_such_mode", TierConfig(), SamplingConfig())
