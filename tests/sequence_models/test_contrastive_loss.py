"""Tests for supervised contrastive retrieval loss helpers."""

from __future__ import annotations

import torch

from humpback.sequence_models.contrastive_loss import (
    ContrastiveEventMetadata,
    build_contrastive_masks,
    supervised_contrastive_loss,
)


def _meta(
    event_id: str,
    region_id: str,
    labels: tuple[str, ...],
) -> ContrastiveEventMetadata:
    return ContrastiveEventMetadata(
        event_id=event_id,
        region_id=region_id,
        human_types=labels,
        start_index=0,
        end_index=1,
    )


def test_positive_mask_uses_multilabel_intersection() -> None:
    metadata = [
        _meta("a", "r1", ("Moan", "Whup")),
        _meta("b", "r2", ("Whup",)),
        _meta("c", "r3", ("Creak",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
    )

    assert masks.positive_mask[0, 1]
    assert masks.positive_mask[1, 0]
    assert not masks.positive_mask[0, 2]
    assert masks.negative_mask[0, 2]


def test_related_label_pairs_are_not_negatives() -> None:
    metadata = [
        _meta("a", "r1", ("Moan",)),
        _meta("b", "r2", ("Ascending Moan",)),
        _meta("c", "r3", ("Whup",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
    )

    assert not masks.negative_mask[0, 1]
    assert masks.negative_mask[0, 2]


def test_rare_labels_are_excluded_from_contrastive_masks() -> None:
    metadata = [
        _meta("a", "r1", ("Rare",)),
        _meta("b", "r2", ("Common",)),
        _meta("c", "r3", ("Common",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=2,
        min_regions_per_label=2,
    )

    assert not masks.eligible_mask[0]
    assert masks.eligible_mask[1]
    assert masks.positive_mask[1, 2]


def test_cross_region_positive_preference_excludes_same_region_when_available() -> None:
    metadata = [
        _meta("a", "r1", ("Moan",)),
        _meta("b", "r1", ("Moan",)),
        _meta("c", "r2", ("Moan",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
        require_cross_region_positive=True,
    )

    assert not masks.positive_mask[0, 1]
    assert masks.positive_mask[0, 2]


def test_supervised_contrastive_loss_is_finite_for_multilabel_events() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    metadata = [
        _meta("a", "r1", ("Moan", "Whup")),
        _meta("b", "r2", ("Whup",)),
        _meta("c", "r3", ("Creak",)),
    ]

    loss, masks = supervised_contrastive_loss(
        embeddings,
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
    )

    assert torch.isfinite(loss)
    assert masks.valid_anchor_count == 2
    loss.backward()
    assert embeddings.grad is not None


def test_supervised_contrastive_loss_zero_when_no_valid_positive() -> None:
    embeddings = torch.randn(2, 4, requires_grad=True)
    metadata = [
        _meta("a", "r1", ("Moan",)),
        _meta("b", "r2", ("Whup",)),
    ]

    loss, masks = supervised_contrastive_loss(
        embeddings,
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
    )

    assert loss.item() == 0.0
    assert masks.valid_anchor_count == 0
    loss.backward()
    assert embeddings.grad is not None


def test_global_eligible_labels_allow_smaller_local_batch_support() -> None:
    metadata = [
        _meta("a", "r1", ("Moan",)),
        _meta("b", "r2", ("Moan",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=4,
        min_regions_per_label=2,
        eligible_labels={"Moan"},
    )

    assert masks.valid_anchor_count == 2
    assert masks.positive_mask[0, 1]
    assert masks.positive_mask[1, 0]


def test_explicit_eligible_labels_exclude_otherwise_supported_label() -> None:
    metadata = [
        _meta("a", "r1", ("Moan",)),
        _meta("b", "r2", ("Moan",)),
        _meta("c", "r1", ("Whup",)),
        _meta("d", "r2", ("Whup",)),
    ]

    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=1,
        min_regions_per_label=1,
        eligible_labels={"Moan"},
    )

    assert masks.eligible_mask.tolist() == [True, True, False, False]
    assert masks.positive_mask[0, 1]
    assert not masks.positive_mask[2, 3]
