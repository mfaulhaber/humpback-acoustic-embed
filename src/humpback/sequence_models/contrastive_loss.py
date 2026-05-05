"""Supervised contrastive loss helpers for retrieval-aware transformers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

DEFAULT_RELATED_LABEL_EXCLUSIONS: tuple[tuple[str, str], ...] = (
    ("Creak", "Vibrate"),
    ("Moan", "Ascending Moan"),
    ("Moan", "Descending Moan"),
    ("Growl", "Buzz"),
    ("Whup", "Grunt"),
)


@dataclass(frozen=True)
class ContrastiveEventMetadata:
    """Event metadata aligned to one training sequence."""

    event_id: str
    region_id: str
    human_types: tuple[str, ...]
    start_index: int
    end_index: int


@dataclass(frozen=True)
class ContrastiveMaskResult:
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    eligible_mask: torch.Tensor
    valid_anchor_mask: torch.Tensor

    @property
    def valid_anchor_count(self) -> int:
        return int(self.valid_anchor_mask.sum().item())


def parse_related_label_policy(
    policy_json: Optional[str],
) -> frozenset[frozenset[str]]:
    """Return unordered related-label pairs excluded from negatives."""
    if not policy_json:
        pairs = DEFAULT_RELATED_LABEL_EXCLUSIONS
    else:
        parsed = json.loads(policy_json)
        raw_pairs = parsed.get("exclude_pairs", []) if isinstance(parsed, dict) else []
        pairs = tuple((str(a), str(b)) for a, b in raw_pairs)
    return frozenset(frozenset(pair) for pair in pairs if len(pair) == 2)


def compute_eligible_contrastive_labels(
    metadata: list[ContrastiveEventMetadata],
    *,
    min_events_per_label: int,
    min_regions_per_label: int,
) -> set[str]:
    """Compute label support against the full split, not a local batch."""
    counts: Counter[str] = Counter()
    regions: dict[str, set[str]] = defaultdict(set)
    for item in metadata:
        for label in item.human_types:
            counts[label] += 1
            regions[label].add(item.region_id)
    return {
        label
        for label, count in counts.items()
        if count >= min_events_per_label
        and len(regions[label]) >= min_regions_per_label
    }


def build_contrastive_masks(
    metadata: list[ContrastiveEventMetadata],
    *,
    min_events_per_label: int = 4,
    min_regions_per_label: int = 2,
    require_cross_region_positive: bool = True,
    related_label_pairs: Iterable[frozenset[str]] | None = None,
    eligible_labels: set[str] | frozenset[str] | None = None,
    device: torch.device | str = "cpu",
) -> ContrastiveMaskResult:
    """Build positive and negative masks from human label-set semantics."""
    n = len(metadata)
    device_obj = torch.device(device)
    positive = torch.zeros((n, n), dtype=torch.bool, device=device_obj)
    negative = torch.zeros((n, n), dtype=torch.bool, device=device_obj)
    eligible = torch.zeros(n, dtype=torch.bool, device=device_obj)
    valid_anchor = torch.zeros(n, dtype=torch.bool, device=device_obj)
    if n == 0:
        return ContrastiveMaskResult(positive, negative, eligible, valid_anchor)

    related = (
        parse_related_label_policy(None)
        if related_label_pairs is None
        else frozenset(related_label_pairs)
    )
    resolved_eligible_labels = (
        compute_eligible_contrastive_labels(
            metadata,
            min_events_per_label=min_events_per_label,
            min_regions_per_label=min_regions_per_label,
        )
        if eligible_labels is None
        else set(eligible_labels)
    )
    label_sets = [
        frozenset(item.human_types) & resolved_eligible_labels for item in metadata
    ]
    for i, labels in enumerate(label_sets):
        eligible[i] = bool(labels)

    for i, anchor_labels in enumerate(label_sets):
        if not anchor_labels:
            continue
        positive_candidates: list[int] = []
        cross_region_candidates: list[int] = []
        for j, candidate_labels in enumerate(label_sets):
            if i == j or not candidate_labels:
                continue
            if anchor_labels & candidate_labels:
                positive_candidates.append(j)
                if metadata[i].region_id != metadata[j].region_id:
                    cross_region_candidates.append(j)
        selected = (
            cross_region_candidates
            if require_cross_region_positive and cross_region_candidates
            else positive_candidates
        )
        for j in selected:
            positive[i, j] = True
        valid_anchor[i] = bool(selected)

        for j, candidate_labels in enumerate(label_sets):
            if i == j or not candidate_labels:
                continue
            if anchor_labels & candidate_labels:
                continue
            if any(pair <= (anchor_labels | candidate_labels) for pair in related):
                continue
            negative[i, j] = True

    return ContrastiveMaskResult(positive, negative, eligible, valid_anchor)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    metadata: list[ContrastiveEventMetadata],
    *,
    temperature: float = 0.07,
    min_events_per_label: int = 4,
    min_regions_per_label: int = 2,
    require_cross_region_positive: bool = True,
    related_label_policy_json: Optional[str] = None,
    eligible_labels: set[str] | frozenset[str] | None = None,
) -> tuple[torch.Tensor, ContrastiveMaskResult]:
    """Compute a finite supervised contrastive loss for event embeddings."""
    if embeddings.shape[0] != len(metadata):
        raise ValueError("embedding rows must align with contrastive metadata")
    related_pairs = parse_related_label_policy(related_label_policy_json)
    masks = build_contrastive_masks(
        metadata,
        min_events_per_label=min_events_per_label,
        min_regions_per_label=min_regions_per_label,
        require_cross_region_positive=require_cross_region_positive,
        related_label_pairs=related_pairs,
        eligible_labels=eligible_labels,
        device=embeddings.device,
    )
    if embeddings.numel() == 0 or masks.valid_anchor_count == 0:
        return embeddings.sum() * 0.0, masks

    normalized = F.normalize(embeddings, p=2, dim=-1, eps=1e-12)
    logits = normalized @ normalized.T / max(float(temperature), 1e-12)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    denominator_mask = masks.positive_mask | masks.negative_mask
    exp_logits = torch.exp(logits) * denominator_mask.to(logits.dtype)
    pos_exp = torch.exp(logits) * masks.positive_mask.to(logits.dtype)
    numerator = pos_exp.sum(dim=1)
    denominator = exp_logits.sum(dim=1)
    valid = masks.valid_anchor_mask & (numerator > 0.0) & (denominator > 0.0)
    if not bool(valid.any()):
        return embeddings.sum() * 0.0, masks
    loss = -(torch.log(numerator[valid]) - torch.log(denominator[valid])).mean()
    return loss, masks


__all__ = [
    "ContrastiveEventMetadata",
    "ContrastiveMaskResult",
    "build_contrastive_masks",
    "compute_eligible_contrastive_labels",
    "parse_related_label_policy",
    "supervised_contrastive_loss",
]
