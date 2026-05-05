"""Retrieval diagnostics for masked-transformer sequence jobs.

Phase 0 of the retrieval-aware transformer work moves the old standalone
nearest-neighbor report into backend code. The diagnostics here are intentionally
read-only: they load completed masked-transformer artifacts, assign
human-correction labels to effective events, and return structured metrics for
the API layer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.models.call_parsing import (
    RegionDetectionJob,
    VocalizationCorrection,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, MaskedTransformerJob
from humpback.services.masked_transformer_service import parse_k_values
from humpback.storage import (
    continuous_embedding_parquet_path,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_k_decoded_path,
    masked_transformer_retrieval_embeddings_path,
)

FloatArray = NDArray[np.floating[Any]]
EmbeddingSpace = Literal["contextual", "retrieval"]
RetrievalMode = Literal[
    "unrestricted", "exclude_same_event", "exclude_same_event_and_region"
]
EmbeddingVariant = Literal[
    "raw_l2",
    "centered_l2",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "remove_pc10",
    "whiten_pca",
]

DEFAULT_RETRIEVAL_MODES: tuple[RetrievalMode, ...] = (
    "unrestricted",
    "exclude_same_event",
    "exclude_same_event_and_region",
)
DEFAULT_EMBEDDING_VARIANTS: tuple[EmbeddingVariant, ...] = (
    "raw_l2",
    "centered_l2",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "remove_pc10",
    "whiten_pca",
)


class RetrievalDiagnosticsError(Exception):
    """Base class for API-mapped retrieval diagnostic failures."""


class RetrievalDiagnosticsNotFound(RetrievalDiagnosticsError):
    """Requested job, k value, or entity does not exist."""


class RetrievalDiagnosticsConflict(RetrievalDiagnosticsError):
    """Job state or artifact availability prevents diagnostics."""


class RetrievalDiagnosticsInvalid(RetrievalDiagnosticsError):
    """Caller supplied invalid diagnostic options."""


@dataclass(frozen=True)
class RetrievalReportOptions:
    """Options consumed by :func:`build_nearest_neighbor_report`."""

    k: int | None = None
    embedding_space: EmbeddingSpace = "contextual"
    samples: int = 50
    topn: int = 10
    seed: int = 20260504
    retrieval_modes: tuple[RetrievalMode, ...] = DEFAULT_RETRIEVAL_MODES
    embedding_variants: tuple[EmbeddingVariant, ...] = DEFAULT_EMBEDDING_VARIANTS
    include_query_rows: bool = False
    include_neighbor_rows: bool = False
    include_event_level: bool = False


@dataclass(frozen=True)
class HumanLabeledEvent:
    """One effective event with human-correction labels only."""

    event_id: str
    region_id: str
    start_utc: float
    end_utc: float
    human_types: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end_utc - self.start_utc


@dataclass(frozen=True)
class _JobContext:
    job: MaskedTransformerJob
    continuous_embedding_job: ContinuousEmbeddingJob
    region_detection_job: RegionDetectionJob
    k: int
    k_values: list[int]
    events: list[HumanLabeledEvent]
    correction_meta: dict[str, Any]


def _labels(types: tuple[str, ...] | set[str] | frozenset[str]) -> str:
    return ",".join(sorted(types)) if types else "(none)"


def _finite_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_float(value: Any, default: float = 0.0) -> float:
    converted = _finite_or_none(value)
    return default if converted is None else converted


def _similar_duration(a: float | None, b: float | None) -> bool:
    if a is None or b is None or a <= 0 or b <= 0:
        return False
    ratio = min(a, b) / max(a, b)
    return abs(a - b) <= 0.5 or ratio >= 0.67


def _verdict(summary: dict[str, Any]) -> str:
    if (
        summary["same_human_label_rate"] >= 0.5
        and summary["similar_duration_rate"] >= 0.5
    ):
        if summary["adjacent_1s_rate"] < 0.5:
            return "good"
        return "mixed_label_plus_adjacent"
    if summary["adjacent_1s_rate"] >= 0.5:
        return "bad_time_adjacent"
    if summary["neighbor_without_human_label_rate"] >= 0.5:
        return "bad_unlabeled_or_background"
    return "mixed"


def _read_table_rows(
    path: Path, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def _assign_events_to_rows(
    rows: list[dict[str, Any]], events: list[HumanLabeledEvent]
) -> list[HumanLabeledEvent | None]:
    annotations: list[HumanLabeledEvent | None] = [None] * len(rows)
    order = sorted(
        range(len(rows)),
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
    for event in sorted(events, key=lambda e: (e.start_utc, e.end_utc, e.event_id)):
        while cursor < len(order) and centers[cursor] < event.start_utc:
            cursor += 1
        i = cursor
        while i < len(order) and centers[i] < event.end_utc:
            annotations[order[i]] = event
            i += 1
        cursor = i
    return annotations


async def load_human_correction_events(
    session: AsyncSession,
    *,
    storage_root: Path,
    event_segmentation_job_id: str,
    region_detection_job_id: str,
    region_start_timestamp: float | None,
) -> tuple[list[HumanLabeledEvent], dict[str, Any]]:
    """Load effective events annotated only with human correction labels."""

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
    offset = float(region_start_timestamp or 0.0)
    corrections_by_type: Counter[str] = Counter()

    events: list[HumanLabeledEvent] = []
    for event in effective_events:
        added_labels: set[str] = set()
        removed_labels: set[str] = set()
        for correction in corrections:
            if (
                correction.start_sec < event.end_sec
                and correction.end_sec > event.start_sec
            ):
                key = f"{correction.correction_type}:{correction.type_name}"
                corrections_by_type[key] += 1
                if correction.correction_type == "add":
                    added_labels.add(correction.type_name)
                elif correction.correction_type == "remove":
                    removed_labels.add(correction.type_name)
        labels = added_labels - removed_labels

        events.append(
            HumanLabeledEvent(
                event_id=event.event_id,
                region_id=event.region_id,
                start_utc=float(event.start_sec) + offset,
                end_utc=float(event.end_sec) + offset,
                human_types=tuple(sorted(labels)),
            )
        )

    label_counter: Counter[str] = Counter()
    for event in events:
        for label in event.human_types:
            label_counter[label] += 1

    return events, {
        "total_correction_rows": len(corrections),
        "events_with_human_labels": sum(1 for e in events if e.human_types),
        "corrections_by_type": dict(corrections_by_type.most_common()),
        "event_label_counts": dict(label_counter.most_common()),
    }


async def _load_job_context(
    session: AsyncSession,
    *,
    storage_root: Path,
    job_id: str,
    requested_k: int | None,
) -> _JobContext:
    job = await session.get(MaskedTransformerJob, job_id)
    if job is None:
        raise RetrievalDiagnosticsNotFound("masked transformer job not found")
    if job.status != JobStatus.complete.value:
        raise RetrievalDiagnosticsConflict("masked transformer job is not complete")

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise RetrievalDiagnosticsConflict(
            f"continuous embedding job not found: {job.continuous_embedding_job_id}"
        )
    if not cej.region_detection_job_id:
        raise RetrievalDiagnosticsConflict(
            "masked-transformer diagnostics require a region-scoped upstream"
        )
    if not cej.event_segmentation_job_id:
        raise RetrievalDiagnosticsConflict(
            "continuous embedding job has no event_segmentation_job_id"
        )

    rdj = await session.get(RegionDetectionJob, cej.region_detection_job_id)
    if rdj is None:
        raise RetrievalDiagnosticsConflict(
            f"region detection job not found: {cej.region_detection_job_id}"
        )

    try:
        k_values = parse_k_values(job.k_values)
    except ValueError as exc:
        raise RetrievalDiagnosticsConflict(f"invalid k_values payload: {exc}") from exc
    if not k_values:
        raise RetrievalDiagnosticsConflict("masked transformer job has no k values")

    k = k_values[0] if requested_k is None else int(requested_k)
    if k not in k_values:
        raise RetrievalDiagnosticsNotFound(f"k={k} is not in this job's k_values")

    events, correction_meta = await load_human_correction_events(
        session,
        storage_root=storage_root,
        event_segmentation_job_id=cej.event_segmentation_job_id,
        region_detection_job_id=cej.region_detection_job_id,
        region_start_timestamp=rdj.start_timestamp,
    )

    return _JobContext(
        job=job,
        continuous_embedding_job=cej,
        region_detection_job=rdj,
        k=k,
        k_values=k_values,
        events=events,
        correction_meta=correction_meta,
    )


def _embedding_path(storage_root: Path, job_id: str, space: EmbeddingSpace) -> Path:
    if space == "contextual":
        return masked_transformer_contextual_embeddings_path(storage_root, job_id)
    if space == "retrieval":
        return masked_transformer_retrieval_embeddings_path(storage_root, job_id)
    raise RetrievalDiagnosticsInvalid(f"unsupported embedding_space: {space}")


def _build_rows(
    *,
    storage_root: Path,
    context: _JobContext,
    embedding_space: EmbeddingSpace,
) -> tuple[list[dict[str, Any]], FloatArray, dict[str, str]]:
    job_id = context.job.id
    cej_id = context.continuous_embedding_job.id
    embedding_path = _embedding_path(storage_root, job_id, embedding_space)
    decoded_path = masked_transformer_k_decoded_path(storage_root, job_id, context.k)
    ce_path = continuous_embedding_parquet_path(storage_root, cej_id)

    if not embedding_path.exists():
        raise RetrievalDiagnosticsConflict(
            f"{embedding_space} embeddings artifact not found"
        )
    if not decoded_path.exists():
        raise RetrievalDiagnosticsConflict(
            f"decoded.parquet not found for k={context.k}"
        )
    if not ce_path.exists():
        raise RetrievalDiagnosticsConflict(
            f"continuous embedding parquet not found for job {cej_id}"
        )

    embedding_rows = _read_table_rows(embedding_path)
    decoded_rows = _read_table_rows(decoded_path)
    ce_rows = _read_table_rows(ce_path)

    decoded_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in decoded_rows:
        region_id = str(row.get("region_id") or row.get("sequence_id") or "")
        position = int(row.get("chunk_index_in_region", row.get("position", 0)))
        decoded_by_key[(region_id, position)] = row

    ce_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in ce_rows:
        region_id = str(row.get("region_id") or "")
        position = int(row.get("chunk_index_in_region", 0))
        ce_by_key[(region_id, position)] = row

    event_for_row = _assign_events_to_rows(embedding_rows, context.events)
    raw_vectors = np.asarray(
        [row["embedding"] for row in embedding_rows], dtype=np.float32
    )

    rows: list[dict[str, Any]] = []
    for idx, (emb_row, event) in enumerate(zip(embedding_rows, event_for_row)):
        region_id = str(emb_row["region_id"])
        chunk_index = int(emb_row["chunk_index_in_region"])
        key = (region_id, chunk_index)
        decoded = decoded_by_key.get(key, {})
        ce = ce_by_key.get(key, {})
        start = float(emb_row["start_timestamp"])
        end = float(emb_row["end_timestamp"])
        human_types = event.human_types if event else tuple()
        rows.append(
            {
                "idx": idx,
                "region_id": region_id,
                "chunk_index": chunk_index,
                "start_timestamp": start,
                "end_timestamp": end,
                "center_timestamp": (start + end) / 2.0,
                "tier": str(ce.get("tier") or emb_row.get("tier") or ""),
                "hydrophone_id": str(ce.get("hydrophone_id") or ""),
                "token": int(decoded.get("label", -1)),
                "token_confidence": _json_float(decoded.get("confidence"), default=0.0),
                "call_probability": _json_float(
                    ce.get("call_probability"), default=0.0
                ),
                "event_overlap_fraction": _json_float(
                    ce.get("event_overlap_fraction"), default=0.0
                ),
                "nearest_event_id": ce.get("nearest_event_id"),
                "event_id": event.event_id if event else None,
                "event_duration": event.duration if event else None,
                "human_types": human_types,
            }
        )

    artifacts = {
        "embedding_path": str(embedding_path),
        "decoded_path": str(decoded_path),
        "continuous_embedding_path": str(ce_path),
    }
    return rows, raw_vectors, artifacts


def _normalize_rows(vectors: FloatArray) -> FloatArray:
    return cast(FloatArray, normalize(vectors, norm="l2"))


def _mean_centered_vectors(raw_vectors: FloatArray) -> FloatArray:
    if raw_vectors.size == 0:
        return raw_vectors
    centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    return _normalize_rows(centered)


def _safe_pca_components(raw_vectors: FloatArray, requested: int) -> int:
    return max(0, min(int(requested), raw_vectors.shape[0], raw_vectors.shape[1]))


def _remove_top_pcs(raw_vectors: FloatArray, n_remove: int, *, seed: int) -> FloatArray:
    x_centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    n_components = _safe_pca_components(raw_vectors, n_remove)
    if n_components <= 0:
        return _normalize_rows(x_centered)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(x_centered)
    x_proj = pca.inverse_transform(pca.transform(x_centered))
    x_resid = x_centered - x_proj
    return _normalize_rows(x_resid)


def _whiten_embeddings(
    raw_vectors: FloatArray,
    *,
    n_components: int,
    seed: int,
) -> FloatArray:
    x_centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    safe_components = _safe_pca_components(raw_vectors, n_components)
    if safe_components <= 0:
        return _normalize_rows(x_centered)
    pca = PCA(n_components=safe_components, whiten=True, random_state=seed)
    x_white = pca.fit_transform(x_centered)
    return _normalize_rows(x_white)


def build_embedding_variants(
    raw_vectors: FloatArray,
    *,
    seed: int,
    variants: tuple[EmbeddingVariant, ...],
) -> dict[str, FloatArray]:
    out: dict[str, FloatArray] = {}
    for variant in variants:
        if variant == "raw_l2":
            out[variant] = _normalize_rows(raw_vectors.astype(np.float32, copy=False))
        elif variant == "centered_l2":
            out[variant] = _mean_centered_vectors(raw_vectors)
        elif variant.startswith("remove_pc"):
            count = int(variant.replace("remove_pc", ""))
            out[variant] = _remove_top_pcs(raw_vectors, count, seed=seed)
        elif variant == "whiten_pca":
            out[variant] = _whiten_embeddings(raw_vectors, n_components=128, seed=seed)
        else:
            raise RetrievalDiagnosticsInvalid(
                f"unsupported embedding variant: {variant}"
            )
    return out


def _sample_query_indices(
    rows: list[dict[str, Any]], *, samples: int, seed: int
) -> tuple[list[int], int]:
    human_candidates = [int(row["idx"]) for row in rows if row["human_types"]]
    candidate_pool = human_candidates if human_candidates else list(range(len(rows)))
    if not candidate_pool:
        return [], 0
    rng = np.random.default_rng(seed)
    sample_size = min(int(samples), len(candidate_pool))
    sampled = rng.choice(candidate_pool, size=sample_size, replace=False).tolist()
    return sorted(int(i) for i in sampled), len(human_candidates)


def _mask_candidates_for_mode(
    sims: NDArray[np.floating[Any]],
    *,
    rows: list[dict[str, Any]],
    query_idx: int,
    mode: RetrievalMode,
) -> None:
    q = rows[query_idx]
    sims[query_idx] = -np.inf
    for candidate_idx, candidate in enumerate(rows):
        if candidate_idx == query_idx:
            continue
        if mode in {"exclude_same_event", "exclude_same_event_and_region"}:
            if (
                q["event_id"]
                and candidate["event_id"]
                and q["event_id"] == candidate["event_id"]
            ):
                sims[candidate_idx] = -np.inf
                continue
        if mode == "exclude_same_event_and_region":
            if candidate["region_id"] == q["region_id"]:
                sims[candidate_idx] = -np.inf


def analyze_neighbors(
    rows: list[dict[str, Any]],
    vectors: FloatArray,
    *,
    topn: int,
    sample_indices: list[int],
    mode: RetrievalMode,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not sample_indices:
        return [], []

    vectors = _normalize_rows(vectors)
    all_sims = vectors[sample_indices] @ vectors.T
    neighbor_records: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []

    for query_order, query_idx in enumerate(sample_indices, start=1):
        q = rows[query_idx]
        q_human = set(q["human_types"])
        sims = np.array(all_sims[query_order - 1], copy=True)
        _mask_candidates_for_mode(sims, rows=rows, query_idx=query_idx, mode=mode)
        valid = np.where(np.isfinite(sims))[0]
        ranked = valid[np.argsort(np.multiply(sims[valid], -1.0))][:topn]
        counts: Counter[str] = Counter()
        duration_comparable = 0

        for rank, neighbor_idx in enumerate(ranked.tolist(), start=1):
            n = rows[int(neighbor_idx)]
            n_human = set(n["human_types"])
            center_delta = abs(
                float(q["center_timestamp"]) - float(n["center_timestamp"])
            )
            same_region = q["region_id"] == n["region_id"]
            adjacent_1s = same_region and center_delta <= 1.0
            nearby_5s = same_region and center_delta <= 5.0
            same_human = bool(q_human and n_human and q_human.intersection(n_human))
            exact_human = bool(q_human and n_human and q_human == n_human)
            same_event = bool(q["event_id"] and q["event_id"] == n["event_id"])
            similar_duration = _similar_duration(
                q["event_duration"], n["event_duration"]
            )
            if q["event_duration"] is not None and n["event_duration"] is not None:
                duration_comparable += 1

            counts["same_human_label"] += int(same_human)
            counts["exact_human_label_set"] += int(exact_human)
            counts["same_event"] += int(same_event)
            counts["same_region"] += int(same_region)
            counts["adjacent_1s"] += int(adjacent_1s)
            counts["nearby_5s"] += int(nearby_5s)
            counts["same_token"] += int(q["token"] == n["token"])
            counts["similar_duration"] += int(similar_duration)
            counts["neighbor_without_human_label"] += int(not n_human)
            counts["neighbor_low_event_overlap"] += int(
                float(n["event_overlap_fraction"]) < 0.25
            )

            neighbor_records.append(
                {
                    "query_order": query_order,
                    "query_idx": query_idx,
                    "rank": rank,
                    "neighbor_idx": int(neighbor_idx),
                    "cosine": float(sims[int(neighbor_idx)]),
                    "query_region": q["region_id"],
                    "neighbor_region": n["region_id"],
                    "query_chunk": int(q["chunk_index"]),
                    "neighbor_chunk": int(n["chunk_index"]),
                    "center_delta_sec": float(center_delta),
                    "same_region": same_region,
                    "adjacent_1s": adjacent_1s,
                    "nearby_5s": nearby_5s,
                    "query_human_types": _labels(q["human_types"]),
                    "neighbor_human_types": _labels(n["human_types"]),
                    "same_human_label": same_human,
                    "exact_human_label_set": exact_human,
                    "query_event_id": q["event_id"] or "",
                    "neighbor_event_id": n["event_id"] or "",
                    "same_event": same_event,
                    "query_duration": q["event_duration"],
                    "neighbor_duration": n["event_duration"],
                    "similar_duration": similar_duration,
                    "query_token": int(q["token"]),
                    "neighbor_token": int(n["token"]),
                    "same_token": q["token"] == n["token"],
                    "query_tier": q["tier"],
                    "neighbor_tier": n["tier"],
                    "query_overlap": float(q["event_overlap_fraction"]),
                    "neighbor_overlap": float(n["event_overlap_fraction"]),
                    "query_call_probability": float(q["call_probability"]),
                    "neighbor_call_probability": float(n["call_probability"]),
                    "query_start_timestamp": float(q["start_timestamp"]),
                    "neighbor_start_timestamp": float(n["start_timestamp"]),
                }
            )

        neighbor_count = len(ranked)
        denom = float(max(neighbor_count, 1))
        avg_cosine = (
            float(
                np.mean(
                    [
                        r["cosine"]
                        for r in neighbor_records
                        if r["query_order"] == query_order
                    ]
                )
            )
            if neighbor_count
            else 0.0
        )
        summary = {
            "query_order": query_order,
            "query_idx": query_idx,
            "query_region": q["region_id"],
            "query_chunk": int(q["chunk_index"]),
            "query_start_timestamp": float(q["start_timestamp"]),
            "query_human_types": _labels(q["human_types"]),
            "query_event_id": q["event_id"] or "",
            "query_duration": q["event_duration"],
            "query_token": int(q["token"]),
            "neighbor_count": neighbor_count,
            "same_human_label_rate": counts["same_human_label"] / denom,
            "exact_human_label_set_rate": counts["exact_human_label_set"] / denom,
            "same_event_rate": counts["same_event"] / denom,
            "same_region_rate": counts["same_region"] / denom,
            "adjacent_1s_rate": counts["adjacent_1s"] / denom,
            "nearby_5s_rate": counts["nearby_5s"] / denom,
            "same_token_rate": counts["same_token"] / denom,
            "similar_duration_rate": counts["similar_duration"]
            / max(duration_comparable, 1),
            "neighbor_without_human_label_rate": counts["neighbor_without_human_label"]
            / denom,
            "neighbor_low_event_overlap_rate": counts["neighbor_low_event_overlap"]
            / denom,
            "avg_cosine": avg_cosine,
        }
        summary["verdict"] = _verdict(summary)
        query_summaries.append(summary)

    return query_summaries, neighbor_records


def _cosine_baseline(
    vectors: FloatArray, *, seed: int, n_pairs: int = 20_000
) -> dict[str, float]:
    if vectors.shape[0] < 2:
        return {str(p): 0.0 for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]}
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vectors.shape[0], size=n_pairs)
    b = rng.integers(0, vectors.shape[0], size=n_pairs)
    mask = a != b
    cosine = np.sum(vectors[a[mask]] * vectors[b[mask]], axis=1)
    return {
        str(p): float(v)
        for p, v in zip(
            [0, 1, 5, 25, 50, 75, 95, 99, 100],
            np.percentile(cosine, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
        )
    }


def _label_specific_metrics(
    rows: list[dict[str, Any]], neighbor_records: list[dict[str, Any]]
) -> dict[str, dict[str, float | int]]:
    labels = sorted({label for row in rows for label in row["human_types"]})
    out: dict[str, dict[str, float | int]] = {}
    query_types = {
        int(row["idx"]): set(row["human_types"]) for row in rows if row["human_types"]
    }
    query_counts = Counter()
    for label in labels:
        query_counts[label] = sum(1 for types in query_types.values() if label in types)

    for label in labels:
        records = [
            record
            for record in neighbor_records
            if label in query_types.get(int(record["query_idx"]), set())
        ]
        denom = max(len(records), 1)
        out[label] = {
            "query_count": int(query_counts[label]),
            "neighbor_count": len(records),
            "same_human_label": sum(
                int(record["same_human_label"]) for record in records
            )
            / denom,
        }
    return out


def _aggregate_neighbor_metrics(
    rows: list[dict[str, Any]],
    query_summaries: list[dict[str, Any]],
    neighbor_records: list[dict[str, Any]],
    vectors: FloatArray,
    *,
    seed: int,
) -> dict[str, Any]:
    all_neighbor_count = max(len(neighbor_records), 1)

    def _rate(name: str) -> float:
        return (
            sum(int(record[name]) for record in neighbor_records) / all_neighbor_count
        )

    cosines = [float(record["cosine"]) for record in neighbor_records]
    return {
        "same_human_label": _rate("same_human_label"),
        "exact_human_label_set": _rate("exact_human_label_set"),
        "same_event": _rate("same_event"),
        "same_region": _rate("same_region"),
        "adjacent_1s": _rate("adjacent_1s"),
        "nearby_5s": _rate("nearby_5s"),
        "same_token": _rate("same_token"),
        "similar_duration": float(
            np.mean([q["similar_duration_rate"] for q in query_summaries])
        )
        if query_summaries
        else 0.0,
        "without_human_label": sum(
            record["neighbor_human_types"] == "(none)" for record in neighbor_records
        )
        / all_neighbor_count,
        "low_event_overlap": sum(
            float(record["neighbor_overlap"]) < 0.25 for record in neighbor_records
        )
        / all_neighbor_count,
        "avg_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "median_cosine": float(np.median(cosines)) if cosines else 0.0,
        "random_pair_percentiles": _cosine_baseline(vectors, seed=seed),
        "verdicts": dict(Counter(q["verdict"] for q in query_summaries)),
        "label_specific_same_human_label": _label_specific_metrics(
            rows, neighbor_records
        ),
    }


def run_variant_matrix(
    rows: list[dict[str, Any]],
    raw_vectors: FloatArray,
    *,
    options: RetrievalReportOptions,
) -> tuple[
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[int],
    int,
]:
    sample_indices, human_candidate_count = _sample_query_indices(
        rows, samples=options.samples, seed=options.seed
    )
    variants = build_embedding_variants(
        raw_vectors, seed=options.seed, variants=options.embedding_variants
    )
    results: dict[str, dict[str, Any]] = {}
    first_query_summaries: list[dict[str, Any]] = []
    first_neighbor_records: list[dict[str, Any]] = []

    for mode in options.retrieval_modes:
        results[mode] = {}
        for variant_name, vectors in variants.items():
            query_summaries, neighbor_records = analyze_neighbors(
                rows,
                vectors,
                topn=options.topn,
                sample_indices=sample_indices,
                mode=mode,
            )
            results[mode][variant_name] = _aggregate_neighbor_metrics(
                rows,
                query_summaries,
                neighbor_records,
                vectors,
                seed=options.seed,
            )
            if not first_query_summaries:
                first_query_summaries = query_summaries
                first_neighbor_records = neighbor_records

    return (
        results,
        first_query_summaries,
        first_neighbor_records,
        sample_indices,
        human_candidate_count,
    )


def _build_event_level_inputs(
    rows: list[dict[str, Any]], raw_vectors: FloatArray
) -> tuple[list[dict[str, Any]], FloatArray]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        if row["event_id"]:
            groups.setdefault(str(row["event_id"]), []).append(i)

    event_rows: list[dict[str, Any]] = []
    event_vectors: list[FloatArray] = []
    for event_idx, (event_id, indices) in enumerate(sorted(groups.items())):
        first = rows[indices[0]]
        event_vectors.append(
            cast(FloatArray, raw_vectors[np.asarray(indices)].mean(axis=0))
        )
        event_rows.append(
            {
                **first,
                "idx": event_idx,
                "chunk_index": event_idx,
                "start_timestamp": min(
                    float(rows[i]["start_timestamp"]) for i in indices
                ),
                "end_timestamp": max(float(rows[i]["end_timestamp"]) for i in indices),
                "center_timestamp": (
                    min(float(rows[i]["start_timestamp"]) for i in indices)
                    + max(float(rows[i]["end_timestamp"]) for i in indices)
                )
                / 2.0,
                "event_id": event_id,
                "token": -1,
            }
        )
    if not event_vectors:
        return [], np.empty((0, raw_vectors.shape[1]), dtype=np.float32)
    return event_rows, np.vstack(event_vectors).astype(np.float32)


def _representative_queries(
    query_summaries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good = sorted(
        query_summaries,
        key=lambda q: (
            q["verdict"] != "good",
            -q["same_human_label_rate"],
            q["adjacent_1s_rate"],
        ),
    )[:8]
    risky = sorted(
        query_summaries,
        key=lambda q: (
            q["verdict"] == "good",
            -max(q["adjacent_1s_rate"], q["neighbor_without_human_label_rate"]),
            q["same_human_label_rate"],
        ),
    )[:8]
    return good, risky


async def build_nearest_neighbor_report(
    session: AsyncSession,
    *,
    storage_root: Path,
    job_id: str,
    options: RetrievalReportOptions,
) -> dict[str, Any]:
    """Build a structured nearest-neighbor report for one MT job."""

    if options.samples <= 0:
        raise RetrievalDiagnosticsInvalid("samples must be > 0")
    if options.topn <= 0:
        raise RetrievalDiagnosticsInvalid("topn must be > 0")

    context = await _load_job_context(
        session, storage_root=storage_root, job_id=job_id, requested_k=options.k
    )
    rows, raw_vectors, artifacts = _build_rows(
        storage_root=storage_root,
        context=context,
        embedding_space=options.embedding_space,
    )
    if raw_vectors.shape[0] == 0:
        raise RetrievalDiagnosticsConflict("embedding artifact contains no rows")

    (
        chunk_results,
        query_summaries,
        neighbor_records,
        sample_indices,
        human_candidate_count,
    ) = run_variant_matrix(rows, raw_vectors, options=options)
    good, risky = _representative_queries(query_summaries)

    human_label_counter: Counter[str] = Counter()
    for row in rows:
        for label in row["human_types"]:
            human_label_counter[label] += 1

    event_level_results = None
    if options.include_event_level:
        event_rows, event_vectors = _build_event_level_inputs(rows, raw_vectors)
        if len(event_rows) > 0:
            event_level_results, _, _, _, _ = run_variant_matrix(
                event_rows, event_vectors, options=options
            )
        else:
            event_level_results = {}

    response = {
        "job": {
            "job_id": context.job.id,
            "status": context.job.status,
            "continuous_embedding_job_id": context.job.continuous_embedding_job_id,
            "event_classification_job_id": context.job.event_classification_job_id,
            "region_detection_job_id": context.continuous_embedding_job.region_detection_job_id,
            "k_values": context.k_values,
            "k": context.k,
            "total_sequences": context.job.total_sequences,
            "total_chunks": context.job.total_chunks,
            "final_train_loss": context.job.final_train_loss,
            "final_val_loss": context.job.final_val_loss,
            "total_epochs": context.job.total_epochs,
        },
        "options": {
            "embedding_space": options.embedding_space,
            "samples": options.samples,
            "topn": options.topn,
            "seed": options.seed,
            "retrieval_modes": list(options.retrieval_modes),
            "embedding_variants": list(options.embedding_variants),
            "include_query_rows": options.include_query_rows,
            "include_neighbor_rows": options.include_neighbor_rows,
            "include_event_level": options.include_event_level,
        },
        "artifacts": artifacts,
        "label_coverage": {
            "embedding_rows": len(rows),
            "sampled_queries": len(sample_indices),
            "human_labeled_query_pool_rows": human_candidate_count,
            "human_labeled_effective_events": context.correction_meta[
                "events_with_human_labels"
            ],
            "vocalization_correction_rows": context.correction_meta[
                "total_correction_rows"
            ],
            "human_label_chunk_counts": dict(human_label_counter.most_common()),
            "human_label_event_counts": context.correction_meta["event_label_counts"],
            "corrections_by_type": context.correction_meta["corrections_by_type"],
        },
        "results": chunk_results,
        "event_level_results": event_level_results,
        "representative_good_queries": good,
        "representative_risky_queries": risky,
        "query_rows": query_summaries if options.include_query_rows else [],
        "neighbor_rows": neighbor_records if options.include_neighbor_rows else [],
    }
    return response


__all__ = [
    "DEFAULT_EMBEDDING_VARIANTS",
    "DEFAULT_RETRIEVAL_MODES",
    "EmbeddingSpace",
    "EmbeddingVariant",
    "HumanLabeledEvent",
    "RetrievalDiagnosticsConflict",
    "RetrievalDiagnosticsError",
    "RetrievalDiagnosticsInvalid",
    "RetrievalDiagnosticsNotFound",
    "RetrievalMode",
    "RetrievalReportOptions",
    "analyze_neighbors",
    "build_embedding_variants",
    "build_nearest_neighbor_report",
    "load_human_correction_events",
    "run_variant_matrix",
]
