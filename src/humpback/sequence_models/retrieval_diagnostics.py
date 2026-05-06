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
from sqlalchemy.orm import selectinload

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
    masked_transformer_retrieval_head_outputs_path,
)

FloatArray = NDArray[np.floating[Any]]
EmbeddingSpace = Literal["contextual", "retrieval"]
GeometryEmbeddingSpace = Literal[
    "contextual.raw_l2",
    "contextual.remove_pc10",
    "contextual.whiten_pca",
    "retrieval.raw_l2",
    "retrieval.remove_pc10",
    "retrieval.whiten_pca",
]
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
DEFAULT_GEOMETRY_SPACES: tuple[GeometryEmbeddingSpace, ...] = (
    "contextual.raw_l2",
    "contextual.remove_pc10",
    "contextual.whiten_pca",
    "retrieval.raw_l2",
    "retrieval.remove_pc10",
    "retrieval.whiten_pca",
)
_PERCENTILES: tuple[int, ...] = (0, 1, 5, 25, 50, 75, 95, 99, 100)


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
    include_geometry_report: bool = False
    geometry_embedding_spaces: tuple[GeometryEmbeddingSpace, ...] | None = None
    geometry_random_pairs: int = 20_000
    geometry_pca_components: int = 20


@dataclass(frozen=True)
class HumanLabeledEvent:
    """One effective event with human-correction labels only."""

    event_id: str
    region_id: str
    start_utc: float
    end_utc: float
    human_types: tuple[str, ...]
    source_index: int | None = None
    continuous_embedding_job_id: str | None = None
    event_classification_job_id: str | None = None
    original_event_id: str | None = None
    original_region_id: str | None = None

    @property
    def duration(self) -> float:
        return self.end_utc - self.start_utc


@dataclass(frozen=True)
class _DiagnosticSource:
    source_index: int
    continuous_embedding_job_id: str
    event_classification_job_id: str | None
    continuous_embedding_job: ContinuousEmbeddingJob
    region_detection_job: RegionDetectionJob


@dataclass(frozen=True)
class _JobContext:
    job: MaskedTransformerJob
    continuous_embedding_job: ContinuousEmbeddingJob
    region_detection_job: RegionDetectionJob
    sources: list[_DiagnosticSource]
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
    sorted_events = sorted(events, key=lambda e: (e.start_utc, e.end_utc, e.event_id))
    for row_idx, row in enumerate(rows):
        center = (float(row["start_timestamp"]) + float(row["end_timestamp"])) / 2.0
        for event in sorted_events:
            if center < event.start_utc:
                break
            if center >= event.end_utc:
                continue
            if event.source_index is None or int(row.get("source_index", 0)) == int(
                event.source_index
            ):
                event_region = event.region_id
                row_region = str(row.get("region_id") or "")
                row_original_region = str(row.get("original_region_id") or row_region)
                if not row_region or event_region in {row_region, row_original_region}:
                    annotations[row_idx] = event
                    break
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
    unlabeled_events = sum(1 for event in events if not event.human_types)
    single_label_events = sum(1 for event in events if len(event.human_types) == 1)
    multi_label_events = sum(1 for event in events if len(event.human_types) > 1)

    return events, {
        "total_correction_rows": len(corrections),
        "events_with_human_labels": sum(1 for e in events if e.human_types),
        "unlabeled_effective_events": unlabeled_events,
        "single_label_effective_events": single_label_events,
        "multi_label_effective_events": multi_label_events,
        "corrections_by_type": dict(corrections_by_type.most_common()),
        "event_label_counts": dict(label_counter.most_common()),
    }


def _namespace_event(
    event: HumanLabeledEvent,
    *,
    source_index: int,
    continuous_embedding_job_id: str,
    event_classification_job_id: str | None,
    namespace_regions: bool,
) -> HumanLabeledEvent:
    event_id = (
        f"{source_index}:{event.event_id}" if namespace_regions else event.event_id
    )
    region_id = (
        f"{source_index}:{event.region_id}" if namespace_regions else event.region_id
    )
    return HumanLabeledEvent(
        event_id=event_id,
        region_id=region_id,
        start_utc=event.start_utc,
        end_utc=event.end_utc,
        human_types=event.human_types,
        source_index=source_index,
        continuous_embedding_job_id=continuous_embedding_job_id,
        event_classification_job_id=event_classification_job_id,
        original_event_id=event.event_id,
        original_region_id=event.region_id,
    )


def _merge_correction_meta(metas: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "total_correction_rows": 0,
        "events_with_human_labels": 0,
        "unlabeled_effective_events": 0,
        "single_label_effective_events": 0,
        "multi_label_effective_events": 0,
        "corrections_by_type": {},
        "event_label_counts": {},
    }
    correction_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    for meta in metas:
        for key in (
            "total_correction_rows",
            "events_with_human_labels",
            "unlabeled_effective_events",
            "single_label_effective_events",
            "multi_label_effective_events",
        ):
            merged[key] += int(meta.get(key, 0))
        correction_counter.update(meta.get("corrections_by_type", {}) or {})
        label_counter.update(meta.get("event_label_counts", {}) or {})
    merged["corrections_by_type"] = dict(correction_counter.most_common())
    merged["event_label_counts"] = dict(label_counter.most_common())
    return merged


async def _diagnostic_sources_for_job(
    session: AsyncSession, job: MaskedTransformerJob
) -> list[_DiagnosticSource]:
    source_rows = sorted(job.sources, key=lambda source: source.source_order)
    if not source_rows:
        source_rows = []

    raw_sources: list[tuple[int, str, str | None]] = (
        [
            (
                int(source.source_order),
                source.continuous_embedding_job_id,
                source.event_classification_job_id,
            )
            for source in source_rows
        ]
        if source_rows
        else [(0, job.continuous_embedding_job_id, job.event_classification_job_id)]
    )
    if len(raw_sources) == 1 and job.event_classification_job_id:
        idx, cej_id, _classify_id = raw_sources[0]
        raw_sources[0] = (idx, cej_id, job.event_classification_job_id)

    out: list[_DiagnosticSource] = []
    for source_index, cej_id, classify_id in raw_sources:
        cej = await session.get(ContinuousEmbeddingJob, cej_id)
        if cej is None:
            raise RetrievalDiagnosticsConflict(
                f"continuous embedding job not found: {cej_id}"
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
        out.append(
            _DiagnosticSource(
                source_index=source_index,
                continuous_embedding_job_id=cej.id,
                event_classification_job_id=classify_id,
                continuous_embedding_job=cej,
                region_detection_job=rdj,
            )
        )
    return out


async def _load_job_context(
    session: AsyncSession,
    *,
    storage_root: Path,
    job_id: str,
    requested_k: int | None,
) -> _JobContext:
    result = await session.execute(
        select(MaskedTransformerJob)
        .options(selectinload(MaskedTransformerJob.sources))
        .where(MaskedTransformerJob.id == job_id)
    )
    job = result.scalars().one_or_none()
    if job is None:
        raise RetrievalDiagnosticsNotFound("masked transformer job not found")
    if job.status != JobStatus.complete.value:
        raise RetrievalDiagnosticsConflict("masked transformer job is not complete")

    sources = await _diagnostic_sources_for_job(session, job)
    primary = sources[0]
    cej = primary.continuous_embedding_job
    rdj = primary.region_detection_job

    try:
        k_values = parse_k_values(job.k_values)
    except ValueError as exc:
        raise RetrievalDiagnosticsConflict(f"invalid k_values payload: {exc}") from exc
    if not k_values:
        raise RetrievalDiagnosticsConflict("masked transformer job has no k values")

    k = k_values[0] if requested_k is None else int(requested_k)
    if k not in k_values:
        raise RetrievalDiagnosticsNotFound(f"k={k} is not in this job's k_values")

    namespace_regions = len(sources) > 1
    all_events: list[HumanLabeledEvent] = []
    all_meta: list[dict[str, Any]] = []
    for source in sources:
        source_events, source_meta = await load_human_correction_events(
            session,
            storage_root=storage_root,
            event_segmentation_job_id=source.continuous_embedding_job.event_segmentation_job_id
            or "",
            region_detection_job_id=source.continuous_embedding_job.region_detection_job_id
            or "",
            region_start_timestamp=source.region_detection_job.start_timestamp,
        )
        all_events.extend(
            _namespace_event(
                event,
                source_index=source.source_index,
                continuous_embedding_job_id=source.continuous_embedding_job_id,
                event_classification_job_id=source.event_classification_job_id,
                namespace_regions=namespace_regions,
            )
            for event in source_events
        )
        all_meta.append(source_meta)

    return _JobContext(
        job=job,
        continuous_embedding_job=cej,
        region_detection_job=rdj,
        sources=sources,
        k=k,
        k_values=k_values,
        events=all_events,
        correction_meta=_merge_correction_meta(all_meta),
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
    embedding_path = _embedding_path(storage_root, job_id, embedding_space)
    decoded_path = masked_transformer_k_decoded_path(storage_root, job_id, context.k)

    if not embedding_path.exists():
        raise RetrievalDiagnosticsConflict(
            f"{embedding_space} embeddings artifact not found"
        )
    if not decoded_path.exists():
        raise RetrievalDiagnosticsConflict(
            f"decoded.parquet not found for k={context.k}"
        )

    embedding_rows = _read_table_rows(embedding_path)
    decoded_rows = _read_table_rows(decoded_path)

    decoded_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in decoded_rows:
        region_id = str(row.get("region_id") or row.get("sequence_id") or "")
        position = int(row.get("chunk_index_in_region", row.get("position", 0)))
        decoded_by_key[(region_id, position)] = row

    ce_paths: dict[int, Path] = {}
    ce_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for source in context.sources:
        ce_path = continuous_embedding_parquet_path(
            storage_root, source.continuous_embedding_job_id
        )
        if not ce_path.exists():
            raise RetrievalDiagnosticsConflict(
                "continuous embedding parquet not found for job "
                f"{source.continuous_embedding_job_id}"
            )
        ce_paths[int(source.source_index)] = ce_path
        for row in _read_table_rows(ce_path):
            region_id = str(row.get("region_id") or "")
            position = int(row.get("chunk_index_in_region", 0))
            ce_by_key[(int(source.source_index), region_id, position)] = row

    event_for_row = _assign_events_to_rows(embedding_rows, context.events)
    raw_vectors = np.asarray(
        [row["embedding"] for row in embedding_rows], dtype=np.float32
    )

    rows: list[dict[str, Any]] = []
    for idx, (emb_row, event) in enumerate(zip(embedding_rows, event_for_row)):
        region_id = str(emb_row["region_id"])
        source_index = int(emb_row.get("source_index", 0))
        original_region_id = str(emb_row.get("original_region_id") or region_id)
        chunk_index = int(emb_row["chunk_index_in_region"])
        key = (region_id, chunk_index)
        decoded = decoded_by_key.get(key, {})
        ce = ce_by_key.get((source_index, original_region_id, chunk_index), {})
        start = float(emb_row["start_timestamp"])
        end = float(emb_row["end_timestamp"])
        human_types = event.human_types if event else tuple()
        rows.append(
            {
                "idx": idx,
                "region_id": region_id,
                "original_region_id": original_region_id,
                "source_index": source_index,
                "continuous_embedding_job_id": str(
                    emb_row.get("continuous_embedding_job_id")
                    or context.continuous_embedding_job.id
                ),
                "event_classification_job_id": emb_row.get(
                    "event_classification_job_id"
                )
                or context.job.event_classification_job_id,
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
        "continuous_embedding_path": str(
            ce_paths.get(0)
            or continuous_embedding_parquet_path(
                storage_root, context.continuous_embedding_job.id
            )
        ),
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


def _geometry_variant_matrices(
    raw_vectors: FloatArray,
    *,
    variant: EmbeddingVariant,
    seed: int,
) -> tuple[FloatArray, FloatArray]:
    """Return ``(pre_l2_matrix, evaluation_matrix)`` for geometry diagnostics."""
    raw = raw_vectors.astype(np.float32, copy=False)
    if variant == "raw_l2":
        pre_l2 = raw
    elif variant.startswith("remove_pc"):
        count = int(variant.replace("remove_pc", ""))
        x_centered = raw - raw.mean(axis=0, keepdims=True)
        n_components = _safe_pca_components(raw, count)
        if n_components <= 0:
            pre_l2 = x_centered
        else:
            pca = PCA(n_components=n_components, random_state=seed)
            pca.fit(x_centered)
            x_proj = pca.inverse_transform(pca.transform(x_centered))
            pre_l2 = cast(FloatArray, x_centered - x_proj)
    elif variant == "whiten_pca":
        x_centered = raw - raw.mean(axis=0, keepdims=True)
        safe_components = _safe_pca_components(raw, 128)
        if safe_components <= 0:
            pre_l2 = x_centered
        else:
            pca = PCA(n_components=safe_components, whiten=True, random_state=seed)
            pre_l2 = cast(FloatArray, pca.fit_transform(x_centered))
    else:
        raise RetrievalDiagnosticsInvalid(
            f"unsupported geometry embedding variant: {variant}"
        )
    return pre_l2.astype(np.float32, copy=False), _normalize_rows(pre_l2)


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


def _cosine_percentiles(
    vectors: FloatArray, *, seed: int, n_pairs: int = 20_000
) -> dict[str, float]:
    if vectors.shape[0] < 2:
        return {f"p{p}": 0.0 for p in _PERCENTILES}
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vectors.shape[0], size=n_pairs)
    b = rng.integers(0, vectors.shape[0], size=n_pairs)
    mask = a != b
    cosine = np.sum(vectors[a[mask]] * vectors[b[mask]], axis=1)
    return {
        f"p{p}": float(v)
        for p, v in zip(_PERCENTILES, np.percentile(cosine, _PERCENTILES))
    }


def _percentile_summary(values: FloatArray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        out = {name: 0.0 for name in ("min", "max", "mean")}
        out.update({f"p{p}": 0.0 for p in _PERCENTILES if p not in {0, 100}})
        return out
    percentiles = {
        f"p{p}": float(v)
        for p, v in zip(_PERCENTILES, np.percentile(flat, _PERCENTILES))
        if p not in {0, 100}
    }
    return {
        "min": float(np.min(flat)),
        **percentiles,
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
    }


def _dimension_std_summary(pre_l2_vectors: FloatArray) -> dict[str, float]:
    if pre_l2_vectors.size == 0:
        summary = _percentile_summary(np.asarray([], dtype=np.float32))
        summary["near_zero_fraction"] = 0.0
        summary["dominance_ratio"] = 0.0
        return summary
    std = np.asarray(pre_l2_vectors, dtype=np.float64).std(axis=0)
    summary = _percentile_summary(cast(FloatArray, std))
    mean_std = float(np.mean(std)) if std.size else 0.0
    summary["near_zero_fraction"] = float(np.mean(std < 1e-5)) if std.size else 0.0
    summary["dominance_ratio"] = (
        float(np.max(std) / mean_std) if mean_std > 0.0 and std.size else 0.0
    )
    return summary


def _mean_vector_norm(vectors: FloatArray) -> float:
    if vectors.size == 0:
        return 0.0
    normalized = _normalize_rows(vectors)
    mean_vec = normalized.mean(axis=0)
    return float(np.linalg.norm(mean_vec))


def _mean_vector_band(value: float) -> str:
    if value < 0.05:
        return "good"
    if value < 0.15:
        return "okay"
    if value < 0.30:
        return "suspicious"
    return "collapse_risk"


def _effective_rank(vectors: FloatArray) -> float:
    if vectors.shape[0] == 0 or vectors.shape[1] == 0:
        return 0.0
    centered = vectors.astype(np.float64, copy=False) - vectors.mean(
        axis=0, keepdims=True
    )
    s = np.linalg.svd(centered, compute_uv=False)
    total = float(s.sum())
    if total <= 0.0:
        return 0.0
    p = s / total
    entropy = float(-(p * np.log(p + 1e-12)).sum())
    return float(np.exp(entropy))


def _effective_rank_band(value: float) -> str:
    if value < 10.0:
        return "severe_collapse"
    if value < 30.0:
        return "weak"
    if value < 80.0:
        return "plausible"
    return "broad"


def _pca_explained_variance(
    vectors: FloatArray, *, n_components: int, seed: int
) -> dict[str, float | int]:
    safe_components = _safe_pca_components(vectors, n_components)
    if safe_components <= 0:
        return {"pc1": 0.0, "pc1_5": 0.0, "pc1_10": 0.0, "components_available": 0}
    centered = vectors.astype(np.float32, copy=False) - vectors.mean(
        axis=0, keepdims=True
    )
    pca = PCA(n_components=safe_components, random_state=seed)
    pca.fit(centered)
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)

    def _clamped_sum(count: int) -> float:
        return float(np.clip(ratios[: min(count, ratios.size)].sum(), 0.0, 1.0))

    return {
        "pc1": _clamped_sum(1) if ratios.size >= 1 else 0.0,
        "pc1_5": _clamped_sum(5),
        "pc1_10": _clamped_sum(10),
        "components_available": int(safe_components),
    }


def _norm_distribution(
    vectors: FloatArray | None,
    *,
    available: bool,
    source: str,
) -> dict[str, float | bool | str]:
    if not available or vectors is None:
        return {"available": False, "source": source}
    norms = np.linalg.norm(vectors, axis=1)
    return {"available": True, "source": source, **_percentile_summary(norms)}


def _dimension_std_vectors(
    *,
    source_space: EmbeddingSpace,
    variant_pre_l2_matrix: FloatArray,
    retrieval_pre_l2_vectors: FloatArray | None,
    retrieval_pre_l2_source: str | None,
) -> tuple[FloatArray, str]:
    if source_space == "retrieval":
        if retrieval_pre_l2_vectors is not None:
            return retrieval_pre_l2_vectors, (
                retrieval_pre_l2_source or "retrieval_head_outputs"
            )
        return variant_pre_l2_matrix, "retrieval_post_l2_artifact"
    return variant_pre_l2_matrix, "contextual_artifact"


def _geometry_warnings(
    *,
    cosine: dict[str, float],
    mean_norm: float,
    effective_rank: float,
    pca_variance: dict[str, float | int],
) -> list[str]:
    warnings: list[str] = []
    if cosine.get("p50", 0.0) > 0.30:
        warnings.append("median_gt_0p3")
    if cosine.get("p75", 0.0) > 0.70:
        warnings.append("p75_gt_0p7")
    if cosine.get("p95", 0.0) > 0.95:
        warnings.append("p95_gt_0p95")
    if mean_norm >= 0.30:
        warnings.append("mean_norm_collapse_risk")
    elif mean_norm >= 0.15:
        warnings.append("mean_norm_suspicious")
    if effective_rank < 10.0:
        warnings.append("effective_rank_severe_collapse")
    elif effective_rank < 30.0:
        warnings.append("effective_rank_weak")
    pc1 = float(pca_variance.get("pc1", 0.0))
    pc1_5 = float(pca_variance.get("pc1_5", 0.0))
    pc1_10 = float(pca_variance.get("pc1_10", 0.0))
    if pc1 >= 0.30:
        warnings.append("pc1_dominant")
    if pc1_5 >= 0.70:
        warnings.append("pc5_dominant")
    if pc1_10 >= 0.85:
        warnings.append("pc10_dominant")
    return warnings


def _is_saturated_retrieval_raw(report: dict[str, Any]) -> bool:
    cosine = report.get("random_pair_percentiles", {}) or {}
    pca_variance = report.get("pca_explained_variance", {}) or {}
    return bool(
        float(cosine.get("p75", 0.0)) > 0.70
        or float(cosine.get("p95", 0.0)) > 0.95
        or float(report.get("mean_vector_norm") or 0.0) >= 0.30
        or float(report.get("effective_rank") or 0.0) < 10.0
        or float(pca_variance.get("pc1", 0.0)) >= 0.30
        or float(pca_variance.get("pc1_5", 0.0)) >= 0.70
    )


def build_geometry_space_report(
    *,
    raw_vectors: FloatArray,
    source_space: EmbeddingSpace,
    variant: EmbeddingVariant,
    seed: int,
    random_pairs: int,
    pca_components: int,
    artifact_path: str | None = None,
    pre_l2_vectors: FloatArray | None = None,
    pre_l2_source: str | None = None,
) -> dict[str, Any]:
    pre_l2_matrix, vectors = _geometry_variant_matrices(
        raw_vectors, variant=variant, seed=seed
    )
    cosine = _cosine_percentiles(vectors, seed=seed, n_pairs=random_pairs)
    mean_norm = _mean_vector_norm(vectors)
    rank = _effective_rank(vectors)
    pca_variance = _pca_explained_variance(
        vectors, n_components=pca_components, seed=seed
    )
    vector_dim = int(vectors.shape[1]) if vectors.ndim == 2 else 0
    norm_source = pre_l2_source or (
        "contextual_artifact" if source_space == "contextual" else "unavailable"
    )
    norm_vectors = (
        pre_l2_vectors
        if source_space == "retrieval" and pre_l2_vectors is not None
        else pre_l2_matrix
        if source_space == "contextual"
        else None
    )
    norm_available = bool(source_space == "contextual" or pre_l2_vectors is not None)
    dimension_vectors, dimension_source = _dimension_std_vectors(
        source_space=source_space,
        variant_pre_l2_matrix=pre_l2_matrix,
        retrieval_pre_l2_vectors=pre_l2_vectors,
        retrieval_pre_l2_source=pre_l2_source,
    )
    return {
        "available": True,
        "reason": None,
        "artifact_path": artifact_path,
        "source_space": source_space,
        "variant": variant,
        "row_count": int(vectors.shape[0]),
        "vector_dim": vector_dim,
        "random_pair_percentiles": cosine,
        "mean_vector_norm": mean_norm,
        "mean_vector_band": _mean_vector_band(mean_norm),
        "effective_rank": rank,
        "effective_rank_fraction": rank / max(vector_dim, 1),
        "effective_rank_band": _effective_rank_band(rank),
        "pca_explained_variance": pca_variance,
        "dimension_std": _dimension_std_summary(dimension_vectors),
        "dimension_std_source": dimension_source,
        "pre_l2_norm_distribution": _norm_distribution(
            norm_vectors, available=norm_available, source=norm_source
        ),
        "warnings": _geometry_warnings(
            cosine=cosine,
            mean_norm=mean_norm,
            effective_rank=rank,
            pca_variance=pca_variance,
        ),
    }


def _unavailable_geometry_space_report(
    *,
    space: GeometryEmbeddingSpace,
    reason: str,
    artifact_path: str | None,
) -> dict[str, Any]:
    source_space, variant = space.split(".", 1)
    return {
        "available": False,
        "reason": reason,
        "artifact_path": artifact_path,
        "source_space": source_space,
        "variant": variant,
        "row_count": 0,
        "vector_dim": 0,
        "random_pair_percentiles": {},
        "mean_vector_norm": None,
        "mean_vector_band": None,
        "effective_rank": None,
        "effective_rank_fraction": None,
        "effective_rank_band": None,
        "pca_explained_variance": {},
        "dimension_std": {},
        "dimension_std_source": "unavailable",
        "pre_l2_norm_distribution": {"available": False, "source": "unavailable"},
        "warnings": [reason],
    }


def _read_embedding_vectors(path: Path) -> FloatArray:
    rows = _read_table_rows(path)
    return np.asarray([row["embedding"] for row in rows], dtype=np.float32)


def _retrieval_head_outputs_path(storage_root: Path, job_id: str) -> Path:
    return masked_transformer_retrieval_head_outputs_path(storage_root, job_id)


def build_geometry_report(
    *,
    storage_root: Path,
    job_id: str,
    options: RetrievalReportOptions,
) -> dict[str, Any]:
    """Build geometry diagnostics across requested contextual/retrieval spaces."""
    requested_spaces = options.geometry_embedding_spaces or DEFAULT_GEOMETRY_SPACES
    spaces: dict[str, dict[str, Any]] = {}
    raw_cache: dict[EmbeddingSpace, tuple[Path, FloatArray] | None] = {}
    retrieval_pre_l2: FloatArray | None = None
    retrieval_pre_l2_path = _retrieval_head_outputs_path(storage_root, job_id)
    if retrieval_pre_l2_path.exists():
        retrieval_pre_l2 = _read_embedding_vectors(retrieval_pre_l2_path)

    for space in requested_spaces:
        source_name, variant_name = space.split(".", 1)
        source_space = cast(EmbeddingSpace, source_name)
        variant = cast(EmbeddingVariant, variant_name)
        if source_space not in raw_cache:
            path = _embedding_path(storage_root, job_id, source_space)
            if path.exists():
                raw_cache[source_space] = (path, _read_embedding_vectors(path))
            else:
                raw_cache[source_space] = None
        cached = raw_cache[source_space]
        if cached is None:
            spaces[space] = _unavailable_geometry_space_report(
                space=space,
                reason=f"{source_space}_artifact_unavailable",
                artifact_path=str(_embedding_path(storage_root, job_id, source_space)),
            )
            continue
        path, raw_vectors = cached
        spaces[space] = build_geometry_space_report(
            raw_vectors=raw_vectors,
            source_space=source_space,
            variant=variant,
            seed=options.seed,
            random_pairs=options.geometry_random_pairs,
            pca_components=options.geometry_pca_components,
            artifact_path=str(path),
            pre_l2_vectors=retrieval_pre_l2 if source_space == "retrieval" else None,
            pre_l2_source=(
                "retrieval_head_outputs"
                if source_space == "retrieval" and retrieval_pre_l2 is not None
                else None
            ),
        )

    retrieval_raw = spaces.get("retrieval.raw_l2")
    retrieval_raw_saturated = False
    if retrieval_raw is not None and retrieval_raw.get("available"):
        retrieval_raw_saturated = _is_saturated_retrieval_raw(retrieval_raw)
    warnings: list[str] = []
    if retrieval_raw_saturated:
        warnings.append("retrieval_raw_saturated")
    for name, report in spaces.items():
        for warning in report.get("warnings", []):
            warnings.append(f"{name}:{warning}")
    return {
        "spaces": spaces,
        "summary": {
            "retrieval_raw_saturated": retrieval_raw_saturated,
            "lambda_sweeps_blocked": retrieval_raw_saturated,
            "warnings": warnings,
        },
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

    geometry_report = (
        build_geometry_report(
            storage_root=storage_root,
            job_id=job_id,
            options=options,
        )
        if options.include_geometry_report
        else None
    )

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
            "include_geometry_report": options.include_geometry_report,
            "geometry_embedding_spaces": list(
                options.geometry_embedding_spaces or DEFAULT_GEOMETRY_SPACES
            ),
            "geometry_random_pairs": options.geometry_random_pairs,
            "geometry_pca_components": options.geometry_pca_components,
        },
        "artifacts": artifacts,
        "label_coverage": {
            "embedding_rows": len(rows),
            "sampled_queries": len(sample_indices),
            "human_labeled_query_pool_rows": human_candidate_count,
            "human_labeled_effective_events": context.correction_meta[
                "events_with_human_labels"
            ],
            "unlabeled_effective_events": context.correction_meta[
                "unlabeled_effective_events"
            ],
            "single_label_effective_events": context.correction_meta[
                "single_label_effective_events"
            ],
            "multi_label_effective_events": context.correction_meta[
                "multi_label_effective_events"
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
        "geometry_report": geometry_report,
    }
    return response


__all__ = [
    "DEFAULT_EMBEDDING_VARIANTS",
    "DEFAULT_GEOMETRY_SPACES",
    "DEFAULT_RETRIEVAL_MODES",
    "EmbeddingSpace",
    "EmbeddingVariant",
    "GeometryEmbeddingSpace",
    "HumanLabeledEvent",
    "RetrievalDiagnosticsConflict",
    "RetrievalDiagnosticsError",
    "RetrievalDiagnosticsInvalid",
    "RetrievalDiagnosticsNotFound",
    "RetrievalMode",
    "RetrievalReportOptions",
    "analyze_neighbors",
    "build_embedding_variants",
    "build_geometry_report",
    "build_geometry_space_report",
    "build_nearest_neighbor_report",
    "load_human_correction_events",
    "run_variant_matrix",
]
