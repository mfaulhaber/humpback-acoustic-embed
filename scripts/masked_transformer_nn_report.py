#!/usr/bin/env python
"""Nearest-neighbor diagnostics for a masked-transformer job.

Reads the app DB/storage from .env via Settings.from_repo_env().
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sqlalchemy import select

from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import RegionDetectionJob, VocalizationCorrection
from humpback.models.sequence_models import ContinuousEmbeddingJob, MaskedTransformerJob
from humpback.sequence_models.label_distribution import load_effective_event_labels
from humpback.storage import (
    continuous_embedding_parquet_path,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_k_decoded_path,
)


DEFAULT_JOB_ID = "9fd95e63-9f06-4cfb-8242-63a03dbbedd0"
FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class EventSpan:
    event_id: str
    start_utc: float
    end_utc: float
    effective_types: tuple[str, ...]
    human_types: tuple[str, ...]

    @property
    def duration(self) -> float:
        return self.end_utc - self.start_utc


def _format_ts(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return ""
    return f"{seconds:.3f}"


def _labels(types: tuple[str, ...] | set[str] | frozenset[str]) -> str:
    return ",".join(sorted(types)) if types else "(none)"


def _sqlite_path(database_url: str) -> str:
    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        return database_url
    return database_url[len(prefix) - 1 :]


def _read_table_rows(
    path: Path, columns: list[str] | None = None
) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=columns)
    return table.to_pylist()


def _assign_events_to_rows(
    rows: list[dict[str, Any]], events: list[EventSpan]
) -> list[EventSpan | None]:
    """Assign each row to the event containing its center timestamp."""
    annotations: list[EventSpan | None] = [None] * len(rows)
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


async def _load_job_context(settings: Settings, job_id: str, k: int):
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            job = await session.get(MaskedTransformerJob, job_id)
            if job is None:
                raise SystemExit(f"masked_transformer_job not found: {job_id}")
            cej = await session.get(
                ContinuousEmbeddingJob, job.continuous_embedding_job_id
            )
            if cej is None:
                raise SystemExit(
                    "continuous_embedding_job not found: "
                    f"{job.continuous_embedding_job_id}"
                )
            rdj = None
            if cej.region_detection_job_id:
                rdj = await session.get(RegionDetectionJob, cej.region_detection_job_id)
            if rdj is None:
                raise SystemExit(
                    "region_detection_job not found for continuous embedding job "
                    f"{cej.id}"
                )
            if not job.event_classification_job_id:
                raise SystemExit("masked transformer job has no classify binding")

            effective = await load_effective_event_labels(
                session,
                event_classification_job_id=job.event_classification_job_id,
                storage_root=settings.storage_root,
            )
            result = await session.execute(
                select(VocalizationCorrection).where(
                    VocalizationCorrection.region_detection_job_id
                    == cej.region_detection_job_id
                )
            )
            corrections = list(result.scalars().all())

            offset = float(rdj.start_timestamp or 0.0)
            human_by_event: dict[str, set[str]] = defaultdict(set)
            corrections_by_type: Counter[str] = Counter()
            for event in effective:
                event_start_sec = event.start_utc - offset
                event_end_sec = event.end_utc - offset
                labels: set[str] = set()
                for correction in corrections:
                    if (
                        correction.start_sec < event_end_sec
                        and correction.end_sec > event_start_sec
                    ):
                        key = f"{correction.correction_type}:{correction.type_name}"
                        corrections_by_type[key] += 1
                        if correction.correction_type == "add":
                            labels.add(correction.type_name)
                        elif correction.correction_type == "remove":
                            labels.discard(correction.type_name)
                if labels:
                    human_by_event[event.event_id] = labels

            events = [
                EventSpan(
                    event_id=e.event_id,
                    start_utc=float(e.start_utc),
                    end_utc=float(e.end_utc),
                    effective_types=tuple(sorted(e.types)),
                    human_types=tuple(sorted(human_by_event.get(e.event_id, set()))),
                )
                for e in effective
            ]

            job_meta = {
                "job_id": job.id,
                "status": job.status,
                "continuous_embedding_job_id": job.continuous_embedding_job_id,
                "event_classification_job_id": job.event_classification_job_id,
                "region_detection_job_id": cej.region_detection_job_id,
                "k_values": job.k_values,
                "k": k,
                "total_sequences": job.total_sequences,
                "total_chunks": job.total_chunks,
                "final_train_loss": job.final_train_loss,
                "final_val_loss": job.final_val_loss,
                "total_epochs": job.total_epochs,
            }
            correction_meta = {
                "total_correction_rows": len(corrections),
                "events_with_human_labels": sum(1 for e in events if e.human_types),
                "corrections_by_type": dict(corrections_by_type.most_common()),
            }
            return job_meta, correction_meta, events
    finally:
        await engine.dispose()


def _build_rows(settings: Settings, job_meta: dict[str, Any], events: list[EventSpan]):
    job_id = str(job_meta["job_id"])
    k = int(job_meta["k"])
    cej_id = str(job_meta["continuous_embedding_job_id"])

    z_path = masked_transformer_contextual_embeddings_path(
        settings.storage_root, job_id
    )
    decoded_path = masked_transformer_k_decoded_path(settings.storage_root, job_id, k)
    ce_path = continuous_embedding_parquet_path(settings.storage_root, cej_id)

    z_rows = _read_table_rows(z_path)
    decoded_rows = _read_table_rows(decoded_path)
    ce_rows = _read_table_rows(
        ce_path,
        columns=[
            "region_id",
            "hydrophone_id",
            "chunk_index_in_region",
            "is_in_pad",
            "call_probability",
            "event_overlap_fraction",
            "nearest_event_id",
            "tier",
        ],
    )

    decoded_by_key = {
        (r["region_id"], int(r["chunk_index_in_region"])): r for r in decoded_rows
    }
    ce_by_key = {(r["region_id"], int(r["chunk_index_in_region"])): r for r in ce_rows}
    event_for_row = _assign_events_to_rows(z_rows, events)

    raw_vectors = np.asarray([r["embedding"] for r in z_rows], dtype=np.float32)
    vectors = _normalize_rows(raw_vectors)

    rows: list[dict[str, Any]] = []
    for idx, (zr, event) in enumerate(zip(z_rows, event_for_row)):
        key = (zr["region_id"], int(zr["chunk_index_in_region"]))
        decoded = decoded_by_key.get(key, {})
        ce = ce_by_key.get(key, {})
        center = (float(zr["start_timestamp"]) + float(zr["end_timestamp"])) / 2.0
        rows.append(
            {
                "idx": idx,
                "region_id": zr["region_id"],
                "chunk_index": int(zr["chunk_index_in_region"]),
                "start_timestamp": float(zr["start_timestamp"]),
                "end_timestamp": float(zr["end_timestamp"]),
                "center_timestamp": center,
                "tier": str(ce.get("tier") or zr.get("tier") or ""),
                "hydrophone_id": ce.get("hydrophone_id") or "",
                "token": int(decoded.get("label", -1)),
                "token_confidence": float(decoded.get("confidence", math.nan)),
                "call_probability": float(ce.get("call_probability", math.nan)),
                "event_overlap_fraction": float(
                    ce.get("event_overlap_fraction", math.nan)
                ),
                "nearest_event_id": ce.get("nearest_event_id"),
                "event_id": event.event_id if event else None,
                "event_duration": event.duration if event else None,
                "human_types": event.human_types if event else tuple(),
                "effective_types": event.effective_types if event else tuple(),
            }
        )

    return (
        rows,
        vectors,
        raw_vectors,
        {"z_path": z_path, "decoded_path": decoded_path, "ce_path": ce_path},
    )


def _analyze_neighbors(
    rows: list[dict[str, Any]],
    vectors: np.ndarray,
    *,
    n_samples: int,
    topn: int,
    seed: int,
    sample_indices: list[int] | None = None,
    exclude_same_event_region: bool = False,
):
    human_candidates = [r["idx"] for r in rows if r["human_types"]]
    if sample_indices is None:
        rng = np.random.default_rng(seed)
        candidate_pool = (
            human_candidates
            if len(human_candidates) >= n_samples
            else list(range(len(rows)))
        )
        sample_indices = sorted(
            rng.choice(
                candidate_pool,
                size=min(n_samples, len(candidate_pool)),
                replace=False,
            ).tolist()
        )

    sims = vectors[sample_indices] @ vectors.T
    for row_num, idx in enumerate(sample_indices):
        q = rows[idx]
        sims[row_num, idx] = -np.inf
        if exclude_same_event_region:
            for candidate_idx, candidate in enumerate(rows):
                if candidate["region_id"] == q["region_id"]:
                    sims[row_num, candidate_idx] = -np.inf
                    continue
                if (
                    q["event_id"]
                    and candidate["event_id"]
                    and candidate["event_id"] == q["event_id"]
                ):
                    sims[row_num, candidate_idx] = -np.inf
    top_indices = np.argsort(np.multiply(sims, -1.0), axis=1)[:, :topn]

    neighbor_records: list[dict[str, Any]] = []
    query_summaries: list[dict[str, Any]] = []

    for query_order, (query_idx, nn_idxs) in enumerate(
        zip(sample_indices, top_indices), start=1
    ):
        q = rows[query_idx]
        q_human = set(q["human_types"])
        q_effective = set(q["effective_types"])
        counts = Counter()
        duration_comparable = 0

        for rank, neighbor_idx in enumerate(nn_idxs.tolist(), start=1):
            n = rows[neighbor_idx]
            n_human = set(n["human_types"])
            n_effective = set(n["effective_types"])
            center_delta = abs(
                float(q["center_timestamp"]) - float(n["center_timestamp"])
            )
            same_region = q["region_id"] == n["region_id"]
            adjacent_1s = same_region and center_delta <= 1.0
            nearby_5s = same_region and center_delta <= 5.0
            same_human = bool(q_human and n_human and q_human.intersection(n_human))
            exact_human = bool(q_human and n_human and q_human == n_human)
            same_effective = bool(
                q_effective and n_effective and q_effective.intersection(n_effective)
            )
            same_event = bool(q["event_id"] and q["event_id"] == n["event_id"])
            similar_duration = _similar_duration(
                q["event_duration"], n["event_duration"]
            )
            if q["event_duration"] is not None and n["event_duration"] is not None:
                duration_comparable += 1

            counts["same_human_label"] += int(same_human)
            counts["exact_human_label_set"] += int(exact_human)
            counts["same_effective_label"] += int(same_effective)
            counts["same_event"] += int(same_event)
            counts["same_region"] += int(same_region)
            counts["adjacent_1s"] += int(adjacent_1s)
            counts["nearby_5s"] += int(nearby_5s)
            counts["same_token"] += int(q["token"] == n["token"])
            counts["similar_duration"] += int(similar_duration)
            counts["neighbor_without_human_label"] += int(not n_human)
            counts["neighbor_low_event_overlap"] += int(
                not math.isfinite(n["event_overlap_fraction"])
                or n["event_overlap_fraction"] < 0.25
            )

            neighbor_records.append(
                {
                    "query_order": query_order,
                    "query_idx": query_idx,
                    "rank": rank,
                    "neighbor_idx": neighbor_idx,
                    "cosine": float(sims[query_order - 1, neighbor_idx]),
                    "query_region": q["region_id"],
                    "neighbor_region": n["region_id"],
                    "query_chunk": q["chunk_index"],
                    "neighbor_chunk": n["chunk_index"],
                    "center_delta_sec": center_delta,
                    "same_region": same_region,
                    "adjacent_1s": adjacent_1s,
                    "nearby_5s": nearby_5s,
                    "query_human_types": _labels(q["human_types"]),
                    "neighbor_human_types": _labels(n["human_types"]),
                    "same_human_label": same_human,
                    "exact_human_label_set": exact_human,
                    "query_effective_types": _labels(q["effective_types"]),
                    "neighbor_effective_types": _labels(n["effective_types"]),
                    "same_effective_label": same_effective,
                    "query_event_id": q["event_id"] or "",
                    "neighbor_event_id": n["event_id"] or "",
                    "same_event": same_event,
                    "query_duration": q["event_duration"],
                    "neighbor_duration": n["event_duration"],
                    "similar_duration": similar_duration,
                    "query_token": q["token"],
                    "neighbor_token": n["token"],
                    "same_token": q["token"] == n["token"],
                    "query_tier": q["tier"],
                    "neighbor_tier": n["tier"],
                    "query_overlap": q["event_overlap_fraction"],
                    "neighbor_overlap": n["event_overlap_fraction"],
                    "query_call_probability": q["call_probability"],
                    "neighbor_call_probability": n["call_probability"],
                    "query_start_timestamp": q["start_timestamp"],
                    "neighbor_start_timestamp": n["start_timestamp"],
                }
            )

        denom = float(topn)
        summary = {
            "query_order": query_order,
            "query_idx": query_idx,
            "query_region": q["region_id"],
            "query_chunk": q["chunk_index"],
            "query_start_timestamp": q["start_timestamp"],
            "query_human_types": _labels(q["human_types"]),
            "query_effective_types": _labels(q["effective_types"]),
            "query_event_id": q["event_id"] or "",
            "query_duration": q["event_duration"],
            "query_token": q["token"],
            "same_human_label_rate": counts["same_human_label"] / denom,
            "exact_human_label_set_rate": counts["exact_human_label_set"] / denom,
            "same_effective_label_rate": counts["same_effective_label"] / denom,
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
            "avg_cosine": float(
                np.mean(
                    [
                        r["cosine"]
                        for r in neighbor_records
                        if r["query_order"] == query_order
                    ]
                )
            ),
        }
        summary["verdict"] = _verdict(summary)
        query_summaries.append(summary)

    return query_summaries, neighbor_records, sample_indices, len(human_candidates)


def _aggregate_neighbor_metrics(
    query_summaries: list[dict[str, Any]],
    neighbor_records: list[dict[str, Any]],
) -> dict[str, Any]:
    all_neighbor_count = max(len(neighbor_records), 1)
    return {
        "same_human_label": sum(int(r["same_human_label"]) for r in neighbor_records)
        / all_neighbor_count,
        "same_region": sum(int(r["same_region"]) for r in neighbor_records)
        / all_neighbor_count,
        "similar_duration": float(
            np.mean([q["similar_duration_rate"] for q in query_summaries])
        ),
        "avg_cosine": float(np.mean([r["cosine"] for r in neighbor_records])),
    }


def _cosine_baseline(
    vectors: FloatArray, *, seed: int, n_pairs: int = 20_000
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, vectors.shape[0], size=n_pairs)
    b = rng.integers(0, vectors.shape[0], size=n_pairs)
    mask = a != b
    cosine = np.sum(vectors[a[mask]] * vectors[b[mask]], axis=1)
    return {
        "random_pair_percentiles": {
            str(p): float(v)
            for p, v in zip(
                [0, 1, 5, 25, 50, 75, 95, 99, 100],
                np.percentile(cosine, [0, 1, 5, 25, 50, 75, 95, 99, 100]),
            )
        },
    }


def _normalize_rows(vectors: FloatArray) -> FloatArray:
    return cast(FloatArray, normalize(vectors, norm="l2"))


def _mean_centered_vectors(raw_vectors: FloatArray) -> FloatArray:
    centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    return _normalize_rows(centered)


def _remove_top_pcs(raw_vectors: FloatArray, n_remove: int, *, seed: int) -> FloatArray:
    x_centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    pca = PCA(n_components=int(n_remove), random_state=seed)
    pca.fit(x_centered)
    x_proj = pca.inverse_transform(pca.transform(x_centered))
    x_resid = x_centered - x_proj
    return _normalize_rows(x_resid)


def _build_pc_removal_variants(
    raw_vectors: FloatArray,
    *,
    seed: int,
    remove_counts: list[int],
) -> dict[str, FloatArray]:
    variants = {
        "raw_l2": _normalize_rows(raw_vectors.astype(np.float32, copy=False)),
        "centered_l2": _mean_centered_vectors(raw_vectors),
    }
    for n_remove in remove_counts:
        variants[f"remove_pc{n_remove}"] = _remove_top_pcs(
            raw_vectors, n_remove, seed=seed
        )
    return variants


def _whiten_embeddings(
    raw_vectors: FloatArray,
    *,
    n_components: int,
    seed: int,
) -> FloatArray:
    x_centered = raw_vectors.astype(np.float32, copy=False) - raw_vectors.mean(
        axis=0, keepdims=True
    )
    n_components = min(int(n_components), raw_vectors.shape[0], raw_vectors.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    x_white = pca.fit_transform(x_centered)
    return _normalize_rows(x_white)


def _variant_sweep(
    rows: list[dict[str, Any]],
    variants: dict[str, FloatArray],
    *,
    n_samples: int,
    topn: int,
    seed: int,
    sample_indices: list[int],
    exclude_same_event_region: bool = False,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, vectors in variants.items():
        query_summaries, neighbor_records, _, _ = _analyze_neighbors(
            rows,
            vectors,
            n_samples=n_samples,
            topn=topn,
            seed=seed,
            sample_indices=sample_indices,
            exclude_same_event_region=exclude_same_event_region,
        )
        metrics = _aggregate_neighbor_metrics(query_summaries, neighbor_records)
        metrics["random_pair_percentiles"] = _cosine_baseline(vectors, seed=seed)[
            "random_pair_percentiles"
        ]
        metrics["verdicts"] = dict(Counter(q["verdict"] for q in query_summaries))
        out[name] = metrics
    return out


def _pca_diagnostics(
    raw_vectors: FloatArray,
    *,
    n_components: int,
    seed: int,
    center: bool,
) -> dict[str, Any]:
    n_components = min(int(n_components), raw_vectors.shape[0], raw_vectors.shape[1])
    x = raw_vectors.astype(np.float32, copy=False)
    if center:
        x = x - x.mean(axis=0, keepdims=True)

    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(x)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    def _cum(component_count: int) -> float:
        idx = min(component_count, len(cumulative)) - 1
        return float(cumulative[idx]) if idx >= 0 else float("nan")

    return {
        "centered": center,
        "n_components": n_components,
        "pc1_explained_variance": float(explained[0])
        if len(explained)
        else float("nan"),
        "pc1_pc5_cumulative_variance": _cum(5),
        "pc1_pc10_cumulative_variance": _cum(10),
        "pc1_pc50_cumulative_variance": _cum(50),
        "top_components": [
            {
                "component": i + 1,
                "explained_variance": float(explained[i]),
                "cumulative_variance": float(cumulative[i]),
            }
            for i in range(min(10, len(explained)))
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


def _write_report(
    path: Path,
    *,
    job_meta: dict[str, Any],
    correction_meta: dict[str, Any],
    artifact_paths: dict[str, Path],
    rows: list[dict[str, Any]],
    query_summaries: list[dict[str, Any]],
    neighbor_records: list[dict[str, Any]],
    cosine_baseline: dict[str, Any],
    mean_centered_comparison: dict[str, Any] | None,
    pc_removal_sweep: dict[str, Any] | None,
    excluded_neighbor_sweep: dict[str, Any] | None,
    pca_diagnostics: dict[str, Any] | None,
    human_candidate_count: int,
    neighbor_csv: Path,
    query_csv: Path,
    centered_neighbor_csv: Path | None = None,
    centered_query_csv: Path | None = None,
):
    verdicts = Counter(q["verdict"] for q in query_summaries)
    all_neighbor_count = max(len(neighbor_records), 1)
    agg = {
        "same_human_label": sum(int(r["same_human_label"]) for r in neighbor_records)
        / all_neighbor_count,
        "exact_human_label_set": sum(
            int(r["exact_human_label_set"]) for r in neighbor_records
        )
        / all_neighbor_count,
        "same_effective_label": sum(
            int(r["same_effective_label"]) for r in neighbor_records
        )
        / all_neighbor_count,
        "same_event": sum(int(r["same_event"]) for r in neighbor_records)
        / all_neighbor_count,
        "same_region": sum(int(r["same_region"]) for r in neighbor_records)
        / all_neighbor_count,
        "adjacent_1s": sum(int(r["adjacent_1s"]) for r in neighbor_records)
        / all_neighbor_count,
        "nearby_5s": sum(int(r["nearby_5s"]) for r in neighbor_records)
        / all_neighbor_count,
        "same_token": sum(int(r["same_token"]) for r in neighbor_records)
        / all_neighbor_count,
        "without_human_label": sum(
            r["neighbor_human_types"] == "(none)" for r in neighbor_records
        )
        / all_neighbor_count,
        "low_event_overlap": sum(
            float(r["neighbor_overlap"]) < 0.25
            for r in neighbor_records
            if math.isfinite(float(r["neighbor_overlap"]))
        )
        / all_neighbor_count,
        "avg_cosine": float(np.mean([r["cosine"] for r in neighbor_records])),
        "median_cosine": float(np.median([r["cosine"] for r in neighbor_records])),
    }

    human_label_counter: Counter[str] = Counter()
    for row in rows:
        for label in row["human_types"]:
            human_label_counter[label] += 1

    good_examples = sorted(
        query_summaries,
        key=lambda q: (
            q["verdict"] != "good",
            -q["same_human_label_rate"],
            q["adjacent_1s_rate"],
        ),
    )[:8]
    bad_examples = sorted(
        query_summaries,
        key=lambda q: (
            q["verdict"] == "good",
            -max(q["adjacent_1s_rate"], q["neighbor_without_human_label_rate"]),
            q["same_human_label_rate"],
        ),
    )[:8]

    lines: list[str] = []
    lines.append(f"# Masked Transformer Nearest-Neighbor Report, k={job_meta['k']}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- job_id: `{job_meta['job_id']}`")
    lines.append(f"- DB: `{_sqlite_path(Settings.from_repo_env().database_url)}`")
    lines.append(f"- storage_root: `{Settings.from_repo_env().storage_root}`")
    lines.append(
        f"- continuous_embedding_job_id: `{job_meta['continuous_embedding_job_id']}`"
    )
    lines.append(
        f"- event_classification_job_id: `{job_meta['event_classification_job_id']}`"
    )
    lines.append(f"- region_detection_job_id: `{job_meta['region_detection_job_id']}`")
    lines.append(f"- contextual embeddings: `{artifact_paths['z_path']}`")
    lines.append(f"- decoded tokens: `{artifact_paths['decoded_path']}`")
    lines.append(f"- upstream chunks: `{artifact_paths['ce_path']}`")
    lines.append("")
    lines.append("## Label Source")
    lines.append("")
    lines.append(
        "Primary call-type metric uses human correction labels: overlapping "
        "`vocalization_corrections` `add` rows, minus overlapping `remove` rows, "
        "assigned to boundary-corrected effective events. The Sequence Models "
        "effective classifier-overlay label set is also reported as a secondary "
        "metric, but this job's classifier output is broad enough that it is not "
        "very diagnostic for nearest-neighbor quality."
    )
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(
        _markdown_table(
            ["metric", "value"],
            [
                ["embedding rows", len(rows)],
                ["sampled queries", len(query_summaries)],
                ["human-labeled query pool rows", human_candidate_count],
                [
                    "human-labeled effective events",
                    correction_meta["events_with_human_labels"],
                ],
                [
                    "vocalization correction rows",
                    correction_meta["total_correction_rows"],
                ],
                [
                    "vector dim",
                    len(
                        pq.read_table(artifact_paths["z_path"], columns=["embedding"])
                        .column("embedding")[0]
                        .as_py()
                    ),
                ],
                ["job total_sequences", job_meta["total_sequences"]],
                ["job total_chunks", job_meta["total_chunks"]],
                ["final train loss", f"{job_meta['final_train_loss']:.4f}"],
                ["final val loss", f"{job_meta['final_val_loss']:.4f}"],
                ["epochs", job_meta["total_epochs"]],
            ],
        )
    )
    lines.append("")
    lines.append("Top human labels by chunk coverage:")
    lines.append("")
    lines.append(
        _markdown_table(
            ["label", "chunks"],
            [[label, count] for label, count in human_label_counter.most_common(20)],
        )
    )
    lines.append("")
    lines.append("## Aggregate Top-10 Neighbor Metrics")
    lines.append("")
    lines.append(
        _markdown_table(
            ["metric", "value"],
            [
                ["same human label overlap", f"{agg['same_human_label']:.1%}"],
                ["exact human label-set match", f"{agg['exact_human_label_set']:.1%}"],
                [
                    "same effective classifier/corrected label overlap",
                    f"{agg['same_effective_label']:.1%}",
                ],
                [
                    "similar event duration",
                    f"{np.mean([q['similar_duration_rate'] for q in query_summaries]):.1%}",
                ],
                ["same event", f"{agg['same_event']:.1%}"],
                ["same token", f"{agg['same_token']:.1%}"],
                ["same region", f"{agg['same_region']:.1%}"],
                ["same region within 1s", f"{agg['adjacent_1s']:.1%}"],
                ["same region within 5s", f"{agg['nearby_5s']:.1%}"],
                ["neighbor without human label", f"{agg['without_human_label']:.1%}"],
                ["neighbor event-overlap < 0.25", f"{agg['low_event_overlap']:.1%}"],
                ["average cosine", f"{agg['avg_cosine']:.4f}"],
                ["median cosine", f"{agg['median_cosine']:.4f}"],
            ],
        )
    )
    lines.append("")
    lines.append(
        "Verdicts are query-level heuristics: `good` means at least half of top-10 neighbors share a human label and similar duration, without being mostly adjacent in time."
    )
    lines.append("")
    lines.append(
        _markdown_table(
            ["verdict", "queries"], [[k, v] for k, v in verdicts.most_common()]
        )
    )
    lines.append("")
    lines.append("## Cosine Baseline")
    lines.append("")
    lines.append(
        "Random pair cosine is highly bimodal: many unrelated pairs are far apart, "
        "but the upper quartile is already very high. That makes a top-10 cosine "
        "near 1.0 less meaningful by itself; label and time diagnostics matter more."
    )
    lines.append("")
    pct = cosine_baseline["random_pair_percentiles"]
    lines.append(
        _markdown_table(
            ["random-pair percentile", "cosine"],
            [
                [p, f"{pct[p]:.4f}"]
                for p in ["0", "1", "5", "25", "50", "75", "95", "99", "100"]
            ],
        )
    )
    lines.append("")
    if mean_centered_comparison is not None:
        lines.append("## Mean-Centered Comparison")
        lines.append("")
        lines.append(
            "Mean-centering tests whether nearest neighbors are mostly riding a "
            "global region/context direction. Both rows use the same sampled query "
            "chunks and cosine top-10 retrieval; the centered row uses "
            "`X_centered = X - X.mean(axis=0, keepdims=True)` followed by L2 "
            "normalization."
        )
        lines.append("")
        comparison_rows = []
        for label in (
            "raw contextual embeddings",
            "mean-centered contextual embeddings",
        ):
            metrics = mean_centered_comparison[label]
            comparison_rows.append(
                [
                    label,
                    f"{metrics['same_human_label']:.1%}",
                    f"{metrics['same_region']:.1%}",
                    f"{metrics['similar_duration']:.1%}",
                    f"{metrics['avg_cosine']:.4f}",
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "space",
                    "same human label overlap",
                    "same region",
                    "similar duration",
                    "average cosine",
                ],
                comparison_rows,
            )
        )
        lines.append("")
        lines.append("Random-pair cosine percentiles:")
        lines.append("")
        percentile_rows = []
        for p in ["0", "1", "5", "25", "50", "75", "95", "99", "100"]:
            percentile_rows.append(
                [
                    p,
                    f"{mean_centered_comparison['raw contextual embeddings']['random_pair_percentiles'][p]:.4f}",
                    f"{mean_centered_comparison['mean-centered contextual embeddings']['random_pair_percentiles'][p]:.4f}",
                ]
            )
        lines.append(
            _markdown_table(
                [
                    "random-pair percentile",
                    "raw contextual embeddings",
                    "mean-centered contextual embeddings",
                ],
                percentile_rows,
            )
        )
        lines.append("")
    if pc_removal_sweep is not None:
        lines.append("## PC-Removal Variant Sweep")
        lines.append("")
        lines.append(
            "This sweep removes dominant PCA directions from mean-centered raw "
            "contextual embeddings, then L2-normalizes the residuals for cosine "
            "retrieval. Each variant uses the same sampled query chunks."
        )
        lines.append("")
        variant_order = [
            name
            for name in [
                "raw_l2",
                "centered_l2",
                "remove_pc1",
                "remove_pc3",
                "remove_pc5",
                "remove_pc10",
                "whiten_pca",
            ]
            if name in pc_removal_sweep
        ]
        lines.append(
            _markdown_table(
                [
                    "variant",
                    "same human label overlap",
                    "same region",
                    "similar duration",
                    "average cosine",
                    "good queries",
                ],
                [
                    [
                        name,
                        f"{pc_removal_sweep[name]['same_human_label']:.1%}",
                        f"{pc_removal_sweep[name]['same_region']:.1%}",
                        f"{pc_removal_sweep[name]['similar_duration']:.1%}",
                        f"{pc_removal_sweep[name]['avg_cosine']:.4f}",
                        pc_removal_sweep[name]["verdicts"].get("good", 0),
                    ]
                    for name in variant_order
                ],
            )
        )
        lines.append("")
        lines.append("Random-pair cosine percentiles:")
        lines.append("")
        lines.append(
            _markdown_table(
                [
                    "variant",
                    "p0",
                    "p1",
                    "p5",
                    "p25",
                    "p50",
                    "p75",
                    "p95",
                    "p99",
                    "p100",
                ],
                [
                    [
                        name,
                        *[
                            f"{pc_removal_sweep[name]['random_pair_percentiles'][p]:.4f}"
                            for p in [
                                "0",
                                "1",
                                "5",
                                "25",
                                "50",
                                "75",
                                "95",
                                "99",
                                "100",
                            ]
                        ],
                    ]
                    for name in variant_order
                ],
            )
        )
        lines.append("")
    if excluded_neighbor_sweep is not None:
        lines.append("## Excluding Same Event And Same Region")
        lines.append("")
        lines.append(
            "This treatment retrieves top-k neighbors after masking candidates "
            "from the same event and the same region as the query. It is meant "
            "to test whether nearest-neighbor quality survives after removing "
            "obvious local/context matches."
        )
        lines.append("")
        variant_order = [
            name
            for name in [
                "raw_l2",
                "centered_l2",
                "remove_pc1",
                "remove_pc3",
                "remove_pc5",
                "remove_pc10",
                "whiten_pca",
            ]
            if name in excluded_neighbor_sweep
        ]
        lines.append(
            _markdown_table(
                [
                    "variant",
                    "same human label overlap",
                    "same region",
                    "similar duration",
                    "average cosine",
                    "good queries",
                ],
                [
                    [
                        name,
                        f"{excluded_neighbor_sweep[name]['same_human_label']:.1%}",
                        f"{excluded_neighbor_sweep[name]['same_region']:.1%}",
                        f"{excluded_neighbor_sweep[name]['similar_duration']:.1%}",
                        f"{excluded_neighbor_sweep[name]['avg_cosine']:.4f}",
                        excluded_neighbor_sweep[name]["verdicts"].get("good", 0),
                    ]
                    for name in variant_order
                ],
            )
        )
        lines.append("")
        lines.append(
            "Random-pair cosine percentiles are unchanged from the corresponding embedding variants; only the neighbor candidate set is masked."
        )
        lines.append("")
    if pca_diagnostics is not None:
        lines.append("## PCA Diagnostics")
        lines.append("")
        lines.append(
            "PCA helps test whether the vector space is dominated by a few "
            "region/context axes. Values below are fit on raw contextual "
            "embeddings, before cosine normalization."
        )
        lines.append("")
        lines.append(
            _markdown_table(
                ["metric", "value"],
                [
                    ["mean centered", str(pca_diagnostics["centered"])],
                    ["components fit", pca_diagnostics["n_components"]],
                    [
                        "PC1 explained variance",
                        f"{pca_diagnostics['pc1_explained_variance']:.2%}",
                    ],
                    [
                        "PC1-PC5 cumulative variance",
                        f"{pca_diagnostics['pc1_pc5_cumulative_variance']:.2%}",
                    ],
                    [
                        "PC1-PC10 cumulative variance",
                        f"{pca_diagnostics['pc1_pc10_cumulative_variance']:.2%}",
                    ],
                    [
                        "PC1-PC50 cumulative variance",
                        f"{pca_diagnostics['pc1_pc50_cumulative_variance']:.2%}",
                    ],
                ],
            )
        )
        lines.append("")
        lines.append("Top PCA variance ratios:")
        lines.append("")
        lines.append(
            _markdown_table(
                ["PC", "explained", "cumulative"],
                [
                    [
                        row["component"],
                        f"{row['explained_variance']:.4%}",
                        f"{row['cumulative_variance']:.4%}",
                    ]
                    for row in pca_diagnostics["top_components"]
                ],
            )
        )
        lines.append("")
    lines.append("## Representative Good Queries")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                "q",
                "human labels",
                "duration",
                "same label",
                "similar dur",
                "adjacent 1s",
                "same region",
                "avg cosine",
                "verdict",
            ],
            [
                [
                    q["query_order"],
                    q["query_human_types"],
                    _format_ts(q["query_duration"]),
                    f"{q['same_human_label_rate']:.0%}",
                    f"{q['similar_duration_rate']:.0%}",
                    f"{q['adjacent_1s_rate']:.0%}",
                    f"{q['same_region_rate']:.0%}",
                    f"{q['avg_cosine']:.3f}",
                    q["verdict"],
                ]
                for q in good_examples
            ],
        )
    )
    lines.append("")
    lines.append("## Representative Risky Queries")
    lines.append("")
    lines.append(
        _markdown_table(
            [
                "q",
                "human labels",
                "duration",
                "same label",
                "unlabeled nn",
                "adjacent 1s",
                "nearby 5s",
                "same region",
                "avg cosine",
                "verdict",
            ],
            [
                [
                    q["query_order"],
                    q["query_human_types"],
                    _format_ts(q["query_duration"]),
                    f"{q['same_human_label_rate']:.0%}",
                    f"{q['neighbor_without_human_label_rate']:.0%}",
                    f"{q['adjacent_1s_rate']:.0%}",
                    f"{q['nearby_5s_rate']:.0%}",
                    f"{q['same_region_rate']:.0%}",
                    f"{q['avg_cosine']:.3f}",
                    q["verdict"],
                ]
                for q in bad_examples
            ],
        )
    )
    lines.append("")
    lines.append("## Full Detail")
    lines.append("")
    lines.append(f"- Query summary CSV: `{query_csv}`")
    lines.append(f"- All top-10 neighbor rows CSV: `{neighbor_csv}`")
    if centered_query_csv is not None and centered_neighbor_csv is not None:
        lines.append(f"- Mean-centered query summary CSV: `{centered_query_csv}`")
        lines.append(f"- Mean-centered top-10 neighbors CSV: `{centered_neighbor_csv}`")
    lines.append("")
    lines.append(
        "The detailed CSV contains all 50 sampled query points and their top-10 cosine neighbors with human labels, effective labels, timing, token, tier, event overlap, and adjacency flags."
    )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--k", type=int, default=150)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/nn-report"))
    parser.add_argument(
        "--pca-diagnostics",
        action="store_true",
        help="Include PCA variance diagnostics for raw contextual embeddings.",
    )
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument(
        "--pca-no-center",
        action="store_true",
        help="Do not subtract the global mean before fitting PCA.",
    )
    parser.add_argument(
        "--mean-centered-comparison",
        action="store_true",
        help=(
            "Compare nearest-neighbor metrics for raw L2-normalized contextual "
            "embeddings versus mean-centered then L2-normalized embeddings."
        ),
    )
    parser.add_argument(
        "--pc-removal-sweep",
        action="store_true",
        help=(
            "Run nearest-neighbor metrics for raw_l2, centered_l2, and residual "
            "spaces after removing top principal components."
        ),
    )
    parser.add_argument(
        "--pc-removal-counts",
        default="1,3,5,10",
        help="Comma-separated top-PC counts to remove for --pc-removal-sweep.",
    )
    parser.add_argument(
        "--whitening-sweep",
        action="store_true",
        help=(
            "Add a PCA whitening variant to the sweep: center, PCA(whiten=True), "
            "then L2-normalize."
        ),
    )
    parser.add_argument(
        "--whiten-components",
        type=int,
        default=128,
        help="Number of PCA components for --whitening-sweep.",
    )
    parser.add_argument(
        "--exclude-same-event-region",
        action="store_true",
        help=(
            "Add a separate nearest-neighbor treatment that excludes candidate "
            "neighbors from the same event or same region as each query."
        ),
    )
    args = parser.parse_args()

    settings = Settings.from_repo_env()
    job_meta, correction_meta, events = await _load_job_context(
        settings, args.job_id, args.k
    )
    rows, vectors, raw_vectors, artifact_paths = _build_rows(settings, job_meta, events)
    query_summaries, neighbor_records, _sample_indices, human_candidate_count = (
        _analyze_neighbors(
            rows,
            vectors,
            n_samples=args.samples,
            topn=args.topn,
            seed=args.seed,
        )
    )
    cosine_baseline = _cosine_baseline(vectors, seed=args.seed)
    mean_centered_comparison = None
    centered_query_summaries = None
    centered_neighbor_records = None
    if args.mean_centered_comparison:
        centered_vectors = _mean_centered_vectors(raw_vectors)
        (
            centered_query_summaries,
            centered_neighbor_records,
            _,
            _,
        ) = _analyze_neighbors(
            rows,
            centered_vectors,
            n_samples=args.samples,
            topn=args.topn,
            seed=args.seed,
            sample_indices=_sample_indices,
        )
        raw_metrics = _aggregate_neighbor_metrics(query_summaries, neighbor_records)
        centered_metrics = _aggregate_neighbor_metrics(
            centered_query_summaries, centered_neighbor_records
        )
        raw_metrics["random_pair_percentiles"] = cosine_baseline[
            "random_pair_percentiles"
        ]
        centered_metrics["random_pair_percentiles"] = _cosine_baseline(
            centered_vectors, seed=args.seed
        )["random_pair_percentiles"]
        mean_centered_comparison = {
            "raw contextual embeddings": raw_metrics,
            "mean-centered contextual embeddings": centered_metrics,
        }
    pc_removal_sweep = None
    excluded_neighbor_sweep = None
    variant_vectors = None
    if args.pc_removal_sweep or args.whitening_sweep or args.exclude_same_event_region:
        remove_counts = [
            int(part) for part in str(args.pc_removal_counts).split(",") if part.strip()
        ]
        variant_vectors = _build_pc_removal_variants(
            raw_vectors,
            seed=args.seed,
            remove_counts=remove_counts if args.pc_removal_sweep else [],
        )
        if args.whitening_sweep:
            variant_vectors["whiten_pca"] = _whiten_embeddings(
                raw_vectors,
                n_components=args.whiten_components,
                seed=args.seed,
            )
        if args.pc_removal_sweep or args.whitening_sweep:
            pc_removal_sweep = _variant_sweep(
                rows,
                variant_vectors,
                n_samples=args.samples,
                topn=args.topn,
                seed=args.seed,
                sample_indices=_sample_indices,
            )
        if args.exclude_same_event_region:
            excluded_neighbor_sweep = _variant_sweep(
                rows,
                variant_vectors,
                n_samples=args.samples,
                topn=args.topn,
                seed=args.seed,
                sample_indices=_sample_indices,
                exclude_same_event_region=True,
            )
    pca_diagnostics = (
        _pca_diagnostics(
            raw_vectors,
            n_components=args.pca_components,
            seed=args.seed,
            center=not args.pca_no_center,
        )
        if args.pca_diagnostics
        else None
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.job_id}_k{args.k}"
    query_csv = args.out_dir / f"{prefix}_query_summary.csv"
    neighbor_csv = args.out_dir / f"{prefix}_neighbors.csv"
    centered_query_csv = args.out_dir / f"{prefix}_mean_centered_query_summary.csv"
    centered_neighbor_csv = args.out_dir / f"{prefix}_mean_centered_neighbors.csv"
    report_md = args.out_dir / f"{prefix}_report.md"
    meta_json = args.out_dir / f"{prefix}_metadata.json"

    _write_csv(query_csv, query_summaries)
    _write_csv(neighbor_csv, neighbor_records)
    if centered_query_summaries is not None and centered_neighbor_records is not None:
        _write_csv(centered_query_csv, centered_query_summaries)
        _write_csv(centered_neighbor_csv, centered_neighbor_records)
    meta_json.write_text(
        json.dumps(
            {
                "job": job_meta,
                "corrections": correction_meta,
                "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
                "cosine_baseline": cosine_baseline,
                "mean_centered_comparison": mean_centered_comparison,
                "pc_removal_sweep": pc_removal_sweep,
                "excluded_neighbor_sweep": excluded_neighbor_sweep,
                "pca_diagnostics": pca_diagnostics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_report(
        report_md,
        job_meta=job_meta,
        correction_meta=correction_meta,
        artifact_paths=artifact_paths,
        rows=rows,
        query_summaries=query_summaries,
        neighbor_records=neighbor_records,
        cosine_baseline=cosine_baseline,
        mean_centered_comparison=mean_centered_comparison,
        pc_removal_sweep=pc_removal_sweep,
        excluded_neighbor_sweep=excluded_neighbor_sweep,
        pca_diagnostics=pca_diagnostics,
        human_candidate_count=human_candidate_count,
        neighbor_csv=neighbor_csv,
        query_csv=query_csv,
        centered_neighbor_csv=centered_neighbor_csv
        if centered_neighbor_records is not None
        else None,
        centered_query_csv=centered_query_csv
        if centered_query_summaries is not None
        else None,
    )

    print(report_md)
    print(query_csv)
    print(neighbor_csv)
    if centered_query_summaries is not None and centered_neighbor_records is not None:
        print(centered_query_csv)
        print(centered_neighbor_csv)
    print(meta_json)


if __name__ == "__main__":
    asyncio.run(async_main())
