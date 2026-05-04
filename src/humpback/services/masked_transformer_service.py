"""Service layer for masked-transformer sequence jobs (ADR-061).

Mirrors :mod:`humpback.services.hmm_sequence_service` but with
training-signature idempotency and the extend-k-sweep entry point.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    MaskedTransformerJob,
)
from humpback.sequence_models.exemplars import WindowMeta, select_exemplars
from humpback.sequence_models.label_distribution import (
    EffectiveEventLabels,
    WindowAnnotation,
    assign_labels_to_windows,
    compute_label_distribution,
    load_effective_event_labels,
)
from humpback.sequence_models.overlay import OverlayMetadata, compute_overlay
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.storage import (
    atomic_rename,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_dir,
    masked_transformer_k_decoded_path,
    masked_transformer_k_exemplars_path,
    masked_transformer_k_label_distribution_path,
    masked_transformer_k_overlay_path,
)

logger = logging.getLogger(__name__)


class CancelTerminalJobError(Exception):
    """Raised when caller attempts to cancel a job in a terminal state."""


class ExtendKSweepError(Exception):
    """Raised when extend-k-sweep is invoked on a non-completed job."""


# ---------------------------------------------------------------------------
# Training signature
# ---------------------------------------------------------------------------


def compute_training_signature(
    *,
    continuous_embedding_job_id: str,
    preset: str,
    mask_fraction: float,
    span_length_min: int,
    span_length_max: int,
    dropout: float,
    mask_weight_bias: bool,
    cosine_loss_weight: float,
    max_epochs: int,
    early_stop_patience: int,
    val_split: float,
    seed: int,
) -> str:
    """Stable signature over training-only config (excludes ``k_values``).

    Excluding ``k_values`` keeps the same trained transformer reusable
    across extend-k-sweep calls — only tokenization is re-run.
    """
    payload = {
        "continuous_embedding_job_id": continuous_embedding_job_id,
        "preset": preset,
        "mask_fraction": float(mask_fraction),
        "span_length_min": int(span_length_min),
        "span_length_max": int(span_length_max),
        "dropout": float(dropout),
        "mask_weight_bias": bool(mask_weight_bias),
        "cosine_loss_weight": float(cosine_loss_weight),
        "max_epochs": int(max_epochs),
        "early_stop_patience": int(early_stop_patience),
        "val_split": float(val_split),
        "seed": int(seed),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_k_values(payload: str | list[int]) -> list[int]:
    if isinstance(payload, list):
        values = list(payload)
    else:
        values = json.loads(payload)
        if not isinstance(values, list):
            raise ValueError("k_values must be a JSON list")
    deduped: list[int] = []
    seen: set[int] = set()
    for raw_v in values:
        v = int(raw_v)
        if v < 2:
            raise ValueError(f"k must be >= 2, got {v}")
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    if not deduped:
        raise ValueError("k_values must be a non-empty list")
    return deduped


def serialize_k_values(values: list[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


async def create_masked_transformer_job(
    session: AsyncSession,
    *,
    continuous_embedding_job_id: str,
    event_classification_job_id: Optional[str] = None,
    preset: str = "default",
    mask_fraction: float = 0.20,
    span_length_min: int = 2,
    span_length_max: int = 6,
    dropout: float = 0.1,
    mask_weight_bias: bool = True,
    cosine_loss_weight: float = 0.0,
    max_epochs: int = 30,
    early_stop_patience: int = 3,
    val_split: float = 0.1,
    seed: int = 42,
    k_values: list[int] | None = None,
) -> tuple[MaskedTransformerJob, bool]:
    """Create a masked-transformer training job.

    Returns ``(job, created)`` where ``created`` is ``False`` when an
    existing job with the same ``training_signature`` is returned.

    The Classify FK is resolved at submit time: when omitted, the most
    recent completed ``EventClassificationJob`` for the upstream
    segmentation is bound; when provided, it must be completed and on
    the same segmentation. An idempotent re-submit returns the existing
    job unchanged regardless of any ``event_classification_job_id``
    value passed (re-binding is done via the regenerate endpoint).
    """
    if preset not in {"small", "default", "large"}:
        raise ValueError(f"preset must be one of small/default/large, got {preset!r}")

    cej = await session.get(ContinuousEmbeddingJob, continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            f"continuous_embedding_job not found: {continuous_embedding_job_id}"
        )
    if cej.status != JobStatus.complete.value:
        raise ValueError(
            "masked-transformer job requires a completed continuous_embedding_job "
            f"(current status: {cej.status!r})"
        )
    if source_kind_for(cej.model_version) != SOURCE_KIND_REGION_CRNN:
        raise ValueError(
            "masked-transformer job requires a CRNN region-based upstream "
            f"continuous_embedding_job (got source_kind={source_kind_for(cej.model_version)!r})"
        )

    k_list = parse_k_values(k_values if k_values is not None else [100])

    signature = compute_training_signature(
        continuous_embedding_job_id=continuous_embedding_job_id,
        preset=preset,
        mask_fraction=mask_fraction,
        span_length_min=span_length_min,
        span_length_max=span_length_max,
        dropout=dropout,
        mask_weight_bias=mask_weight_bias,
        cosine_loss_weight=cosine_loss_weight,
        max_epochs=max_epochs,
        early_stop_patience=early_stop_patience,
        val_split=val_split,
        seed=seed,
    )

    existing = await session.execute(
        select(MaskedTransformerJob).where(
            MaskedTransformerJob.training_signature == signature
        )
    )
    found = existing.scalar_one_or_none()
    if found is not None:
        return found, False

    if not cej.event_segmentation_job_id:
        raise ValueError(
            "continuous_embedding_job has no event_segmentation_job_id; "
            "cannot resolve Classify binding"
        )
    from humpback.services.hmm_sequence_service import (
        resolve_event_classification_job_id,
    )

    classify_id = await resolve_event_classification_job_id(
        session,
        event_segmentation_job_id=cej.event_segmentation_job_id,
        requested_id=event_classification_job_id,
    )

    job = MaskedTransformerJob(
        continuous_embedding_job_id=continuous_embedding_job_id,
        event_classification_job_id=classify_id,
        training_signature=signature,
        preset=preset,
        mask_fraction=float(mask_fraction),
        span_length_min=int(span_length_min),
        span_length_max=int(span_length_max),
        dropout=float(dropout),
        mask_weight_bias=bool(mask_weight_bias),
        cosine_loss_weight=float(cosine_loss_weight),
        max_epochs=int(max_epochs),
        early_stop_patience=int(early_stop_patience),
        val_split=float(val_split),
        seed=int(seed),
        k_values=serialize_k_values(k_list),
        status=JobStatus.queued.value,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job, True


async def list_masked_transformer_jobs(
    session: AsyncSession,
    *,
    status: Optional[str] = None,
    continuous_embedding_job_id: Optional[str] = None,
) -> list[MaskedTransformerJob]:
    stmt = select(MaskedTransformerJob).order_by(MaskedTransformerJob.created_at.desc())
    if status is not None:
        stmt = stmt.where(MaskedTransformerJob.status == status)
    if continuous_embedding_job_id is not None:
        stmt = stmt.where(
            MaskedTransformerJob.continuous_embedding_job_id
            == continuous_embedding_job_id
        )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_masked_transformer_job(
    session: AsyncSession, job_id: str
) -> Optional[MaskedTransformerJob]:
    return await session.get(MaskedTransformerJob, job_id)


async def cancel_masked_transformer_job(
    session: AsyncSession, job_id: str
) -> Optional[MaskedTransformerJob]:
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
        return job
    raise CancelTerminalJobError(
        f"masked_transformer_job {job_id} is in terminal state {job.status!r}"
    )


async def delete_masked_transformer_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(MaskedTransformerJob, job_id)
    if job is None:
        return False
    artifact_dir = masked_transformer_dir(settings.storage_root, job_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir, ignore_errors=True)
    await session.delete(job)
    await session.commit()
    return True


async def extend_k_sweep_job(
    session: AsyncSession, job_id: str, additional_k: list[int]
) -> MaskedTransformerJob:
    """Extend the k-sweep on a completed job.

    Appends k values not already present in ``k_values`` and requeues
    the job for a follow-up worker pass that runs tokenization +
    interpretation only (the trained transformer + Z are untouched).
    """
    job = await get_masked_transformer_job(session, job_id)
    if job is None:
        raise ValueError(f"masked_transformer_job not found: {job_id}")
    if job.status != JobStatus.complete.value:
        raise ExtendKSweepError(
            f"masked_transformer_job {job_id} must be completed to extend k-sweep "
            f"(current status: {job.status!r})"
        )

    additional = parse_k_values(additional_k)
    current = parse_k_values(job.k_values)
    merged: list[int] = list(current)
    seen = set(current)
    for k in additional:
        if k not in seen:
            merged.append(k)
            seen.add(k)

    if merged == current:
        # No new k values — return the job unchanged.
        return job

    job.k_values = serialize_k_values(merged)
    job.status = JobStatus.queued.value
    job.status_reason = "extend_k_sweep"
    await session.commit()
    await session.refresh(job)
    return job


# ---------------------------------------------------------------------------
# Per-k interpretation artifact generation
# ---------------------------------------------------------------------------


def _read_contextual_embeddings_grouped(
    storage_root: Path, job_id: str
) -> tuple[
    list[str],
    list[np.ndarray],
    list[np.ndarray],
    list[list[float]],
    list[list[float]],
    list[list[Optional[int]]],
    list[list[str]],
]:
    """Group contextual_embeddings.parquet by region_id (ordered).

    Returns ordered region_ids and per-region arrays/lists for embeddings,
    chunk indexes, starts, ends, audio file ids, and tiers.
    """
    path = masked_transformer_contextual_embeddings_path(storage_root, job_id)
    if not path.exists():
        raise FileNotFoundError(f"contextual_embeddings.parquet not found: {path}")
    table = pq.read_table(path)
    region_col = table.column("region_id").to_pylist()
    starts_col = table.column("start_timestamp").to_pylist()

    min_start: dict[str, float] = {}
    for rid, st in zip(region_col, starts_col):
        if rid not in min_start or st < min_start[rid]:
            min_start[rid] = float(st)
    ordered = sorted(min_start.keys(), key=lambda r: (min_start[r], r))

    seqs: list[np.ndarray] = []
    chunk_idxs: list[np.ndarray] = []
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    audio_ids: list[list[Optional[int]]] = []
    tiers: list[list[str]] = []
    for rid in ordered:
        indices = [i for i, v in enumerate(region_col) if v == rid]
        sub = table.take(indices).sort_by("chunk_index_in_region")
        seqs.append(np.asarray(sub.column("embedding").to_pylist(), dtype=np.float32))
        chunk_idxs.append(
            np.asarray(sub.column("chunk_index_in_region").to_pylist(), dtype=np.int32)
        )
        starts.append([float(v) for v in sub.column("start_timestamp").to_pylist()])
        ends.append([float(v) for v in sub.column("end_timestamp").to_pylist()])
        audio_ids.append(list(sub.column("audio_file_id").to_pylist()))
        tier_col = sub.column("tier").to_pylist() if "tier" in sub.column_names else []
        tiers.append([str(t) if t is not None else "" for t in tier_col])
    return ordered, seqs, chunk_idxs, starts, ends, audio_ids, tiers


def _read_decoded_per_k(
    storage_root: Path, job_id: str, k: int
) -> dict[tuple[str, int], dict[str, Any]]:
    """Read k<N>/decoded.parquet keyed by (region_id, chunk_index_in_region)."""
    decoded_path = masked_transformer_k_decoded_path(storage_root, job_id, k)
    if not decoded_path.exists():
        raise FileNotFoundError(f"decoded.parquet not found for k={k}: {decoded_path}")
    table = pq.read_table(decoded_path)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    cols = table.to_pydict()
    n = table.num_rows
    for i in range(n):
        rid = str(cols["region_id"][i])
        cidx = int(cols["chunk_index_in_region"][i])
        out[(rid, cidx)] = {
            "label": int(cols["label"][i]),
            "confidence": float(cols["confidence"][i]),
            "tier": str(cols["tier"][i]) if cols["tier"][i] is not None else "",
        }
    return out


def _atomic_write_parquet(table: Any, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    atomic_rename(tmp, dst)


def _atomic_write_json(payload: dict[str, Any], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    atomic_rename(tmp, dst)


async def generate_interpretations(
    session: AsyncSession,
    storage_root: Path,
    job: MaskedTransformerJob,
    k: int,
    *,
    events_cache: list[EffectiveEventLabels] | None = None,
) -> dict[str, Any]:
    """Generate per-k overlay, exemplars, and label distribution artifacts.

    Fits PCA on the contextual embeddings ``Z`` (the masked-transformer
    workflow does not persist a PCA model), computes UMAP on the PCA
    projection, and uses the event-scoped helper for label distribution
    and exemplar annotation.

    Labels come from the bound ``event_classification_job_id`` via
    ``load_effective_event_labels``. The submit endpoint guarantees the
    FK is non-NULL after job creation succeeds.

    ``events_cache`` lets callers iterating over k_values load effective
    events once and reuse the result; the per-row annotations recompute
    cheaply. Returns the persisted ``label_distribution`` payload.
    """
    if not job.event_classification_job_id:
        raise ValueError(
            f"MaskedTransformerJob {job.id} has no event_classification_job_id; "
            "label distribution cannot be generated. Ensure the submit "
            "endpoint resolves a Classify job before the worker runs."
        )

    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            "source continuous_embedding_job not found: "
            f"{job.continuous_embedding_job_id}"
        )

    (
        region_ids,
        z_by_region,
        chunk_idxs_by_region,
        starts,
        ends,
        audio_ids,
        tiers,
    ) = _read_contextual_embeddings_grouped(storage_root, job.id)

    if not z_by_region:
        raise ValueError("no contextual embeddings found for masked-transformer job")

    decoded_lookup = _read_decoded_per_k(storage_root, job.id, k)

    # Build per-region tokens (treat tokens as the "Viterbi state" labels).
    viterbi_states: list[np.ndarray] = []
    max_probs: list[np.ndarray] = []
    sequence_ids: list[str] = []
    positions: list[int] = []
    starts_flat: list[float] = []
    ends_flat: list[float] = []
    window_metas: list[WindowMeta] = []
    for region_id, z_seq, cidx_seq, start_seq, end_seq, audio_seq, tier_seq in zip(
        region_ids, z_by_region, chunk_idxs_by_region, starts, ends, audio_ids, tiers
    ):
        v = np.zeros(z_seq.shape[0], dtype=np.int16)
        p = np.zeros(z_seq.shape[0], dtype=np.float32)
        for i in range(z_seq.shape[0]):
            cidx = int(cidx_seq[i])
            entry = decoded_lookup.get((region_id, cidx))
            if entry is None:
                continue
            v[i] = entry["label"]
            p[i] = entry["confidence"]
            sequence_ids.append(region_id)
            positions.append(cidx)
            starts_flat.append(float(start_seq[i]))
            ends_flat.append(float(end_seq[i]))
            window_metas.append(
                WindowMeta(
                    sequence_id=region_id,
                    position_in_sequence=cidx,
                    audio_file_id=audio_seq[i] if i < len(audio_seq) else None,
                    start_timestamp=float(start_seq[i]),
                    end_timestamp=float(end_seq[i]),
                    max_state_probability=float(p[i]),
                    extras={"tier": tier_seq[i] if i < len(tier_seq) else ""},
                )
            )
        viterbi_states.append(v)
        max_probs.append(p)

    # Fit PCA on stacked Z. Use 2 components if dim >= 2.
    z_stack = np.concatenate(z_by_region, axis=0)
    n_components = min(2, z_stack.shape[1])
    pca_model = PCA(n_components=n_components, random_state=int(job.seed))
    pca_model.fit(z_stack.astype(np.float32))

    metadata = OverlayMetadata(
        sequence_ids=sequence_ids,
        positions_in_sequence=positions,
        start_timestamps=starts_flat,
        end_timestamps=ends_flat,
    )
    overlay_table, pca_full = compute_overlay(
        pca_model,
        z_by_region,
        viterbi_states,
        max_probs,
        metadata,
        l2_normalize=False,
        random_state=int(job.seed),
    )
    _atomic_write_parquet(
        overlay_table, masked_transformer_k_overlay_path(storage_root, job.id, k)
    )

    all_states_flat = np.concatenate(viterbi_states).astype(np.int16)
    n_labels = int(np.max(all_states_flat)) + 1 if all_states_flat.size else int(k)

    # Per-window rows for the event-scoped join (parallel to window_metas
    # and all_states_flat).
    window_rows: list[dict[str, Any]] = [
        {
            "start_timestamp": float(m.start_timestamp),
            "end_timestamp": float(m.end_timestamp),
            "viterbi_state": int(s),
        }
        for m, s in zip(window_metas, all_states_flat)
    ]

    if events_cache is None:
        events = await load_effective_event_labels(
            session,
            event_classification_job_id=job.event_classification_job_id,
            storage_root=storage_root,
        )
    else:
        events = events_cache
    annotations = assign_labels_to_windows(window_rows, events)

    # Annotate exemplars after selection.
    exemplars = select_exemplars(pca_full, all_states_flat, window_metas, n_labels)
    annotation_by_key: dict[tuple[str, int], WindowAnnotation] = {
        (m.sequence_id, m.position_in_sequence): a
        for m, a in zip(window_metas, annotations)
    }
    for state_records in exemplars["states"].values():
        for record in state_records:
            key = (record["sequence_id"], int(record["position_in_sequence"]))
            ann = annotation_by_key.get(key)
            if ann is not None:
                record["extras"]["event_id"] = ann.event_id
                record["extras"]["event_types"] = list(ann.event_types)
                record["extras"]["event_confidence"] = dict(ann.event_confidence)
    _atomic_write_json(
        exemplars, masked_transformer_k_exemplars_path(storage_root, job.id, k)
    )

    dist = compute_label_distribution(window_rows, annotations, n_labels)
    _atomic_write_json(
        dist, masked_transformer_k_label_distribution_path(storage_root, job.id, k)
    )

    return dist


async def generate_interpretations_all_k(
    session: AsyncSession,
    storage_root: Path,
    job: MaskedTransformerJob,
    k_values: list[int],
) -> dict[int, dict[str, Any]]:
    """Run ``generate_interpretations`` for every ``k`` in one call.

    Loads effective events once and threads the cache through each call,
    so the DB + parquet read for the bound Classify job happens exactly
    once regardless of how many k values are present.

    Returns a ``{k: label_distribution_payload}`` dict so callers (e.g.,
    the regenerate endpoint) can return the active k's payload directly.
    """
    if not job.event_classification_job_id:
        raise ValueError(
            f"MaskedTransformerJob {job.id} has no event_classification_job_id; "
            "label distribution cannot be generated."
        )

    events = await load_effective_event_labels(
        session,
        event_classification_job_id=job.event_classification_job_id,
        storage_root=storage_root,
    )

    out: dict[int, dict[str, Any]] = {}
    for k in k_values:
        dist = await generate_interpretations(
            session, storage_root, job, k, events_cache=events
        )
        out[k] = dist
    return out


async def regenerate_label_distribution(
    session: AsyncSession,
    storage_root: Path,
    job: MaskedTransformerJob,
    *,
    requested_classify_id: Optional[str] = None,
) -> dict[int, dict[str, Any]]:
    """Regenerate every per-k label-distribution artifact; optionally re-bind.

    Mirrors :func:`humpback.services.hmm_sequence_service.regenerate_label_distribution`:
    validate → write artifacts → commit FK update. The MT regenerate
    rebuilds **all** ``k<N>/label_distribution.json`` files in one call
    so the per-k caches stay coherent. Effective events are loaded once
    by ``generate_interpretations_all_k``.
    """
    cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
    if cej is None:
        raise ValueError(
            "source continuous_embedding_job not found: "
            f"{job.continuous_embedding_job_id}"
        )

    previous_classify_id = job.event_classification_job_id

    if requested_classify_id is not None:
        if not cej.event_segmentation_job_id:
            raise ValueError(
                "continuous_embedding_job has no event_segmentation_job_id; "
                "cannot validate Classify re-bind"
            )
        from humpback.services.hmm_sequence_service import (
            resolve_event_classification_job_id,
        )

        await resolve_event_classification_job_id(
            session,
            event_segmentation_job_id=cej.event_segmentation_job_id,
            requested_id=requested_classify_id,
        )
        job.event_classification_job_id = requested_classify_id

    try:
        out = await generate_interpretations_all_k(
            session, storage_root, job, parse_k_values(job.k_values)
        )
    except Exception:
        job.event_classification_job_id = previous_classify_id
        raise

    if (
        requested_classify_id is not None
        and previous_classify_id != requested_classify_id
    ):
        await session.commit()

    return out


__all__ = [
    "CancelTerminalJobError",
    "ExtendKSweepError",
    "cancel_masked_transformer_job",
    "compute_training_signature",
    "create_masked_transformer_job",
    "delete_masked_transformer_job",
    "extend_k_sweep_job",
    "generate_interpretations",
    "generate_interpretations_all_k",
    "get_masked_transformer_job",
    "list_masked_transformer_jobs",
    "parse_k_values",
    "regenerate_label_distribution",
    "serialize_k_values",
]
