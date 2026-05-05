"""Worker for masked-transformer sequence jobs (ADR-061).

The first pass over a queued job:

1. validates the upstream CRNN region-based ``ContinuousEmbeddingJob``,
2. validates the chosen accelerator with a forward+backward sanity check
   on a small synthetic batch (falls back to CPU on divergence),
3. trains the masked-span transformer and persists ``transformer.pt``,
   ``loss_curve.json`` and ``reconstruction_error.parquet``,
4. extracts contextual embeddings (Z) and persists
   ``contextual_embeddings.parquet``,
5. for each ``k`` in ``k_values``: fits k-means + decodes tokens with
   softmax-temperature confidences, stages the per-k bundle to
   ``k<N>.tmp/`` and atomically renames it to ``k<N>/``.

If the worker re-claims a completed job whose ``k_values`` was extended
via the service layer's :func:`extend_k_sweep_job`, only the new ``k``
values are processed; the trained transformer and ``Z`` are left
untouched.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.segmentation.extraction import load_effective_events
from humpback.config import Settings
from humpback.ml.device import select_device
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (
    ContinuousEmbeddingJob,
    MaskedTransformerJob,
)
from humpback.sequence_models.masked_transformer import (
    MaskedTransformer,
    MaskedTransformerConfig,
    compute_reconstruction_error,
    extract_transformer_embeddings,
    train_masked_transformer,
)
from humpback.sequence_models.contrastive_labels import load_contrastive_event_labels
from humpback.sequence_models.contrastive_loss import (
    ContrastiveEventMetadata,
    build_contrastive_masks,
    parse_related_label_policy,
)
from humpback.sequence_models.masked_transformer_sequences import (
    EffectiveEventInterval,
    build_masked_transformer_training_sequences,
)
from humpback.sequence_models.tokenization import (
    compute_run_lengths,
    decode_tokens,
    fit_kmeans_token_model,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    source_kind_for,
)
from humpback.services.masked_transformer_service import (
    generate_interpretations,
    parse_k_values,
)
from humpback.storage import (
    atomic_rename,
    continuous_embedding_parquet_path,
    ensure_dir,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_dir,
    masked_transformer_k_decoded_path,
    masked_transformer_k_dir,
    masked_transformer_k_exemplars_path,
    masked_transformer_k_kmeans_path,
    masked_transformer_k_label_distribution_path,
    masked_transformer_k_overlay_path,
    masked_transformer_k_run_lengths_path,
    masked_transformer_k_tmp_dir,
    masked_transformer_loss_curve_path,
    masked_transformer_model_path,
    masked_transformer_reconstruction_error_path,
    masked_transformer_retrieval_embeddings_path,
)
from humpback.workers.queue import claim_masked_transformer_job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region grouping (parquet → list of arrays + tier lists)
# ---------------------------------------------------------------------------


def _group_region_sequences(
    table: pa.Table,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[list[str]],
    list[list[float]],
    list[list[float]],
    list[list[Optional[int]]],
]:
    """Group a CRNN embeddings parquet by ``region_id``.

    Returns ``(region_ids, sequences, tier_lists, start_ts, end_ts, audio_file_ids)``
    aligned 1:1; rows within a region are sorted by ``chunk_index_in_region``.
    """
    region_col = table.column("region_id").to_pylist()
    unique_regions = sorted(set(region_col))
    seqs: list[np.ndarray] = []
    tiers: list[list[str]] = []
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    audio_ids: list[list[Optional[int]]] = []
    for region_id in unique_regions:
        indices = [i for i, v in enumerate(region_col) if v == region_id]
        sub = table.take(indices).sort_by("chunk_index_in_region")
        embeddings = np.asarray(sub.column("embedding").to_pylist(), dtype=np.float32)
        seqs.append(embeddings)
        tier_col = sub.column("tier").to_pylist() if "tier" in sub.column_names else []
        tiers.append([str(t) if t is not None else "" for t in tier_col])
        starts.append([float(v) for v in sub.column("start_timestamp").to_pylist()])
        ends.append([float(v) for v in sub.column("end_timestamp").to_pylist()])
        audio_ids.append(list(sub.column("audio_file_id").to_pylist()))
    return unique_regions, seqs, tiers, starts, ends, audio_ids


# ---------------------------------------------------------------------------
# Device validation
# ---------------------------------------------------------------------------


def _make_synthetic_batch(feature_dim: int, T: int = 8, batch: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(0)
    return torch.from_numpy(
        rng.standard_normal((batch, T, feature_dim)).astype(np.float32)
    )


def validate_training_device(
    feature_dim: int,
    *,
    rtol: float = 0.05,
    seed: int = 0,
    force_fail: bool = False,
) -> tuple[torch.device, Optional[str]]:
    """Run forward+backward on CPU and the chosen accelerator and compare.

    Returns ``(device, fallback_reason)``. ``force_fail`` is a test hook
    that simulates accelerator failure without altering the environment.
    """
    target_device = select_device()
    if target_device.type == "cpu" and not force_fail:
        return target_device, None
    if force_fail and target_device.type == "cpu":
        # No accelerator anyway; honor the force-fail signal for tests.
        return target_device, None

    backend = target_device.type
    sample = _make_synthetic_batch(feature_dim)

    def _forward_loss(device: torch.device) -> float:
        torch.manual_seed(seed)
        model = MaskedTransformer(
            input_dim=feature_dim,
            d_model=32,
            num_layers=1,
            num_heads=4,
            ff_dim=64,
            dropout=0.0,
        ).to(device)
        x = sample.to(device)
        rec, _ = model(x)
        loss = ((rec - x) ** 2).mean()
        loss.backward()
        return float(loss.detach().to("cpu").item())

    try:
        cpu_loss = _forward_loss(torch.device("cpu"))
    except Exception:
        logger.warning(
            "CPU validation forward failed; staying on CPU",
            exc_info=True,
        )
        return torch.device("cpu"), f"{backend}_load_error"

    if force_fail:
        return torch.device("cpu"), f"{backend}_output_mismatch"

    try:
        acc_loss = _forward_loss(target_device)
    except Exception:
        logger.warning(
            "Target-device forward failed on %s; falling back to CPU",
            backend,
            exc_info=True,
        )
        return torch.device("cpu"), f"{backend}_load_error"

    denom = max(abs(cpu_loss), 1e-6)
    if abs(cpu_loss - acc_loss) / denom > rtol:
        logger.warning(
            "Loss mismatch between CPU (%.6f) and %s (%.6f); falling back to CPU",
            cpu_loss,
            backend,
            acc_loss,
        )
        return torch.device("cpu"), f"{backend}_output_mismatch"

    return target_device, None


# ---------------------------------------------------------------------------
# Per-k atomic writes
# ---------------------------------------------------------------------------


_DECODED_SCHEMA = pa.schema(
    [
        pa.field("sequence_id", pa.string()),
        pa.field("position", pa.int32()),
        pa.field("label", pa.int16()),
        pa.field("confidence", pa.float32()),
        pa.field("audio_file_id", pa.int32(), nullable=True),
        pa.field("start_timestamp", pa.float64()),
        pa.field("end_timestamp", pa.float64()),
        pa.field("tier", pa.string()),
        pa.field("chunk_index_in_region", pa.int32()),
        pa.field("region_id", pa.string()),
    ]
)


def _write_per_k_bundle(
    *,
    storage_root: Path,
    job_id: str,
    k: int,
    Z_by_seq: list[np.ndarray],
    region_ids: list[str],
    tier_lists: list[list[str]],
    starts: list[list[float]],
    ends: list[list[float]],
    audio_ids: list[list[Optional[int]]],
    seed: int,
) -> None:
    """Fit k-means on stacked Z, decode tokens, and persist k<N>/ atomically."""
    tmp_dir = masked_transformer_k_tmp_dir(storage_root, job_id, k)
    final_dir = masked_transformer_k_dir(storage_root, job_id, k)

    # Clean any stale tmp from a prior crash so the rename can proceed.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    Z_stack = np.concatenate(Z_by_seq, axis=0).astype(np.float64)
    kmeans, tau = fit_kmeans_token_model(Z_stack, k=k, seed=seed)

    # Token decoding per sequence keeps ordering aligned with parquet rows.
    token_sequences: list[np.ndarray] = []
    rows: list[dict] = []
    for region_id, Z_seq, tier_seq, start_seq, end_seq, audio_seq in zip(
        region_ids, Z_by_seq, tier_lists, starts, ends, audio_ids
    ):
        labels, confidences = decode_tokens(Z_seq, kmeans, tau)
        token_sequences.append(labels)
        for pos in range(labels.shape[0]):
            rows.append(
                {
                    "sequence_id": region_id,
                    "position": int(pos),
                    "label": int(labels[pos]),
                    "confidence": float(confidences[pos]),
                    "audio_file_id": audio_seq[pos] if pos < len(audio_seq) else None,
                    "start_timestamp": float(start_seq[pos])
                    if pos < len(start_seq)
                    else 0.0,
                    "end_timestamp": float(end_seq[pos]) if pos < len(end_seq) else 0.0,
                    "tier": tier_seq[pos] if pos < len(tier_seq) else "",
                    "chunk_index_in_region": int(pos),
                    "region_id": region_id,
                }
            )

    decoded_table = pa.Table.from_pylist(rows, schema=_DECODED_SCHEMA)
    pq.write_table(decoded_table, tmp_dir / "decoded.parquet")
    joblib.dump({"kmeans": kmeans, "tau": float(tau)}, tmp_dir / "kmeans.joblib")

    run_lengths = compute_run_lengths(token_sequences, k=k)
    (tmp_dir / "run_lengths.json").write_text(
        json.dumps(
            {
                "k": int(k),
                "tau": float(tau),
                "run_lengths": run_lengths,
            },
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Atomic finalize: rename the staged dir into place.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)


def _write_contextual_embeddings(
    *,
    storage_root: Path,
    job_id: str,
    region_ids: list[str],
    Z_by_seq: list[np.ndarray],
    starts: list[list[float]],
    ends: list[list[float]],
    audio_ids: list[list[Optional[int]]],
    tier_lists: list[list[str]],
) -> None:
    _write_embedding_artifact(
        dst=masked_transformer_contextual_embeddings_path(storage_root, job_id),
        region_ids=region_ids,
        embeddings_by_seq=Z_by_seq,
        starts=starts,
        ends=ends,
        audio_ids=audio_ids,
        tier_lists=tier_lists,
    )


def _write_retrieval_embeddings(
    *,
    storage_root: Path,
    job_id: str,
    region_ids: list[str],
    R_by_seq: list[np.ndarray],
    starts: list[list[float]],
    ends: list[list[float]],
    audio_ids: list[list[Optional[int]]],
    tier_lists: list[list[str]],
) -> None:
    _write_embedding_artifact(
        dst=masked_transformer_retrieval_embeddings_path(storage_root, job_id),
        region_ids=region_ids,
        embeddings_by_seq=R_by_seq,
        starts=starts,
        ends=ends,
        audio_ids=audio_ids,
        tier_lists=tier_lists,
    )


def _write_embedding_artifact(
    *,
    dst: Path,
    region_ids: list[str],
    embeddings_by_seq: list[np.ndarray],
    starts: list[list[float]],
    ends: list[list[float]],
    audio_ids: list[list[Optional[int]]],
    tier_lists: list[list[str]],
) -> None:
    rows: list[dict] = []
    for region_id, Z_seq, start_seq, end_seq, audio_seq, tier_seq in zip(
        region_ids, embeddings_by_seq, starts, ends, audio_ids, tier_lists
    ):
        for pos in range(Z_seq.shape[0]):
            rows.append(
                {
                    "region_id": region_id,
                    "chunk_index_in_region": int(pos),
                    "audio_file_id": audio_seq[pos] if pos < len(audio_seq) else None,
                    "start_timestamp": float(start_seq[pos])
                    if pos < len(start_seq)
                    else 0.0,
                    "end_timestamp": float(end_seq[pos]) if pos < len(end_seq) else 0.0,
                    "tier": tier_seq[pos] if pos < len(tier_seq) else "",
                    "embedding": Z_seq[pos].astype(np.float32).tolist(),
                }
            )
    table = pa.Table.from_pylist(rows)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    atomic_rename(tmp, dst)


def _write_reconstruction_error(
    *,
    storage_root: Path,
    job_id: str,
    region_ids: list[str],
    rec_per_chunk: list[np.ndarray],
    starts: list[list[float]],
    ends: list[list[float]],
) -> None:
    rows: list[dict] = []
    for region_id, scores, start_seq, end_seq in zip(
        region_ids, rec_per_chunk, starts, ends
    ):
        for pos in range(len(scores)):
            rows.append(
                {
                    "sequence_id": region_id,
                    "position": int(pos),
                    "score": float(scores[pos]),
                    "start_timestamp": float(start_seq[pos])
                    if pos < len(start_seq)
                    else 0.0,
                    "end_timestamp": float(end_seq[pos]) if pos < len(end_seq) else 0.0,
                }
            )
    table = pa.Table.from_pylist(rows)
    dst = masked_transformer_reconstruction_error_path(storage_root, job_id)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    atomic_rename(tmp, dst)


def _write_loss_curve(
    *, storage_root: Path, job_id: str, loss_curve: dict, val_metrics: dict
) -> None:
    train_total = list(loss_curve.get("train_total", loss_curve.get("train", [])))
    val_total = list(loss_curve.get("val_total", loss_curve.get("val", [])))
    payload = {
        "epochs": list(range(1, len(train_total) + 1)),
        "train_loss": train_total,
        "val_loss": val_total,
        "train_masked_loss": list(loss_curve.get("train_masked", train_total)),
        "train_contrastive_loss": list(loss_curve.get("train_contrastive", [])),
        "train_total_loss": train_total,
        "val_masked_loss": list(loss_curve.get("val_masked", val_total)),
        "val_contrastive_loss": list(loss_curve.get("val_contrastive", [])),
        "val_total_loss": val_total,
        "train_contrastive_skipped_batches": list(
            loss_curve.get("train_contrastive_skipped_batches", [])
        ),
        "val_contrastive_skipped_batches": list(
            loss_curve.get("val_contrastive_skipped_batches", [])
        ),
        "val_metrics": val_metrics,
    }
    dst = masked_transformer_loss_curve_path(storage_root, job_id)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    atomic_rename(tmp, dst)


def _save_model_state(
    *,
    storage_root: Path,
    job_id: str,
    model: MaskedTransformer,
    feature_dim: int,
    config: MaskedTransformerConfig,
) -> None:
    dst = masked_transformer_model_path(storage_root, job_id)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "input_dim": feature_dim,
            "d_model": model.d_model,
            "retrieval_head_enabled": model.retrieval_head_enabled,
            "retrieval_dim": model.retrieval_dim,
            "retrieval_hidden_dim": model.retrieval_hidden_dim,
            "retrieval_l2_normalize": model.retrieval_l2_normalize,
            "batch_size": config.batch_size,
            "sequence_construction_mode": config.sequence_construction_mode,
            "event_centered_fraction": config.event_centered_fraction,
            "pre_event_context_sec": config.pre_event_context_sec,
            "post_event_context_sec": config.post_event_context_sec,
            "contrastive_loss_weight": config.contrastive_loss_weight,
            "contrastive_temperature": config.contrastive_temperature,
            "contrastive_label_source": config.contrastive_label_source,
            "contrastive_min_events_per_label": config.contrastive_min_events_per_label,
            "contrastive_min_regions_per_label": config.contrastive_min_regions_per_label,
            "require_cross_region_positive": config.require_cross_region_positive,
            "related_label_policy_json": config.related_label_policy_json,
        },
    }
    torch.save(payload, tmp)
    atomic_rename(tmp, dst)


# ---------------------------------------------------------------------------
# Read helpers for follow-up extend-k-sweep passes
# ---------------------------------------------------------------------------


def _read_contextual_embeddings(
    storage_root: Path, job_id: str
) -> tuple[
    list[str],
    list[np.ndarray],
    list[list[float]],
    list[list[float]],
    list[list[Optional[int]]],
    list[list[str]],
]:
    return _read_embedding_artifact(
        masked_transformer_contextual_embeddings_path(storage_root, job_id)
    )


def _read_retrieval_embeddings(
    storage_root: Path, job_id: str
) -> tuple[
    list[str],
    list[np.ndarray],
    list[list[float]],
    list[list[float]],
    list[list[Optional[int]]],
    list[list[str]],
]:
    path = masked_transformer_retrieval_embeddings_path(storage_root, job_id)
    if not path.exists():
        raise FileNotFoundError(
            f"retrieval_embeddings.parquet not found for retrieval-head job {job_id}"
        )
    return _read_embedding_artifact(path)


def _read_embedding_artifact(
    path: Path,
) -> tuple[
    list[str],
    list[np.ndarray],
    list[list[float]],
    list[list[float]],
    list[list[Optional[int]]],
    list[list[str]],
]:
    table = pq.read_table(path)
    region_col = table.column("region_id").to_pylist()
    unique_regions = sorted(set(region_col))
    seqs: list[np.ndarray] = []
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    audio_ids: list[list[Optional[int]]] = []
    tiers: list[list[str]] = []
    for rid in unique_regions:
        indices = [i for i, v in enumerate(region_col) if v == rid]
        sub = table.take(indices).sort_by("chunk_index_in_region")
        embeddings = np.asarray(sub.column("embedding").to_pylist(), dtype=np.float32)
        seqs.append(embeddings)
        starts.append([float(v) for v in sub.column("start_timestamp").to_pylist()])
        ends.append([float(v) for v in sub.column("end_timestamp").to_pylist()])
        audio_ids.append(list(sub.column("audio_file_id").to_pylist()))
        tier_col = sub.column("tier").to_pylist() if "tier" in sub.column_names else []
        tiers.append([str(t) if t is not None else "" for t in tier_col])
    return unique_regions, seqs, starts, ends, audio_ids, tiers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _config_from_job(job: MaskedTransformerJob) -> MaskedTransformerConfig:
    return MaskedTransformerConfig(
        preset=job.preset,  # type: ignore[arg-type]
        mask_fraction=float(job.mask_fraction),
        span_length_min=int(job.span_length_min),
        span_length_max=int(job.span_length_max),
        dropout=float(job.dropout),
        mask_weight_bias=bool(job.mask_weight_bias),
        cosine_loss_weight=float(job.cosine_loss_weight),
        batch_size=int(job.batch_size),
        retrieval_head_enabled=bool(job.retrieval_head_enabled),
        retrieval_dim=job.retrieval_dim,
        retrieval_hidden_dim=job.retrieval_hidden_dim,
        retrieval_l2_normalize=bool(job.retrieval_l2_normalize),
        sequence_construction_mode=job.sequence_construction_mode,  # type: ignore[arg-type]
        event_centered_fraction=float(job.event_centered_fraction),
        pre_event_context_sec=job.pre_event_context_sec,
        post_event_context_sec=job.post_event_context_sec,
        contrastive_loss_weight=float(job.contrastive_loss_weight),
        contrastive_temperature=float(job.contrastive_temperature),
        contrastive_label_source=job.contrastive_label_source,  # type: ignore[arg-type]
        contrastive_min_events_per_label=int(job.contrastive_min_events_per_label),
        contrastive_min_regions_per_label=int(job.contrastive_min_regions_per_label),
        require_cross_region_positive=bool(job.require_cross_region_positive),
        related_label_policy_json=job.related_label_policy_json,
        max_epochs=int(job.max_epochs),
        early_stop_patience=int(job.early_stop_patience),
        val_split=float(job.val_split),
        seed=int(job.seed),
    )


async def _load_effective_event_intervals(
    session: AsyncSession,
    *,
    storage_root: Path,
    cej: ContinuousEmbeddingJob,
) -> list[EffectiveEventInterval]:
    if not cej.event_segmentation_job_id:
        raise ValueError(
            "event-centered masked-transformer training requires "
            "event_segmentation_job_id on the upstream continuous embedding job"
        )
    if not cej.region_detection_job_id:
        raise ValueError(
            "event-centered masked-transformer training requires "
            "region_detection_job_id on the upstream continuous embedding job"
        )
    region_job = await session.get(RegionDetectionJob, cej.region_detection_job_id)
    if region_job is None:
        raise ValueError(f"RegionDetectionJob {cej.region_detection_job_id} not found")
    offset = float(region_job.start_timestamp or 0.0)
    events = await load_effective_events(
        session,
        event_segmentation_job_id=cej.event_segmentation_job_id,
        storage_root=storage_root,
    )
    return [
        EffectiveEventInterval(
            region_id=event.region_id,
            start_timestamp=float(event.start_sec) + offset,
            end_timestamp=float(event.end_sec) + offset,
        )
        for event in events
    ]


async def _load_contrastive_event_intervals(
    session: AsyncSession,
    *,
    storage_root: Path,
    cej: ContinuousEmbeddingJob,
) -> list[EffectiveEventInterval]:
    if not cej.event_segmentation_job_id:
        raise ValueError(
            "contrastive masked-transformer training requires "
            "event_segmentation_job_id on the upstream continuous embedding job"
        )
    if not cej.region_detection_job_id:
        raise ValueError(
            "contrastive masked-transformer training requires "
            "region_detection_job_id on the upstream continuous embedding job"
        )
    region_job = await session.get(RegionDetectionJob, cej.region_detection_job_id)
    if region_job is None:
        raise ValueError(f"RegionDetectionJob {cej.region_detection_job_id} not found")
    labels, _ = await load_contrastive_event_labels(
        session,
        storage_root=storage_root,
        event_segmentation_job_id=cej.event_segmentation_job_id,
        region_detection_job_id=cej.region_detection_job_id,
        region_start_timestamp=region_job.start_timestamp,
    )
    return [
        EffectiveEventInterval(
            region_id=event.region_id,
            start_timestamp=event.start_timestamp,
            end_timestamp=event.end_timestamp,
            event_id=event.event_id,
            human_types=event.human_types,
        )
        for event in labels
    ]


def _contrastive_metadata_from_candidates(
    candidates: list,
) -> list[Optional[ContrastiveEventMetadata]]:
    out: list[Optional[ContrastiveEventMetadata]] = []
    for candidate in candidates:
        if (
            candidate.event_id is None
            or candidate.event_start_index is None
            or candidate.event_end_index is None
        ):
            out.append(None)
            continue
        out.append(
            ContrastiveEventMetadata(
                event_id=candidate.event_id,
                region_id=candidate.region_id,
                human_types=candidate.human_types,
                start_index=int(candidate.event_start_index),
                end_index=int(candidate.event_end_index),
            )
        )
    return out


def _previously_done_k(storage_root: Path, job_id: str, k_list: list[int]) -> set[int]:
    done: set[int] = set()
    for k in k_list:
        required = (
            masked_transformer_k_decoded_path(storage_root, job_id, k),
            masked_transformer_k_kmeans_path(storage_root, job_id, k),
            masked_transformer_k_run_lengths_path(storage_root, job_id, k),
            masked_transformer_k_overlay_path(storage_root, job_id, k),
            masked_transformer_k_exemplars_path(storage_root, job_id, k),
            masked_transformer_k_label_distribution_path(storage_root, job_id, k),
        )
        if all(path.exists() for path in required):
            done.add(k)
    return done


async def run_masked_transformer_job(
    session: AsyncSession,
    job: MaskedTransformerJob,
    settings: Settings,
    *,
    device_validation_force_fail: bool = False,
) -> None:
    """Execute one masked-transformer job end-to-end.

    Splits into the "first pass" (train + Z + per-k) and the
    "extend-k-sweep follow-up" (per-k only) based on whether
    ``transformer.pt`` and ``contextual_embeddings.parquet`` already
    exist on disk for the job.
    """
    job_id = job.id
    job_dir = ensure_dir(masked_transformer_dir(settings.storage_root, job_id))
    try:
        job = await session.merge(job)

        # Validate upstream.
        cej = await session.get(ContinuousEmbeddingJob, job.continuous_embedding_job_id)
        if cej is None or cej.status != JobStatus.complete.value:
            raise ValueError(
                f"upstream continuous_embedding_job {job.continuous_embedding_job_id}"
                " is not complete"
            )
        if source_kind_for(cej.model_version) != SOURCE_KIND_REGION_CRNN:
            raise ValueError(
                "masked-transformer requires a CRNN region-based upstream "
                f"(got {cej.model_version!r})"
            )

        embeddings_path = continuous_embedding_parquet_path(
            settings.storage_root, cej.id
        )
        if not embeddings_path.exists():
            raise FileNotFoundError(f"embeddings.parquet not found for cej {cej.id}")

        # Mark running.
        job.status = JobStatus.running.value
        job.error_message = None
        await session.commit()
        await session.refresh(job)
        if job.status == JobStatus.canceled.value:
            return

        k_list = parse_k_values(job.k_values)
        config = _config_from_job(job)

        transformer_path = masked_transformer_model_path(settings.storage_root, job_id)
        z_path = masked_transformer_contextual_embeddings_path(
            settings.storage_root, job_id
        )
        first_pass = not (transformer_path.exists() and z_path.exists())

        if first_pass:
            embeddings_table = pq.read_table(embeddings_path)
            (
                region_ids,
                sequences,
                tier_lists,
                starts,
                ends,
                audio_ids,
            ) = _group_region_sequences(embeddings_table)
            if not sequences:
                raise ValueError("no embedding sequences found in upstream parquet")

            feature_dim = int(sequences[0].shape[1])
            chosen_device, fallback_reason = validate_training_device(
                feature_dim, force_fail=device_validation_force_fail
            )
            job.chosen_device = chosen_device.type
            job.fallback_reason = fallback_reason
            await session.commit()
            await session.refresh(job)

            event_intervals: list[EffectiveEventInterval] = []
            contrastive_enabled = (
                config.contrastive_loss_weight > 0.0
                and config.contrastive_label_source == "human_corrections"
            )
            if contrastive_enabled:
                event_intervals = await _load_contrastive_event_intervals(
                    session,
                    storage_root=settings.storage_root,
                    cej=cej,
                )
            elif config.sequence_construction_mode != "region":
                event_intervals = await _load_effective_event_intervals(
                    session,
                    storage_root=settings.storage_root,
                    cej=cej,
                )
            constructed = build_masked_transformer_training_sequences(
                region_ids=region_ids,
                sequences=sequences,
                tier_lists=tier_lists,
                starts=starts,
                ends=ends,
                effective_events=event_intervals,
                mode=config.sequence_construction_mode,
                event_centered_fraction=config.event_centered_fraction,
                pre_event_context_sec=config.pre_event_context_sec,
                post_event_context_sec=config.post_event_context_sec,
                seed=int(job.seed),
            )
            if not constructed.sequences:
                raise ValueError(
                    "sequence construction produced no trainable masked-transformer "
                    f"windows for mode {config.sequence_construction_mode!r}"
                )

            tier_payload: list[Optional[list[str]]] = [
                t for t in constructed.tier_lists
            ]
            contrastive_events: list[Optional[ContrastiveEventMetadata]] | None = None
            if contrastive_enabled:
                contrastive_events = _contrastive_metadata_from_candidates(
                    constructed.candidates
                )
                labeled_events = [
                    event
                    for event in contrastive_events
                    if event is not None and event.human_types
                ]
                masks = build_contrastive_masks(
                    labeled_events,
                    min_events_per_label=config.contrastive_min_events_per_label,
                    min_regions_per_label=config.contrastive_min_regions_per_label,
                    require_cross_region_positive=config.require_cross_region_positive,
                    related_label_pairs=parse_related_label_policy(
                        config.related_label_policy_json
                    ),
                )
                if masks.valid_anchor_count == 0:
                    raise ValueError(
                        "contrastive masked-transformer training found no eligible "
                        "human-correction positive pairs"
                    )
            train_result = train_masked_transformer(
                constructed.sequences,
                config,
                device=chosen_device,
                tier_lists=tier_payload,
                contrastive_events=contrastive_events,
            )
            full_region_tier_payload: list[Optional[list[str]]] = [
                t for t in tier_lists
            ]
            full_region_reconstruction_error = compute_reconstruction_error(
                train_result.model,
                sequences,
                config,
                device=chosen_device,
                tier_lists=full_region_tier_payload,
            )

            _save_model_state(
                storage_root=settings.storage_root,
                job_id=job_id,
                model=train_result.model,
                feature_dim=feature_dim,
                config=config,
            )
            _write_loss_curve(
                storage_root=settings.storage_root,
                job_id=job_id,
                loss_curve=train_result.loss_curve,
                val_metrics=train_result.val_metrics,
            )
            _write_reconstruction_error(
                storage_root=settings.storage_root,
                job_id=job_id,
                region_ids=region_ids,
                rec_per_chunk=full_region_reconstruction_error,
                starts=starts,
                ends=ends,
            )

            Z, R, _ = extract_transformer_embeddings(
                train_result.model, sequences, device=chosen_device
            )
            _write_contextual_embeddings(
                storage_root=settings.storage_root,
                job_id=job_id,
                region_ids=region_ids,
                Z_by_seq=Z,
                starts=starts,
                ends=ends,
                audio_ids=audio_ids,
                tier_lists=tier_lists,
            )
            if config.retrieval_head_enabled:
                if R is None:
                    raise RuntimeError("retrieval head enabled but no retrieval output")
                _write_retrieval_embeddings(
                    storage_root=settings.storage_root,
                    job_id=job_id,
                    region_ids=region_ids,
                    R_by_seq=R,
                    starts=starts,
                    ends=ends,
                    audio_ids=audio_ids,
                    tier_lists=tier_lists,
                )
                token_embeddings = R
            else:
                token_embeddings = Z

            job.final_train_loss = train_result.val_metrics.get("final_train_loss")
            final_val = train_result.val_metrics.get("final_val_loss")
            if isinstance(final_val, float) and not np.isnan(final_val):
                job.final_val_loss = float(final_val)
            job.total_epochs = int(train_result.stopped_epoch)
            job.total_sequences = len(region_ids)
            job.total_chunks = int(sum(s.shape[0] for s in sequences))
            await session.commit()
            await session.refresh(job)
        else:
            # Extend-k-sweep follow-up: read Z back from disk; transformer
            # state is left untouched.
            (
                region_ids,
                token_embeddings,
                starts,
                ends,
                audio_ids,
                tier_lists,
            ) = (
                _read_retrieval_embeddings(settings.storage_root, job_id)
                if config.retrieval_head_enabled
                else _read_contextual_embeddings(settings.storage_root, job_id)
            )

        previously_done = _previously_done_k(settings.storage_root, job_id, k_list)

        for k in k_list:
            await session.refresh(job)
            if job.status == JobStatus.canceled.value:
                return
            if k in previously_done:
                continue
            _write_per_k_bundle(
                storage_root=settings.storage_root,
                job_id=job_id,
                k=k,
                Z_by_seq=token_embeddings,
                region_ids=region_ids,
                tier_lists=tier_lists,
                starts=starts,
                ends=ends,
                audio_ids=audio_ids,
                seed=int(job.seed),
            )
            # Spec §4.2: produce overlay/exemplars/label-distribution per k so
            # the detail page is fully populated when the job lands in
            # `complete`. Without this, frontend reads 404 until the user hits
            # POST /generate-interpretations manually.
            await generate_interpretations(session, settings.storage_root, job, int(k))

        # Mark completed.
        now = datetime.now(timezone.utc)
        refreshed = await session.get(MaskedTransformerJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.status_reason = None
        target.error_message = None
        target.job_dir = str(job_dir)
        target.updated_at = now
        await session.commit()

        logger.info(
            "masked_transformer | job=%s | complete | seqs=%d k=%s",
            job_id,
            len(region_ids),
            k_list,
        )

    except Exception as exc:
        logger.exception("masked_transformer job %s failed", job_id)
        # Clean any half-written tmp dirs.
        for child in masked_transformer_dir(settings.storage_root, job_id).glob(
            "*.tmp"
        ):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
            except OSError:
                logger.debug("cleanup failed for %s", child, exc_info=True)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(MaskedTransformerJob, job_id)
            if refreshed is not None:
                refreshed.status = JobStatus.failed.value
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = datetime.now(timezone.utc)
                await session.commit()
        except Exception:
            logger.exception(
                "failed to mark masked_transformer job %s as failed", job_id
            )


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> Optional[MaskedTransformerJob]:
    job = await claim_masked_transformer_job(session)
    if job is None:
        return None
    await run_masked_transformer_job(session, job, settings)
    return job


__all__ = [
    "run_masked_transformer_job",
    "run_one_iteration",
    "validate_training_device",
]
