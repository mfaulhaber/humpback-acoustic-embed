"""Dataclass types and pyarrow schemas for the call parsing pipeline.

These types form the cross-pass data contract. Each pass writes its own
per-job parquet files using the schemas defined here:

- Pass 1 writes ``trace.parquet`` (WindowScore rows) and
  ``regions.parquet`` (Region rows).
- Pass 2 writes ``events.parquet`` (Event rows).
- Pass 3 writes ``typed_events.parquet`` (TypedEvent rows).
- Pass 4 reads typed events from Pass 3 and exports a sorted sequence.

Cross-pass linkage uses UUID strings (``region_id``, ``event_id``)
embedded in the parquet — the same pattern ``vocalization_labels``
already uses to reference detection ``row_id``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import pyarrow as pa


def new_uuid() -> str:
    """Return a UUID4 string for region_id / event_id generation."""
    return str(uuid.uuid4())


@dataclass(frozen=True)
class WindowScore:
    """One row of the dense per-window confidence trace (Pass 1 output).

    Written to ``trace.parquet``. One row per detector window across the
    audio source; preserves the raw classifier score without NMS or
    hysteresis collapsing, so downstream passes (and debugging) can see
    the full timeline.
    """

    time_sec: float
    score: float


@dataclass(frozen=True)
class WindowEmbedding:
    """One row of the per-window embedding cache (Pass 1 output).

    Written to ``embeddings.parquet`` alongside the scalar trace.
    Stores the full Perch embedding vector for each detector window
    so downstream sidecar jobs can score cached embeddings without
    re-running inference.
    """

    time_sec: float
    embedding: list[float]


@dataclass(frozen=True)
class Region:
    """One padded whale-active region (Pass 1 output).

    A region represents a contiguous span of time identified as likely
    to contain humpback vocalizations, padded on both sides to avoid
    clipping events. Pass 2 runs segmentation inference over the region
    audio to produce per-event onset/offset bounds.

    ``start_sec`` / ``end_sec`` are the raw hysteresis-merged bounds.
    ``padded_start_sec`` / ``padded_end_sec`` are the expanded bounds
    clamped to ``[0.0, audio_duration_sec]``.
    """

    region_id: str
    start_sec: float
    end_sec: float
    padded_start_sec: float
    padded_end_sec: float
    max_score: float
    mean_score: float
    n_windows: int


@dataclass(frozen=True)
class Event:
    """One segmented vocalization event (Pass 2 output).

    Produced by framewise segmentation inference over a region. The
    ``region_id`` FK points back to the Pass 1 region the event was
    decoded from. Absolute timestamps (``start_sec`` / ``end_sec``) are
    stored in the source audio's timeline so downstream consumers don't
    need to join against the region's padded bounds.
    """

    event_id: str
    region_id: str
    start_sec: float
    end_sec: float
    center_sec: float
    segmentation_confidence: float


@dataclass(frozen=True)
class TypedEvent:
    """One per-type classification score for an event (Pass 3 output).

    A single event can produce multiple TypedEvent rows — one per call
    type in the vocabulary — because Pass 3 uses multi-label sigmoid
    output. ``above_threshold`` applies the per-type threshold from the
    ``vocalization_models`` row at inference time.
    """

    event_id: str
    start_sec: float
    end_sec: float
    type_name: str
    score: float
    above_threshold: bool


# ---- pyarrow schemas ----------------------------------------------------
#
# Field order matches dataclass field order so zip-based serialization is
# straightforward. Types are the most compact/obvious choice (float64 for
# time and scores, string for ids and type names, int64 for counts).

TRACE_SCHEMA = pa.schema(
    [
        pa.field("time_sec", pa.float64(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
    ]
)

EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("time_sec", pa.float64(), nullable=False),
        pa.field("embedding", pa.list_(pa.float32()), nullable=False),
    ]
)

REGION_SCHEMA = pa.schema(
    [
        pa.field("region_id", pa.string(), nullable=False),
        pa.field("start_sec", pa.float64(), nullable=False),
        pa.field("end_sec", pa.float64(), nullable=False),
        pa.field("padded_start_sec", pa.float64(), nullable=False),
        pa.field("padded_end_sec", pa.float64(), nullable=False),
        pa.field("max_score", pa.float64(), nullable=False),
        pa.field("mean_score", pa.float64(), nullable=False),
        pa.field("n_windows", pa.int64(), nullable=False),
    ]
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("region_id", pa.string(), nullable=False),
        pa.field("start_sec", pa.float64(), nullable=False),
        pa.field("end_sec", pa.float64(), nullable=False),
        pa.field("center_sec", pa.float64(), nullable=False),
        pa.field("segmentation_confidence", pa.float64(), nullable=False),
    ]
)

TYPED_EVENT_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("start_sec", pa.float64(), nullable=False),
        pa.field("end_sec", pa.float64(), nullable=False),
        pa.field("type_name", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("above_threshold", pa.bool_(), nullable=False),
    ]
)
