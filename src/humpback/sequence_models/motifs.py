"""Motif extraction over decoded HMM state sequences.

The extractor is intentionally source-normalized: SurfPerch jobs group decoded
rows by ``merged_span_id`` while CRNN jobs group them by ``region_id``. Both
become collapsed symbolic state sequences before n-gram mining.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_KIND_SURFPERCH = "surfperch"
SOURCE_KIND_REGION_CRNN = "region_crnn"
SCHEMA_VERSION = 1

SourceKind = Literal["surfperch", "region_crnn"]


@dataclass(frozen=True)
class MotifExtractionConfig:
    """Configuration for one motif extraction run."""

    min_ngram: int = 2
    max_ngram: int = 8
    minimum_occurrences: int = 5
    minimum_event_sources: int = 2
    frequency_weight: float = 0.40
    event_source_weight: float = 0.30
    event_core_weight: float = 0.20
    low_background_weight: float = 0.10
    call_probability_weight: float | None = None

    def validate(self) -> None:
        if self.min_ngram < 1:
            raise ValueError("min_ngram must be >= 1")
        if self.max_ngram < self.min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        if self.max_ngram > 16:
            raise ValueError("max_ngram must be <= 16")
        if self.minimum_occurrences < 1:
            raise ValueError("minimum_occurrences must be >= 1")
        if self.minimum_event_sources < 1:
            raise ValueError("minimum_event_sources must be >= 1")
        weights = [
            self.frequency_weight,
            self.event_source_weight,
            self.event_core_weight,
            self.low_background_weight,
        ]
        if self.call_probability_weight is not None:
            weights.append(self.call_probability_weight)
        if any(w < 0 for w in weights):
            raise ValueError("rank weights must be non-negative")
        if not any(w > 0 for w in weights):
            raise ValueError("at least one rank weight must be > 0")

    def to_signature_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "min_ngram": self.min_ngram,
            "max_ngram": self.max_ngram,
            "minimum_occurrences": self.minimum_occurrences,
            "minimum_event_sources": self.minimum_event_sources,
            "frequency_weight": self.frequency_weight,
            "event_source_weight": self.event_source_weight,
            "event_core_weight": self.event_core_weight,
            "low_background_weight": self.low_background_weight,
            "call_probability_weight": self.call_probability_weight,
        }


@dataclass
class CollapsedToken:
    state: int
    run_start_index: int
    run_end_index: int
    start_timestamp: float
    end_timestamp: float
    event_source_key: str
    audio_source_key: str | None
    event_core_duration: float
    background_duration: float
    mean_call_probability: float | None = None

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)


@dataclass
class CollapsedSequence:
    group_key: str
    tokens: list[CollapsedToken]


@dataclass
class MotifOccurrence:
    occurrence_id: str
    motif_key: str
    states: list[int]
    source_kind: str
    group_key: str
    event_source_key: str
    audio_source_key: str | None
    token_start_index: int
    token_end_index: int
    raw_start_index: int
    raw_end_index: int
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    event_core_fraction: float
    background_fraction: float
    mean_call_probability: float | None
    anchor_event_id: str | None
    anchor_timestamp: float
    relative_start_seconds: float
    relative_end_seconds: float
    anchor_strategy: str


@dataclass
class MotifSummary:
    motif_key: str
    states: list[int]
    length: int
    occurrence_count: int
    event_source_count: int
    audio_source_count: int
    group_count: int
    event_core_fraction: float
    background_fraction: float
    mean_call_probability: float | None
    mean_duration_seconds: float
    median_duration_seconds: float
    rank_score: float
    example_occurrence_ids: list[str] = field(default_factory=list)


@dataclass
class MotifExtractionResult:
    hmm_sequence_job_id: str
    continuous_embedding_job_id: str
    source_kind: str
    config: MotifExtractionConfig
    config_signature: str
    total_groups: int
    total_collapsed_tokens: int
    total_candidate_occurrences: int
    total_motifs: int
    event_source_key_strategy: str
    motifs: list[MotifSummary]
    occurrences: list[MotifOccurrence]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def manifest(self, motif_extraction_job_id: str) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "motif_extraction_job_id": motif_extraction_job_id,
            "hmm_sequence_job_id": self.hmm_sequence_job_id,
            "continuous_embedding_job_id": self.continuous_embedding_job_id,
            "source_kind": self.source_kind,
            "config": self.config.to_signature_dict(),
            "config_signature": self.config_signature,
            "generated_at": self.generated_at,
            "total_groups": self.total_groups,
            "total_collapsed_tokens": self.total_collapsed_tokens,
            "total_candidate_occurrences": self.total_candidate_occurrences,
            "total_motifs": self.total_motifs,
            "event_source_key_strategy": self.event_source_key_strategy,
        }


def config_signature(hmm_sequence_job_id: str, config: MotifExtractionConfig) -> str:
    """Return a stable signature for source job + extraction config."""
    config.validate()
    payload = {
        "hmm_sequence_job_id": hmm_sequence_job_id,
        "config": config.to_signature_dict(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _table_rows(table: pa.Table) -> list[dict[str, Any]]:
    data = table.to_pydict()
    return [
        {col: data[col][i] for col in table.column_names} for i in range(table.num_rows)
    ]


def _embedding_lookup(embedding_table: pa.Table | None) -> dict[tuple[str, int], dict]:
    if embedding_table is None:
        return {}
    rows = _table_rows(embedding_table)
    out: dict[tuple[str, int], dict] = {}
    for row in rows:
        region_id = row.get("region_id")
        chunk_idx = row.get("chunk_index_in_region")
        if region_id is None or chunk_idx is None:
            continue
        out[(str(region_id), int(chunk_idx))] = row
    return out


def _mode(values: list[str | None]) -> str | None:
    counts = Counter(v for v in values if v not in (None, ""))
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _row_duration(row: dict[str, Any]) -> float:
    return max(0.0, float(row["end_timestamp"]) - float(row["start_timestamp"]))


def _row_event_source(
    row: dict[str, Any],
    source_kind: str,
    lookup: dict[tuple[str, int], dict],
) -> tuple[str | None, str]:
    if source_kind == SOURCE_KIND_SURFPERCH:
        event_id = row.get("event_id")
        if event_id not in (None, ""):
            return str(event_id), "event_id"
        return None, "group_fallback"

    region_id = row.get("region_id")
    chunk_idx = row.get("chunk_index_in_region")
    emb = lookup.get((str(region_id), int(chunk_idx))) if chunk_idx is not None else {}
    event_id = emb.get("nearest_event_id") if emb else None
    if event_id not in (None, ""):
        return str(event_id), "nearest_event_id"
    return None, "group_fallback"


def _row_call_probability(
    row: dict[str, Any],
    source_kind: str,
    lookup: dict[tuple[str, int], dict],
) -> float | None:
    if source_kind != SOURCE_KIND_REGION_CRNN:
        return None
    if "call_probability" in row and row["call_probability"] is not None:
        return float(row["call_probability"])
    region_id = row.get("region_id")
    chunk_idx = row.get("chunk_index_in_region")
    emb = lookup.get((str(region_id), int(chunk_idx))) if chunk_idx is not None else {}
    prob = emb.get("call_probability") if emb else None
    return float(prob) if prob is not None else None


def _row_core_background(row: dict[str, Any], source_kind: str) -> tuple[float, float]:
    duration = _row_duration(row)
    if source_kind == SOURCE_KIND_SURFPERCH:
        return (0.0, duration) if bool(row.get("is_in_pad")) else (duration, 0.0)
    tier = row.get("tier")
    if tier == "event_core":
        return duration, 0.0
    if tier == "background":
        return 0.0, duration
    return 0.0, 0.0


def _source_columns(source_kind: str) -> tuple[str, str]:
    if source_kind == SOURCE_KIND_REGION_CRNN:
        return "region_id", "chunk_index_in_region"
    return "merged_span_id", "window_index_in_span"


def collapse_state_runs(
    states_table: pa.Table,
    *,
    source_kind: str,
    embedding_table: pa.Table | None = None,
) -> tuple[list[CollapsedSequence], str]:
    """Collapse consecutive repeated states for each decoded source sequence."""
    if "viterbi_state" not in states_table.column_names:
        raise ValueError("states table must include viterbi_state")
    group_col, sort_col = _source_columns(source_kind)
    required = {group_col, sort_col, "start_timestamp", "end_timestamp"}
    missing = required.difference(states_table.column_names)
    if missing:
        raise ValueError(f"states table missing required columns: {sorted(missing)}")

    lookup = _embedding_lookup(embedding_table)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _table_rows(states_table):
        grouped[str(row[group_col])].append(row)

    sequences: list[CollapsedSequence] = []
    strategies: set[str] = set()
    for group_key in sorted(grouped):
        rows = sorted(grouped[group_key], key=lambda r: int(r[sort_col]))
        tokens: list[CollapsedToken] = []
        run_rows: list[dict[str, Any]] = []
        run_state: int | None = None

        def flush() -> None:
            if not run_rows or run_state is None:
                return
            first = run_rows[0]
            last = run_rows[-1]
            event_sources: list[str | None] = []
            audio_sources: list[str | None] = []
            core = 0.0
            background = 0.0
            probs: list[float] = []
            run_strategies: set[str] = set()
            for item in run_rows:
                event_source, strategy = _row_event_source(item, source_kind, lookup)
                run_strategies.add(strategy)
                event_sources.append(event_source)
                audio = item.get("audio_file_id")
                audio_sources.append(str(audio) if audio is not None else None)
                c, b = _row_core_background(item, source_kind)
                core += c
                background += b
                prob = _row_call_probability(item, source_kind, lookup)
                if prob is not None:
                    probs.append(prob)

            event_source_key = _mode(event_sources)
            if event_source_key is None:
                event_source_key = group_key
                run_strategies.add("group_fallback")
            strategies.update(run_strategies)
            tokens.append(
                CollapsedToken(
                    state=run_state,
                    run_start_index=int(first[sort_col]),
                    run_end_index=int(last[sort_col]),
                    start_timestamp=float(first["start_timestamp"]),
                    end_timestamp=float(last["end_timestamp"]),
                    event_source_key=event_source_key,
                    audio_source_key=_mode(audio_sources),
                    event_core_duration=core,
                    background_duration=background,
                    mean_call_probability=float(np.mean(probs)) if probs else None,
                )
            )

        for row in rows:
            state = int(row["viterbi_state"])
            if run_state is None:
                run_state = state
            if state != run_state:
                flush()
                run_rows = []
                run_state = state
            run_rows.append(row)
        flush()
        sequences.append(CollapsedSequence(group_key=group_key, tokens=tokens))

    strategy = (
        "group_fallback"
        if "group_fallback" in strategies
        else (
            "nearest_event_id"
            if "nearest_event_id" in strategies
            else "event_id"
            if "event_id" in strategies
            else "group_fallback"
        )
    )
    return sequences, strategy


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights or sum(weights) <= 0:
        return None
    return float(np.average(values, weights=weights))


def _anchor_for_occurrence(
    event_source_key: str,
    start: float,
    end: float,
    core_duration: float,
    event_lookup: dict[str, tuple[float, float]] | None,
) -> tuple[str | None, float, str]:
    if event_lookup and event_source_key in event_lookup:
        ev_start, ev_end = event_lookup[event_source_key]
        return (
            event_source_key,
            (float(ev_start) + float(ev_end)) / 2.0,
            "event_midpoint",
        )
    midpoint = (start + end) / 2.0
    if core_duration > 0:
        return None, midpoint, "event_core_midpoint"
    return None, midpoint, "occurrence_midpoint"


def _occurrence_from_tokens(
    *,
    source_kind: str,
    group_key: str,
    token_start: int,
    tokens: list[CollapsedToken],
    event_lookup: dict[str, tuple[float, float]] | None,
) -> MotifOccurrence:
    states = [t.state for t in tokens]
    motif_key = "-".join(str(s) for s in states)
    start = tokens[0].start_timestamp
    end = tokens[-1].end_timestamp
    duration = max(0.0, end - start)
    core = sum(t.event_core_duration for t in tokens)
    background = sum(t.background_duration for t in tokens)
    event_source_key = _mode([t.event_source_key for t in tokens]) or group_key
    audio_source_key = _mode([t.audio_source_key for t in tokens])
    probs = [
        t.mean_call_probability for t in tokens if t.mean_call_probability is not None
    ]
    prob_weights = [
        t.duration_seconds for t in tokens if t.mean_call_probability is not None
    ]
    mean_call_probability = _weighted_mean(probs, prob_weights)
    anchor_event_id, anchor_timestamp, anchor_strategy = _anchor_for_occurrence(
        event_source_key, start, end, core, event_lookup
    )
    occurrence_id = hashlib.sha1(
        f"{group_key}:{token_start}:{token_start + len(tokens) - 1}:{motif_key}".encode(
            "utf-8"
        )
    ).hexdigest()
    return MotifOccurrence(
        occurrence_id=occurrence_id,
        motif_key=motif_key,
        states=states,
        source_kind=source_kind,
        group_key=group_key,
        event_source_key=event_source_key,
        audio_source_key=audio_source_key,
        token_start_index=token_start,
        token_end_index=token_start + len(tokens) - 1,
        raw_start_index=tokens[0].run_start_index,
        raw_end_index=tokens[-1].run_end_index,
        start_timestamp=start,
        end_timestamp=end,
        duration_seconds=duration,
        event_core_fraction=(core / duration) if duration > 0 else 0.0,
        background_fraction=(background / duration) if duration > 0 else 0.0,
        mean_call_probability=mean_call_probability,
        anchor_event_id=anchor_event_id,
        anchor_timestamp=anchor_timestamp,
        relative_start_seconds=start - anchor_timestamp,
        relative_end_seconds=end - anchor_timestamp,
        anchor_strategy=anchor_strategy,
    )


def _rank_summaries(
    motifs: list[MotifSummary],
    config: MotifExtractionConfig,
) -> list[MotifSummary]:
    max_log_count = max((np.log1p(m.occurrence_count) for m in motifs), default=1.0)
    max_sources = max((m.event_source_count for m in motifs), default=1)
    for motif in motifs:
        frequency_norm = float(np.log1p(motif.occurrence_count) / max_log_count)
        source_norm = motif.event_source_count / max_sources
        score = (
            config.frequency_weight * frequency_norm
            + config.event_source_weight * source_norm
            + config.event_core_weight * motif.event_core_fraction
            + config.low_background_weight * (1.0 - motif.background_fraction)
        )
        if (
            config.call_probability_weight is not None
            and motif.mean_call_probability is not None
        ):
            score += config.call_probability_weight * motif.mean_call_probability
        motif.rank_score = float(score)
    return sorted(
        motifs,
        key=lambda m: (
            -m.rank_score,
            -m.event_source_count,
            -m.occurrence_count,
            -m.event_core_fraction,
            m.background_fraction,
            -m.length,
            m.motif_key,
        ),
    )


def extract_motifs(
    states_table: pa.Table,
    *,
    source_kind: str,
    config: MotifExtractionConfig,
    hmm_sequence_job_id: str = "",
    continuous_embedding_job_id: str = "",
    event_lookup: dict[str, tuple[float, float]] | None = None,
    embedding_table: pa.Table | None = None,
) -> MotifExtractionResult:
    """Extract ranked motifs from a decoded HMM states table."""
    config.validate()
    sequences, strategy = collapse_state_runs(
        states_table, source_kind=source_kind, embedding_table=embedding_table
    )

    all_occurrences: list[MotifOccurrence] = []
    for sequence in sequences:
        tokens = sequence.tokens
        for n in range(config.min_ngram, config.max_ngram + 1):
            if len(tokens) < n:
                continue
            for start_idx in range(0, len(tokens) - n + 1):
                all_occurrences.append(
                    _occurrence_from_tokens(
                        source_kind=source_kind,
                        group_key=sequence.group_key,
                        token_start=start_idx,
                        tokens=tokens[start_idx : start_idx + n],
                        event_lookup=event_lookup,
                    )
                )

    by_key: dict[str, list[MotifOccurrence]] = defaultdict(list)
    for occ in all_occurrences:
        by_key[occ.motif_key].append(occ)

    summaries: list[MotifSummary] = []
    kept_occurrences: list[MotifOccurrence] = []
    for motif_key, occs in by_key.items():
        event_sources = {o.event_source_key for o in occs if o.event_source_key}
        if len(occs) < config.minimum_occurrences:
            continue
        if len(event_sources) < config.minimum_event_sources:
            continue
        durations = [o.duration_seconds for o in occs]
        duration_total = sum(durations)
        probs = [
            o.mean_call_probability for o in occs if o.mean_call_probability is not None
        ]
        prob_weights = [
            o.duration_seconds for o in occs if o.mean_call_probability is not None
        ]
        summaries.append(
            MotifSummary(
                motif_key=motif_key,
                states=occs[0].states,
                length=len(occs[0].states),
                occurrence_count=len(occs),
                event_source_count=len(event_sources),
                audio_source_count=len(
                    {o.audio_source_key for o in occs if o.audio_source_key}
                ),
                group_count=len({o.group_key for o in occs}),
                event_core_fraction=(
                    sum(o.event_core_fraction * o.duration_seconds for o in occs)
                    / duration_total
                    if duration_total > 0
                    else 0.0
                ),
                background_fraction=(
                    sum(o.background_fraction * o.duration_seconds for o in occs)
                    / duration_total
                    if duration_total > 0
                    else 0.0
                ),
                mean_call_probability=_weighted_mean(probs, prob_weights),
                mean_duration_seconds=float(np.mean(durations)) if durations else 0.0,
                median_duration_seconds=float(median(durations)) if durations else 0.0,
                rank_score=0.0,
                example_occurrence_ids=[
                    o.occurrence_id
                    for o in sorted(
                        occs,
                        key=lambda o: (
                            -o.event_core_fraction,
                            o.background_fraction,
                            o.event_source_key,
                            o.start_timestamp,
                        ),
                    )[:10]
                ],
            )
        )
        kept_occurrences.extend(occs)

    summaries = _rank_summaries(summaries, config)
    sig = config_signature(hmm_sequence_job_id, config) if hmm_sequence_job_id else ""
    return MotifExtractionResult(
        hmm_sequence_job_id=hmm_sequence_job_id,
        continuous_embedding_job_id=continuous_embedding_job_id,
        source_kind=source_kind,
        config=config,
        config_signature=sig,
        total_groups=len(sequences),
        total_collapsed_tokens=sum(len(seq.tokens) for seq in sequences),
        total_candidate_occurrences=len(all_occurrences),
        total_motifs=len(summaries),
        event_source_key_strategy=strategy,
        motifs=summaries,
        occurrences=kept_occurrences,
    )


MOTIFS_SCHEMA = pa.schema(
    [
        pa.field("motif_key", pa.string()),
        pa.field("states", pa.list_(pa.int16())),
        pa.field("length", pa.int16()),
        pa.field("occurrence_count", pa.int32()),
        pa.field("event_source_count", pa.int32()),
        pa.field("audio_source_count", pa.int32()),
        pa.field("group_count", pa.int32()),
        pa.field("event_core_fraction", pa.float32()),
        pa.field("background_fraction", pa.float32()),
        pa.field("mean_call_probability", pa.float32(), nullable=True),
        pa.field("mean_duration_seconds", pa.float32()),
        pa.field("median_duration_seconds", pa.float32()),
        pa.field("rank_score", pa.float32()),
        pa.field("example_occurrence_ids", pa.list_(pa.string())),
    ]
)

OCCURRENCES_SCHEMA = pa.schema(
    [
        pa.field("occurrence_id", pa.string()),
        pa.field("motif_key", pa.string()),
        pa.field("states", pa.list_(pa.int16())),
        pa.field("source_kind", pa.string()),
        pa.field("group_key", pa.string()),
        pa.field("event_source_key", pa.string()),
        pa.field("audio_source_key", pa.string(), nullable=True),
        pa.field("token_start_index", pa.int32()),
        pa.field("token_end_index", pa.int32()),
        pa.field("raw_start_index", pa.int32()),
        pa.field("raw_end_index", pa.int32()),
        pa.field("start_timestamp", pa.float64()),
        pa.field("end_timestamp", pa.float64()),
        pa.field("duration_seconds", pa.float32()),
        pa.field("event_core_fraction", pa.float32()),
        pa.field("background_fraction", pa.float32()),
        pa.field("mean_call_probability", pa.float32(), nullable=True),
        pa.field("anchor_event_id", pa.string(), nullable=True),
        pa.field("anchor_timestamp", pa.float64()),
        pa.field("relative_start_seconds", pa.float32()),
        pa.field("relative_end_seconds", pa.float32()),
        pa.field("anchor_strategy", pa.string()),
    ]
)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_parquet(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp)
    tmp.replace(path)


def write_motif_artifacts(
    result: MotifExtractionResult,
    output_dir: Path,
    *,
    motif_extraction_job_id: str,
) -> None:
    """Write manifest and parquet artifacts for a motif result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        result.manifest(motif_extraction_job_id), output_dir / "manifest.json"
    )
    _atomic_write_parquet(
        pa.Table.from_pylist([asdict(m) for m in result.motifs], schema=MOTIFS_SCHEMA),
        output_dir / "motifs.parquet",
    )
    _atomic_write_parquet(
        pa.Table.from_pylist(
            [asdict(o) for o in result.occurrences], schema=OCCURRENCES_SCHEMA
        ),
        output_dir / "occurrences.parquet",
    )


def read_motif_artifacts(output_dir: Path) -> dict[str, Any]:
    """Read motif artifacts from disk into JSON-friendly dictionaries."""
    manifest_path = output_dir / "manifest.json"
    motifs_path = output_dir / "motifs.parquet"
    occurrences_path = output_dir / "occurrences.parquet"
    return {
        "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
        "motifs": _table_rows(pq.read_table(motifs_path))
        if motifs_path.exists()
        else [],
        "occurrences": (
            _table_rows(pq.read_table(occurrences_path))
            if occurrences_path.exists()
            else []
        ),
    }
