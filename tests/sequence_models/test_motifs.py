from pathlib import Path

import pyarrow as pa

from humpback.sequence_models.motifs import (
    MotifExtractionConfig,
    collapse_state_runs,
    config_signature,
    extract_motifs,
    read_motif_artifacts,
    write_motif_artifacts,
)


def _surf_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows)


def _surf_rows(events: list[str] | None = None) -> list[dict]:
    states = [1, 1, 1, 2, 2, 3, 1, 2, 3, 1, 2, 3]
    event_ids = events or ["e1"] * 6 + ["e2"] * 6
    return [
        {
            "merged_span_id": 0 if i < 6 else 1,
            "window_index_in_span": i if i < 6 else i - 6,
            "audio_file_id": 10 if i < 6 else 11,
            "start_timestamp": float(i),
            "end_timestamp": float(i + 1),
            "is_in_pad": i in {0, 5, 6},
            "event_id": event_ids[i],
            "viterbi_state": state,
            "state_posterior": [0.9],
            "max_state_probability": 0.9,
            "was_used_for_training": True,
        }
        for i, state in enumerate(states)
    ]


def test_collapse_state_runs_reduces_repeated_states() -> None:
    sequences, strategy = collapse_state_runs(
        _surf_table(_surf_rows()[:6]),
        source_kind="surfperch",
    )

    assert strategy == "event_id"
    assert [[t.state for t in seq.tokens] for seq in sequences] == [[1, 2, 3]]
    assert sequences[0].tokens[0].run_start_index == 0
    assert sequences[0].tokens[0].run_end_index == 2


def test_extracts_ngrams_and_filters_by_event_source() -> None:
    result = extract_motifs(
        _surf_table(_surf_rows()),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=3,
            minimum_occurrences=2,
            minimum_event_sources=2,
        ),
        hmm_sequence_job_id="hmm-1",
        continuous_embedding_job_id="ce-1",
        event_lookup={"e1": (1.0, 5.0), "e2": (7.0, 11.0)},
    )

    keys = [m.motif_key for m in result.motifs]
    assert "1-2" in keys
    assert "2-3" in keys
    motif = next(m for m in result.motifs if m.motif_key == "1-2")
    assert motif.occurrence_count == 3
    assert motif.event_source_count == 2
    assert motif.background_fraction < 1.0
    occ = next(o for o in result.occurrences if o.motif_key == "1-2")
    assert occ.anchor_strategy == "event_midpoint"


def test_minimum_event_sources_filters_single_event_recurrence() -> None:
    result = extract_motifs(
        _surf_table(_surf_rows(events=["e1"] * 12)),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=2,
            minimum_event_sources=2,
        ),
    )

    assert result.motifs == []


def test_rank_weights_change_ordering() -> None:
    rows = _surf_rows() + [
        {
            "merged_span_id": 2,
            "window_index_in_span": 0,
            "audio_file_id": 12,
            "start_timestamp": 20.0,
            "end_timestamp": 21.0,
            "is_in_pad": True,
            "event_id": "e3",
            "viterbi_state": 4,
            "state_posterior": [0.9],
            "max_state_probability": 0.9,
            "was_used_for_training": True,
        },
        {
            "merged_span_id": 2,
            "window_index_in_span": 1,
            "audio_file_id": 12,
            "start_timestamp": 21.0,
            "end_timestamp": 22.0,
            "is_in_pad": True,
            "event_id": "e3",
            "viterbi_state": 5,
            "state_posterior": [0.9],
            "max_state_probability": 0.9,
            "was_used_for_training": True,
        },
    ]
    frequency_first = extract_motifs(
        _surf_table(rows),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=1,
            minimum_event_sources=1,
            frequency_weight=1.0,
            event_source_weight=0.0,
            event_core_weight=0.0,
            low_background_weight=0.0,
        ),
    )
    core_first = extract_motifs(
        _surf_table(rows),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=1,
            minimum_event_sources=1,
            frequency_weight=0.0,
            event_source_weight=0.0,
            event_core_weight=1.0,
            low_background_weight=0.0,
        ),
    )

    assert frequency_first.motifs[0].motif_key != core_first.motifs[0].motif_key


def test_crnn_uses_tiers_nearest_events_and_call_probability() -> None:
    states = pa.Table.from_pylist(
        [
            {
                "region_id": "r1",
                "chunk_index_in_region": 0,
                "audio_file_id": 1,
                "start_timestamp": 0.0,
                "end_timestamp": 0.25,
                "is_in_pad": False,
                "tier": "event_core",
                "viterbi_state": 1,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
            {
                "region_id": "r1",
                "chunk_index_in_region": 1,
                "audio_file_id": 1,
                "start_timestamp": 0.25,
                "end_timestamp": 0.5,
                "is_in_pad": False,
                "tier": "background",
                "viterbi_state": 2,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 0,
                "audio_file_id": 2,
                "start_timestamp": 10.0,
                "end_timestamp": 10.25,
                "is_in_pad": False,
                "tier": "event_core",
                "viterbi_state": 1,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 1,
                "audio_file_id": 2,
                "start_timestamp": 10.25,
                "end_timestamp": 10.5,
                "is_in_pad": False,
                "tier": "background",
                "viterbi_state": 2,
                "state_posterior": [0.9],
                "max_state_probability": 0.9,
                "was_used_for_training": True,
            },
        ]
    )
    embeddings = pa.Table.from_pylist(
        [
            {
                "region_id": "r1",
                "chunk_index_in_region": 0,
                "nearest_event_id": "e1",
                "call_probability": 0.8,
            },
            {
                "region_id": "r1",
                "chunk_index_in_region": 1,
                "nearest_event_id": "e1",
                "call_probability": 0.2,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 0,
                "nearest_event_id": "e2",
                "call_probability": 1.0,
            },
            {
                "region_id": "r2",
                "chunk_index_in_region": 1,
                "nearest_event_id": "e2",
                "call_probability": 0.4,
            },
        ]
    )

    result = extract_motifs(
        states,
        source_kind="region_crnn",
        embedding_table=embeddings,
        event_lookup={"e1": (0.0, 1.0), "e2": (10.0, 11.0)},
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=2,
            minimum_event_sources=2,
            call_probability_weight=0.5,
        ),
    )

    assert result.event_source_key_strategy == "nearest_event_id"
    motif = result.motifs[0]
    assert motif.motif_key == "1-2"
    assert motif.event_core_fraction == 0.5
    assert motif.background_fraction == 0.5
    assert motif.mean_call_probability == 0.6


def test_anchor_falls_back_without_event_lookup() -> None:
    result = extract_motifs(
        _surf_table(_surf_rows()),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=1,
            minimum_event_sources=1,
        ),
    )

    assert {o.anchor_strategy for o in result.occurrences} == {"event_core_midpoint"}

    background_rows = _surf_rows()
    for row in background_rows:
        row["is_in_pad"] = True
    background_result = extract_motifs(
        _surf_table(background_rows),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=1,
            minimum_event_sources=1,
        ),
    )
    assert {o.anchor_strategy for o in background_result.occurrences} == {
        "occurrence_midpoint"
    }


def test_signature_is_stable_and_changes_with_config() -> None:
    base = MotifExtractionConfig()
    same = MotifExtractionConfig()
    other = MotifExtractionConfig(minimum_event_sources=3)

    assert config_signature("hmm-1", base) == config_signature("hmm-1", same)
    assert config_signature("hmm-1", base) != config_signature("hmm-1", other)


def test_artifact_round_trip(tmp_path: Path) -> None:
    result = extract_motifs(
        _surf_table(_surf_rows()),
        source_kind="surfperch",
        config=MotifExtractionConfig(
            min_ngram=2,
            max_ngram=2,
            minimum_occurrences=2,
            minimum_event_sources=2,
        ),
        hmm_sequence_job_id="hmm-1",
        continuous_embedding_job_id="ce-1",
    )

    write_motif_artifacts(result, tmp_path, motif_extraction_job_id="motif-1")
    payload = read_motif_artifacts(tmp_path)

    assert payload["manifest"]["motif_extraction_job_id"] == "motif-1"
    assert payload["motifs"]
    assert payload["occurrences"]
