"""Tests for embedding diff and audio resolution in detector.py."""

import struct
import wave
from pathlib import Path

import numpy as np

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    write_detection_row_store,
)
from humpback.classifier.detector import (
    diff_row_store_vs_embeddings,
    resolve_audio_for_window,
    resolve_audio_for_window_hydrophone,
    write_detection_embeddings,
)

_ROW_COUNTER = 0


def _make_row(
    start_utc: float, end_utc: float, label: str = "", row_id: str = ""
) -> dict[str, str]:
    """Build a minimal row-store row dict."""
    global _ROW_COUNTER  # noqa: PLW0603
    _ROW_COUNTER += 1
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id or f"row-{_ROW_COUNTER}"
    row["start_utc"] = str(start_utc)
    row["end_utc"] = str(end_utc)
    if label:
        row[label] = "1"
    return row


def _make_emb(row_id: str) -> dict:
    """Build an embedding record with a dummy vector."""
    return {
        "row_id": row_id,
        "embedding": np.zeros(8, dtype=np.float32).tolist(),
        "confidence": 0.9,
    }


# The synthetic filenames use YYYYMMDDTHHMMSSZ format.
# 2021-11-01T08:50:00Z → epoch 1635756600.0
_BASE_EPOCH = 1635756600.0


class TestDiffRowStoreVsEmbeddings:
    def test_all_matched(self, tmp_path):
        """Every row-store entry has a matching embedding."""
        rs_path = tmp_path / "row_store.parquet"
        emb_path = tmp_path / "emb.parquet"

        rows = [
            _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, row_id="a"),
            _make_row(_BASE_EPOCH + 10, _BASE_EPOCH + 15, row_id="b"),
        ]
        write_detection_row_store(rs_path, rows)

        embs = [_make_emb("a"), _make_emb("b")]
        write_detection_embeddings(embs, emb_path)

        result = diff_row_store_vs_embeddings(rs_path, emb_path)
        assert result.matched_count == 2
        assert result.missing == []
        assert result.orphaned_indices == []

    def test_all_missing(self, tmp_path):
        """Row store has entries but embeddings have different row_ids."""
        rs_path = tmp_path / "row_store.parquet"
        emb_path = tmp_path / "emb.parquet"

        rows = [
            _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, row_id="a"),
            _make_row(_BASE_EPOCH + 60, _BASE_EPOCH + 65, row_id="b"),
        ]
        write_detection_row_store(rs_path, rows)

        embs = [_make_emb("z")]
        write_detection_embeddings(embs, emb_path)

        result = diff_row_store_vs_embeddings(rs_path, emb_path)
        assert result.matched_count == 0
        assert len(result.missing) == 2
        assert result.orphaned_indices == [0]

    def test_all_orphaned(self, tmp_path):
        """Embeddings exist but row store is empty."""
        rs_path = tmp_path / "row_store.parquet"
        emb_path = tmp_path / "emb.parquet"

        write_detection_row_store(rs_path, [])

        embs = [_make_emb("x"), _make_emb("y")]
        write_detection_embeddings(embs, emb_path)

        result = diff_row_store_vs_embeddings(rs_path, emb_path)
        assert result.matched_count == 0
        assert result.missing == []
        assert result.orphaned_indices == [0, 1]

    def test_mixed(self, tmp_path):
        """Some matched, some missing, some orphaned."""
        rs_path = tmp_path / "row_store.parquet"
        emb_path = tmp_path / "emb.parquet"

        rows = [
            _make_row(_BASE_EPOCH + 0, _BASE_EPOCH + 5, row_id="a"),  # matched
            _make_row(_BASE_EPOCH + 100, _BASE_EPOCH + 105, row_id="b"),  # missing
        ]
        write_detection_row_store(rs_path, rows)

        embs = [
            _make_emb("a"),  # matched
            _make_emb("z"),  # orphaned (no row)
        ]
        write_detection_embeddings(embs, emb_path)

        result = diff_row_store_vs_embeddings(rs_path, emb_path)
        assert result.matched_count == 1
        assert len(result.missing) == 1
        assert result.missing[0]["row_id"] == "b"
        assert result.orphaned_indices == [1]

    def test_exact_row_id_match_required(self, tmp_path):
        """Matching is by exact row_id, not by timestamp proximity."""
        rs_path = tmp_path / "row_store.parquet"
        emb_path = tmp_path / "emb.parquet"

        rows = [_make_row(_BASE_EPOCH + 10, _BASE_EPOCH + 15, row_id="a")]
        write_detection_row_store(rs_path, rows)

        # Embedding with different row_id — should NOT match even if timestamps
        # would have matched under old tolerance logic.
        embs = [_make_emb("different-id")]
        write_detection_embeddings(embs, emb_path)

        result = diff_row_store_vs_embeddings(rs_path, emb_path)
        assert result.matched_count == 0
        assert len(result.missing) == 1
        assert result.orphaned_indices == [0]


# ---------------------------------------------------------------------------
# Audio resolution helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _write_wav(path: Path, duration: float = 10.0, sample_rate: int = SAMPLE_RATE):
    """Write a simple sine wave WAV file."""
    import math as m

    n_samples = int(sample_rate * duration)
    samples = [
        int(32767 * m.sin(2 * m.pi * 440 * i / sample_rate)) for i in range(n_samples)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))


class TestResolveAudioForWindow:
    def test_correct_file_selection(self, tmp_path):
        """Selects the correct file when multiple files span different times."""
        folder = tmp_path / "audio"
        folder.mkdir()
        # File A: 08:50:00 - 08:50:10 (10s)
        _write_wav(folder / "20211101T085000Z.wav", duration=10.0)
        # File B: 08:51:00 - 08:51:10 (10s)
        _write_wav(folder / "20211101T085100Z.wav", duration=10.0)

        # Request window inside file B: 08:51:02 - 08:51:07
        base_b = 1635756660.0  # 2021-11-01T08:51:00Z
        audio, reason = resolve_audio_for_window(
            base_b + 2.0, base_b + 7.0, folder, SAMPLE_RATE
        )
        assert reason is None
        assert audio is not None
        expected_samples = int(5.0 * SAMPLE_RATE)
        assert len(audio) == expected_samples

    def test_correct_offset_within_file(self, tmp_path):
        """The extracted window starts at the correct offset."""
        folder = tmp_path / "audio"
        folder.mkdir()
        _write_wav(folder / "20211101T085000Z.wav", duration=10.0)

        base = _BASE_EPOCH  # 2021-11-01T08:50:00Z
        # Window at 3s-8s into the file
        audio, reason = resolve_audio_for_window(
            base + 3.0, base + 8.0, folder, SAMPLE_RATE
        )
        assert reason is None
        assert audio is not None
        assert len(audio) == int(5.0 * SAMPLE_RATE)

    def test_no_covering_file(self, tmp_path):
        """Returns None with reason when no file covers the UTC range."""
        folder = tmp_path / "audio"
        folder.mkdir()
        # File covers 08:50:00 - 08:50:10
        _write_wav(folder / "20211101T085000Z.wav", duration=10.0)

        # Request window at 09:00:00 — no file covers this
        audio, reason = resolve_audio_for_window(
            1635757200.0, 1635757205.0, folder, SAMPLE_RATE
        )
        assert audio is None
        assert reason is not None
        assert "no file covers" in reason

    def test_empty_folder(self, tmp_path):
        """Returns None when folder has no audio files."""
        folder = tmp_path / "audio"
        folder.mkdir()
        audio, reason = resolve_audio_for_window(
            _BASE_EPOCH, _BASE_EPOCH + 5.0, folder, SAMPLE_RATE
        )
        assert audio is None
        assert reason is not None and "no audio files" in reason

    def test_prebuilt_timeline(self, tmp_path):
        """Accepts a pre-built file timeline to avoid re-scanning."""
        from humpback.classifier.detector import _build_file_timeline

        folder = tmp_path / "audio"
        folder.mkdir()
        _write_wav(folder / "20211101T085000Z.wav", duration=10.0)

        timeline = _build_file_timeline(folder, SAMPLE_RATE)
        audio, reason = resolve_audio_for_window(
            _BASE_EPOCH + 1.0,
            _BASE_EPOCH + 6.0,
            folder,
            SAMPLE_RATE,
            _file_timeline=timeline,
        )
        assert reason is None
        assert audio is not None


class TestResolveAudioForWindowHydrophoneUnit:
    """Unit tests using a mock provider via iter_audio_chunks."""

    def test_returns_none_when_no_chunks(self, monkeypatch):
        """Returns None when provider yields nothing."""

        def _fake_iter(*args, **kwargs):
            return iter([])

        import humpback.classifier.s3_stream as ss

        monkeypatch.setattr(ss, "iter_audio_chunks", _fake_iter)

        audio, reason = resolve_audio_for_window_hydrophone(
            _BASE_EPOCH, _BASE_EPOCH + 5.0, object(), SAMPLE_RATE
        )
        assert audio is None
        assert reason is not None

    def test_extracts_correct_window(self, monkeypatch):
        """Extracts the correct sub-window from a returned chunk."""
        from datetime import datetime, timezone

        chunk_start_ts = _BASE_EPOCH
        chunk_audio = np.ones(int(30.0 * SAMPLE_RATE), dtype=np.float32)
        chunk_dt = datetime.fromtimestamp(chunk_start_ts, tz=timezone.utc)

        def _fake_iter(*args, **kwargs):
            yield chunk_audio, chunk_dt, 1, 1

        import humpback.classifier.s3_stream as ss

        monkeypatch.setattr(ss, "iter_audio_chunks", _fake_iter)

        # Request window 10s into the chunk
        audio, reason = resolve_audio_for_window_hydrophone(
            chunk_start_ts + 10.0,
            chunk_start_ts + 15.0,
            object(),
            SAMPLE_RATE,
        )
        assert reason is None
        assert audio is not None
        assert len(audio) == int(5.0 * SAMPLE_RATE)
