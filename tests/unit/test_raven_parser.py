"""Tests for Raven selection table parser."""

import textwrap
from pathlib import Path

import pytest

from humpback.classifier.raven_parser import (
    _annotation_stem,
    normalize_call_type,
    pair_annotations_with_recordings,
    parse_raven_tsv,
)


class TestNormalizeCallType:
    def test_basic_title_case(self):
        assert normalize_call_type("moan") == "Moan"
        assert normalize_call_type("whup") == "Whup"

    def test_already_title_case(self):
        assert normalize_call_type("Moan") == "Moan"
        assert normalize_call_type("Growl") == "Growl"

    def test_multi_word_title_case(self):
        assert normalize_call_type("Descending moan") == "Descending Moan"
        assert normalize_call_type("Ascending moan") == "Ascending Moan"

    def test_strips_whitespace(self):
        assert normalize_call_type("  Whup ") == "Whup"
        assert normalize_call_type("Grunt\t") == "Grunt"

    def test_typo_corrections(self):
        assert normalize_call_type("Chrip") == "Chirp"
        assert normalize_call_type("chrip") == "Chirp"
        assert normalize_call_type("Whuo") == "Whup"
        assert normalize_call_type("whuo") == "Whup"
        assert normalize_call_type("Piccalo") == "Piccolo"

    def test_empty_passthrough(self):
        assert normalize_call_type("") == ""


class TestAnnotationStem:
    def test_standard_raven_name(self):
        p = Path("211026-133018-OS-humpback-47min-clip.Table.1.selections.txt")
        assert _annotation_stem(p) == "211026-133018-OS-humpback-47min-clip"

    def test_different_table_number(self):
        p = Path("recording.Table.2.selections.txt")
        assert _annotation_stem(p) == "recording"

    def test_no_raven_suffix(self):
        # Should just strip .txt
        p = Path("recording.txt")
        assert _annotation_stem(p) == "recording"


class TestParseRavenTsv:
    def test_parses_standard_format(self, tmp_path):
        content = textwrap.dedent("""\
            Selection\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tCall Type
            1\t245.992\t247.366\t263.518\t2845.995\tDescending moan
            2\t253.277\t257.039\t263.518\t1791.923\tDescending moan
            3\t259.368\t259.836\t2898.699\t3531.142\tWhistle
        """)
        tsv_path = tmp_path / "test.Table.1.selections.txt"
        tsv_path.write_text(content)

        annotations = parse_raven_tsv(tsv_path)
        assert len(annotations) == 3
        assert annotations[0].selection == 1
        assert annotations[0].begin_time == pytest.approx(245.992)
        assert annotations[0].end_time == pytest.approx(247.366)
        assert annotations[0].low_freq == pytest.approx(263.518)
        assert annotations[0].high_freq == pytest.approx(2845.995)
        assert annotations[0].call_type == "Descending Moan"
        assert annotations[2].call_type == "Whistle"

    def test_handles_bom(self, tmp_path):
        # Write with utf-8-sig which adds a BOM prefix automatically
        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        tsv_path = tmp_path / "bom.txt"
        tsv_path.write_text(content, encoding="utf-8-sig")

        annotations = parse_raven_tsv(tsv_path)
        assert len(annotations) == 1
        assert annotations[0].call_type == "Moan"

    def test_skips_empty_call_type(self, tmp_path):
        content = "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\t\n2\t3.0\t4.0\tMoan\n"
        tsv_path = tmp_path / "empty.txt"
        tsv_path.write_text(content)

        annotations = parse_raven_tsv(tsv_path)
        assert len(annotations) == 1
        assert annotations[0].call_type == "Moan"

    def test_missing_required_columns(self, tmp_path):
        content = "Selection\tBegin Time (s)\n1\t1.0\n"
        tsv_path = tmp_path / "bad.txt"
        tsv_path.write_text(content)

        with pytest.raises(ValueError, match="Missing required columns"):
            parse_raven_tsv(tsv_path)

    def test_normalizes_call_types(self, tmp_path):
        content = "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tchrip\n2\t3.0\t4.0\t  whup \n"
        tsv_path = tmp_path / "normalize.txt"
        tsv_path.write_text(content)

        annotations = parse_raven_tsv(tsv_path)
        assert annotations[0].call_type == "Chirp"
        assert annotations[1].call_type == "Whup"

    def test_optional_freq_columns(self, tmp_path):
        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        tsv_path = tmp_path / "no_freq.txt"
        tsv_path.write_text(content)

        annotations = parse_raven_tsv(tsv_path)
        assert len(annotations) == 1
        assert annotations[0].low_freq == 0.0
        assert annotations[0].high_freq == 0.0


class TestPairAnnotationsWithRecordings:
    def test_pairs_by_stem(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        aud_dir = tmp_path / "audio"
        ann_dir.mkdir()
        aud_dir.mkdir()

        # Create annotation file
        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        (ann_dir / "recording1.Table.1.selections.txt").write_text(content)
        (aud_dir / "recording1.flac").write_bytes(b"fake")

        pairs = pair_annotations_with_recordings(ann_dir, aud_dir)
        assert len(pairs) == 1
        assert pairs[0].audio_path.name == "recording1.flac"
        assert len(pairs[0].annotations) == 1

    def test_no_annotations_raises(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        aud_dir = tmp_path / "audio"
        ann_dir.mkdir()
        aud_dir.mkdir()
        (aud_dir / "recording.flac").write_bytes(b"fake")

        with pytest.raises(ValueError, match="No annotation"):
            pair_annotations_with_recordings(ann_dir, aud_dir)

    def test_no_audio_raises(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        aud_dir = tmp_path / "audio"
        ann_dir.mkdir()
        aud_dir.mkdir()
        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        (ann_dir / "test.Table.1.selections.txt").write_text(content)

        with pytest.raises(ValueError, match="No audio"):
            pair_annotations_with_recordings(ann_dir, aud_dir)

    def test_no_matching_pairs_raises(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        aud_dir = tmp_path / "audio"
        ann_dir.mkdir()
        aud_dir.mkdir()
        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        (ann_dir / "file_a.Table.1.selections.txt").write_text(content)
        (aud_dir / "file_b.flac").write_bytes(b"fake")

        with pytest.raises(ValueError, match="No annotation-audio pairs"):
            pair_annotations_with_recordings(ann_dir, aud_dir)

    def test_multiple_pairs(self, tmp_path):
        ann_dir = tmp_path / "annotations"
        aud_dir = tmp_path / "audio"
        ann_dir.mkdir()
        aud_dir.mkdir()

        content = (
            "Selection\tBegin Time (s)\tEnd Time (s)\tCall Type\n1\t1.0\t2.0\tMoan\n"
        )
        for name in ["rec1", "rec2", "rec3"]:
            (ann_dir / f"{name}.Table.1.selections.txt").write_text(content)
            (aud_dir / f"{name}.flac").write_bytes(b"fake")

        pairs = pair_annotations_with_recordings(ann_dir, aud_dir)
        assert len(pairs) == 3
