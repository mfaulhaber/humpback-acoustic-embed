"""Tests for scripts/noaa_detection_metadata.py."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from scripts.noaa_detection_metadata import (
    PresenceDay,
    _date_to_epoch,
    build_job_payloads,
    build_parser,
    group_consecutive_days,
    load_and_post_job,
    parse_noaa_detection_csv,
    split_into_job_ranges,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = """\
ISOStartTime,Presence
2018-10-31T00:00:00.000Z,0
2018-11-01T00:00:00.000Z,1
2018-11-02T00:00:00.000Z,1
2018-11-03T00:00:00.000Z,1
2018-11-04T00:00:00.000Z,0
2018-11-05T00:00:00.000Z,1
2018-11-06T00:00:00.000Z,1
2018-11-07T00:00:00.000Z,0
2018-11-08T00:00:00.000Z,0
2018-11-09T00:00:00.000Z,1
"""


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


class TestParseNoaaDetectionCsv:
    def test_filters_presence_only(self) -> None:
        days = parse_noaa_detection_csv(SAMPLE_CSV, deployment="01", source="test")
        dates = [d.date for d in days]
        assert date(2018, 10, 31) not in dates  # Presence=0
        assert date(2018, 11, 1) in dates  # Presence=1
        assert date(2018, 11, 4) not in dates  # Presence=0
        assert len(days) == 6

    def test_handles_bom_and_cr_line_endings(self) -> None:
        """Real NOAA CSVs have BOM + \\r\\n line endings."""
        csv_text = (
            "\ufeffISOStartTime,Presence\r\n"
            "2018-11-01T00:00:00.000Z,1\r\n"
            "2018-11-02T00:00:00.000Z,0\r\n"
        )
        days = parse_noaa_detection_csv(csv_text)
        assert len(days) == 1
        assert days[0].date == date(2018, 11, 1)

    def test_empty_csv(self) -> None:
        days = parse_noaa_detection_csv("ISOStartTime,Presence\n")
        assert days == []

    def test_all_zero_presence(self) -> None:
        csv_text = (
            "ISOStartTime,Presence\n"
            "2018-11-01T00:00:00.000Z,0\n"
            "2018-11-02T00:00:00.000Z,0\n"
        )
        days = parse_noaa_detection_csv(csv_text)
        assert days == []

    def test_deployment_and_source_propagated(self) -> None:
        csv_text = "ISOStartTime,Presence\n2018-11-01T00:00:00.000Z,1\n"
        days = parse_noaa_detection_csv(csv_text, deployment="03", source="gcs://foo")
        assert len(days) == 1
        assert days[0].deployment == "03"
        assert days[0].source == "gcs://foo"

    def test_mixed_case_column_header(self) -> None:
        """OC01 CSVs use 'IsoStartTime' instead of 'ISOStartTime'."""
        csv_text = (
            "IsoStartTime,Presence\n"
            "2020-11-02T00:00:00,1\n"
            "2020-11-03T00:00:00,0\n"
            "2020-11-04T00:00:00,1\n"
        )
        days = parse_noaa_detection_csv(csv_text)
        assert len(days) == 2
        assert days[0].date == date(2020, 11, 2)
        assert days[1].date == date(2020, 11, 4)

    def test_timestamp_without_z_suffix(self) -> None:
        """OC01 CSVs omit the Z timezone suffix: '2020-11-02T00:00:00'."""
        csv_text = "IsoStartTime,Presence\n2020-11-02T00:00:00,1\n"
        days = parse_noaa_detection_csv(csv_text)
        assert len(days) == 1
        assert days[0].date == date(2020, 11, 2)

    def test_sorted_output(self) -> None:
        csv_text = (
            "ISOStartTime,Presence\n"
            "2018-11-05T00:00:00.000Z,1\n"
            "2018-11-01T00:00:00.000Z,1\n"
            "2018-11-03T00:00:00.000Z,1\n"
        )
        days = parse_noaa_detection_csv(csv_text)
        dates = [d.date for d in days]
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGroupConsecutiveDays:
    def test_single_run(self) -> None:
        days = [
            PresenceDay(date(2018, 11, 1), "01", ""),
            PresenceDay(date(2018, 11, 2), "01", ""),
            PresenceDay(date(2018, 11, 3), "01", ""),
        ]
        groups = group_consecutive_days(days)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_multiple_runs_with_gaps(self) -> None:
        days = [
            PresenceDay(date(2018, 11, 1), "01", ""),
            PresenceDay(date(2018, 11, 2), "01", ""),
            # gap: Nov 3 missing
            PresenceDay(date(2018, 11, 5), "01", ""),
            PresenceDay(date(2018, 11, 6), "01", ""),
        ]
        groups = group_consecutive_days(days)
        assert len(groups) == 2
        assert len(groups[0]) == 2  # Nov 1-2
        assert len(groups[1]) == 2  # Nov 5-6

    def test_empty_input(self) -> None:
        assert group_consecutive_days([]) == []

    def test_single_day(self) -> None:
        days = [PresenceDay(date(2018, 11, 1), "01", "")]
        groups = group_consecutive_days(days)
        assert len(groups) == 1
        assert len(groups[0]) == 1


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


class TestSplitIntoJobRanges:
    def test_default_one_per_day(self) -> None:
        days = [
            PresenceDay(date(2018, 11, 1), "01", ""),
            PresenceDay(date(2018, 11, 2), "01", ""),
            PresenceDay(date(2018, 11, 3), "01", ""),
        ]
        groups = [days]
        ranges = split_into_job_ranges(groups, days_per_job=1)
        assert len(ranges) == 3
        for start, end, chunk in ranges:
            assert start == end
            assert len(chunk) == 1

    def test_days_per_job_consolidation(self) -> None:
        days = [PresenceDay(date(2018, 11, i), "01", "") for i in range(1, 15)]
        groups = [days]  # 14 consecutive days
        ranges = split_into_job_ranges(groups, days_per_job=7)
        assert len(ranges) == 2
        assert len(ranges[0][2]) == 7
        assert len(ranges[1][2]) == 7

    def test_capped_at_7_days(self) -> None:
        """days_per_job > 7 is capped to 7 by API constraint."""
        days = [PresenceDay(date(2018, 11, i), "01", "") for i in range(1, 15)]
        groups = [days]
        ranges = split_into_job_ranges(groups, days_per_job=14)
        assert len(ranges) == 2  # still splits at 7
        assert len(ranges[0][2]) == 7

    def test_short_run_preserved(self) -> None:
        days = [
            PresenceDay(date(2018, 11, 1), "01", ""),
            PresenceDay(date(2018, 11, 2), "01", ""),
            PresenceDay(date(2018, 11, 3), "01", ""),
        ]
        groups = [days]
        ranges = split_into_job_ranges(groups, days_per_job=7)
        assert len(ranges) == 1
        assert len(ranges[0][2]) == 3


# ---------------------------------------------------------------------------
# Payload generation
# ---------------------------------------------------------------------------


class TestBuildJobPayloads:
    def test_timestamps_are_utc_day_boundaries(self) -> None:
        days = [PresenceDay(date(2018, 11, 1), "01", "")]
        groups = [[days[0]]]
        ranges = split_into_job_ranges(groups, days_per_job=1)
        payloads = build_job_payloads(ranges, classifier_model_id="test-id")
        p = payloads[0]
        # Nov 1 00:00:00 UTC
        assert p["start_timestamp"] == _date_to_epoch(date(2018, 11, 1))
        # Nov 2 00:00:00 UTC (exclusive end)
        assert p["end_timestamp"] == _date_to_epoch(date(2018, 11, 2))

    def test_multi_day_range_timestamps(self) -> None:
        days = [
            PresenceDay(date(2018, 11, 1), "01", ""),
            PresenceDay(date(2018, 11, 2), "01", ""),
            PresenceDay(date(2018, 11, 3), "01", ""),
        ]
        groups = [days]
        ranges = split_into_job_ranges(groups, days_per_job=7)
        payloads = build_job_payloads(ranges, classifier_model_id="test-id")
        p = payloads[0]
        assert p["start_timestamp"] == _date_to_epoch(date(2018, 11, 1))
        assert p["end_timestamp"] == _date_to_epoch(date(2018, 11, 4))

    def test_metadata_included(self) -> None:
        days = [PresenceDay(date(2018, 11, 1), "01", "")]
        ranges = split_into_job_ranges([[days[0]]], days_per_job=1)
        payloads = build_job_payloads(ranges, classifier_model_id="test-id")
        meta = payloads[0]["_metadata"]
        assert meta["index"] == 0
        assert meta["presence_days"] == 1
        assert meta["deployment"] == "01"
        assert "2018-11-01" in meta["date_range"]

    def test_thresholds_passed_through(self) -> None:
        days = [PresenceDay(date(2018, 11, 1), "01", "")]
        ranges = split_into_job_ranges([[days[0]]], days_per_job=1)
        payloads = build_job_payloads(
            ranges,
            classifier_model_id="test-id",
            high_threshold=0.80,
            low_threshold=0.50,
            hop_seconds=0.5,
        )
        p = payloads[0]
        assert p["high_threshold"] == 0.80
        assert p["low_threshold"] == 0.50
        assert p["hop_seconds"] == 0.5

    def test_range_within_7_day_api_limit(self) -> None:
        """Every generated payload must have end - start <= 7 days."""
        days = [PresenceDay(date(2018, 11, i), "01", "") for i in range(1, 21)]
        groups = group_consecutive_days(days)
        ranges = split_into_job_ranges(groups, days_per_job=7)
        payloads = build_job_payloads(ranges, classifier_model_id="test-id")
        max_range = 7 * 24 * 3600
        for p in payloads:
            assert p["end_timestamp"] - p["start_timestamp"] <= max_range

    def test_index_sequential(self) -> None:
        days = [PresenceDay(date(2018, 11, i), "01", "") for i in range(1, 6)]
        ranges = split_into_job_ranges([[d] for d in days], days_per_job=1)
        payloads = build_job_payloads(ranges, classifier_model_id="test-id")
        indices = [p["_metadata"]["index"] for p in payloads]
        assert indices == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Epoch conversion
# ---------------------------------------------------------------------------


class TestDateToEpoch:
    def test_known_epoch(self) -> None:
        # 2018-11-01 00:00:00 UTC = 1541030400
        assert _date_to_epoch(date(2018, 11, 1)) == 1541030400.0

    def test_utc_semantics(self) -> None:
        d = date(2020, 1, 1)
        expected = datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()
        assert _date_to_epoch(d) == expected


# ---------------------------------------------------------------------------
# load_and_post_job validation
# ---------------------------------------------------------------------------


class TestLoadAndPostJob:
    def test_file_not_found(self, tmp_path: object) -> None:
        with pytest.raises(FileNotFoundError):
            load_and_post_job("/nonexistent/path.json", 0, "http://localhost:8000")

    def test_index_out_of_range(self, tmp_path: object) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"jobs": [{"classifier_model_id": "a"}]}, f)
            f.flush()
            with pytest.raises(IndexError, match="out of range"):
                load_and_post_job(f.name, 5, "http://localhost:8000")

    def test_empty_jobs(self, tmp_path: object) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"jobs": []}, f)
            f.flush()
            with pytest.raises(ValueError, match="No jobs"):
                load_and_post_job(f.name, 0, "http://localhost:8000")


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--classifier-model-id", "test-id"])
        assert args.classifier_model_id == "test-id"
        assert args.hydrophone_id is None
        assert args.csv_url is None
        assert args.deployment == "01"
        assert args.days_per_job == 1
        assert args.strategy == "consecutive"
        assert args.high_threshold == 0.70
        assert args.low_threshold == 0.45
        assert args.hop_seconds == 1.0
        assert args.output == "detection_jobs.json"
        assert args.post is False
        assert args.job_index is None
        assert args.api_url == "http://localhost:8000"
        assert args.dry_run is False

    def test_csv_url_arg(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--csv-url",
                "https://example.com/SanctSound_OC01_03_humpbackwhale_1d.csv",
                "--hydrophone-id",
                "sanctsound_oc01",
                "--deployment",
                "03",
            ]
        )
        assert (
            args.csv_url
            == "https://example.com/SanctSound_OC01_03_humpbackwhale_1d.csv"
        )
        assert args.hydrophone_id == "sanctsound_oc01"
        assert args.deployment == "03"

    def test_post_mode_args(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--post", "--job-index", "3", "--output", "my.json"])
        assert args.post is True
        assert args.job_index == 3
        assert args.output == "my.json"
