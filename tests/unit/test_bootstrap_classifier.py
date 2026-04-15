"""Tests for scripts/bootstrap_classifier.py."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.bootstrap_classifier import (
    _BootstrapSample,
    assign_random_types,
    flatten_events,
    run_bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeSample:
    """Mimics SegmentationTrainingSample for unit tests."""

    hydrophone_id: Optional[str]
    start_timestamp: Optional[float]
    end_timestamp: Optional[float]
    crop_start_sec: float
    events_json: str


@dataclass
class _FakeDataset:
    id: str = "ds-1"
    name: str = "test-dataset"


@dataclass
class _FakeVocType:
    name: str


# ---------------------------------------------------------------------------
# flatten_events
# ---------------------------------------------------------------------------


class TestFlattenEvents:
    def test_basic_offset(self):
        sample = _FakeSample(
            hydrophone_id="hydro-1",
            start_timestamp=1000.0,
            end_timestamp=2000.0,
            crop_start_sec=5.0,
            events_json=json.dumps(
                [
                    {"start_sec": 1.0, "end_sec": 2.5},
                    {"start_sec": 3.0, "end_sec": 4.0},
                ]
            ),
        )
        events = flatten_events([sample])  # type: ignore[arg-type]

        assert len(events) == 2
        # First event: 1.0 + 5.0 = 6.0, 2.5 + 5.0 = 7.5
        assert events[0] == (6.0, 7.5, "hydro-1", 1000.0, 2000.0)
        # Second event: 3.0 + 5.0 = 8.0, 4.0 + 5.0 = 9.0
        assert events[1] == (8.0, 9.0, "hydro-1", 1000.0, 2000.0)

    def test_multiple_samples(self):
        s1 = _FakeSample(
            hydrophone_id="h1",
            start_timestamp=100.0,
            end_timestamp=200.0,
            crop_start_sec=0.0,
            events_json=json.dumps([{"start_sec": 1.0, "end_sec": 2.0}]),
        )
        s2 = _FakeSample(
            hydrophone_id="h2",
            start_timestamp=300.0,
            end_timestamp=400.0,
            crop_start_sec=10.0,
            events_json=json.dumps([{"start_sec": 0.5, "end_sec": 1.5}]),
        )
        events = flatten_events([s1, s2])  # type: ignore[arg-type]

        assert len(events) == 2
        assert events[0][2] == "h1"
        assert events[1] == (10.5, 11.5, "h2", 300.0, 400.0)

    def test_empty_events_json(self):
        sample = _FakeSample(
            hydrophone_id="h1",
            start_timestamp=100.0,
            end_timestamp=200.0,
            crop_start_sec=5.0,
            events_json=json.dumps([]),
        )
        events = flatten_events([sample])  # type: ignore[arg-type]
        assert events == []

    def test_skips_non_hydrophone_sample(self):
        sample = _FakeSample(
            hydrophone_id=None,
            start_timestamp=100.0,
            end_timestamp=200.0,
            crop_start_sec=0.0,
            events_json=json.dumps([{"start_sec": 1.0, "end_sec": 2.0}]),
        )
        events = flatten_events([sample])  # type: ignore[arg-type]
        assert events == []

    def test_skips_missing_timestamps(self):
        sample = _FakeSample(
            hydrophone_id="h1",
            start_timestamp=None,
            end_timestamp=200.0,
            crop_start_sec=0.0,
            events_json=json.dumps([{"start_sec": 1.0, "end_sec": 2.0}]),
        )
        events = flatten_events([sample])  # type: ignore[arg-type]
        assert events == []


# ---------------------------------------------------------------------------
# assign_random_types
# ---------------------------------------------------------------------------


class TestAssignRandomTypes:
    def test_guarantees_min_coverage(self):
        assignments = assign_random_types(n_events=100, n_types=5, min_per_type=10)

        assert len(assignments) == 100
        counts = Counter(assignments)
        for type_idx in range(5):
            assert counts[type_idx] >= 10, (
                f"Type {type_idx} has only {counts[type_idx]} events"
            )

    def test_all_indices_valid(self):
        assignments = assign_random_types(n_events=50, n_types=3, min_per_type=5)

        assert len(assignments) == 50
        for idx in assignments:
            assert 0 <= idx < 3

    def test_fewer_events_than_full_coverage(self):
        # 15 events, 5 types, min_per_type=10 → can't guarantee 10 each
        assignments = assign_random_types(n_events=15, n_types=5, min_per_type=10)

        assert len(assignments) == 15
        counts = Counter(assignments)
        # All 5 types should be represented (round-robin assigns 3 of each type)
        assert len(counts) == 5
        for idx in assignments:
            assert 0 <= idx < 5

    def test_zero_events(self):
        assert assign_random_types(n_events=0, n_types=5) == []

    def test_zero_types(self):
        assert assign_random_types(n_events=10, n_types=0) == []

    def test_deterministic_with_seed(self):
        a1 = assign_random_types(n_events=50, n_types=5, seed=123)
        a2 = assign_random_types(n_events=50, n_types=5, seed=123)
        assert a1 == a2

    def test_different_seeds_differ(self):
        a1 = assign_random_types(n_events=50, n_types=5, seed=1)
        a2 = assign_random_types(n_events=50, n_types=5, seed=2)
        assert a1 != a2

    def test_single_type(self):
        assignments = assign_random_types(n_events=20, n_types=1, min_per_type=10)
        assert len(assignments) == 20
        assert all(idx == 0 for idx in assignments)


# ---------------------------------------------------------------------------
# _BootstrapSample interface
# ---------------------------------------------------------------------------


class TestBootstrapSample:
    def test_has_trainer_interface(self):
        sample = _BootstrapSample(
            start_sec=1.0,
            end_sec=2.0,
            type_index=3,
            hydrophone_id="h1",
            start_timestamp=100.0,
            end_timestamp=200.0,
        )
        assert hasattr(sample, "start_sec")
        assert hasattr(sample, "end_sec")
        assert hasattr(sample, "type_index")
        assert hasattr(sample, "hydrophone_id")
        assert hasattr(sample, "start_timestamp")
        assert hasattr(sample, "end_timestamp")
        assert sample.type_index == 3


# ---------------------------------------------------------------------------
# run_bootstrap (mocked DB + trainer)
# ---------------------------------------------------------------------------


class TestRunBootstrap:
    @pytest.fixture()
    def mock_session(self):
        session = AsyncMock()
        return session

    @pytest.fixture()
    def mock_settings(self, tmp_path):
        settings = MagicMock()
        settings.storage_root = tmp_path
        settings.s3_cache_path = None
        settings.noaa_cache_path = None
        return settings

    @pytest.mark.asyncio()
    async def test_missing_dataset_exits(self, mock_session, mock_settings):
        mock_session.get.return_value = None
        with pytest.raises(SystemExit, match="not found"):
            await run_bootstrap(
                mock_session, dataset_id="nonexistent", settings=mock_settings
            )

    @pytest.mark.asyncio()
    async def test_empty_dataset_exits(self, mock_session, mock_settings):
        mock_session.get.return_value = _FakeDataset()

        # First execute call returns empty samples
        empty_result = MagicMock()
        empty_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = empty_result

        with pytest.raises(SystemExit, match="no samples"):
            await run_bootstrap(mock_session, dataset_id="ds-1", settings=mock_settings)

    @pytest.mark.asyncio()
    async def test_no_vocalization_types_exits(self, mock_session, mock_settings):
        mock_session.get.return_value = _FakeDataset()

        samples_result = MagicMock()
        samples_result.scalars.return_value.all.return_value = [
            _FakeSample(
                hydrophone_id="h1",
                start_timestamp=100.0,
                end_timestamp=200.0,
                crop_start_sec=0.0,
                events_json=json.dumps([{"start_sec": 1.0, "end_sec": 2.0}]),
            )
        ]
        types_result = MagicMock()
        types_result.scalars.return_value.all.return_value = []
        mock_session.execute.side_effect = [samples_result, types_result]

        with pytest.raises(SystemExit, match="no vocalization types"):
            await run_bootstrap(mock_session, dataset_id="ds-1", settings=mock_settings)

    @pytest.mark.asyncio()
    async def test_successful_bootstrap(self, mock_session, mock_settings):
        mock_session.get.return_value = _FakeDataset()

        fake_samples = [
            _FakeSample(
                hydrophone_id="h1",
                start_timestamp=100.0,
                end_timestamp=200.0,
                crop_start_sec=5.0,
                events_json=json.dumps(
                    [
                        {"start_sec": 1.0, "end_sec": 2.0},
                        {"start_sec": 3.0, "end_sec": 4.0},
                    ]
                ),
            )
        ]
        samples_result = MagicMock()
        samples_result.scalars.return_value.all.return_value = fake_samples
        types_result = MagicMock()
        types_result.scalars.return_value.all.return_value = [
            _FakeVocType(name="Song"),
            _FakeVocType(name="Call"),
        ]
        mock_session.execute.side_effect = [samples_result, types_result]

        fake_training_result = MagicMock(
            spec=["vocabulary", "per_type_thresholds", "per_type_metrics", "to_summary"]
        )
        fake_training_result.vocabulary = ["Call", "Song"]
        fake_training_result.per_type_thresholds = {"Call": 0.5, "Song": 0.5}
        fake_training_result.per_type_metrics = {
            "Call": {"f1": 0.1},
            "Song": {"f1": 0.1},
        }
        fake_training_result.to_summary.return_value = {"vocabulary": ["Call", "Song"]}

        with (
            patch(
                "scripts.bootstrap_classifier.train_event_classifier",
                return_value=fake_training_result,
            ) as mock_train,
            patch("scripts.bootstrap_classifier.select_device"),
            patch(
                "scripts.bootstrap_classifier._build_audio_loader",
                return_value=lambda s: None,
            ),
        ):
            model_id = await run_bootstrap(
                mock_session, dataset_id="ds-1", settings=mock_settings
            )

        assert model_id is not None
        assert len(model_id) == 36  # UUID format

        # Verify trainer was called with correct args
        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args
        samples_arg = call_kwargs.kwargs.get("samples") or call_kwargs[1].get("samples")
        if samples_arg is None:
            samples_arg = call_kwargs[0][0]
        assert len(samples_arg) == 2
        assert all(hasattr(s, "type_index") for s in samples_arg)

        # Verify model was added to session
        mock_session.add.assert_called_once()
        model = mock_session.add.call_args[0][0]
        assert model.model_family == "pytorch_event_cnn"
        assert model.input_mode == "segmented_event"
