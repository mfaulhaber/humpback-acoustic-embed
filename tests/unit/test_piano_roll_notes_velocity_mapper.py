"""Unit tests for worker helpers in ``piano_roll_notes_worker``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from humpback.workers.piano_roll_notes_worker import (
    _cleanup_partial_parquet,
    _velocity_mapper,
    _VelocityParams,
)


def test_velocity_mapper_empty_distribution_returns_floor():
    params = _VelocityParams()
    mapper = _velocity_mapper([], params)
    assert mapper(0.0) == params.floor
    assert mapper(1000.0) == params.floor


def test_velocity_mapper_degenerate_distribution_returns_floor():
    params = _VelocityParams()
    mapper = _velocity_mapper([5.0, 5.0, 5.0], params)
    assert mapper(5.0) == params.floor
    assert mapper(50.0) == params.floor


def test_velocity_mapper_nonfinite_percentile_returns_floor():
    params = _VelocityParams()
    mapper = _velocity_mapper([math.nan, math.nan, math.nan], params)
    assert mapper(0.0) == params.floor


def test_velocity_mapper_linear_interpolation_within_range():
    params = _VelocityParams(
        floor_percentile=0.0,
        ceiling_percentile=100.0,
        floor=1,
        ceiling=127,
    )
    distribution = [float(value) for value in range(0, 101)]
    mapper = _velocity_mapper(distribution, params)
    assert mapper(0.0) == params.floor
    assert mapper(100.0) == params.ceiling
    midpoint = mapper(50.0)
    assert 60 <= midpoint <= 70


def test_velocity_mapper_clamps_outside_range():
    params = _VelocityParams(
        floor_percentile=0.0,
        ceiling_percentile=100.0,
        floor=1,
        ceiling=127,
    )
    mapper = _velocity_mapper([0.0, 100.0], params)
    assert mapper(-50.0) == params.floor
    assert mapper(1000.0) == params.ceiling


@pytest.mark.parametrize("floor", [1, 10, 64])
def test_velocity_mapper_respects_custom_floor(floor):
    params = _VelocityParams(floor=floor, ceiling=127)
    mapper = _velocity_mapper([], params)
    assert mapper(0.0) == floor


def test_cleanup_partial_parquet_removes_canonical_and_tmp(tmp_path: Path) -> None:
    canonical = tmp_path / "event_notes_v2.parquet"
    tmp = canonical.with_suffix(canonical.suffix + ".tmp")
    canonical.write_bytes(b"\x00")
    tmp.write_bytes(b"\x00")

    _cleanup_partial_parquet(canonical)

    assert not canonical.exists()
    assert not tmp.exists()


def test_cleanup_partial_parquet_is_no_op_when_missing(tmp_path: Path) -> None:
    canonical = tmp_path / "event_notes_v2.parquet"
    # Should not raise even if neither path exists.
    _cleanup_partial_parquet(canonical)
    assert not canonical.exists()
