"""Unit tests for the window classification sidecar worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from humpback.call_parsing.types import Region
from humpback.workers.window_classification_worker import (
    _select_windows_for_regions,
    _write_window_scores,
)


def _make_region(region_id: str, start: float, end: float, pad: float = 1.0) -> Region:
    return Region(
        region_id=region_id,
        start_sec=start,
        end_sec=end,
        padded_start_sec=max(0.0, start - pad),
        padded_end_sec=end + pad,
        max_score=0.9,
        mean_score=0.7,
        n_windows=3,
    )


def test_select_windows_includes_center_within_padded_bounds() -> None:
    times = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    embs = np.random.randn(7, 8).astype(np.float32)
    region = _make_region("r1", start=1.0, end=5.0, pad=1.0)

    sel_times, sel_regions, sel_embs = _select_windows_for_regions(
        times, embs, [region]
    )

    # Window centers: 2.5, 3.5, 4.5, 5.5 for times 0-3
    # padded bounds: [0.0, 6.0]
    # All of 0..3 have centers 2.5..5.5 which fall in [0.0, 6.0]
    # times 10..12 have centers 12.5..14.5 — outside
    assert len(sel_times) == 4
    assert all(r == "r1" for r in sel_regions)
    assert sel_embs.shape == (4, 8)


def test_select_windows_assigns_to_multiple_overlapping_regions() -> None:
    times = [5.0]
    embs = np.random.randn(1, 4).astype(np.float32)
    r1 = _make_region("r1", start=3.0, end=9.0, pad=1.0)
    r2 = _make_region("r2", start=6.0, end=12.0, pad=1.0)

    sel_times, sel_regions, sel_embs = _select_windows_for_regions(
        times, embs, [r1, r2]
    )

    # center=7.5 falls in both [2.0, 10.0] and [5.0, 13.0]
    assert len(sel_times) == 2
    assert set(sel_regions) == {"r1", "r2"}


def test_select_windows_empty_when_no_overlap() -> None:
    times = [100.0]
    embs = np.random.randn(1, 4).astype(np.float32)
    region = _make_region("r1", start=0.0, end=5.0)

    sel_times, sel_regions, sel_embs = _select_windows_for_regions(
        times, embs, [region]
    )

    assert len(sel_times) == 0
    assert sel_embs.shape[0] == 0


def test_write_window_scores_creates_wide_parquet(tmp_path: Path) -> None:
    path = tmp_path / "window_scores.parquet"
    times = [0.0, 1.0, 2.0]
    regions = ["r1", "r1", "r2"]
    scores = {
        "whup": np.array([0.9, 0.1, 0.5]),
        "moan": np.array([0.2, 0.8, 0.3]),
    }
    vocabulary = ["whup", "moan"]

    _write_window_scores(path, times, regions, scores, vocabulary)

    assert path.exists()
    table = pq.read_table(path)
    assert set(table.column_names) == {"time_sec", "region_id", "whup", "moan"}
    assert table.num_rows == 3


def test_write_window_scores_atomic_no_tmp_left(tmp_path: Path) -> None:
    path = tmp_path / "window_scores.parquet"
    _write_window_scores(path, [1.0], ["r1"], {"t": np.array([0.5])}, ["t"])
    assert not path.with_suffix(".parquet.tmp").exists()
