"""Read-time correction overlay for Pass 1 region boundaries.

Loads the raw ``regions.parquet`` produced by a region detection job and
merges any human corrections from the ``region_boundary_corrections``
table — adjusting boundaries, inserting new regions, and removing
deleted regions — without modifying the original parquet file.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import read_regions
from humpback.call_parsing.types import Region
from humpback.models.feedback_training import RegionBoundaryCorrection


async def load_corrected_regions(
    session: AsyncSession,
    region_detection_job_id: str,
    regions_path: Path,
) -> list[Region]:
    """Load regions from parquet and apply SQL corrections.

    Returns a list of ``Region`` objects with corrections merged:
    - ``adjust``: replaces start_sec/end_sec and padded bounds
    - ``add``: inserts a new region with a generated UUID
    - ``delete``: removes the region from the list
    """
    regions = read_regions(regions_path)

    result = await session.execute(
        select(RegionBoundaryCorrection).where(
            RegionBoundaryCorrection.region_detection_job_id == region_detection_job_id
        )
    )
    corrections = list(result.scalars().all())

    if not corrections:
        return regions

    corrections_by_region: dict[str, RegionBoundaryCorrection] = {
        c.region_id: c for c in corrections
    }

    output: list[Region] = []
    for region in regions:
        correction = corrections_by_region.pop(region.region_id, None)
        if correction is None:
            output.append(region)
            continue

        if correction.correction_type == "delete":
            continue
        elif correction.correction_type == "adjust":
            assert correction.start_sec is not None
            assert correction.end_sec is not None
            output.append(
                Region(
                    region_id=region.region_id,
                    start_sec=correction.start_sec,
                    end_sec=correction.end_sec,
                    padded_start_sec=correction.start_sec,
                    padded_end_sec=correction.end_sec,
                    max_score=region.max_score,
                    mean_score=region.mean_score,
                    n_windows=region.n_windows,
                )
            )

    for region_id, correction in corrections_by_region.items():
        if correction.correction_type == "add":
            assert correction.start_sec is not None
            assert correction.end_sec is not None
            output.append(
                Region(
                    region_id=str(uuid.uuid4()),
                    start_sec=correction.start_sec,
                    end_sec=correction.end_sec,
                    padded_start_sec=correction.start_sec,
                    padded_end_sec=correction.end_sec,
                    max_score=0.0,
                    mean_score=0.0,
                    n_windows=0,
                )
            )

    return sorted(output, key=lambda r: r.start_sec)
