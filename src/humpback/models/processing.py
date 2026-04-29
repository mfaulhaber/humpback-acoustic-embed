"""Retired processing models.

The ``processing_jobs`` and ``embedding_sets`` tables were removed in migration
060. This module remains as a compatibility stub for imports that should be
retired alongside the legacy workflow code.
"""

from __future__ import annotations

import enum


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    complete = "complete"
    failed = "failed"
    canceled = "canceled"


__all__ = ["JobStatus"]
