"""Retire classifier retrain workflows that depended on legacy processing."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.retrain import RetrainWorkflow

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "importing", "processing", "training")
RETIRED_MESSAGE = (
    "Classifier retrain is retired because it depended on the legacy "
    "audio/processing workflow"
)


async def poll_retrain_workflows(
    session: AsyncSession,
    settings: Settings,
    session_factory,
) -> bool:
    """Fail any active retrain workflows with the retirement reason."""
    del settings, session_factory

    result = await session.execute(
        select(RetrainWorkflow.id).where(RetrainWorkflow.status.in_(ACTIVE_STATUSES))
    )
    workflow_ids = [row_id for row_id in result.scalars().all()]
    if not workflow_ids:
        return False

    logger.info("Retiring %d classifier retrain workflow(s)", len(workflow_ids))
    await session.execute(
        update(RetrainWorkflow)
        .where(RetrainWorkflow.id.in_(workflow_ids))
        .values(
            status="failed",
            error_message=RETIRED_MESSAGE,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return True
