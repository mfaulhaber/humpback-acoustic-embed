"""Service layer for vocalization type vocabulary and training orchestration."""

import json
import logging

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
    VocalizationTrainingJob,
    VocalizationType,
)

logger = logging.getLogger(__name__)


# ---- Vocabulary CRUD ----


async def list_types(session: AsyncSession) -> list[VocalizationType]:
    result = await session.execute(
        select(VocalizationType).order_by(VocalizationType.name)
    )
    return list(result.scalars().all())


async def create_type(
    session: AsyncSession, name: str, description: str | None = None
) -> VocalizationType:
    normalized = name.strip().title()
    vt = VocalizationType(name=normalized, description=description)
    session.add(vt)
    await session.commit()
    await session.refresh(vt)
    return vt


async def update_type(
    session: AsyncSession,
    type_id: str,
    name: str | None = None,
    description: str | None = None,
) -> VocalizationType | None:
    result = await session.execute(
        select(VocalizationType).where(VocalizationType.id == type_id)
    )
    vt = result.scalar_one_or_none()
    if vt is None:
        return None
    if name is not None:
        vt.name = name.strip().title()
    if description is not None:
        vt.description = description
    await session.commit()
    await session.refresh(vt)
    return vt


async def delete_type(session: AsyncSession, type_id: str) -> bool:
    """Delete a vocalization type. Fails if referenced by an active model."""
    result = await session.execute(
        select(VocalizationType).where(VocalizationType.id == type_id)
    )
    vt = result.scalar_one_or_none()
    if vt is None:
        return False

    # Check if type is in any active model's vocabulary
    active_result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.is_active.is_(True)
        )
    )
    for model in active_result.scalars().all():
        vocab: list[str] = json.loads(model.vocabulary_snapshot)
        if vt.name in vocab:
            raise ValueError(
                f"Cannot delete type '{vt.name}' — referenced by active model "
                f"'{model.name}'"
            )

    await session.delete(vt)
    await session.commit()
    return True


# ---- Active Model Management ----


async def activate_model(
    session: AsyncSession, model_id: str
) -> VocalizationClassifierModel | None:
    """Set a model as active, deactivating any previously active model."""
    result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.id == model_id
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        return None

    # Deactivate all models
    await session.execute(update(VocalizationClassifierModel).values(is_active=False))
    # Activate this one
    model.is_active = True
    await session.commit()
    await session.refresh(model)
    return model


# ---- Job Queries ----


async def get_training_job(
    session: AsyncSession, job_id: str
) -> VocalizationTrainingJob | None:
    result = await session.execute(
        select(VocalizationTrainingJob).where(VocalizationTrainingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_inference_job(
    session: AsyncSession, job_id: str
) -> VocalizationInferenceJob | None:
    result = await session.execute(
        select(VocalizationInferenceJob).where(VocalizationInferenceJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def get_model(
    session: AsyncSession, model_id: str
) -> VocalizationClassifierModel | None:
    result = await session.execute(
        select(VocalizationClassifierModel).where(
            VocalizationClassifierModel.id == model_id
        )
    )
    return result.scalar_one_or_none()
