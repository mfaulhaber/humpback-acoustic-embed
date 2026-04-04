"""Classifier model CRUD operations."""

import shutil
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierModel


async def list_classifier_models(session: AsyncSession) -> list[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).order_by(ClassifierModel.created_at.desc())
    )
    return list(result.scalars().all())


async def get_classifier_model(
    session: AsyncSession, model_id: str
) -> Optional[ClassifierModel]:
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    return result.scalar_one_or_none()


async def delete_classifier_model(
    session: AsyncSession, model_id: str, storage_root: Path
) -> bool:
    """Delete a classifier model and its files. Returns True if found."""
    result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == model_id)
    )
    cm = result.scalar_one_or_none()
    if cm is None:
        return False

    # Delete files
    from humpback.storage import classifier_dir

    cdir = classifier_dir(storage_root, model_id)
    if cdir.is_dir():
        shutil.rmtree(cdir)

    await session.delete(cm)
    await session.commit()
    return True


async def bulk_delete_classifier_models(
    session: AsyncSession, model_ids: list[str], storage_root: Path
) -> int:
    """Delete multiple classifier models. Returns count of deleted models."""
    count = 0
    for model_id in model_ids:
        if await delete_classifier_model(session, model_id, storage_root):
            count += 1
    return count
