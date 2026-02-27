"""Service layer for TFLite model registry CRUD and file scanning."""

from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.model_registry import TFLiteModelConfig
from humpback.models.processing import EmbeddingSet


async def list_models(session: AsyncSession) -> list[TFLiteModelConfig]:
    result = await session.execute(
        select(TFLiteModelConfig).order_by(TFLiteModelConfig.name)
    )
    return list(result.scalars().all())


async def get_model_by_name(
    session: AsyncSession, name: str
) -> Optional[TFLiteModelConfig]:
    result = await session.execute(
        select(TFLiteModelConfig).where(TFLiteModelConfig.name == name)
    )
    return result.scalar_one_or_none()


async def get_model_by_id(
    session: AsyncSession, model_id: str
) -> Optional[TFLiteModelConfig]:
    result = await session.execute(
        select(TFLiteModelConfig).where(TFLiteModelConfig.id == model_id)
    )
    return result.scalar_one_or_none()


async def get_default_model(session: AsyncSession) -> Optional[TFLiteModelConfig]:
    result = await session.execute(
        select(TFLiteModelConfig).where(TFLiteModelConfig.is_default == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


async def create_model(
    session: AsyncSession,
    *,
    name: str,
    display_name: str,
    path: str,
    vector_dim: int = 1280,
    description: Optional[str] = None,
    is_default: bool = False,
) -> TFLiteModelConfig:
    if is_default:
        await _clear_defaults(session)

    model = TFLiteModelConfig(
        name=name,
        display_name=display_name,
        path=path,
        vector_dim=vector_dim,
        description=description,
        is_default=is_default,
    )
    session.add(model)
    await session.commit()
    return model


async def update_model(
    session: AsyncSession,
    model_id: str,
    *,
    display_name: Optional[str] = None,
    vector_dim: Optional[int] = None,
    description: Optional[str] = None,
    is_default: Optional[bool] = None,
) -> Optional[TFLiteModelConfig]:
    model = await get_model_by_id(session, model_id)
    if model is None:
        return None

    if display_name is not None:
        model.display_name = display_name
    if vector_dim is not None:
        model.vector_dim = vector_dim
    if description is not None:
        model.description = description
    if is_default is not None:
        if is_default:
            await _clear_defaults(session)
        model.is_default = is_default

    await session.commit()
    return model


async def set_default_model(
    session: AsyncSession, model_id: str
) -> Optional[TFLiteModelConfig]:
    model = await get_model_by_id(session, model_id)
    if model is None:
        return None
    await _clear_defaults(session)
    model.is_default = True
    await session.commit()
    return model


async def delete_model(session: AsyncSession, model_id: str) -> None:
    """Delete a model config. Raises ValueError if embeddings reference it."""
    model = await get_model_by_id(session, model_id)
    if model is None:
        raise ValueError("Model not found")

    # Check if any embedding sets reference this model_version
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.model_version == model.name).limit(1)
    )
    if result.scalar_one_or_none():
        raise ValueError(
            f"Cannot delete model '{model.name}': embedding sets reference it"
        )

    await session.delete(model)
    await session.commit()


def scan_model_files(settings: Settings) -> list[dict]:
    """Scan models directory for .tflite files. Returns list of file info dicts."""
    models_dir = Path(settings.models_dir)
    files = []
    if models_dir.is_dir():
        for p in sorted(models_dir.glob("*.tflite")):
            files.append({
                "filename": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            })
    return files


async def scan_model_files_with_status(
    settings: Settings, session: AsyncSession
) -> list[dict]:
    """Scan files and flag which are already registered."""
    files = scan_model_files(settings)
    registered = await list_models(session)
    registered_paths = {m.path for m in registered}
    for f in files:
        f["registered"] = f["path"] in registered_paths
    return files


async def seed_default_model(session: AsyncSession) -> None:
    """Insert the default model if the table is empty."""
    result = await session.execute(select(TFLiteModelConfig).limit(1))
    if result.scalar_one_or_none() is not None:
        return  # Table already has entries

    await create_model(
        session,
        name="multispecies_whale_fp16",
        display_name="Multispecies Whale FP16 Flex",
        path="models/multispecies_whale_fp16_flex.tflite",
        vector_dim=1280,
        is_default=True,
    )


async def _clear_defaults(session: AsyncSession) -> None:
    """Clear is_default on all models."""
    result = await session.execute(
        select(TFLiteModelConfig).where(TFLiteModelConfig.is_default == True)  # noqa: E712
    )
    for m in result.scalars().all():
        m.is_default = False
