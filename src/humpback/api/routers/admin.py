from fastapi import APIRouter, HTTPException
from sqlalchemy import func, select, text

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models import (
    AudioFile,
    AudioMetadata,
    Cluster,
    ClusterAssignment,
    ClusteringJob,
    EmbeddingSet,
    ProcessingJob,
    TFLiteModelConfig,
)
from humpback.schemas.model_registry import (
    AvailableModelFile,
    ModelConfigCreate,
    ModelConfigOut,
    ModelConfigUpdate,
)
from humpback.services import model_registry_service

router = APIRouter(prefix="/admin", tags=["admin"])

_MODELS = [
    ("audio_files", AudioFile),
    ("audio_metadata", AudioMetadata),
    ("model_configs", TFLiteModelConfig),
    ("processing_jobs", ProcessingJob),
    ("embedding_sets", EmbeddingSet),
    ("clustering_jobs", ClusteringJob),
    ("clusters", Cluster),
    ("cluster_assignments", ClusterAssignment),
]


# ---- Model Registry Endpoints ----


@router.get("/models", response_model=list[ModelConfigOut])
async def list_models(session: SessionDep):
    models = await model_registry_service.list_models(session)
    return [ModelConfigOut.model_validate(m) for m in models]


@router.post("/models", status_code=201, response_model=ModelConfigOut)
async def create_model(body: ModelConfigCreate, session: SessionDep):
    try:
        model = await model_registry_service.create_model(
            session,
            name=body.name,
            display_name=body.display_name,
            path=body.path,
            vector_dim=body.vector_dim,
            description=body.description,
            is_default=body.is_default,
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    return ModelConfigOut.model_validate(model)


@router.put("/models/{model_id}", response_model=ModelConfigOut)
async def update_model(
    model_id: str, body: ModelConfigUpdate, session: SessionDep
):
    model = await model_registry_service.update_model(
        session,
        model_id,
        display_name=body.display_name,
        vector_dim=body.vector_dim,
        description=body.description,
        is_default=body.is_default,
    )
    if model is None:
        raise HTTPException(404, "Model not found")
    return ModelConfigOut.model_validate(model)


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, session: SessionDep):
    try:
        await model_registry_service.delete_model(session, model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "ok"}


@router.post("/models/{model_id}/set-default", response_model=ModelConfigOut)
async def set_default_model(model_id: str, session: SessionDep):
    model = await model_registry_service.set_default_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Model not found")
    return ModelConfigOut.model_validate(model)


@router.get("/models/scan", response_model=list[AvailableModelFile])
async def scan_models(session: SessionDep, settings: SettingsDep):
    files = await model_registry_service.scan_model_files_with_status(
        settings, session
    )
    return [AvailableModelFile(**f) for f in files]


# ---- Table Management Endpoints ----


@router.get("/tables")
async def list_tables(session: SessionDep):
    """Return row counts for every table."""
    tables = []
    for table_name, model in _MODELS:
        result = await session.execute(select(func.count()).select_from(model))
        count = result.scalar()
        tables.append({"table": table_name, "count": count})
    return tables


@router.delete("/tables")
async def delete_all(session: SessionDep):
    """Delete all rows from every table (order respects FK constraints)."""
    # Delete in reverse order to respect foreign key dependencies
    for _, model in reversed(_MODELS):
        await session.execute(text(f"DELETE FROM {model.__tablename__}"))
    await session.commit()
    return {"status": "ok", "message": "All records deleted"}
