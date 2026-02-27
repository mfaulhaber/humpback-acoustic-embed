from fastapi import APIRouter
from sqlalchemy import func, select, text

from humpback.api.deps import SessionDep
from humpback.models import (
    AudioFile,
    AudioMetadata,
    Cluster,
    ClusterAssignment,
    ClusteringJob,
    EmbeddingSet,
    ProcessingJob,
)

router = APIRouter(prefix="/admin", tags=["admin"])

_MODELS = [
    ("audio_files", AudioFile),
    ("audio_metadata", AudioMetadata),
    ("processing_jobs", ProcessingJob),
    ("embedding_sets", EmbeddingSet),
    ("clustering_jobs", ClusteringJob),
    ("clusters", Cluster),
    ("cluster_assignments", ClusterAssignment),
]


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
