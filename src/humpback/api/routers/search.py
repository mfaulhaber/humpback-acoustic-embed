import json

import numpy as np
from fastapi import APIRouter, HTTPException
from sqlalchemy import delete, select

from humpback.api.deps import SessionDep
from humpback.models.classifier import DetectionJob
from humpback.models.search import SearchJob
from humpback.schemas.search import (
    AudioSearchRequest,
    SearchJobResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    VectorSearchRequest,
)
from humpback.services.search_service import (
    similarity_search,
    similarity_search_by_vector,
)

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/similar", response_model=SimilaritySearchResponse)
async def search_similar(
    request: SimilaritySearchRequest,
    session: SessionDep,
) -> SimilaritySearchResponse:
    """Find the top-K most similar embeddings to a query window."""
    try:
        return await similarity_search(session, request)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/similar-by-vector", response_model=SimilaritySearchResponse)
async def search_similar_by_vector(
    request: VectorSearchRequest,
    session: SessionDep,
) -> SimilaritySearchResponse:
    """Find the top-K most similar embeddings to a raw query vector."""
    try:
        return await similarity_search_by_vector(session, request)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/similar-by-audio", response_model=SearchJobResponse, status_code=201)
async def create_audio_search(
    request: AudioSearchRequest,
    session: SessionDep,
) -> SearchJobResponse:
    """Queue a search job that encodes detection audio via the worker."""
    # Validate detection job exists
    det_result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == request.detection_job_id)
    )
    if det_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=404,
            detail=f"Detection job {request.detection_job_id} not found",
        )

    job = SearchJob(
        detection_job_id=request.detection_job_id,
        filename=request.filename,
        start_sec=request.start_sec,
        end_sec=request.end_sec,
        top_k=request.top_k,
        metric=request.metric,
        embedding_set_ids=(
            json.dumps(request.embedding_set_ids) if request.embedding_set_ids else None
        ),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    return SearchJobResponse(id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=SearchJobResponse)
async def get_search_job(
    job_id: str,
    session: SessionDep,
) -> SearchJobResponse:
    """Poll a search job for results.

    When complete, runs the similarity search synchronously, returns results,
    and deletes the ephemeral search job row.
    """
    result = await session.execute(select(SearchJob).where(SearchJob.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Search job not found")

    if job.status in ("queued", "running"):
        return SearchJobResponse(id=job.id, status=job.status)

    if job.status == "failed":
        return SearchJobResponse(id=job.id, status="failed", error=job.error_message)

    # status == "complete": run search and clean up
    if not job.embedding_vector or not job.model_version:
        return SearchJobResponse(
            id=job.id, status="failed", error="Missing embedding data"
        )

    vector = np.asarray(json.loads(job.embedding_vector), dtype=np.float32)
    embedding_set_ids = (
        json.loads(job.embedding_set_ids) if job.embedding_set_ids else None
    )

    search_request = VectorSearchRequest(
        vector=vector.tolist(),
        model_version=job.model_version,
        top_k=job.top_k,
        metric=job.metric,
        embedding_set_ids=embedding_set_ids,
    )

    try:
        search_results = await similarity_search_by_vector(session, search_request)
    except (KeyError, ValueError) as e:
        return SearchJobResponse(id=job.id, status="failed", error=str(e))

    # Clean up ephemeral job row
    await session.execute(delete(SearchJob).where(SearchJob.id == job_id))
    await session.commit()

    return SearchJobResponse(
        id=job.id,
        status="complete",
        results=search_results,
    )
