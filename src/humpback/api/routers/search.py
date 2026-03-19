from fastapi import APIRouter, HTTPException

from humpback.api.deps import SessionDep
from humpback.schemas.search import SimilaritySearchRequest, SimilaritySearchResponse
from humpback.services.search_service import similarity_search

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
