"""Embedding similarity search: brute-force cosine/euclidean over parquet files."""

import asyncio
import heapq
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.audio import AudioFile
from humpback.models.processing import EmbeddingSet
from humpback.processing.embeddings import read_embeddings
from humpback.schemas.search import (
    SimilaritySearchHit,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def _cached_read_embeddings(path: str) -> tuple[np.ndarray, np.ndarray]:
    """LRU-cached wrapper around read_embeddings, keyed by path string."""
    return read_embeddings(Path(path))


def _cosine_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Standard cosine similarity (not mean-centered)."""
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    cand_norms = np.linalg.norm(candidates, axis=1)
    cand_norms = np.maximum(cand_norms, 1e-10)
    return (candidates @ query) / (cand_norms * query_norm)


def _euclidean_scores(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Negative euclidean distance (so higher = more similar for max-heap)."""
    return -np.linalg.norm(candidates - query, axis=1)


def _brute_force_search(
    query: np.ndarray,
    candidate_sets: list[tuple[str, str]],  # (es_id, parquet_path)
    top_k: int,
    metric: str,
) -> tuple[list[tuple[float, str, int]], int]:
    """Search all candidate parquet files, return top-K hits and total count.

    Returns (hits, total_candidates) where hits are (score, es_id, row_index).
    """
    score_fn = _cosine_scores if metric == "cosine" else _euclidean_scores
    heap: list[tuple[float, str, int]] = []
    total = 0

    for es_id, parquet_path in candidate_sets:
        try:
            row_indices, embeddings = _cached_read_embeddings(parquet_path)
        except Exception:
            logger.warning("Failed to read parquet %s, skipping", parquet_path)
            continue

        if embeddings.shape[0] == 0:
            continue

        scores = score_fn(query, embeddings)
        total += len(scores)

        for i, score in enumerate(scores):
            entry = (float(score), es_id, int(row_indices[i]))
            if len(heap) < top_k:
                heapq.heappush(heap, entry)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, entry)

    # Return sorted descending by score
    hits = sorted(heap, key=lambda x: x[0], reverse=True)
    return hits, total


async def similarity_search(
    session: AsyncSession,
    request: SimilaritySearchRequest,
) -> SimilaritySearchResponse:
    """Execute embedding similarity search."""
    # 1. Load query embedding set
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id == request.embedding_set_id)
    )
    query_es = result.scalar_one_or_none()
    if query_es is None:
        raise KeyError(f"Embedding set {request.embedding_set_id!r} not found")

    # 2. Extract query vector
    row_indices, embeddings = await asyncio.to_thread(
        _cached_read_embeddings, query_es.parquet_path
    )
    mask = row_indices == request.row_index
    if not mask.any():
        raise KeyError(
            f"Row index {request.row_index} not found in embedding set "
            f"{request.embedding_set_id!r}"
        )
    query_vector = embeddings[mask][0].astype(np.float32)

    # 3. Find candidate embedding sets matching model_version
    stmt = select(EmbeddingSet).where(
        EmbeddingSet.model_version == query_es.model_version
    )
    if request.embedding_set_ids is not None:
        stmt = stmt.where(EmbeddingSet.id.in_(request.embedding_set_ids))
    if request.exclude_self:
        stmt = stmt.where(EmbeddingSet.id != request.embedding_set_id)

    es_rows = (await session.execute(stmt)).scalars().all()
    candidate_sets = [(es.id, es.parquet_path) for es in es_rows]

    # 4. Brute-force search in thread pool
    hits, total_candidates = await asyncio.to_thread(
        _brute_force_search,
        query_vector,
        candidate_sets,
        request.top_k,
        request.metric,
    )

    # 5. Enrich hits with audio file metadata
    es_id_set = {h[1] for h in hits}
    es_map: dict[str, EmbeddingSet] = {}
    if es_id_set:
        es_result = await session.execute(
            select(EmbeddingSet).where(EmbeddingSet.id.in_(es_id_set))
        )
        for es in es_result.scalars():
            es_map[es.id] = es

    audio_ids = {es.audio_file_id for es in es_map.values()}
    audio_map: dict[str, AudioFile] = {}
    if audio_ids:
        audio_result = await session.execute(
            select(AudioFile).where(AudioFile.id.in_(audio_ids))
        )
        for af in audio_result.scalars():
            audio_map[af.id] = af

    results: list[SimilaritySearchHit] = []
    for score, es_id, row_idx in hits:
        es = es_map.get(es_id)
        if es is None:
            continue
        af = audio_map.get(es.audio_file_id)
        results.append(
            SimilaritySearchHit(
                score=score,
                embedding_set_id=es_id,
                row_index=row_idx,
                audio_file_id=es.audio_file_id,
                audio_filename=af.filename if af else "",
                audio_folder_path=af.folder_path if af else None,
                window_offset_seconds=row_idx * query_es.window_size_seconds,
            )
        )

    return SimilaritySearchResponse(
        query_embedding_set_id=request.embedding_set_id,
        query_row_index=request.row_index,
        model_version=query_es.model_version,
        metric=request.metric,
        total_candidates=total_candidates,
        results=results,
    )
