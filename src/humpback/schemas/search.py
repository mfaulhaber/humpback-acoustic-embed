from pydantic import BaseModel, Field


class SimilaritySearchRequest(BaseModel):
    embedding_set_id: str
    row_index: int = Field(ge=0)
    top_k: int = Field(default=20, ge=1, le=500)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    exclude_self: bool = True
    embedding_set_ids: list[str] | None = None


class VectorSearchRequest(BaseModel):
    vector: list[float]
    model_version: str
    top_k: int = Field(default=20, ge=1, le=500)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    embedding_set_ids: list[str] | None = None


class SimilaritySearchHit(BaseModel):
    score: float
    embedding_set_id: str
    row_index: int
    audio_file_id: str
    audio_filename: str
    audio_folder_path: str | None
    window_offset_seconds: float


class SimilaritySearchResponse(BaseModel):
    query_embedding_set_id: str
    query_row_index: int
    model_version: str
    metric: str
    total_candidates: int
    results: list[SimilaritySearchHit]
