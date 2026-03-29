from pydantic import BaseModel, Field


class SimilaritySearchRequest(BaseModel):
    embedding_set_id: str
    row_index: int = Field(ge=0)
    top_k: int = Field(default=20, ge=1, le=500)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    exclude_self: bool = True
    embedding_set_ids: list[str] | None = None
    search_mode: str = Field(default="raw", pattern="^(raw|projected)$")
    classifier_model_id: str | None = None


class VectorSearchRequest(BaseModel):
    vector: list[float]
    model_version: str
    top_k: int = Field(default=20, ge=1, le=500)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    embedding_set_ids: list[str] | None = None
    search_mode: str = Field(default="raw", pattern="^(raw|projected)$")
    classifier_model_id: str | None = None


class ScoreHistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class ScoreDistribution(BaseModel):
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    histogram: list[ScoreHistogramBin] = Field(default_factory=list)


class SimilaritySearchHit(BaseModel):
    score: float
    percentile_rank: float = 0.0
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
    score_distribution: ScoreDistribution = Field(default_factory=ScoreDistribution)


class AudioSearchRequest(BaseModel):
    detection_job_id: str
    start_utc: float
    end_utc: float
    top_k: int = Field(default=20, ge=1, le=500)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    embedding_set_ids: list[str] | None = None
    search_mode: str = Field(default="raw", pattern="^(raw|projected)$")
    classifier_model_id: str | None = None


class SearchJobResponse(BaseModel):
    id: str
    status: str
    error: str | None = None
    results: SimilaritySearchResponse | None = None
    query_vector: list[float] | None = None
    model_version: str | None = None
