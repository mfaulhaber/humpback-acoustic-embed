from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///data/humpback.db"
    storage_root: Path = Path("data")
    model_version: str = "perch_v1"
    window_size_seconds: float = 5.0
    target_sample_rate: int = 32000
    vector_dim: int = 512
    worker_poll_interval: float = 2.0
    use_real_model: bool = False

    model_config = {"env_prefix": "HUMPBACK_"}
