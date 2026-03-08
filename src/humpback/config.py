from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///data/humpback.db"
    storage_root: Path = Path("data")
    model_version: str = "perch_v1"
    window_size_seconds: float = 5.0
    target_sample_rate: int = 32000
    vector_dim: int = 1280
    worker_poll_interval: float = 2.0
    use_real_model: bool = True
    model_path: str = "models/multispecies_whale_fp16_flex.tflite"
    models_dir: str = "models"
    tf_force_cpu: bool = False
    positive_sample_path: str = "/Users/michael/development/data-vocalizations/positives"
    negative_sample_path: str = "/Users/michael/development/data-vocalizations/negatives"

    model_config = {"env_prefix": "HUMPBACK_"}


# ---- Orcasound Hydrophone Configuration ----

ORCASOUND_HYDROPHONES = [
    {"id": "rpi_orcasound_lab", "name": "Orcasound Lab", "location": "San Juan Islands"},
    {"id": "rpi_north_sjc", "name": "North San Juan Channel", "location": "San Juan Channel"},
    {"id": "rpi_port_townsend", "name": "Port Townsend", "location": "Puget Sound"},
    {"id": "rpi_bush_point", "name": "Bush Point", "location": "Whidbey Island"},
]

ORCASOUND_S3_BUCKET = "audio-orcasound-net"

HYDROPHONE_IDS = {h["id"] for h in ORCASOUND_HYDROPHONES}
