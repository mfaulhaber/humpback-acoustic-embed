from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_env_file() -> Path | None:
    env_file = REPO_ROOT / ".env"
    return env_file if env_file.is_file() else None


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///data/humpback.db"
    storage_root: Path = Path("data")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_hosts: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    model_version: str = "perch_v1"
    window_size_seconds: float = 5.0
    target_sample_rate: int = 32000
    vector_dim: int = 1280
    worker_poll_interval: float = 2.0
    use_real_model: bool = True
    model_path: str = "models/multispecies_whale_fp16_flex.tflite"
    models_dir: str = "models"
    tf_force_cpu: bool = False
    positive_sample_path: str | None = None
    negative_sample_path: str | None = None
    s3_cache_path: str | None = None
    hydrophone_timeline_lookback_increment_hours: int = 4
    hydrophone_timeline_max_lookback_hours: int = 7 * 24
    hydrophone_prefetch_enabled: bool = True
    hydrophone_prefetch_workers: int = 4
    hydrophone_prefetch_inflight_segments: int = 16

    # Spectrogram popup settings
    spectrogram_hop_length: int = 256
    spectrogram_dynamic_range_db: float = 80.0
    spectrogram_width_px: int = 640
    spectrogram_height_px: int = 320
    spectrogram_cache_max_items: int = 1000

    @classmethod
    def from_repo_env(cls, **kwargs) -> "Settings":
        return cast(Any, cls)(_env_file=_repo_env_file(), **kwargs)

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def _parse_allowed_hosts(cls, value: object) -> list[str]:
        if value is None:
            return ["*"]
        if isinstance(value, str):
            hosts = [part.strip() for part in value.split(",") if part.strip()]
            return hosts or ["*"]
        if isinstance(value, (list, tuple)):
            return list(value)
        raise TypeError("allowed_hosts must be a comma-separated string or list")

    @model_validator(mode="after")
    def _apply_defaults_and_validate(self):
        if self.positive_sample_path is None:
            self.positive_sample_path = str(self.storage_root / "labeled" / "positives")
        if self.negative_sample_path is None:
            self.negative_sample_path = str(self.storage_root / "labeled" / "negatives")
        if self.s3_cache_path is None:
            self.s3_cache_path = str(self.storage_root / "s3-orcasound-cache")
        if self.api_port <= 0:
            raise ValueError("api_port must be > 0")
        if self.hydrophone_timeline_lookback_increment_hours <= 0:
            raise ValueError("hydrophone_timeline_lookback_increment_hours must be > 0")
        if self.hydrophone_timeline_max_lookback_hours <= 0:
            raise ValueError("hydrophone_timeline_max_lookback_hours must be > 0")
        if (
            self.hydrophone_timeline_max_lookback_hours
            < self.hydrophone_timeline_lookback_increment_hours
        ):
            raise ValueError(
                "hydrophone_timeline_max_lookback_hours must be >= "
                "hydrophone_timeline_lookback_increment_hours"
            )
        if self.hydrophone_prefetch_workers <= 0:
            raise ValueError("hydrophone_prefetch_workers must be > 0")
        if self.hydrophone_prefetch_inflight_segments <= 0:
            raise ValueError("hydrophone_prefetch_inflight_segments must be > 0")
        return self

    model_config = SettingsConfigDict(
        env_prefix="HUMPBACK_",
        extra="ignore",
    )


# ---- Orcasound Hydrophone Configuration ----

ORCASOUND_HYDROPHONES = [
    {
        "id": "rpi_orcasound_lab",
        "name": "Orcasound Lab",
        "location": "San Juan Islands",
        "provider_kind": "orcasound_hls",
    },
    {
        "id": "rpi_north_sjc",
        "name": "North San Juan Channel",
        "location": "San Juan Channel",
        "provider_kind": "orcasound_hls",
    },
    {
        "id": "rpi_port_townsend",
        "name": "Port Townsend",
        "location": "Puget Sound",
        "provider_kind": "orcasound_hls",
    },
    {
        "id": "rpi_bush_point",
        "name": "Bush Point",
        "location": "Whidbey Island",
        "provider_kind": "orcasound_hls",
    },
]

NOAA_ARCHIVE_SOURCES = [
    {
        "id": "noaa_glacier_bay",
        "name": "NOAA Glacier Bay (Bartlett Cove)",
        "location": "Glacier Bay, Alaska",
        "provider_kind": "noaa_gcs",
        "bucket": "noaa-passive-bioacoustic",
        "prefix": (
            "nps/audio/glacier_bay/bartlettcove/"
            "glacierbay_bartlettcove_jul-oct2015/audio/"
        ),
    }
]

ORCASOUND_S3_BUCKET = "audio-orcasound-net"

HYDROPHONE_IDS = {h["id"] for h in ORCASOUND_HYDROPHONES}
ARCHIVE_SOURCES = [*ORCASOUND_HYDROPHONES, *NOAA_ARCHIVE_SOURCES]
ARCHIVE_SOURCE_IDS = {source["id"] for source in ARCHIVE_SOURCES}


def get_archive_source(source_id: str) -> dict[str, str] | None:
    return next(
        (source for source in ARCHIVE_SOURCES if source["id"] == source_id),
        None,
    )
