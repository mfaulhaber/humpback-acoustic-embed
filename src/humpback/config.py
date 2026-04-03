import json
import re
from importlib.resources import files
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
    noaa_cache_path: str | None = None
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

    # Timeline viewer settings
    timeline_tile_width_px: int = 512
    timeline_tile_height_px: int = 256
    timeline_cache_max_jobs: int = 15
    timeline_dynamic_range_db: float = 80.0
    timeline_prepare_workers: int = 2
    timeline_startup_radius_tiles: int = 2
    timeline_startup_coarse_levels: int = 1
    timeline_neighbor_prefetch_radius: int = 1
    timeline_tile_memory_cache_items: int = 256
    timeline_manifest_memory_cache_items: int = 8
    timeline_pcm_memory_cache_mb: int = 128

    # Replay verification settings
    replay_metric_tolerance: float = 0.01

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
        if self.noaa_cache_path is None:
            self.noaa_cache_path = str(self.storage_root / "noaa-gcs-cache")
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
        if self.timeline_prepare_workers <= 0:
            raise ValueError("timeline_prepare_workers must be > 0")
        if self.timeline_startup_radius_tiles < 0:
            raise ValueError("timeline_startup_radius_tiles must be >= 0")
        if self.timeline_startup_coarse_levels < 0:
            raise ValueError("timeline_startup_coarse_levels must be >= 0")
        if self.timeline_neighbor_prefetch_radius < 0:
            raise ValueError("timeline_neighbor_prefetch_radius must be >= 0")
        if self.timeline_tile_memory_cache_items < 0:
            raise ValueError("timeline_tile_memory_cache_items must be >= 0")
        if self.timeline_manifest_memory_cache_items < 0:
            raise ValueError("timeline_manifest_memory_cache_items must be >= 0")
        if self.timeline_pcm_memory_cache_mb < 0:
            raise ValueError("timeline_pcm_memory_cache_mb must be >= 0")
        return self

    model_config = SettingsConfigDict(
        env_prefix="HUMPBACK_",
        extra="ignore",
    )


# ---- Archive Source Configuration ----

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


_SANCTSOUND_SITE_CODE_RE = re.compile(r"^[a-z]{2}\d{2}$")


def _validate_noaa_archive_metadata(records: list[dict[str, Any]]) -> None:
    """Validate metadata invariants for loadable NOAA archive sources."""
    for record in records:
        if record.get("program") != "SanctSound":
            continue
        code = record.get("code")
        if (
            not isinstance(code, str)
            or _SANCTSOUND_SITE_CODE_RE.fullmatch(code) is None
        ):
            continue

        for hint in record.get("child_folder_hints") or []:
            prefix = hint.get("prefix") if isinstance(hint, dict) else None
            if not isinstance(prefix, str) or not prefix.strip():
                continue
            normalized = prefix.strip("/")
            if normalized.startswith(f"{code}/") or f"_{code}_" in normalized:
                continue
            raise ValueError(
                "SanctSound site-scoped record "
                f"{record.get('id', '(unknown)')} contains a child-folder hint "
                f"outside site code {code}: {prefix}"
            )


def _load_noaa_archive_metadata() -> list[dict[str, Any]]:
    """Load packaged NOAA archive metadata records."""
    payload = json.loads(
        files("humpback.data")
        .joinpath("noaa_archive_sources.json")
        .read_text(encoding="utf-8")
    )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("NOAA archive metadata must define a records list")
    parsed = [cast(dict[str, Any], record) for record in records]
    _validate_noaa_archive_metadata(parsed)
    return parsed


NOAA_ARCHIVE_METADATA = _load_noaa_archive_metadata()
NOAA_ARCHIVE_SOURCES = [
    record
    for record in NOAA_ARCHIVE_METADATA
    if record.get("provider_kind") == "noaa_gcs"
    and record.get("bucket")
    and record.get("prefix")
]

ORCASOUND_S3_BUCKET = "audio-orcasound-net"

HYDROPHONE_IDS = {h["id"] for h in ORCASOUND_HYDROPHONES}
ARCHIVE_SOURCES = [*ORCASOUND_HYDROPHONES, *NOAA_ARCHIVE_SOURCES]
HYDROPHONE_UI_SOURCES = [
    *ORCASOUND_HYDROPHONES,
    *[
        source
        for source in NOAA_ARCHIVE_SOURCES
        if bool(source.get("include_in_detection_ui"))
    ],
]
ARCHIVE_SOURCE_IDS = {source["id"] for source in ARCHIVE_SOURCES}


def get_archive_source(source_id: str) -> dict[str, Any] | None:
    return next(
        (source for source in ARCHIVE_SOURCES if source["id"] == source_id),
        None,
    )
