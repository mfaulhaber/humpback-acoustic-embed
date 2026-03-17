from pathlib import Path
from typing import Any, cast

import humpback.config as config_module

from humpback.config import (
    ARCHIVE_SOURCE_IDS,
    NOAA_ARCHIVE_METADATA,
    Settings,
    get_archive_source,
)


def test_storage_root_drives_default_paths(tmp_path):
    storage_root = tmp_path / "storage"

    settings = Settings(storage_root=storage_root)

    assert settings.positive_sample_path == str(storage_root / "labeled" / "positives")
    assert settings.negative_sample_path == str(storage_root / "labeled" / "negatives")
    assert settings.s3_cache_path == str(storage_root / "s3-orcasound-cache")
    assert settings.noaa_cache_path == str(storage_root / "noaa-gcs-cache")
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.allowed_hosts == ["*"]


def test_env_file_loads_hosts_and_path_overrides(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HUMPBACK_API_HOST=0.0.0.0",
                "HUMPBACK_API_PORT=9001",
                "HUMPBACK_ALLOWED_HOSTS=*.trycloudflare.com,localhost,127.0.0.1",
                "HUMPBACK_STORAGE_ROOT=/workspace/data",
                "HUMPBACK_POSITIVE_SAMPLE_PATH=/workspace/custom/positives",
                "HUMPBACK_NEGATIVE_SAMPLE_PATH=/workspace/custom/negatives",
                "HUMPBACK_S3_CACHE_PATH=/workspace/custom/cache",
                "HUMPBACK_NOAA_CACHE_PATH=/workspace/custom/noaa-cache",
                "TF_EXTRA=tf-linux-cpu",
            ]
        ),
        encoding="utf-8",
    )

    settings = cast(Any, Settings)(_env_file=env_file)

    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 9001
    assert settings.allowed_hosts == [
        "*.trycloudflare.com",
        "localhost",
        "127.0.0.1",
    ]
    assert settings.storage_root == Path("/workspace/data")
    assert settings.positive_sample_path == "/workspace/custom/positives"
    assert settings.negative_sample_path == "/workspace/custom/negatives"
    assert settings.s3_cache_path == "/workspace/custom/cache"
    assert settings.noaa_cache_path == "/workspace/custom/noaa-cache"


def test_settings_does_not_auto_load_cwd_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HUMPBACK_ALLOWED_HOSTS=*.trycloudflare.com,localhost,127.0.0.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.allowed_hosts == ["*"]


def test_from_repo_env_loads_repo_root_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "HUMPBACK_API_HOST=0.0.0.0",
                "HUMPBACK_ALLOWED_HOSTS=*.trycloudflare.com,localhost",
                "HUMPBACK_STORAGE_ROOT=/workspace/data",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(config_module, "_repo_env_file", lambda: env_file)

    settings = Settings.from_repo_env()

    assert settings.api_host == "0.0.0.0"
    assert settings.allowed_hosts == ["*.trycloudflare.com", "localhost"]
    assert settings.storage_root == Path("/workspace/data")


def test_noaa_archive_metadata_loads_runtime_and_reference_records():
    record_ids = {record["id"] for record in NOAA_ARCHIVE_METADATA}

    assert "sanctsound_ci01" in record_ids
    assert "sanctsound_pmmn" in record_ids
    assert "noaa_glacier_bay" in record_ids

    assert "sanctsound_ci01" in ARCHIVE_SOURCE_IDS
    assert "noaa_glacier_bay" in ARCHIVE_SOURCE_IDS
    assert "sanctsound_pmmn" not in ARCHIVE_SOURCE_IDS

    ci01 = get_archive_source("sanctsound_ci01")
    assert ci01 is not None
    assert ci01["audio_subpath"] == "audio/"
    assert ci01["include_in_detection_ui"] is True
    assert ci01["supports_segment_prefetch"] is False
    assert len(ci01["child_folder_hints"]) == 27

    glacier_bay = get_archive_source("noaa_glacier_bay")
    assert glacier_bay is not None
    assert glacier_bay["supports_segment_prefetch"] is True
