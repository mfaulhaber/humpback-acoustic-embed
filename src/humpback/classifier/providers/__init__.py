"""Archive providers for the detection pipeline."""

from typing import Any

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers.noaa_gcs import (
    CachingNoaaGCSProvider,
    NoaaGCSProvider,
    build_noaa_detection_provider,
    build_noaa_playback_provider,
)
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
    build_orcasound_detection_provider,
    build_orcasound_local_cache_provider,
)
from humpback.config import get_archive_source


def _require_archive_source(source_id: str) -> dict[str, Any]:
    source = get_archive_source(source_id)
    if source is None:
        raise ValueError(f"Unknown archive source: {source_id}")
    return source


def build_archive_detection_provider(
    source_id: str,
    *,
    local_cache_path: str | None,
    s3_cache_path: str | None,
    noaa_cache_path: str | None = None,
    force_refresh: bool = True,
) -> ArchiveProvider:
    source = _require_archive_source(source_id)
    provider_kind = source["provider_kind"]

    if provider_kind == "orcasound_hls":
        return build_orcasound_detection_provider(
            source["id"],
            source["name"],
            local_cache_path=local_cache_path,
            s3_cache_path=s3_cache_path,
            force_refresh=force_refresh,
        )

    if local_cache_path is not None:
        raise ValueError(
            "local_cache_path is only supported for Orcasound HLS archive sources"
        )
    if provider_kind == "noaa_gcs":
        bucket = source.get("bucket")
        prefix = source.get("prefix")
        if not bucket or not prefix:
            raise ValueError(
                f"NOAA source is missing bucket/prefix config: {source['id']}"
            )
        return build_noaa_detection_provider(
            source["id"],
            source["name"],
            noaa_cache_path=noaa_cache_path,
            bucket=bucket,
            prefix=prefix,
            audio_subpath=source.get("audio_subpath"),
            child_folder_hints=source.get("child_folder_hints"),
            supports_segment_prefetch=bool(
                source.get("supports_segment_prefetch", True)
            ),
        )

    raise ValueError(f"Unsupported archive provider kind: {provider_kind}")


def build_archive_playback_provider(
    source_id: str,
    *,
    cache_path: str | None,
    noaa_cache_path: str | None = None,
) -> ArchiveProvider:
    source = _require_archive_source(source_id)
    provider_kind = source["provider_kind"]

    if provider_kind == "orcasound_hls":
        if cache_path is None:
            raise ValueError(
                "Orcasound playback/extraction requires a configured cache path"
            )
        return build_orcasound_local_cache_provider(
            source["id"],
            source["name"],
            cache_path,
        )

    if provider_kind == "noaa_gcs":
        bucket = source.get("bucket")
        prefix = source.get("prefix")
        if not bucket or not prefix:
            raise ValueError(
                f"NOAA source is missing bucket/prefix config: {source['id']}"
            )
        return build_noaa_playback_provider(
            source["id"],
            source["name"],
            noaa_cache_path=noaa_cache_path,
            bucket=bucket,
            prefix=prefix,
            audio_subpath=source.get("audio_subpath"),
            child_folder_hints=source.get("child_folder_hints"),
            supports_segment_prefetch=bool(
                source.get("supports_segment_prefetch", True)
            ),
        )

    raise ValueError(f"Unsupported archive provider kind: {provider_kind}")


__all__ = [
    "ArchiveProvider",
    "CachingHLSProvider",
    "CachingNoaaGCSProvider",
    "LocalHLSCacheProvider",
    "NoaaGCSProvider",
    "OrcasoundHLSProvider",
    "StreamSegment",
    "build_archive_detection_provider",
    "build_archive_playback_provider",
    "build_noaa_detection_provider",
    "build_noaa_playback_provider",
    "build_orcasound_detection_provider",
    "build_orcasound_local_cache_provider",
]
