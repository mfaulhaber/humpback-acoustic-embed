"""Archive providers for the detection pipeline."""

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers.noaa_gcs import NoaaGCSProvider
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
    build_orcasound_detection_provider,
    build_orcasound_local_cache_provider,
)
from humpback.config import get_archive_source


def _require_archive_source(source_id: str) -> dict[str, str]:
    source = get_archive_source(source_id)
    if source is None:
        raise ValueError(f"Unknown archive source: {source_id}")
    return source


def _build_noaa_provider(source: dict[str, str]) -> NoaaGCSProvider:
    bucket = source.get("bucket")
    prefix = source.get("prefix")
    if not bucket or not prefix:
        raise ValueError(f"NOAA source is missing bucket/prefix config: {source['id']}")
    return NoaaGCSProvider(
        source["id"],
        source["name"],
        bucket=bucket,
        prefix=prefix,
    )


def build_archive_detection_provider(
    source_id: str,
    *,
    local_cache_path: str | None,
    s3_cache_path: str | None,
) -> ArchiveProvider:
    source = _require_archive_source(source_id)
    provider_kind = source["provider_kind"]

    if provider_kind == "orcasound_hls":
        return build_orcasound_detection_provider(
            source["id"],
            source["name"],
            local_cache_path=local_cache_path,
            s3_cache_path=s3_cache_path,
        )

    if local_cache_path is not None:
        raise ValueError(
            "local_cache_path is only supported for Orcasound HLS archive sources"
        )
    if provider_kind == "noaa_gcs":
        return _build_noaa_provider(source)

    raise ValueError(f"Unsupported archive provider kind: {provider_kind}")


def build_archive_playback_provider(
    source_id: str,
    *,
    cache_path: str | None,
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
        return _build_noaa_provider(source)

    raise ValueError(f"Unsupported archive provider kind: {provider_kind}")


__all__ = [
    "ArchiveProvider",
    "CachingHLSProvider",
    "LocalHLSCacheProvider",
    "NoaaGCSProvider",
    "OrcasoundHLSProvider",
    "StreamSegment",
    "build_archive_detection_provider",
    "build_archive_playback_provider",
    "build_orcasound_detection_provider",
    "build_orcasound_local_cache_provider",
]
