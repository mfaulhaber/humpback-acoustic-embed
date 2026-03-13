"""Archive providers for the detection pipeline."""

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
    build_orcasound_detection_provider,
    build_orcasound_local_cache_provider,
)

__all__ = [
    "ArchiveProvider",
    "CachingHLSProvider",
    "LocalHLSCacheProvider",
    "OrcasoundHLSProvider",
    "StreamSegment",
    "build_orcasound_detection_provider",
    "build_orcasound_local_cache_provider",
]
