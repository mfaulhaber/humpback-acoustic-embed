"""Archive providers for the detection pipeline."""

from humpback.classifier.archive import ArchiveProvider, StreamSegment
from humpback.classifier.providers.orcasound_hls import (
    CachingHLSProvider,
    LocalHLSCacheProvider,
    OrcasoundHLSProvider,
)

__all__ = [
    "ArchiveProvider",
    "CachingHLSProvider",
    "LocalHLSCacheProvider",
    "OrcasoundHLSProvider",
    "StreamSegment",
]
