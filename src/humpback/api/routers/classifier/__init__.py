"""Classifier router package — composes sub-routers into a single router.

app.py continues to import ``from humpback.api.routers import classifier``
and use ``classifier.router``.
"""

from __future__ import annotations

from fastapi import APIRouter

from humpback.api.routers.classifier.autoresearch import (
    router as autoresearch_router,
)
from humpback.api.routers.classifier.detection import (
    _DecodedAudioCache,
    _NoaaPlaybackProviderRegistry,
    _noaa_provider_registry,
    download_detections,
    router as detection_router,
)
from humpback.api.routers.classifier.embeddings import (
    router as embeddings_router,
)
from humpback.api.routers.classifier.hyperparameter import (
    router as hyperparameter_router,
)
from humpback.api.routers.classifier.hydrophone import (
    router as hydrophone_router,
)
from humpback.api.routers.classifier.models import router as models_router
from humpback.api.routers.classifier.training import router as training_router

# Re-export for backward compatibility (vocalization.py lazy-imports these)
from humpback.classifier.providers import build_archive_playback_provider

router = APIRouter(prefix="/classifier", tags=["classifier"])
router.include_router(autoresearch_router)
router.include_router(models_router)
router.include_router(training_router)
router.include_router(detection_router)
router.include_router(hydrophone_router)
router.include_router(embeddings_router)
router.include_router(hyperparameter_router)

__all__ = [
    "_DecodedAudioCache",
    "_NoaaPlaybackProviderRegistry",
    "_noaa_provider_registry",
    "build_archive_playback_provider",
    "download_detections",
    "router",
]
