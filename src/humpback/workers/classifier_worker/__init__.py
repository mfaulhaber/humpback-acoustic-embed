"""Classifier worker package — re-exports job runner functions.

runner.py continues to import ``from humpback.workers.classifier_worker import ...``.

Test monkeypatching targets (e.g. ``humpback.workers.classifier_worker.joblib``)
require these names to exist at package level. Sub-modules that need monkeypatch
compatibility import through the package via late-bound lookups.
"""

import joblib  # noqa: F401 — re-exported for monkeypatch compatibility

from humpback.classifier.extractor import (
    extract_labeled_samples,  # noqa: F401 — re-exported for monkeypatch
)
from humpback.classifier.providers import (
    build_archive_detection_provider,  # noqa: F401
    build_archive_playback_provider,  # noqa: F401
)
from humpback.services.model_registry_service import (
    get_model_by_name,  # noqa: F401
)
from humpback.workers.classifier_worker.detection import (
    run_detection_job,
    run_extraction_job,
)
from humpback.workers.classifier_worker.hydrophone import (
    _avg_audio_x_realtime,
    _hydrophone_detection_subprocess_main,
    _hydrophone_provider_mode,
    _run_hydrophone_detection_in_subprocess,
    run_hydrophone_detection_job,
)
from humpback.workers.classifier_worker.training import run_training_job
from humpback.workers.model_cache import get_model_by_version  # noqa: F401

__all__ = [
    "_avg_audio_x_realtime",
    "_hydrophone_detection_subprocess_main",
    "_hydrophone_provider_mode",
    "_run_hydrophone_detection_in_subprocess",
    "run_detection_job",
    "run_extraction_job",
    "run_hydrophone_detection_job",
    "run_training_job",
]
