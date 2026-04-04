"""Classifier service package — re-exports all public functions.

Callers continue to import ``from humpback.services import classifier_service``
or ``from humpback.services.classifier_service import ...``.
"""

from humpback.services.classifier_service.autoresearch import (
    _assess_reproducibility,
    create_training_job_from_autoresearch_candidate,
    get_autoresearch_candidate,
    import_autoresearch_candidate,
    list_autoresearch_candidates,
)
from humpback.services.classifier_service.detection import (
    DetectionJobDependencyError,
    bulk_delete_detection_jobs,
    create_detection_job,
    delete_detection_job,
    get_detection_job,
    list_detection_jobs,
)
from humpback.services.classifier_service.hydrophone import (
    cancel_hydrophone_detection_job,
    create_hydrophone_detection_job,
    list_hydrophone_detection_jobs,
    pause_hydrophone_detection_job,
    resume_hydrophone_detection_job,
)
from humpback.services.classifier_service.models import (
    bulk_delete_classifier_models,
    delete_classifier_model,
    get_classifier_model,
    list_classifier_models,
)
from humpback.services.classifier_service.training import (
    bulk_delete_training_jobs,
    collect_embedding_sets_for_folders,
    create_retrain_workflow,
    create_training_job,
    delete_training_job,
    get_retrain_info,
    get_training_data_summary,
    get_training_job,
    list_retrain_workflows,
    list_training_jobs,
    get_retrain_workflow,
    trace_folder_roots,
)

__all__ = [
    # autoresearch
    "_assess_reproducibility",
    "create_training_job_from_autoresearch_candidate",
    "get_autoresearch_candidate",
    "import_autoresearch_candidate",
    "list_autoresearch_candidates",
    # detection
    "DetectionJobDependencyError",
    "bulk_delete_detection_jobs",
    "create_detection_job",
    "delete_detection_job",
    "get_detection_job",
    "list_detection_jobs",
    # hydrophone
    "cancel_hydrophone_detection_job",
    "create_hydrophone_detection_job",
    "list_hydrophone_detection_jobs",
    "pause_hydrophone_detection_job",
    "resume_hydrophone_detection_job",
    # models
    "bulk_delete_classifier_models",
    "delete_classifier_model",
    "get_classifier_model",
    "list_classifier_models",
    # training
    "bulk_delete_training_jobs",
    "collect_embedding_sets_for_folders",
    "create_retrain_workflow",
    "create_training_job",
    "delete_training_job",
    "get_retrain_info",
    "get_training_data_summary",
    "get_training_job",
    "list_retrain_workflows",
    "list_training_jobs",
    "get_retrain_workflow",
    "trace_folder_roots",
]
