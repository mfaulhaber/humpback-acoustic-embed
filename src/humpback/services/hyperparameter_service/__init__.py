"""Hyperparameter tuning service package — re-exports public functions."""

from humpback.services.hyperparameter_service.comparison import (
    build_prediction_disagreements,
    compare_classifiers,
    resolve_production_classifier,
)
from humpback.services.hyperparameter_service.manifest import (
    generate_manifest,
)
from humpback.services.hyperparameter_service.search import (
    cache_embeddings_by_pooling,
    default_objective,
    run_search,
)
from humpback.services.hyperparameter_service.search_space import (
    DEFAULT_SEARCH_SPACE,
    config_hash,
    sample_config,
)

__all__ = [
    # manifest
    "generate_manifest",
    # search
    "cache_embeddings_by_pooling",
    "default_objective",
    "run_search",
    # search_space
    "DEFAULT_SEARCH_SPACE",
    "config_hash",
    "sample_config",
    # comparison
    "build_prediction_disagreements",
    "compare_classifiers",
    "resolve_production_classifier",
]
