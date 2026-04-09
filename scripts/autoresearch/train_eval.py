"""Single train/eval run for autoresearch.

.. deprecated::
    This module is deprecated. Use
    ``humpback.services.hyperparameter_service.train_eval`` instead.
"""

from __future__ import annotations

from humpback.classifier.replay import ParquetCacheEntry  # noqa: F401
from humpback.services.hyperparameter_service.train_eval import (
    apply_context_pooling,
    apply_transforms,  # noqa: F401
    build_classifier,  # noqa: F401
    build_embedding_lookup as _build_embedding_lookup,
    build_feature_pipeline,  # noqa: F401
    collect_split_arrays,  # noqa: F401
    compute_metrics,  # noqa: F401
    evaluate_classifier_on_split,
    evaluate_on_split,  # noqa: F401
    find_top_false_positives,  # noqa: F401
    fit_autoresearch_classifier,
    load_manifest,  # noqa: F401
    load_parquet_cache as _load_parquet_cache,
    prepare_embeddings,
    score_classifier,  # noqa: F401
    train_eval,
)

__all__ = [
    "ParquetCacheEntry",
    "_build_embedding_lookup",
    "_load_parquet_cache",
    "apply_context_pooling",
    "apply_transforms",
    "build_classifier",
    "build_feature_pipeline",
    "collect_split_arrays",
    "compute_metrics",
    "evaluate_classifier_on_split",
    "evaluate_on_split",
    "find_top_false_positives",
    "fit_autoresearch_classifier",
    "load_manifest",
    "prepare_embeddings",
    "score_classifier",
    "train_eval",
]
