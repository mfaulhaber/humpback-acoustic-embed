"""Convert SQLAlchemy model instances to Pydantic response schemas.

Centralised here so router modules stay focused on endpoint logic and to
reduce file sizes ahead of Phase 3 router splitting.
"""

from __future__ import annotations

import json

from humpback.schemas.audio import AudioFileOut, AudioMetadataOut
from humpback.schemas.classifier import (
    AutoresearchCandidateArtifactPaths,
    AutoresearchCandidateDetailOut,
    AutoresearchCandidateSummaryOut,
    ClassifierModelOut,
    ClassifierTrainingJobOut,
    DetectionJobOut,
    RetrainWorkflowOut,
)
from humpback.schemas.clustering import ClusteringJobOut, ClusterOut
from humpback.schemas.label_processing import LabelProcessingJobOut
from humpback.schemas.processing import ProcessingJobOut
from humpback.schemas.vocalization import (
    TrainingDatasetOut,
    VocalizationInferenceJobOut,
    VocalizationModelOut,
    VocalizationTrainingJobOut,
)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def audio_file_to_out(af) -> AudioFileOut:
    meta = None
    if af.metadata_:
        m = af.metadata_
        meta = AudioMetadataOut(
            id=m.id,
            audio_file_id=m.audio_file_id,
            tag_data=json.loads(m.tag_data) if m.tag_data else None,
            visual_observations=json.loads(m.visual_observations)
            if m.visual_observations
            else None,
            group_composition=json.loads(m.group_composition)
            if m.group_composition
            else None,
            prey_density_proxy=json.loads(m.prey_density_proxy)
            if m.prey_density_proxy
            else None,
        )
    return AudioFileOut(
        id=af.id,
        filename=af.filename,
        folder_path=af.folder_path,
        source_folder=af.source_folder,
        checksum_sha256=af.checksum_sha256,
        duration_seconds=af.duration_seconds,
        sample_rate_original=af.sample_rate_original,
        created_at=af.created_at,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


def processing_job_to_out(job, skipped: bool = False) -> ProcessingJobOut:
    return ProcessingJobOut(
        id=job.id,
        audio_file_id=job.audio_file_id,
        status=job.status,
        encoding_signature=job.encoding_signature,
        model_version=job.model_version,
        window_size_seconds=job.window_size_seconds,
        target_sample_rate=job.target_sample_rate,
        feature_config=json.loads(job.feature_config) if job.feature_config else None,
        error_message=job.error_message,
        warning_message=job.warning_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def clustering_job_to_out(job) -> ClusteringJobOut:
    return ClusteringJobOut(
        id=job.id,
        status=job.status,
        embedding_set_ids=json.loads(job.embedding_set_ids),
        parameters=json.loads(job.parameters) if job.parameters else None,
        error_message=job.error_message,
        metrics=json.loads(job.metrics_json) if job.metrics_json else None,
        refined_from_job_id=job.refined_from_job_id,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def cluster_to_out(c) -> ClusterOut:
    return ClusterOut(
        id=c.id,
        clustering_job_id=c.clustering_job_id,
        cluster_label=c.cluster_label,
        size=c.size,
        metadata_summary=json.loads(c.metadata_summary) if c.metadata_summary else None,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classifier_training_job_to_out(job) -> ClassifierTrainingJobOut:
    return ClassifierTrainingJobOut(
        id=job.id,
        status=job.status,
        name=job.name,
        positive_embedding_set_ids=json.loads(job.positive_embedding_set_ids),
        negative_embedding_set_ids=json.loads(job.negative_embedding_set_ids),
        model_version=job.model_version,
        window_size_seconds=job.window_size_seconds,
        target_sample_rate=job.target_sample_rate,
        feature_config=json.loads(job.feature_config) if job.feature_config else None,
        parameters=json.loads(job.parameters) if job.parameters else None,
        classifier_model_id=job.classifier_model_id,
        error_message=job.error_message,
        source_mode=job.source_mode,
        source_candidate_id=job.source_candidate_id,
        source_model_id=job.source_model_id,
        manifest_path=job.manifest_path,
        training_split_name=job.training_split_name,
        promoted_config=json.loads(job.promoted_config)
        if job.promoted_config
        else None,
        source_comparison_context=json.loads(job.source_comparison_context)
        if job.source_comparison_context
        else None,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def autoresearch_candidate_to_summary(
    candidate,
) -> AutoresearchCandidateSummaryOut:
    return AutoresearchCandidateSummaryOut(
        id=candidate.id,
        name=candidate.name,
        status=candidate.status,
        phase=candidate.phase,
        objective_name=candidate.objective_name,
        threshold=candidate.threshold,
        comparison_target=candidate.comparison_target,
        source_model_id=candidate.source_model_id,
        source_model_name=candidate.source_model_name,
        is_reproducible_exact=candidate.is_reproducible_exact,
        promoted_config=json.loads(candidate.promoted_config),
        best_run_metrics=json.loads(candidate.best_run_metrics)
        if candidate.best_run_metrics
        else None,
        split_metrics=json.loads(candidate.split_metrics)
        if candidate.split_metrics
        else None,
        metric_deltas=json.loads(candidate.metric_deltas)
        if candidate.metric_deltas
        else None,
        replay_summary=json.loads(candidate.replay_summary)
        if candidate.replay_summary
        else None,
        source_counts=json.loads(candidate.source_counts)
        if candidate.source_counts
        else None,
        warnings=json.loads(candidate.warnings) if candidate.warnings else [],
        training_job_id=candidate.training_job_id,
        new_model_id=candidate.new_model_id,
        error_message=candidate.error_message,
        created_at=candidate.created_at,
        updated_at=candidate.updated_at,
    )


def autoresearch_candidate_to_detail(
    candidate,
    *,
    replay_verification: dict | None = None,
) -> AutoresearchCandidateDetailOut:
    summary = autoresearch_candidate_to_summary(candidate)
    return AutoresearchCandidateDetailOut(
        **summary.model_dump(),
        artifact_paths=AutoresearchCandidateArtifactPaths(
            manifest_path=candidate.manifest_path,
            best_run_path=candidate.best_run_path,
            comparison_path=candidate.comparison_path,
            top_false_positives_path=candidate.top_false_positives_path,
        ),
        source_model_metadata=json.loads(candidate.source_model_metadata)
        if candidate.source_model_metadata
        else None,
        top_false_positives_preview=json.loads(candidate.top_false_positives_preview)
        if candidate.top_false_positives_preview
        else None,
        prediction_disagreements_preview=json.loads(
            candidate.prediction_disagreements_preview
        )
        if candidate.prediction_disagreements_preview
        else None,
        replay_verification=replay_verification,
    )


def classifier_model_to_out(m) -> ClassifierModelOut:
    return ClassifierModelOut(
        id=m.id,
        name=m.name,
        model_path=m.model_path,
        model_version=m.model_version,
        vector_dim=m.vector_dim,
        window_size_seconds=m.window_size_seconds,
        target_sample_rate=m.target_sample_rate,
        feature_config=json.loads(m.feature_config) if m.feature_config else None,
        training_summary=json.loads(m.training_summary) if m.training_summary else None,
        training_job_id=m.training_job_id,
        training_source_mode=m.training_source_mode,
        source_candidate_id=m.source_candidate_id,
        source_model_id=m.source_model_id,
        promotion_provenance=json.loads(m.promotion_provenance)
        if m.promotion_provenance
        else None,
        created_at=m.created_at,
        updated_at=m.updated_at,
    )


def detection_job_to_out(job) -> DetectionJobOut:
    return DetectionJobOut(
        id=job.id,
        status=job.status,
        classifier_model_id=job.classifier_model_id,
        audio_folder=job.audio_folder,
        confidence_threshold=job.confidence_threshold,
        hop_seconds=job.hop_seconds,
        high_threshold=job.high_threshold,
        low_threshold=job.low_threshold,
        detection_mode=job.detection_mode,
        output_row_store_path=job.output_row_store_path,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
        files_processed=job.files_processed,
        files_total=job.files_total,
        extract_status=job.extract_status,
        extract_error=job.extract_error,
        extract_summary=json.loads(job.extract_summary)
        if job.extract_summary
        else None,
        hydrophone_id=job.hydrophone_id,
        hydrophone_name=job.hydrophone_name,
        start_timestamp=job.start_timestamp,
        end_timestamp=job.end_timestamp,
        segments_processed=job.segments_processed,
        segments_total=job.segments_total,
        time_covered_sec=job.time_covered_sec,
        alerts=json.loads(job.alerts) if job.alerts else None,
        local_cache_path=job.local_cache_path,
        has_positive_labels=job.has_positive_labels,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def retrain_workflow_to_out(wf) -> RetrainWorkflowOut:
    return RetrainWorkflowOut(
        id=wf.id,
        status=wf.status,
        source_model_id=wf.source_model_id,
        new_model_name=wf.new_model_name,
        model_version=wf.model_version,
        window_size_seconds=wf.window_size_seconds,
        target_sample_rate=wf.target_sample_rate,
        feature_config=json.loads(wf.feature_config) if wf.feature_config else None,
        parameters=json.loads(wf.parameters) if wf.parameters else None,
        positive_folder_roots=json.loads(wf.positive_folder_roots),
        negative_folder_roots=json.loads(wf.negative_folder_roots),
        import_summary=json.loads(wf.import_summary) if wf.import_summary else None,
        processing_job_ids=json.loads(wf.processing_job_ids)
        if wf.processing_job_ids
        else None,
        processing_total=wf.processing_total,
        processing_complete=wf.processing_complete,
        training_job_id=wf.training_job_id,
        new_model_id=wf.new_model_id,
        error_message=wf.error_message,
        created_at=wf.created_at,
        updated_at=wf.updated_at,
    )


# ---------------------------------------------------------------------------
# Label Processing
# ---------------------------------------------------------------------------


def label_processing_job_to_out(job) -> LabelProcessingJobOut:
    """Convert a LabelProcessingJob ORM instance to its response schema."""
    return LabelProcessingJobOut(
        id=job.id,
        status=job.status,
        workflow=job.workflow or "score_based",
        classifier_model_id=job.classifier_model_id,
        annotation_folder=job.annotation_folder,
        audio_folder=job.audio_folder,
        output_root=job.output_root,
        parameters=json.loads(job.parameters) if job.parameters else None,
        files_processed=job.files_processed,
        files_total=job.files_total,
        annotations_total=job.annotations_total,
        result_summary=json.loads(job.result_summary) if job.result_summary else None,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# ---------------------------------------------------------------------------
# Vocalization
# ---------------------------------------------------------------------------


def vocalization_model_to_out(m) -> VocalizationModelOut:
    return VocalizationModelOut(
        id=m.id,
        name=m.name,
        model_dir_path=m.model_dir_path,
        vocabulary_snapshot=json.loads(m.vocabulary_snapshot),
        per_class_thresholds=json.loads(m.per_class_thresholds),
        per_class_metrics=json.loads(m.per_class_metrics)
        if m.per_class_metrics
        else None,
        training_summary=json.loads(m.training_summary) if m.training_summary else None,
        is_active=m.is_active,
        training_dataset_id=m.training_dataset_id,
        created_at=m.created_at,
    )


def vocalization_training_job_to_out(j) -> VocalizationTrainingJobOut:
    return VocalizationTrainingJobOut(
        id=j.id,
        status=j.status,
        source_config=json.loads(j.source_config),
        parameters=json.loads(j.parameters) if j.parameters else None,
        vocalization_model_id=j.vocalization_model_id,
        result_summary=json.loads(j.result_summary) if j.result_summary else None,
        error_message=j.error_message,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


def vocalization_inference_job_to_out(j) -> VocalizationInferenceJobOut:
    return VocalizationInferenceJobOut(
        id=j.id,
        status=j.status,
        vocalization_model_id=j.vocalization_model_id,
        source_type=j.source_type,
        source_id=j.source_id,
        output_path=j.output_path,
        result_summary=json.loads(j.result_summary) if j.result_summary else None,
        error_message=j.error_message,
        created_at=j.created_at,
        updated_at=j.updated_at,
    )


def training_dataset_to_out(d) -> TrainingDatasetOut:
    return TrainingDatasetOut(
        id=d.id,
        name=d.name,
        source_config=json.loads(d.source_config),
        total_rows=d.total_rows,
        vocabulary=json.loads(d.vocabulary),
        created_at=d.created_at,
        updated_at=d.updated_at,
    )
