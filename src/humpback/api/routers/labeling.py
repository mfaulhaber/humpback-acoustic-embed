"""API router for vocalization type labeling workflows."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.labeling import LabelingAnnotation, VocalizationLabel
from humpback.schemas.labeling import (
    ActiveLearningCycleRequest,
    ActiveLearningCycleResponse,
    AnnotationCreate,
    AnnotationOut,
    AnnotationUpdate,
    ConvergenceMetrics,
    DetectionNeighborsRequest,
    DetectionNeighborsResponse,
    LabelingSummary,
    TrainingSummary,
    NeighborHit,
    PredictRequest,
    PredictionRow,
    UncertaintyQueueRow,
    VocalizationLabelCreate,
    VocalizationLabelOut,
    VocalizationLabelUpdate,
    VocalizationModelOut,
    VocalizationTrainingJobCreate,
    VocalizationTrainingJobOut,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeling", tags=["labeling"])


# ---- Vocalization Label CRUD ----


@router.get(
    "/vocalization-labels/{detection_job_id}/{row_id}",
    response_model=list[VocalizationLabelOut],
)
async def list_vocalization_labels(
    detection_job_id: str,
    row_id: str,
    session: SessionDep,
):
    """List all vocalization labels for a detection row."""
    result = await session.execute(
        select(VocalizationLabel)
        .where(VocalizationLabel.detection_job_id == detection_job_id)
        .where(VocalizationLabel.row_id == row_id)
        .order_by(VocalizationLabel.created_at)
    )
    return [VocalizationLabelOut.model_validate(r) for r in result.scalars().all()]


@router.post(
    "/vocalization-labels/{detection_job_id}/{row_id}",
    response_model=VocalizationLabelOut,
    status_code=201,
)
async def create_vocalization_label(
    detection_job_id: str,
    row_id: str,
    body: VocalizationLabelCreate,
    session: SessionDep,
):
    """Add a vocalization type label to a detection row."""
    label = VocalizationLabel(
        detection_job_id=detection_job_id,
        row_id=row_id,
        label=body.label,
        confidence=body.confidence,
        source=body.source,
        notes=body.notes,
    )
    session.add(label)
    await session.commit()
    await session.refresh(label)
    return VocalizationLabelOut.model_validate(label)


@router.put(
    "/vocalization-labels/{label_id}",
    response_model=VocalizationLabelOut,
)
async def update_vocalization_label(
    label_id: str,
    body: VocalizationLabelUpdate,
    session: SessionDep,
):
    """Update an existing vocalization label."""
    result = await session.execute(
        select(VocalizationLabel).where(VocalizationLabel.id == label_id)
    )
    label = result.scalar_one_or_none()
    if label is None:
        raise HTTPException(404, "Vocalization label not found")

    if body.label is not None:
        label.label = body.label
    if body.confidence is not None:
        label.confidence = body.confidence
    if body.notes is not None:
        label.notes = body.notes

    await session.commit()
    await session.refresh(label)
    return VocalizationLabelOut.model_validate(label)


@router.delete("/vocalization-labels/{label_id}", status_code=204)
async def delete_vocalization_label(
    label_id: str,
    session: SessionDep,
):
    """Delete a vocalization label."""
    result = await session.execute(
        select(VocalizationLabel).where(VocalizationLabel.id == label_id)
    )
    label = result.scalar_one_or_none()
    if label is None:
        raise HTTPException(404, "Vocalization label not found")

    await session.delete(label)
    await session.commit()


# ---- Label Vocabulary ----


@router.get("/label-vocabulary", response_model=list[str])
async def get_label_vocabulary(session: SessionDep):
    """Return distinct vocalization type labels used across all labels."""
    result = await session.execute(
        select(VocalizationLabel.label).distinct().order_by(VocalizationLabel.label)
    )
    return [row[0] for row in result.all()]


# ---- Labeling Summary ----


@router.get("/summary/{detection_job_id}", response_model=LabelingSummary)
async def get_labeling_summary(
    detection_job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """Return labeling progress summary for a detection job."""
    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.storage import detection_row_store_path

    row_store = detection_row_store_path(settings.storage_root, detection_job_id)
    if not row_store.exists():
        raise HTTPException(404, "Detection row store not found")

    _fieldnames, rows = read_detection_row_store(row_store)
    total_rows = len(rows)

    # Get all vocalization labels for this job
    result = await session.execute(
        select(VocalizationLabel.row_id, VocalizationLabel.label).where(
            VocalizationLabel.detection_job_id == detection_job_id
        )
    )
    label_rows = result.all()

    labeled_row_ids = set()
    label_counts: dict[str, int] = {}
    for row_id, label in label_rows:
        labeled_row_ids.add(row_id)
        label_counts[label] = label_counts.get(label, 0) + 1

    # Also count annotation labels (sub-window annotations carry vocalization types)
    ann_result = await session.execute(
        select(LabelingAnnotation.row_id, LabelingAnnotation.label).where(
            LabelingAnnotation.detection_job_id == detection_job_id
        )
    )
    for row_id, label in ann_result.all():
        labeled_row_ids.add(row_id)
        label_counts[label] = label_counts.get(label, 0) + 1

    return LabelingSummary(
        total_rows=total_rows,
        labeled_rows=len(labeled_row_ids),
        unlabeled_rows=total_rows - len(labeled_row_ids),
        label_distribution=label_counts,
    )


@router.get("/training-summary", response_model=TrainingSummary)
async def get_training_summary(session: SessionDep):
    """Aggregate label stats across all detection jobs for training readiness."""
    # Vocalization labels
    result = await session.execute(
        select(
            VocalizationLabel.detection_job_id,
            VocalizationLabel.row_id,
            VocalizationLabel.label,
        )
    )
    labeled_job_ids: set[str] = set()
    labeled_row_keys: set[str] = set()
    label_counts: dict[str, int] = {}
    for job_id, row_id, label in result.all():
        labeled_job_ids.add(job_id)
        labeled_row_keys.add(f"{job_id}:{row_id}")
        label_counts[label] = label_counts.get(label, 0) + 1

    # Annotations
    ann_result = await session.execute(
        select(
            LabelingAnnotation.detection_job_id,
            LabelingAnnotation.row_id,
            LabelingAnnotation.label,
        )
    )
    for job_id, row_id, label in ann_result.all():
        labeled_job_ids.add(job_id)
        labeled_row_keys.add(f"{job_id}:{row_id}")
        label_counts[label] = label_counts.get(label, 0) + 1

    return TrainingSummary(
        labeled_job_ids=sorted(labeled_job_ids),
        labeled_rows=len(labeled_row_keys),
        label_distribution=label_counts,
    )


# ---- Detection Neighbors (Vector Search) ----


def _infer_label_from_folder(folder_path: str | None) -> str | None:
    """Extract a vocalization type label from the audio file's folder path.

    Assumes folder structure like: .../dataset_name/call_type/audio.flac
    Returns the last non-empty path segment before the filename level,
    which typically represents the call type or category.
    """
    if not folder_path:
        return None
    parts = Path(folder_path).parts
    if not parts:
        return None
    # The folder_path is the directory containing the audio file.
    # The last segment is typically the call type.
    return parts[-1] if parts else None


@router.post(
    "/detection-neighbors/{detection_job_id}",
    response_model=DetectionNeighborsResponse,
)
async def get_detection_neighbors(
    detection_job_id: str,
    body: DetectionNeighborsRequest,
    session: SessionDep,
    settings: SettingsDep,
):
    """Find similar sounds from reference embedding sets for a detection row."""
    from humpback.classifier.detector import read_detection_embedding
    from humpback.models.classifier import ClassifierModel
    from humpback.schemas.search import VectorSearchRequest
    from humpback.services.classifier_service import get_detection_job
    from humpback.services.search_service import similarity_search_by_vector
    from humpback.storage import detection_embeddings_path

    job = await get_detection_job(session, detection_job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    emb_path = detection_embeddings_path(settings.storage_root, job.id)
    if not emb_path.exists():
        raise HTTPException(404, "No stored embeddings for this detection job")

    embedding = read_detection_embedding(
        emb_path, body.filename, body.start_sec, body.end_sec
    )
    if embedding is None:
        raise HTTPException(404, "Embedding not found for specified detection row")

    # Resolve model_version from the classifier model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    cm = cm_result.scalar_one_or_none()
    if cm is None:
        raise HTTPException(404, "Classifier model not found for detection job")

    search_request = VectorSearchRequest(
        vector=embedding,
        model_version=cm.model_version,
        top_k=body.top_k,
        metric=body.metric,
        embedding_set_ids=body.embedding_set_ids,
    )

    search_response = await similarity_search_by_vector(session, search_request)

    hits = [
        NeighborHit(
            score=hit.score,
            embedding_set_id=hit.embedding_set_id,
            row_index=hit.row_index,
            audio_file_id=hit.audio_file_id,
            audio_filename=hit.audio_filename,
            audio_folder_path=hit.audio_folder_path,
            window_offset_seconds=hit.window_offset_seconds,
            inferred_label=_infer_label_from_folder(hit.audio_folder_path),
        )
        for hit in search_response.results
    ]

    return DetectionNeighborsResponse(
        hits=hits,
        total_candidates=search_response.total_candidates,
    )


# ---- Vocalization Classifier Training ----


@router.post(
    "/training-jobs", response_model=VocalizationTrainingJobOut, status_code=201
)
async def create_vocalization_training_job(
    body: VocalizationTrainingJobCreate,
    session: SessionDep,
):
    """Queue a multi-class vocalization classifier training job."""
    import json

    from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob

    from humpback.services.classifier_service import get_detection_job

    # Validate source detection jobs exist
    for job_id in body.source_detection_job_ids:
        job = await get_detection_job(session, job_id)
        if job is None:
            raise HTTPException(404, f"Detection job {job_id} not found")

    # Check that at least some labels exist across the source jobs
    # (from vocalization labels and/or sub-window annotations)
    result = await session.execute(
        select(VocalizationLabel.label)
        .where(VocalizationLabel.detection_job_id.in_(body.source_detection_job_ids))
        .distinct()
    )
    distinct_labels = set(row[0] for row in result.all())

    ann_result = await session.execute(
        select(LabelingAnnotation.label)
        .where(LabelingAnnotation.detection_job_id.in_(body.source_detection_job_ids))
        .distinct()
    )
    distinct_labels |= set(row[0] for row in ann_result.all())

    if len(distinct_labels) < 2:
        raise HTTPException(
            400,
            f"Need at least 2 distinct vocalization labels across source jobs, "
            f"found {len(distinct_labels)}: {list(distinct_labels)}",
        )

    # Resolve model_version from the first detection job's classifier model
    first_job = await get_detection_job(session, body.source_detection_job_ids[0])
    assert first_job is not None
    cm_result = await session.execute(
        select(ClassifierModel).where(
            ClassifierModel.id == first_job.classifier_model_id
        )
    )
    cm = cm_result.scalar_one_or_none()
    if cm is None:
        raise HTTPException(404, "Classifier model not found for source detection job")

    training_job = ClassifierTrainingJob(
        name=body.name,
        job_purpose="vocalization",
        source_detection_job_ids=json.dumps(body.source_detection_job_ids),
        positive_embedding_set_ids="[]",  # not used for vocalization training
        negative_embedding_set_ids="[]",
        model_version=cm.model_version,
        window_size_seconds=cm.window_size_seconds,
        target_sample_rate=cm.target_sample_rate,
        feature_config=cm.feature_config,
        parameters=json.dumps(body.parameters) if body.parameters else None,
    )
    session.add(training_job)
    await session.commit()
    await session.refresh(training_job)

    return VocalizationTrainingJobOut(
        id=training_job.id,
        status=training_job.status,
        name=training_job.name,
        job_purpose=training_job.job_purpose,
        source_detection_job_ids=body.source_detection_job_ids,
        classifier_model_id=training_job.classifier_model_id,
        error_message=training_job.error_message,
        created_at=training_job.created_at,
        updated_at=training_job.updated_at,
    )


@router.get("/training-jobs/{job_id}", response_model=VocalizationTrainingJobOut)
async def get_vocalization_training_job(job_id: str, session: SessionDep):
    """Fetch a vocalization training job by ID."""
    import json

    from humpback.models.classifier import ClassifierTrainingJob

    result = await session.execute(
        select(ClassifierTrainingJob).where(
            ClassifierTrainingJob.id == job_id,
            ClassifierTrainingJob.job_purpose == "vocalization",
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(404, f"Vocalization training job {job_id} not found")

    source_ids = (
        json.loads(job.source_detection_job_ids) if job.source_detection_job_ids else []
    )
    return VocalizationTrainingJobOut(
        id=job.id,
        status=job.status,
        name=job.name,
        job_purpose=job.job_purpose,
        source_detection_job_ids=source_ids,
        classifier_model_id=job.classifier_model_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/vocalization-models", response_model=list[VocalizationModelOut])
async def list_vocalization_models(session: SessionDep):
    """List classifier models trained for vocalization labeling."""
    import json

    from humpback.models.classifier import ClassifierModel

    result = await session.execute(
        select(ClassifierModel)
        .where(ClassifierModel.classifier_purpose == "vocalization")
        .order_by(ClassifierModel.created_at.desc())
    )
    models = result.scalars().all()
    return [
        VocalizationModelOut(
            id=m.id,
            name=m.name,
            model_version=m.model_version,
            vector_dim=m.vector_dim,
            classifier_purpose=m.classifier_purpose,
            training_summary=(
                json.loads(m.training_summary) if m.training_summary else None
            ),
            created_at=m.created_at,
            updated_at=m.updated_at,
        )
        for m in models
    ]


@router.post(
    "/predict/{detection_job_id}",
    response_model=list[PredictionRow],
)
async def predict_vocalization_labels(
    detection_job_id: str,
    body: PredictRequest,
    session: SessionDep,
    settings: SettingsDep,
):
    """Predict vocalization labels for all detection rows using a trained model."""
    import asyncio
    import json

    import joblib
    import numpy as np
    import pyarrow.parquet as pq

    from humpback.models.classifier import ClassifierModel
    from humpback.services.classifier_service import get_detection_job
    from humpback.storage import detection_embeddings_path

    job = await get_detection_job(session, detection_job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    # Load vocalization model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == body.vocalization_model_id)
    )
    model = cm_result.scalar_one_or_none()
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    if model.classifier_purpose != "vocalization":
        raise HTTPException(400, "Model is not a vocalization classifier")

    # Load detection embeddings
    emb_path = detection_embeddings_path(settings.storage_root, detection_job_id)
    if not emb_path.exists():
        raise HTTPException(404, "No stored embeddings for this detection job")

    table = pq.read_table(str(emb_path))
    filenames = table.column("filename").to_pylist()
    start_secs = table.column("start_sec").to_pylist()
    end_secs = table.column("end_sec").to_pylist()
    embeddings_col = table.column("embedding")
    vectors = np.array([row.as_py() for row in embeddings_col], dtype=np.float32)

    # Load pipeline and predict
    pipeline = await asyncio.to_thread(joblib.load, model.model_path)
    summary = json.loads(model.training_summary) if model.training_summary else {}
    class_names: list[str] = summary.get("class_names", [])

    probas = await asyncio.to_thread(pipeline.predict_proba, vectors)
    predictions = await asyncio.to_thread(pipeline.predict, vectors)

    # Build response
    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.storage import detection_row_store_path

    row_store = detection_row_store_path(settings.storage_root, detection_job_id)
    # Build row_id lookup from row store
    row_id_map: dict[str, str] = {}
    if row_store.exists():
        _fnames, rows = read_detection_row_store(row_store)
        for row in rows:
            key = f"{row.get('filename', '')}:{row.get('start_sec', '')}:{row.get('end_sec', '')}"
            row_id_map[key] = row.get("row_id", "")

    results: list[PredictionRow] = []
    for i in range(len(vectors)):
        key = f"{filenames[i]}:{start_secs[i]}:{end_secs[i]}"
        row_id = row_id_map.get(key, f"emb-{i}")

        pred_idx = int(predictions[i])
        predicted_label = (
            class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        )
        prob_dict = {
            name: round(float(probas[i][j]), 4) for j, name in enumerate(class_names)
        }
        confidence = float(probas[i][pred_idx]) if pred_idx < len(probas[i]) else 0.0

        results.append(
            PredictionRow(
                row_id=row_id,
                predicted_label=predicted_label,
                confidence=round(confidence, 4),
                probabilities=prob_dict,
            )
        )

    # Sort by confidence ascending (least confident first for active learning)
    results.sort(key=lambda r: r.confidence)
    return results


# ---- Sub-Window Annotations ----


@router.get(
    "/annotations/{detection_job_id}/{row_id}",
    response_model=list[AnnotationOut],
)
async def list_annotations(
    detection_job_id: str,
    row_id: str,
    session: SessionDep,
):
    """List all sub-window annotations for a detection row."""
    result = await session.execute(
        select(LabelingAnnotation)
        .where(LabelingAnnotation.detection_job_id == detection_job_id)
        .where(LabelingAnnotation.row_id == row_id)
        .order_by(LabelingAnnotation.start_offset_sec)
    )
    return [AnnotationOut.model_validate(a) for a in result.scalars().all()]


@router.post(
    "/annotations/{detection_job_id}/{row_id}",
    response_model=AnnotationOut,
    status_code=201,
)
async def create_annotation(
    detection_job_id: str,
    row_id: str,
    body: AnnotationCreate,
    session: SessionDep,
):
    """Create a sub-window annotation on a detection row."""
    if body.end_offset_sec <= body.start_offset_sec:
        raise HTTPException(400, "end_offset_sec must be greater than start_offset_sec")

    annotation = LabelingAnnotation(
        detection_job_id=detection_job_id,
        row_id=row_id,
        start_offset_sec=body.start_offset_sec,
        end_offset_sec=body.end_offset_sec,
        label=body.label,
        notes=body.notes,
    )
    session.add(annotation)
    await session.commit()
    await session.refresh(annotation)
    return AnnotationOut.model_validate(annotation)


@router.put(
    "/annotations/{annotation_id}",
    response_model=AnnotationOut,
)
async def update_annotation(
    annotation_id: str,
    body: AnnotationUpdate,
    session: SessionDep,
):
    """Update an existing sub-window annotation."""
    result = await session.execute(
        select(LabelingAnnotation).where(LabelingAnnotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        raise HTTPException(404, "Annotation not found")

    if body.start_offset_sec is not None:
        annotation.start_offset_sec = body.start_offset_sec
    if body.end_offset_sec is not None:
        annotation.end_offset_sec = body.end_offset_sec
    if body.label is not None:
        annotation.label = body.label
    if body.notes is not None:
        annotation.notes = body.notes

    await session.commit()
    await session.refresh(annotation)
    return AnnotationOut.model_validate(annotation)


@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: str,
    session: SessionDep,
):
    """Delete a sub-window annotation."""
    result = await session.execute(
        select(LabelingAnnotation).where(LabelingAnnotation.id == annotation_id)
    )
    annotation = result.scalar_one_or_none()
    if annotation is None:
        raise HTTPException(404, "Annotation not found")

    await session.delete(annotation)
    await session.commit()


# ---- Active Learning ----


@router.post(
    "/active-learning-cycle",
    response_model=ActiveLearningCycleResponse,
    status_code=201,
)
async def start_active_learning_cycle(
    body: ActiveLearningCycleRequest,
    session: SessionDep,
):
    """Queue a retrain cycle: create a new vocalization training job from current labels."""
    import json

    from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob
    from humpback.services.classifier_service import get_detection_job

    # Validate model exists and is vocalization type
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == body.vocalization_model_id)
    )
    model = cm_result.scalar_one_or_none()
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    if model.classifier_purpose != "vocalization":
        raise HTTPException(400, "Model is not a vocalization classifier")

    # Validate detection jobs
    for job_id in body.detection_job_ids:
        job = await get_detection_job(session, job_id)
        if job is None:
            raise HTTPException(404, f"Detection job {job_id} not found")

    # Check labels exist (vocalization labels + annotations)
    result = await session.execute(
        select(VocalizationLabel.label)
        .where(VocalizationLabel.detection_job_id.in_(body.detection_job_ids))
        .distinct()
    )
    distinct_labels = set(row[0] for row in result.all())
    ann_result = await session.execute(
        select(LabelingAnnotation.label)
        .where(LabelingAnnotation.detection_job_id.in_(body.detection_job_ids))
        .distinct()
    )
    distinct_labels |= set(row[0] for row in ann_result.all())
    if len(distinct_labels) < 2:
        raise HTTPException(
            400,
            f"Need at least 2 distinct labels, found {len(distinct_labels)}",
        )

    training_job = ClassifierTrainingJob(
        name=body.name,
        job_purpose="vocalization",
        source_detection_job_ids=json.dumps(body.detection_job_ids),
        positive_embedding_set_ids="[]",
        negative_embedding_set_ids="[]",
        model_version=model.model_version,
        window_size_seconds=model.window_size_seconds,
        target_sample_rate=model.target_sample_rate,
        feature_config=None,
        parameters=None,
    )
    session.add(training_job)
    await session.commit()
    await session.refresh(training_job)

    return ActiveLearningCycleResponse(
        training_job_id=training_job.id,
        status=training_job.status,
    )


@router.get(
    "/uncertainty-queue/{detection_job_id}",
    response_model=list[UncertaintyQueueRow],
)
async def get_uncertainty_queue(
    detection_job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    vocalization_model_id: str = Query(...),
):
    """Return detection rows sorted by prediction uncertainty (least confident first)."""
    import asyncio
    import json

    import joblib
    import numpy as np
    import pyarrow.parquet as pq

    from humpback.classifier.detection_rows import read_detection_row_store
    from humpback.models.classifier import ClassifierModel
    from humpback.services.classifier_service import get_detection_job
    from humpback.storage import detection_embeddings_path, detection_row_store_path

    job = await get_detection_job(session, detection_job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")

    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == vocalization_model_id)
    )
    model = cm_result.scalar_one_or_none()
    if model is None:
        raise HTTPException(404, "Vocalization model not found")

    # Load embeddings
    emb_path = detection_embeddings_path(settings.storage_root, detection_job_id)
    if not emb_path.exists():
        raise HTTPException(404, "No embeddings for this detection job")

    table = pq.read_table(str(emb_path))
    filenames = table.column("filename").to_pylist()
    start_secs = table.column("start_sec").to_pylist()
    end_secs = table.column("end_sec").to_pylist()
    embeddings_col = table.column("embedding")
    vectors = np.array([row.as_py() for row in embeddings_col], dtype=np.float32)

    # Load row store for row_ids and confidence
    row_store = detection_row_store_path(settings.storage_root, detection_job_id)
    row_meta: dict[str, dict[str, str]] = {}
    if row_store.exists():
        _fnames, rows = read_detection_row_store(row_store)
        for row in rows:
            key = f"{row.get('filename', '')}:{row.get('start_sec', '')}:{row.get('end_sec', '')}"
            row_meta[key] = row

    # Predict
    pipeline = await asyncio.to_thread(joblib.load, model.model_path)
    summary = json.loads(model.training_summary) if model.training_summary else {}
    class_names: list[str] = summary.get("class_names", [])

    probas = await asyncio.to_thread(pipeline.predict_proba, vectors)
    predictions = await asyncio.to_thread(pipeline.predict, vectors)

    results: list[UncertaintyQueueRow] = []
    for i in range(len(vectors)):
        key = f"{filenames[i]}:{start_secs[i]}:{end_secs[i]}"
        meta = row_meta.get(key, {})
        row_id = meta.get("row_id", f"emb-{i}")

        pred_idx = int(predictions[i])
        predicted_label = (
            class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        )
        confidence = float(probas[i][pred_idx]) if pred_idx < len(probas[i]) else 0.0
        prob_dict = {
            name: round(float(probas[i][j]), 4) for j, name in enumerate(class_names)
        }

        results.append(
            UncertaintyQueueRow(
                row_id=row_id,
                filename=filenames[i],
                start_sec=float(start_secs[i]),
                end_sec=float(end_secs[i]),
                avg_confidence=float(meta.get("avg_confidence", 0)),
                peak_confidence=float(meta.get("peak_confidence", 0)),
                predicted_label=predicted_label,
                prediction_confidence=round(confidence, 4),
                probabilities=prob_dict,
            )
        )

    # Sort by prediction confidence ascending (most uncertain first)
    results.sort(key=lambda r: r.prediction_confidence or 0)
    return results


@router.get(
    "/convergence/{vocalization_model_id}",
    response_model=ConvergenceMetrics,
)
async def get_convergence_metrics(
    vocalization_model_id: str,
    session: SessionDep,
):
    """Return convergence metrics for a vocalization model lineage."""
    import json

    from humpback.models.classifier import ClassifierModel, ClassifierTrainingJob

    # Get the target model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == vocalization_model_id)
    )
    model = cm_result.scalar_one_or_none()
    if model is None:
        raise HTTPException(404, "Model not found")

    # Find all vocalization models with the same model_version (lineage)
    all_models_result = await session.execute(
        select(ClassifierModel)
        .where(ClassifierModel.classifier_purpose == "vocalization")
        .where(ClassifierModel.model_version == model.model_version)
        .order_by(ClassifierModel.created_at)
    )
    all_models = list(all_models_result.scalars().all())

    # Extract accuracy trend from training summaries
    accuracy_trend: list[float] = []
    for mdl in all_models:
        if mdl.training_summary:
            summary = json.loads(mdl.training_summary)
            acc = summary.get("cv_accuracy")
            if acc is not None:
                accuracy_trend.append(round(float(acc), 4))

    # Get label distribution from the target model's source jobs
    label_dist: dict[str, int] = {}
    if model.training_job_id:
        tj_result = await session.execute(
            select(ClassifierTrainingJob).where(
                ClassifierTrainingJob.id == model.training_job_id
            )
        )
        tj = tj_result.scalar_one_or_none()
        if tj and tj.source_detection_job_ids:
            source_ids = json.loads(tj.source_detection_job_ids)
            label_result = await session.execute(
                select(VocalizationLabel.label).where(
                    VocalizationLabel.detection_job_id.in_(source_ids)
                )
            )
            for (label_val,) in label_result.all():
                label_dist[label_val] = label_dist.get(label_val, 0) + 1

    # Build uncertainty histogram from the target model's training summary
    uncertainty_histogram: list[dict[str, object]] = []
    if model.training_summary:
        summary = json.loads(model.training_summary)
        per_class = summary.get("per_class", {})
        for cls_name, cls_data in per_class.items():
            uncertainty_histogram.append(
                {
                    "class": cls_name,
                    "count": cls_data.get("count", 0),
                    "f1": cls_data.get("f1", 0),
                }
            )

    return ConvergenceMetrics(
        cycles_completed=len(all_models),
        label_distribution=label_dist,
        accuracy_trend=accuracy_trend,
        uncertainty_histogram=uncertainty_histogram,
    )
