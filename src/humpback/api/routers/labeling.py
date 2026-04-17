"""API router for vocalization type labeling workflows."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.classifier.detection_rows import (
    read_detection_row_store,
)
from humpback.models.labeling import VocalizationLabel
from humpback.schemas.labeling import (
    DetectionNeighborsRequest,
    DetectionNeighborsResponse,
    LabelingSummary,
    TimelineVocalizationLabel,
    TrainingSummary,
    NeighborHit,
    VocalizationLabelBatchRequest,
    VocalizationLabelCreate,
    VocalizationLabelOut,
    VocalizationLabelUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeling", tags=["labeling"])


# ---- Vocalization Label CRUD ----


@router.get(
    "/vocalization-labels/{detection_job_id}",
    response_model=list[VocalizationLabelOut],
)
async def list_vocalization_labels(
    detection_job_id: str,
    session: SessionDep,
    row_id: str = Query(...),
):
    """List all vocalization labels for a detection row."""
    result = await session.execute(
        select(VocalizationLabel)
        .where(VocalizationLabel.detection_job_id == detection_job_id)
        .where(VocalizationLabel.row_id == row_id)
        .order_by(VocalizationLabel.created_at)
    )
    return [VocalizationLabelOut.model_validate(r) for r in result.scalars().all()]


@router.get(
    "/vocalization-labels/{detection_job_id}/all",
    response_model=list[TimelineVocalizationLabel],
)
async def list_all_vocalization_labels(
    detection_job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """All vocalization labels for timeline overlay — manual + inference predictions."""
    import json
    from pathlib import Path

    from humpback.classifier.vocalization_inference import read_predictions
    from humpback.models.vocalization import (
        VocalizationClassifierModel,
        VocalizationInferenceJob,
    )
    from humpback.storage import detection_row_store_path

    await _get_detection_job_or_404(session, detection_job_id)

    # Build row_id -> (start_utc, end_utc) lookup from row store
    rs_path = detection_row_store_path(settings.storage_root, detection_job_id)
    utc_by_row_id: dict[str, tuple[float, float]] = {}
    if rs_path.exists():
        _fields, rs_rows = read_detection_row_store(rs_path)
        for r in rs_rows:
            rid = r.get("row_id", "")
            if rid:
                utc_by_row_id[rid] = (
                    float(r.get("start_utc", "0")),
                    float(r.get("end_utc", "0")),
                )

    # 1. Manual labels from the DB — resolve UTC via row store
    result = await session.execute(
        select(VocalizationLabel)
        .where(VocalizationLabel.detection_job_id == detection_job_id)
        .order_by(VocalizationLabel.created_at)
    )
    out: list[TimelineVocalizationLabel] = []
    manual_keys: set[tuple[str, str]] = set()  # (row_id, label)
    for r in result.scalars().all():
        utc = utc_by_row_id.get(r.row_id)
        if utc is None:
            continue  # Row no longer exists in row store
        out.append(
            TimelineVocalizationLabel(
                start_utc=utc[0],
                end_utc=utc[1],
                label=r.label,
                confidence=r.confidence,
                source=r.source,
            )
        )
        manual_keys.add((r.row_id, r.label))

    # 2. Inference predictions — find completed inference jobs for this detection job
    inf_result = await session.execute(
        select(VocalizationInferenceJob)
        .where(VocalizationInferenceJob.source_type == "detection_job")
        .where(VocalizationInferenceJob.source_id == detection_job_id)
        .where(VocalizationInferenceJob.status == "complete")
        .order_by(VocalizationInferenceJob.created_at.desc())
    )
    inf_jobs = inf_result.scalars().all()

    # Use the most recent completed inference job
    if inf_jobs:
        inf_job = inf_jobs[0]
        if inf_job.output_path and Path(inf_job.output_path).exists():
            model_result = await session.execute(
                select(VocalizationClassifierModel).where(
                    VocalizationClassifierModel.id == inf_job.vocalization_model_id
                )
            )
            model = model_result.scalar_one_or_none()
            if model:
                vocabulary: list[str] = json.loads(model.vocabulary_snapshot)
                thresholds: dict[str, float] = json.loads(model.per_class_thresholds)
                predictions = read_predictions(
                    Path(inf_job.output_path), vocabulary, thresholds
                )
                for pred in predictions:
                    rid = pred.get("row_id")
                    if rid is None:
                        continue
                    utc = utc_by_row_id.get(rid)
                    if utc is None:
                        continue
                    for tag in pred["tags"]:
                        if (rid, tag) in manual_keys:
                            continue
                        score = pred["scores"].get(tag)
                        out.append(
                            TimelineVocalizationLabel(
                                start_utc=utc[0],
                                end_utc=utc[1],
                                label=tag,
                                confidence=score,
                                source="inference",
                            )
                        )

    out.sort(key=lambda x: (x.start_utc, x.source, x.label))
    return out


@router.post(
    "/vocalization-labels/{detection_job_id}",
    response_model=VocalizationLabelOut,
    status_code=201,
)
async def create_vocalization_label(
    detection_job_id: str,
    body: VocalizationLabelCreate,
    session: SessionDep,
    row_id: str = Query(...),
):
    """Add a vocalization type label to a detection row."""
    from sqlalchemy import and_, delete

    # Mutual exclusivity: "(Negative)" and type labels cannot coexist
    same_window = and_(
        VocalizationLabel.detection_job_id == detection_job_id,
        VocalizationLabel.row_id == row_id,
    )
    if body.label == "(Negative)":
        # Remove any existing type labels on this window
        await session.execute(
            delete(VocalizationLabel).where(
                same_window, VocalizationLabel.label != "(Negative)"
            )
        )
    else:
        # Remove any existing "(Negative)" label on this window
        await session.execute(
            delete(VocalizationLabel).where(
                same_window, VocalizationLabel.label == "(Negative)"
            )
        )

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


@router.patch(
    "/vocalization-labels/{detection_job_id}/batch",
    response_model=list[TimelineVocalizationLabel],
)
async def batch_vocalization_labels(
    detection_job_id: str,
    body: VocalizationLabelBatchRequest,
    session: SessionDep,
    settings: SettingsDep,
):
    """Atomically apply a batch of vocalization label add/delete edits."""
    from sqlalchemy import and_, delete

    await _get_detection_job_or_404(session, detection_job_id)

    for edit in body.edits:
        same_window = and_(
            VocalizationLabel.detection_job_id == detection_job_id,
            VocalizationLabel.row_id == edit.row_id,
        )

        if edit.action == "add":
            # Idempotent: skip if already exists
            existing = await session.execute(
                select(VocalizationLabel).where(
                    same_window,
                    VocalizationLabel.label == edit.label,
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue

            # Mutual exclusivity
            if edit.label == "(Negative)":
                await session.execute(
                    delete(VocalizationLabel).where(
                        same_window, VocalizationLabel.label != "(Negative)"
                    )
                )
            else:
                await session.execute(
                    delete(VocalizationLabel).where(
                        same_window, VocalizationLabel.label == "(Negative)"
                    )
                )

            session.add(
                VocalizationLabel(
                    detection_job_id=detection_job_id,
                    row_id=edit.row_id,
                    label=edit.label,
                    source=edit.source,
                )
            )

        elif edit.action == "delete":
            await session.execute(
                delete(VocalizationLabel).where(
                    same_window,
                    VocalizationLabel.label == edit.label,
                )
            )

    await session.commit()

    # Return full updated label set (reuse the /all endpoint logic)
    return await list_all_vocalization_labels(detection_job_id, session, settings)


async def _get_detection_job_or_404(session, detection_job_id: str):
    from humpback.models.classifier import DetectionJob

    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == detection_job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return job


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
    from humpback.storage import detection_row_store_path

    row_store = detection_row_store_path(settings.storage_root, detection_job_id)
    if not row_store.exists():
        raise HTTPException(404, "Detection row store not found")

    _fieldnames, rows = read_detection_row_store(row_store)
    total_rows = len(rows)

    # Get all vocalization labels for this job
    result = await session.execute(
        select(
            VocalizationLabel.row_id,
            VocalizationLabel.label,
        ).where(VocalizationLabel.detection_job_id == detection_job_id)
    )
    label_rows = result.all()

    labeled_row_ids: set[str] = set()
    label_counts: dict[str, int] = {}
    for row_id, label in label_rows:
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

    return TrainingSummary(
        labeled_job_ids=sorted(labeled_job_ids),
        labeled_rows=len(labeled_row_keys),
        label_distribution=label_counts,
    )


# ---- Detection Neighbors (Vector Search) ----


def _infer_label_from_folder(folder_path: str | None) -> str | None:
    if not folder_path:
        return None
    parts = Path(folder_path).parts
    if not parts:
        return None
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

    # Resolve model_version from the classifier model
    cm_result = await session.execute(
        select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    cm = cm_result.scalar_one_or_none()
    if cm is None:
        raise HTTPException(404, "Classifier model not found for detection job")

    emb_path = detection_embeddings_path(
        settings.storage_root, job.id, cm.model_version
    )
    if not emb_path.exists():
        raise HTTPException(404, "No stored embeddings for this detection job")

    embedding = read_detection_embedding(emb_path, body.row_id)
    if embedding is None:
        raise HTTPException(404, "Embedding not found for specified detection row")

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
