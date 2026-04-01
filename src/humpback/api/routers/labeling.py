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
    OrphanedLabelDetail,
    RefreshApplyResponse,
    RefreshPreviewResponse,
    TrainingSummary,
    NeighborHit,
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
    start_utc: float = Query(...),
    end_utc: float = Query(...),
):
    """List all vocalization labels for a detection row."""
    result = await session.execute(
        select(VocalizationLabel)
        .where(VocalizationLabel.detection_job_id == detection_job_id)
        .where(VocalizationLabel.start_utc == start_utc)
        .where(VocalizationLabel.end_utc == end_utc)
        .order_by(VocalizationLabel.created_at)
    )
    return [VocalizationLabelOut.model_validate(r) for r in result.scalars().all()]


@router.post(
    "/vocalization-labels/{detection_job_id}",
    response_model=VocalizationLabelOut,
    status_code=201,
)
async def create_vocalization_label(
    detection_job_id: str,
    body: VocalizationLabelCreate,
    session: SessionDep,
    start_utc: float = Query(...),
    end_utc: float = Query(...),
):
    """Add a vocalization type label to a detection row."""
    from sqlalchemy import and_, delete

    from humpback.models.classifier import DetectionJob

    # Look up the detection job's current row_store_version.
    dj_result = await session.execute(
        select(DetectionJob.row_store_version).where(
            DetectionJob.id == detection_job_id
        )
    )
    dj_version = dj_result.scalar_one_or_none()

    # Mutual exclusivity: "(Negative)" and type labels cannot coexist
    same_window = and_(
        VocalizationLabel.detection_job_id == detection_job_id,
        VocalizationLabel.start_utc == start_utc,
        VocalizationLabel.end_utc == end_utc,
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
        start_utc=start_utc,
        end_utc=end_utc,
        label=body.label,
        confidence=body.confidence,
        source=body.source,
        notes=body.notes,
        row_store_version_at_import=dj_version,
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


# ---- Refresh / Reconciliation ----


def _normalize_utc_key(v: float) -> float:
    """Round UTC to 6 decimal places for stable comparison."""
    return round(v, 6)


async def _get_detection_job_or_404(session, detection_job_id: str):
    from humpback.models.classifier import DetectionJob

    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == detection_job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return job


@router.post(
    "/vocalization-labels/{detection_job_id}/refresh",
    response_model=RefreshPreviewResponse,
)
async def refresh_preview(
    detection_job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """Preview reconciliation between row store and vocalization labels."""
    from humpback.storage import detection_row_store_path

    job = await _get_detection_job_or_404(session, detection_job_id)

    rs_path = detection_row_store_path(settings.storage_root, job.id)
    if not rs_path.exists():
        raise HTTPException(400, "Detection row store not available")

    _fieldnames, rows = read_detection_row_store(rs_path)
    row_keys: set[tuple[float, float]] = set()
    for r in rows:
        s = _normalize_utc_key(float(r.get("start_utc", "0")))
        e = _normalize_utc_key(float(r.get("end_utc", "0")))
        row_keys.add((s, e))

    result = await session.execute(
        select(VocalizationLabel).where(
            VocalizationLabel.detection_job_id == detection_job_id
        )
    )
    all_labels = result.scalars().all()

    matched_count = 0
    orphaned: list[OrphanedLabelDetail] = []
    for lbl in all_labels:
        key = (_normalize_utc_key(lbl.start_utc), _normalize_utc_key(lbl.end_utc))
        if key in row_keys:
            matched_count += 1
        else:
            orphaned.append(
                OrphanedLabelDetail(
                    id=lbl.id,
                    start_utc=lbl.start_utc,
                    end_utc=lbl.end_utc,
                    label=lbl.label,
                )
            )

    return RefreshPreviewResponse(
        matched_count=matched_count,
        orphaned_count=len(orphaned),
        orphaned_labels=orphaned,
        current_version=job.row_store_version or 1,
    )


@router.post(
    "/vocalization-labels/{detection_job_id}/refresh/apply",
    response_model=RefreshApplyResponse,
)
async def refresh_apply(
    detection_job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """Delete orphaned vocalization labels and update version on survivors."""
    from sqlalchemy import delete as sa_delete

    from humpback.storage import detection_row_store_path

    job = await _get_detection_job_or_404(session, detection_job_id)

    rs_path = detection_row_store_path(settings.storage_root, job.id)
    if not rs_path.exists():
        raise HTTPException(400, "Detection row store not available")

    _fieldnames, rows = read_detection_row_store(rs_path)
    row_keys: set[tuple[float, float]] = set()
    for r in rows:
        s = _normalize_utc_key(float(r.get("start_utc", "0")))
        e = _normalize_utc_key(float(r.get("end_utc", "0")))
        row_keys.add((s, e))

    result = await session.execute(
        select(VocalizationLabel).where(
            VocalizationLabel.detection_job_id == detection_job_id
        )
    )
    all_labels = result.scalars().all()

    orphaned_ids: list[str] = []
    surviving_ids: list[str] = []
    for lbl in all_labels:
        key = (_normalize_utc_key(lbl.start_utc), _normalize_utc_key(lbl.end_utc))
        if key in row_keys:
            surviving_ids.append(lbl.id)
        else:
            orphaned_ids.append(lbl.id)

    current_version = job.row_store_version or 1

    # Delete orphaned labels
    if orphaned_ids:
        await session.execute(
            sa_delete(VocalizationLabel).where(VocalizationLabel.id.in_(orphaned_ids))
        )

    # Update version on surviving labels
    if surviving_ids:
        from sqlalchemy import update

        await session.execute(
            update(VocalizationLabel)
            .where(VocalizationLabel.id.in_(surviving_ids))
            .values(row_store_version_at_import=current_version)
        )

    await session.commit()

    return RefreshApplyResponse(
        deleted_count=len(orphaned_ids),
        surviving_count=len(surviving_ids),
        current_version=current_version,
    )


# ---- Label Vocabulary ----


@router.get("/label-vocabulary", response_model=list[str])
async def get_label_vocabulary(session: SessionDep):
    """Return distinct vocalization type labels used across all labels."""
    result = await session.execute(
        select(VocalizationLabel.label).distinct().order_by(VocalizationLabel.label)
    )
    return [row[0] for row in result.all()]


# ---- Labeling Summary ----


def _utc_key(start_utc: float, end_utc: float) -> str:
    return f"{start_utc}:{end_utc}"


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
            VocalizationLabel.start_utc,
            VocalizationLabel.end_utc,
            VocalizationLabel.label,
        ).where(VocalizationLabel.detection_job_id == detection_job_id)
    )
    label_rows = result.all()

    labeled_keys: set[str] = set()
    label_counts: dict[str, int] = {}
    for s_utc, e_utc, label in label_rows:
        labeled_keys.add(_utc_key(s_utc, e_utc))
        label_counts[label] = label_counts.get(label, 0) + 1

    return LabelingSummary(
        total_rows=total_rows,
        labeled_rows=len(labeled_keys),
        unlabeled_rows=total_rows - len(labeled_keys),
        label_distribution=label_counts,
    )


@router.get("/training-summary", response_model=TrainingSummary)
async def get_training_summary(session: SessionDep):
    """Aggregate label stats across all detection jobs for training readiness."""
    # Vocalization labels
    result = await session.execute(
        select(
            VocalizationLabel.detection_job_id,
            VocalizationLabel.start_utc,
            VocalizationLabel.end_utc,
            VocalizationLabel.label,
        )
    )
    labeled_job_ids: set[str] = set()
    labeled_row_keys: set[str] = set()
    label_counts: dict[str, int] = {}
    for job_id, s_utc, e_utc, label in result.all():
        labeled_job_ids.add(job_id)
        labeled_row_keys.add(f"{job_id}:{_utc_key(s_utc, e_utc)}")
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
    from humpback.classifier.detection_rows import parse_recording_timestamp
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

    # Resolve UTC to file-relative coords for embedding lookup
    import pyarrow.parquet as pq

    emb_table = pq.read_table(
        str(emb_path), columns=["filename", "start_sec", "end_sec"]
    )
    lookup_filename = None
    lookup_start = None
    lookup_end = None
    for i in range(emb_table.num_rows):
        fname = emb_table.column("filename")[i].as_py()
        s = float(emb_table.column("start_sec")[i].as_py())
        e = float(emb_table.column("end_sec")[i].as_py())
        ts = parse_recording_timestamp(fname)
        base_epoch = ts.timestamp() if ts else 0.0
        if (
            abs(base_epoch + s - body.start_utc) < 0.01
            and abs(base_epoch + e - body.end_utc) < 0.01
        ):
            lookup_filename = fname
            lookup_start = s
            lookup_end = e
            break

    if lookup_filename is None or lookup_start is None or lookup_end is None:
        raise HTTPException(404, "Embedding not found for specified UTC range")

    embedding = read_detection_embedding(
        emb_path, lookup_filename, lookup_start, lookup_end
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
