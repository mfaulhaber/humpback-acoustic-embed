"""API router for vocalization type classification."""

import json
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationInferenceJob,
    VocalizationTrainingJob,
)
from humpback.schemas.vocalization import (
    VocalizationInferenceJobCreate,
    VocalizationInferenceJobOut,
    VocalizationModelOut,
    VocalizationPredictionRow,
    VocalizationTrainingJobCreate,
    VocalizationTrainingJobOut,
    VocalizationTrainingSourceOut,
    VocalizationTypeCreate,
    VocalizationTypeImportRequest,
    VocalizationTypeImportResponse,
    VocalizationTypeOut,
    VocalizationTypeUpdate,
)
from humpback.services.vocalization_service import (
    activate_model,
    create_type,
    delete_type,
    get_inference_job,
    get_model,
    get_training_job,
    import_types_from_embedding_sets,
    list_types,
    update_type,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vocalization", tags=["vocalization"])


# ---- Helpers ----


def _model_to_out(m: VocalizationClassifierModel) -> VocalizationModelOut:
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
        created_at=m.created_at,
    )


def _training_job_to_out(j: VocalizationTrainingJob) -> VocalizationTrainingJobOut:
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


def _inference_job_to_out(j: VocalizationInferenceJob) -> VocalizationInferenceJobOut:
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


# ---- Vocabulary ----


@router.get("/types", response_model=list[VocalizationTypeOut])
async def list_vocalization_types(session: SessionDep):
    types = await list_types(session)
    return [VocalizationTypeOut.model_validate(t) for t in types]


@router.post("/types", response_model=VocalizationTypeOut, status_code=201)
async def create_vocalization_type(body: VocalizationTypeCreate, session: SessionDep):
    if body.name.strip().lower() == "(negative)":
        raise HTTPException(
            400, '"(Negative)" is reserved and cannot be used as a type name'
        )
    try:
        vt = await create_type(session, body.name, body.description)
    except Exception as e:
        if "UNIQUE" in str(e).upper():
            raise HTTPException(409, f"Type '{body.name}' already exists") from e
        raise
    return VocalizationTypeOut.model_validate(vt)


@router.put("/types/{type_id}", response_model=VocalizationTypeOut)
async def update_vocalization_type(
    type_id: str, body: VocalizationTypeUpdate, session: SessionDep
):
    if body.name is not None and body.name.strip().lower() == "(negative)":
        raise HTTPException(
            400, '"(Negative)" is reserved and cannot be used as a type name'
        )
    vt = await update_type(session, type_id, body.name, body.description)
    if vt is None:
        raise HTTPException(404, "Vocalization type not found")
    return VocalizationTypeOut.model_validate(vt)


@router.delete("/types/{type_id}", status_code=204)
async def delete_vocalization_type(type_id: str, session: SessionDep):
    try:
        deleted = await delete_type(session, type_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    if not deleted:
        raise HTTPException(404, "Vocalization type not found")


@router.post("/types/import", response_model=VocalizationTypeImportResponse)
async def import_vocalization_types(
    body: VocalizationTypeImportRequest, session: SessionDep
):
    added, skipped = await import_types_from_embedding_sets(
        session, body.embedding_set_ids
    )
    return VocalizationTypeImportResponse(added=added, skipped=skipped)


# ---- Models ----


@router.get("/models", response_model=list[VocalizationModelOut])
async def list_vocalization_models(session: SessionDep):
    result = await session.execute(
        select(VocalizationClassifierModel).order_by(
            VocalizationClassifierModel.created_at.desc()
        )
    )
    return [_model_to_out(m) for m in result.scalars().all()]


@router.get("/models/{model_id}", response_model=VocalizationModelOut)
async def get_vocalization_model(model_id: str, session: SessionDep):
    model = await get_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    return _model_to_out(model)


@router.put("/models/{model_id}/activate", response_model=VocalizationModelOut)
async def activate_vocalization_model(model_id: str, session: SessionDep):
    model = await activate_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")
    return _model_to_out(model)


@router.get(
    "/models/{model_id}/training-source",
    response_model=VocalizationTrainingSourceOut,
)
async def get_model_training_source(model_id: str, session: SessionDep):
    """Return the source_config and parameters from the training job that
    produced this model."""
    model = await get_model(session, model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")

    # Find the training job that produced this model
    result = await session.execute(
        select(VocalizationTrainingJob).where(
            VocalizationTrainingJob.vocalization_model_id == model_id
        )
    )
    training_job = result.scalar_one_or_none()
    if training_job is None:
        return VocalizationTrainingSourceOut()

    return VocalizationTrainingSourceOut(
        source_config=json.loads(training_job.source_config),
        parameters=json.loads(training_job.parameters)
        if training_job.parameters
        else None,
    )


# ---- Training Jobs ----


@router.post(
    "/training-jobs", response_model=VocalizationTrainingJobOut, status_code=201
)
async def create_vocalization_training_job(
    body: VocalizationTrainingJobCreate, session: SessionDep
):
    job = VocalizationTrainingJob(
        source_config=json.dumps(body.source_config.model_dump()),
        parameters=json.dumps(body.parameters) if body.parameters else None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return _training_job_to_out(job)


@router.get("/training-jobs", response_model=list[VocalizationTrainingJobOut])
async def list_vocalization_training_jobs(session: SessionDep):
    result = await session.execute(
        select(VocalizationTrainingJob).order_by(
            VocalizationTrainingJob.created_at.desc()
        )
    )
    return [_training_job_to_out(j) for j in result.scalars().all()]


@router.get("/training-jobs/{job_id}", response_model=VocalizationTrainingJobOut)
async def get_vocalization_training_job_detail(job_id: str, session: SessionDep):
    job = await get_training_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization training job not found")
    return _training_job_to_out(job)


# ---- Inference Jobs ----


@router.post(
    "/inference-jobs", response_model=VocalizationInferenceJobOut, status_code=201
)
async def create_vocalization_inference_job(
    body: VocalizationInferenceJobCreate, session: SessionDep
):
    # Validate model exists
    model = await get_model(session, body.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Vocalization model not found")

    job = VocalizationInferenceJob(
        vocalization_model_id=body.vocalization_model_id,
        source_type=body.source_type,
        source_id=body.source_id,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return _inference_job_to_out(job)


@router.get("/inference-jobs", response_model=list[VocalizationInferenceJobOut])
async def list_vocalization_inference_jobs(session: SessionDep):
    result = await session.execute(
        select(VocalizationInferenceJob).order_by(
            VocalizationInferenceJob.created_at.desc()
        )
    )
    return [_inference_job_to_out(j) for j in result.scalars().all()]


@router.get("/inference-jobs/{job_id}", response_model=VocalizationInferenceJobOut)
async def get_vocalization_inference_job_detail(job_id: str, session: SessionDep):
    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    return _inference_job_to_out(job)


@router.get(
    "/inference-jobs/{job_id}/results",
    response_model=list[VocalizationPredictionRow],
)
async def get_vocalization_inference_results(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    thresholds: str | None = Query(None),
    sort: str | None = Query(None),
):
    """Paginated results with optional threshold overrides.

    thresholds query param: JSON string like {"whup": 0.3, "moan": 0.6}
    """
    from pathlib import Path

    from humpback.classifier.vocalization_inference import read_predictions

    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    if job.status != "complete":
        raise HTTPException(400, f"Job is {job.status}, not complete")
    if not job.output_path:
        raise HTTPException(404, "No output for this job")

    model = await get_model(session, job.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Model not found")

    vocabulary: list[str] = json.loads(model.vocabulary_snapshot)
    stored_thresholds: dict[str, float] = json.loads(model.per_class_thresholds)

    threshold_overrides: dict[str, float] | None = None
    if thresholds:
        try:
            threshold_overrides = json.loads(thresholds)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid thresholds JSON")

    rows = read_predictions(
        Path(job.output_path), vocabulary, stored_thresholds, threshold_overrides
    )

    # Server-side sort
    if sort == "confidence_desc":
        rows.sort(key=lambda r: r.get("confidence") or -1.0, reverse=True)

    # Paginate
    page = rows[offset : offset + limit]
    return [
        VocalizationPredictionRow(
            filename=r["filename"],
            start_sec=r["start_sec"],
            end_sec=r["end_sec"],
            start_utc=r.get("start_utc"),
            end_utc=r.get("end_utc"),
            confidence=r.get("confidence"),
            scores=r["scores"],
            tags=r["tags"],
        )
        for r in page
    ]


@router.get("/inference-jobs/{job_id}/export")
async def export_vocalization_inference(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    thresholds: str | None = Query(None),
):
    """Export results as TSV with applied thresholds."""
    import io
    from pathlib import Path

    from humpback.classifier.vocalization_inference import read_predictions

    job = await get_inference_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Vocalization inference job not found")
    if job.status != "complete":
        raise HTTPException(400, f"Job is {job.status}, not complete")
    if not job.output_path:
        raise HTTPException(404, "No output for this job")

    model = await get_model(session, job.vocalization_model_id)
    if model is None:
        raise HTTPException(404, "Model not found")

    vocabulary: list[str] = json.loads(model.vocabulary_snapshot)
    stored_thresholds: dict[str, float] = json.loads(model.per_class_thresholds)

    threshold_overrides: dict[str, float] | None = None
    if thresholds:
        try:
            threshold_overrides = json.loads(thresholds)
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid thresholds JSON")

    rows = read_predictions(
        Path(job.output_path), vocabulary, stored_thresholds, threshold_overrides
    )

    # Build TSV
    buf = io.StringIO()
    header_cols = ["filename", "start_sec", "end_sec"]
    if rows and "start_utc" in rows[0]:
        header_cols.extend(["start_utc", "end_utc"])
    header_cols.extend(vocabulary)
    header_cols.append("tags")
    buf.write("\t".join(header_cols) + "\n")

    for r in rows:
        vals: list[str] = [r["filename"], str(r["start_sec"]), str(r["end_sec"])]
        if "start_utc" in r:
            vals.extend([str(r["start_utc"]), str(r["end_utc"])])
        for t in vocabulary:
            vals.append(str(r["scores"].get(t, 0.0)))
        vals.append(",".join(r["tags"]))
        buf.write("\t".join(vals) + "\n")

    content = buf.getvalue()

    return StreamingResponse(
        iter([content]),
        media_type="text/tab-separated-values",
        headers={
            "Content-Disposition": f'attachment; filename="vocalization_{job_id}.tsv"'
        },
    )
