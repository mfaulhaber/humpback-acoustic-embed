"""Autoresearch candidate endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.classifier import (
    AutoresearchCandidateDetailOut,
    AutoresearchCandidateImport,
    AutoresearchCandidateSummaryOut,
    AutoresearchCandidateTrainingJobCreate,
    ClassifierTrainingJobOut,
)
from humpback.schemas.converters import (
    autoresearch_candidate_to_detail as _autoresearch_candidate_to_detail,
    autoresearch_candidate_to_summary as _autoresearch_candidate_to_summary,
    classifier_training_job_to_out as _training_job_to_out,
)
from humpback.services import classifier_service

router = APIRouter()


@router.post("/autoresearch-candidates/import", status_code=201)
async def import_autoresearch_candidate(
    body: AutoresearchCandidateImport,
    session: SessionDep,
    settings: SettingsDep,
) -> AutoresearchCandidateDetailOut:
    try:
        candidate = await classifier_service.import_autoresearch_candidate(
            session,
            settings.storage_root,
            body.manifest_path,
            body.best_run_path,
            comparison_path=body.comparison_path,
            top_false_positives_path=body.top_false_positives_path,
            name=body.name,
            source_model_id_override=body.source_model_id_override,
            source_model_name_override=body.source_model_name_override,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _autoresearch_candidate_to_detail(candidate)


@router.get("/autoresearch-candidates")
async def list_autoresearch_candidates(
    session: SessionDep,
) -> list[AutoresearchCandidateSummaryOut]:
    candidates = await classifier_service.list_autoresearch_candidates(session)
    return [_autoresearch_candidate_to_summary(candidate) for candidate in candidates]


@router.get("/autoresearch-candidates/{candidate_id}")
async def get_autoresearch_candidate(
    candidate_id: str,
    session: SessionDep,
) -> AutoresearchCandidateDetailOut:
    candidate = await classifier_service.get_autoresearch_candidate(
        session, candidate_id
    )
    if candidate is None:
        raise HTTPException(404, "Autoresearch candidate not found")

    replay_verification = None
    if candidate.new_model_id:
        from humpback.models.classifier import ClassifierModel

        model = await session.get(ClassifierModel, candidate.new_model_id)
        if model and model.training_summary:
            summary = json.loads(model.training_summary)
            replay_verification = summary.get("replay_verification")

    return _autoresearch_candidate_to_detail(
        candidate, replay_verification=replay_verification
    )


@router.post("/autoresearch-candidates/{candidate_id}/training-jobs", status_code=201)
async def create_training_job_from_autoresearch_candidate(
    candidate_id: str,
    body: AutoresearchCandidateTrainingJobCreate,
    session: SessionDep,
) -> ClassifierTrainingJobOut:
    try:
        job = await classifier_service.create_training_job_from_autoresearch_candidate(
            session,
            candidate_id,
            body.new_model_name,
            notes=body.notes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _training_job_to_out(job)
