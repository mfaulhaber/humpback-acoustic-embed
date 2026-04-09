"""Hyperparameter tuning endpoints — manifests, searches, and search-space defaults."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.hyperparameter import (
    HyperparameterManifest,
    HyperparameterSearchJob,
)
from humpback.schemas.hyperparameter import (
    ManifestCreate,
    ManifestDetail,
    ManifestSummary,
    SearchCreate,
    SearchDetail,
    SearchSpaceDefaults,
    SearchSummary,
)
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
from humpback.services.hyperparameter_service.search_space import DEFAULT_SEARCH_SPACE
from humpback.storage import (
    hyperparameter_manifest_dir,
    hyperparameter_search_results_dir,
)

router = APIRouter(prefix="/hyperparameter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json(val: str | None) -> Any:
    if val is None:
        return None
    return json.loads(val)


def _manifest_to_summary(m: HyperparameterManifest) -> ManifestSummary:
    return ManifestSummary(
        id=m.id,
        name=m.name,
        status=m.status,
        training_job_ids=json.loads(m.training_job_ids),
        detection_job_ids=json.loads(m.detection_job_ids),
        split_ratio=json.loads(m.split_ratio),
        seed=m.seed,
        example_count=m.example_count,
        error_message=m.error_message,
        created_at=m.created_at,
        completed_at=m.completed_at,
    )


def _manifest_to_detail(m: HyperparameterManifest) -> ManifestDetail:
    return ManifestDetail(
        id=m.id,
        name=m.name,
        status=m.status,
        training_job_ids=json.loads(m.training_job_ids),
        detection_job_ids=json.loads(m.detection_job_ids),
        split_ratio=json.loads(m.split_ratio),
        seed=m.seed,
        example_count=m.example_count,
        error_message=m.error_message,
        created_at=m.created_at,
        completed_at=m.completed_at,
        manifest_path=m.manifest_path,
        split_summary=_parse_json(m.split_summary),
        detection_job_summaries=_parse_json(m.detection_job_summaries),
    )


def _search_to_summary(
    s: HyperparameterSearchJob, manifest_name: str | None = None
) -> SearchSummary:
    return SearchSummary(
        id=s.id,
        name=s.name,
        status=s.status,
        manifest_id=s.manifest_id,
        manifest_name=manifest_name,
        n_trials=s.n_trials,
        seed=s.seed,
        objective_name=s.objective_name,
        trials_completed=s.trials_completed,
        best_objective=s.best_objective,
        comparison_model_id=s.comparison_model_id,
        error_message=s.error_message,
        created_at=s.created_at,
        completed_at=s.completed_at,
    )


def _search_to_detail(
    s: HyperparameterSearchJob, manifest_name: str | None = None
) -> SearchDetail:
    return SearchDetail(
        id=s.id,
        name=s.name,
        status=s.status,
        manifest_id=s.manifest_id,
        manifest_name=manifest_name,
        n_trials=s.n_trials,
        seed=s.seed,
        objective_name=s.objective_name,
        trials_completed=s.trials_completed,
        best_objective=s.best_objective,
        comparison_model_id=s.comparison_model_id,
        error_message=s.error_message,
        created_at=s.created_at,
        completed_at=s.completed_at,
        search_space=json.loads(s.search_space),
        results_dir=s.results_dir,
        best_config=_parse_json(s.best_config),
        best_metrics=_parse_json(s.best_metrics),
        comparison_threshold=s.comparison_threshold,
        comparison_result=_parse_json(s.comparison_result),
    )


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


@router.post("/manifests", status_code=201)
async def create_manifest(
    body: ManifestCreate,
    session: SessionDep,
) -> ManifestSummary:
    manifest = HyperparameterManifest(
        id=str(uuid.uuid4()),
        name=body.name,
        status="queued",
        training_job_ids=json.dumps(body.training_job_ids),
        detection_job_ids=json.dumps(body.detection_job_ids),
        split_ratio=json.dumps(body.split_ratio),
        seed=body.seed,
    )
    session.add(manifest)
    await session.commit()
    await session.refresh(manifest)
    return _manifest_to_summary(manifest)


@router.get("/manifests")
async def list_manifests(session: SessionDep) -> list[ManifestSummary]:
    stmt = select(HyperparameterManifest).order_by(
        HyperparameterManifest.created_at.desc()
    )
    result = await session.execute(stmt)
    return [_manifest_to_summary(m) for m in result.scalars().all()]


@router.get("/manifests/{manifest_id}")
async def get_manifest(manifest_id: str, session: SessionDep) -> ManifestDetail:
    manifest = await session.get(HyperparameterManifest, manifest_id)
    if manifest is None:
        raise HTTPException(404, "Manifest not found")
    return _manifest_to_detail(manifest)


@router.delete("/manifests/{manifest_id}")
async def delete_manifest(
    manifest_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, str]:
    manifest = await session.get(HyperparameterManifest, manifest_id)
    if manifest is None:
        raise HTTPException(404, "Manifest not found")
    # Check for referencing search jobs
    stmt = select(HyperparameterSearchJob).where(
        HyperparameterSearchJob.manifest_id == manifest_id
    )
    result = await session.execute(stmt)
    refs = result.scalars().all()
    if refs:
        raise HTTPException(
            409,
            f"Manifest is referenced by {len(refs)} search job(s); delete them first",
        )
    # Remove artifact directory
    artifact_dir = hyperparameter_manifest_dir(settings.storage_root, manifest_id)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    await session.delete(manifest)
    await session.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------


@router.post("/searches", status_code=201)
async def create_search(
    body: SearchCreate,
    session: SessionDep,
) -> SearchSummary:
    # Verify manifest exists
    manifest = await session.get(HyperparameterManifest, body.manifest_id)
    if manifest is None:
        raise HTTPException(400, "Manifest not found")
    search_space = (
        body.search_space if body.search_space is not None else DEFAULT_SEARCH_SPACE
    )
    search = HyperparameterSearchJob(
        id=str(uuid.uuid4()),
        name=body.name,
        status="queued",
        manifest_id=body.manifest_id,
        search_space=json.dumps(search_space),
        n_trials=body.n_trials,
        seed=body.seed,
        comparison_model_id=body.comparison_model_id,
        comparison_threshold=body.comparison_threshold,
    )
    session.add(search)
    await session.commit()
    await session.refresh(search)
    return _search_to_summary(search, manifest_name=manifest.name)


@router.get("/searches")
async def list_searches(session: SessionDep) -> list[SearchSummary]:
    stmt = select(HyperparameterSearchJob).order_by(
        HyperparameterSearchJob.created_at.desc()
    )
    result = await session.execute(stmt)
    searches = result.scalars().all()
    # Batch-load manifest names
    manifest_ids = {s.manifest_id for s in searches}
    manifest_names: dict[str, str] = {}
    if manifest_ids:
        m_stmt = select(HyperparameterManifest).where(
            HyperparameterManifest.id.in_(manifest_ids)
        )
        m_result = await session.execute(m_stmt)
        for m in m_result.scalars().all():
            manifest_names[m.id] = m.name
    return [
        _search_to_summary(s, manifest_name=manifest_names.get(s.manifest_id))
        for s in searches
    ]


@router.get("/searches/{search_id}")
async def get_search(search_id: str, session: SessionDep) -> SearchDetail:
    search = await session.get(HyperparameterSearchJob, search_id)
    if search is None:
        raise HTTPException(404, "Search not found")
    manifest = await session.get(HyperparameterManifest, search.manifest_id)
    manifest_name = manifest.name if manifest else None
    return _search_to_detail(search, manifest_name=manifest_name)


@router.get("/searches/{search_id}/history")
async def get_search_history(
    search_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> list[dict[str, Any]]:
    search = await session.get(HyperparameterSearchJob, search_id)
    if search is None:
        raise HTTPException(404, "Search not found")
    results_dir = hyperparameter_search_results_dir(settings.storage_root, search_id)
    history_path = results_dir / "search_history.json"
    if not history_path.exists():
        raise HTTPException(404, "Search history not available")
    return json.loads(history_path.read_text())


@router.delete("/searches/{search_id}")
async def delete_search(
    search_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, str]:
    search = await session.get(HyperparameterSearchJob, search_id)
    if search is None:
        raise HTTPException(404, "Search not found")
    # Remove artifact directory
    results_dir = hyperparameter_search_results_dir(settings.storage_root, search_id)
    if results_dir.exists():
        shutil.rmtree(results_dir)
    await session.delete(search)
    await session.commit()
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Search space defaults
# ---------------------------------------------------------------------------


@router.get("/search-space-defaults")
async def get_search_space_defaults() -> SearchSpaceDefaults:
    return SearchSpaceDefaults(search_space=DEFAULT_SEARCH_SPACE)


# ---------------------------------------------------------------------------
# Import candidate from search
# ---------------------------------------------------------------------------


@router.post("/searches/{search_id}/import-candidate", status_code=201)
async def import_candidate_from_search(
    search_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, Any]:
    search = await session.get(HyperparameterSearchJob, search_id)
    if search is None:
        raise HTTPException(404, "Search not found")
    if search.status != "complete":
        raise HTTPException(
            400, "Search must be complete before importing as candidate"
        )

    results_dir = hyperparameter_search_results_dir(settings.storage_root, search_id)
    manifest = await session.get(HyperparameterManifest, search.manifest_id)
    if manifest is None:
        raise HTTPException(400, "Manifest not found for search")

    manifest_path = manifest.manifest_path
    if manifest_path is None:
        raise HTTPException(400, "Manifest has no artifact path")

    best_run_path = results_dir / "best_run.json"
    if not best_run_path.exists():
        raise HTTPException(400, "best_run.json not found in search results")

    comparison_path: Path | None = None
    comparison_file = results_dir / "comparison.json"
    if comparison_file.exists():
        comparison_path = comparison_file

    top_fp_path: Path | None = None
    top_fp_file = results_dir / "top_false_positives.json"
    if top_fp_file.exists():
        top_fp_path = top_fp_file

    try:
        candidate = await classifier_service.import_autoresearch_candidate(
            session,
            settings.storage_root,
            str(manifest_path),
            str(best_run_path),
            comparison_path=str(comparison_path) if comparison_path else None,
            top_false_positives_path=str(top_fp_path) if top_fp_path else None,
            name=f"{search.name} (imported)",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    detail = _autoresearch_candidate_to_detail(candidate)
    return detail.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Candidates (relocated from autoresearch router)
# ---------------------------------------------------------------------------


@router.post("/candidates/import", status_code=201)
async def import_candidate(
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


@router.get("/candidates")
async def list_candidates(
    session: SessionDep,
) -> list[AutoresearchCandidateSummaryOut]:
    candidates = await classifier_service.list_autoresearch_candidates(session)
    return [_autoresearch_candidate_to_summary(c) for c in candidates]


@router.get("/candidates/{candidate_id}")
async def get_candidate(
    candidate_id: str,
    session: SessionDep,
) -> AutoresearchCandidateDetailOut:
    candidate = await classifier_service.get_autoresearch_candidate(
        session, candidate_id
    )
    if candidate is None:
        raise HTTPException(404, "Candidate not found")

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


@router.post("/candidates/{candidate_id}/training-jobs", status_code=201)
async def create_training_job_from_candidate(
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
