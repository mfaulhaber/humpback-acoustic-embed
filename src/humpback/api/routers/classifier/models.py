"""Classifier model CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from humpback.api.deps import SessionDep, SettingsDep
from humpback.schemas.classifier import ClassifierModelOut
from humpback.schemas.converters import classifier_model_to_out as _model_to_out
from humpback.services import classifier_service

router = APIRouter()


class BulkDeleteRequest(BaseModel):
    ids: list[str]


@router.get("/models")
async def list_models(session: SessionDep) -> list[ClassifierModelOut]:
    models = await classifier_service.list_classifier_models(session)
    return [_model_to_out(m) for m in models]


@router.get("/models/{model_id}")
async def get_model(model_id: str, session: SessionDep) -> ClassifierModelOut:
    m = await classifier_service.get_classifier_model(session, model_id)
    if m is None:
        raise HTTPException(404, "Classifier model not found")
    return _model_to_out(m)


@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str, session: SessionDep, settings: SettingsDep
) -> dict:
    deleted = await classifier_service.delete_classifier_model(
        session, model_id, settings.storage_root
    )
    if not deleted:
        raise HTTPException(404, "Classifier model not found")
    return {"status": "deleted"}


@router.post("/models/bulk-delete")
async def bulk_delete_models(
    body: BulkDeleteRequest, session: SessionDep, settings: SettingsDep
) -> dict:
    count = await classifier_service.bulk_delete_classifier_models(
        session, body.ids, settings.storage_root
    )
    return {"status": "deleted", "count": count}
