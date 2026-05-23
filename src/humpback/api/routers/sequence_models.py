"""FastAPI router for retained Sequence Models / Continuous Embedding APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from humpback.api.deps import SessionDep, SettingsDep
from humpback.clustering.reducer import reduce_projection_2d
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.sequence_models.event_encoder import DESCRIPTOR_ORDER, descriptor_units
from humpback.schemas.piano_roll_midi_export import (
    PianoRollMidiExportCreateRequest,
    PianoRollMidiExportRead,
    PianoRollMidiExportStatusAbsent,
    PianoRollMidiExportStatusResponse,
)
from humpback.schemas.piano_roll_notes import (
    PianoRollNote,
    PianoRollNoteContourFrame,
    PianoRollNoteContourRequest,
    PianoRollNoteContourResponse,
    PianoRollNotesJobCreateRequest,
    PianoRollNotesJobRead,
    PianoRollNotesResponse,
    PianoRollNotesStatusAbsent,
    PianoRollNotesStatusResponse,
)
from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobDetail,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingJobOut,
    EventEncoderJobCreate,
    EventEncoderJobDetail,
    EventEncoderJobOut,
    EventEncoderProjectionMethod,
    EventEncoderProjectionPoint,
    EventEncoderProjectionResponse,
    EventEncoderTimelineEvent,
    EventEncoderTimelineResponse,
)
from humpback.services.continuous_embedding_service import (
    CancelTerminalJobError,
    SOURCE_KIND_REGION_CRNN,
    cancel_continuous_embedding_job,
    create_continuous_embedding_job,
    delete_continuous_embedding_job,
    get_continuous_embedding_job,
    list_continuous_embedding_jobs,
    source_kind_for,
)
from humpback.services.event_encoder_service import (
    CancelEventEncoderTerminalJobError,
    cancel_event_encoder_job,
    create_event_encoder_job,
    delete_event_encoder_job,
    get_event_encoder_job,
    list_event_encoder_jobs,
)
from humpback.services.piano_roll_midi_export_service import (
    PianoRollMidiExportConflict,
    complete_for_encoder_job_version as midi_export_complete_for_encoder_job_version,
    enqueue_piano_roll_midi_export,
    latest_for_encoder_job as midi_export_latest_for_encoder_job,
)
from humpback.services.piano_roll_notes_service import (
    PianoRollNotesJobConflict,
    complete_for_encoder_job_version,
    enqueue_piano_roll_notes_job,
    latest_for_encoder_job,
)
from humpback.storage import (
    event_encoder_audio_export_path,
    event_encoder_midi_export_path,
    event_encoder_note_contours_path,
)

router = APIRouter(prefix="/sequence-models", tags=["sequence-models"])


def _to_out(job) -> ContinuousEmbeddingJobOut:
    return ContinuousEmbeddingJobOut.model_validate(job)


def _to_event_encoder_out(job) -> EventEncoderJobOut:
    return EventEncoderJobOut.model_validate(job)


@router.post("/continuous-embeddings")
async def create_continuous_embedding(
    body: ContinuousEmbeddingJobCreate,
    session: SessionDep,
    response: Response,
) -> ContinuousEmbeddingJobOut:
    try:
        job, created = await create_continuous_embedding_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _to_out(job)


@router.get("/continuous-embeddings")
async def list_continuous_embeddings(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
) -> list[ContinuousEmbeddingJobOut]:
    jobs = await list_continuous_embedding_jobs(session, status=status)
    return [_to_out(j) for j in jobs]


@router.get("/continuous-embeddings/{job_id}")
async def get_continuous_embedding(
    job_id: str,
    session: SessionDep,
) -> ContinuousEmbeddingJobDetail:
    job = await get_continuous_embedding_job(session, job_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )

    manifest: Optional[ContinuousEmbeddingJobManifest] = None
    if job.parquet_path:
        manifest_path = Path(job.parquet_path).with_name("manifest.json")
        if manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest = ContinuousEmbeddingJobManifest.model_validate(payload)
            except Exception:
                manifest = None

    return ContinuousEmbeddingJobDetail(job=_to_out(job), manifest=manifest)


@router.post("/continuous-embeddings/{job_id}/cancel")
async def cancel_continuous_embedding(
    job_id: str,
    session: SessionDep,
) -> ContinuousEmbeddingJobOut:
    try:
        job = await cancel_continuous_embedding_job(session, job_id)
    except CancelTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )
    return _to_out(job)


@router.delete("/continuous-embeddings/{job_id}", status_code=204)
async def delete_continuous_embedding(
    job_id: str, session: SessionDep, settings: SettingsDep
):
    deleted = await delete_continuous_embedding_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(
            status_code=404, detail="continuous embedding job not found"
        )
    return None


@router.post("/event-encoders")
async def create_event_encoder(
    body: EventEncoderJobCreate,
    session: SessionDep,
    response: Response,
) -> EventEncoderJobOut:
    try:
        job, created = await create_event_encoder_job(session, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    response.status_code = 201 if created else 200
    return _to_event_encoder_out(job)


@router.get("/event-encoders")
async def list_event_encoders(
    session: SessionDep,
    status: Optional[str] = Query(default=None),
) -> list[EventEncoderJobOut]:
    jobs = await list_event_encoder_jobs(session, status=status)
    return [_to_event_encoder_out(j) for j in jobs]


@router.get("/event-encoders/{job_id}")
async def get_event_encoder(job_id: str, session: SessionDep) -> EventEncoderJobDetail:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    manifest = _load_json_sidecar(job.manifest_path)
    report = _load_json_sidecar(job.report_path)
    return EventEncoderJobDetail(
        job=_to_event_encoder_out(job),
        manifest=manifest,
        report=report,
    )


@router.get("/event-encoders/{job_id}/timeline")
async def get_event_encoder_timeline(
    job_id: str,
    session: SessionDep,
    k: Optional[int] = Query(default=None, gt=0),
) -> EventEncoderTimelineResponse:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(
            status_code=409,
            detail=f"event encoder job status is {job.status!r}, not 'complete'",
        )
    if not job.event_tokens_path:
        raise HTTPException(status_code=404, detail="event_tokens.parquet not found")

    token_path = Path(job.event_tokens_path)
    if not token_path.exists():
        raise HTTPException(status_code=404, detail="event_tokens.parquet not found")

    continuous = await session.get(
        ContinuousEmbeddingJob, job.continuous_embedding_job_id
    )
    try:
        source_kind = (
            source_kind_for(continuous.model_version)
            if continuous is not None
            else None
        )
    except ValueError:
        source_kind = None
    if (
        continuous is None
        or continuous.region_detection_job_id is None
        or source_kind != SOURCE_KIND_REGION_CRNN
    ):
        raise HTTPException(
            status_code=409,
            detail="event encoder timeline requires region_crnn provenance",
        )
    region_job = await session.get(
        RegionDetectionJob, continuous.region_detection_job_id
    )
    if region_job is None:
        raise HTTPException(status_code=409, detail="region detection job not found")

    rows = pq.read_table(token_path).to_pylist()
    token_schema_names = set(pq.read_schema(token_path).names)
    valid_k_values = sorted({int(row["k"]) for row in rows})
    if not valid_k_values:
        raise HTTPException(
            status_code=404, detail="event_tokens.parquet contains no token rows"
        )
    selected_k = int(k) if k is not None else valid_k_values[0]
    if selected_k not in valid_k_values:
        raise HTTPException(
            status_code=422,
            detail=f"k={selected_k} is not available for this event encoder job",
        )

    selected_rows = sorted(
        (row for row in rows if int(row["k"]) == selected_k),
        key=lambda row: (
            str(row["source_sequence_key"]),
            float(row["start_timestamp"]),
            float(row["end_timestamp"]),
            str(row["event_id"]),
        ),
    )
    manifest = _load_json_sidecar(job.manifest_path)
    descriptor_feature_names = _descriptor_feature_names(
        manifest,
        token_schema_names,
    )
    vector_features = _load_descriptor_vector_values(
        job.event_vectors_path,
        descriptor_feature_names,
    )
    events = [
        EventEncoderTimelineEvent(
            event_id=str(row["event_id"]),
            region_id=str(row["region_id"]),
            source_sequence_key=str(row["source_sequence_key"]),
            sequence_index=int(row["sequence_index"]),
            start_timestamp=float(row["start_timestamp"]),
            end_timestamp=float(row["end_timestamp"]),
            token_id=int(row["token_id"]),
            token_label=str(row["token_label"]),
            token_confidence=float(row["token_confidence"]),
            distance_to_centroid=float(row["distance_to_centroid"]),
            second_centroid_distance=(
                None
                if row.get("second_centroid_distance") is None
                else float(row["second_centroid_distance"])
            ),
            descriptor_values=_descriptor_values(row, descriptor_feature_names),
            descriptor_vector_values=vector_features.get(
                _event_vector_key(row),
                {},
            ),
        )
        for row in selected_rows
    ]

    notes_status = await _notes_status_for(session, job.id)

    return EventEncoderTimelineResponse(
        job_id=job.id,
        event_segmentation_job_id=job.event_segmentation_job_id,
        event_source_mode=(
            "effective" if job.event_source_mode == "effective" else "raw"
        ),
        continuous_embedding_job_id=job.continuous_embedding_job_id,
        region_detection_job_id=continuous.region_detection_job_id,
        selected_k=selected_k,
        valid_k_values=valid_k_values,
        descriptor_feature_names=descriptor_feature_names,
        descriptor_feature_units=descriptor_units(),
        job_start_timestamp=float(region_job.start_timestamp or 0.0),
        job_end_timestamp=float(region_job.end_timestamp or 0.0),
        events=events,
        notes_status=notes_status,
    )


@router.get("/event-encoders/{job_id}/projection")
async def get_event_encoder_projection(
    job_id: str,
    session: SessionDep,
    k: Optional[int] = Query(default=None, gt=0),
    method: EventEncoderProjectionMethod = Query(default="umap"),
) -> EventEncoderProjectionResponse:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(
            status_code=409,
            detail=f"event encoder job status is {job.status!r}, not 'complete'",
        )
    if not job.event_tokens_path:
        raise HTTPException(status_code=404, detail="event_tokens.parquet not found")
    if not job.event_vectors_path:
        raise HTTPException(status_code=404, detail="event_vectors.parquet not found")

    token_path = Path(job.event_tokens_path)
    if not token_path.exists():
        raise HTTPException(status_code=404, detail="event_tokens.parquet not found")
    vector_path = Path(job.event_vectors_path)
    if not vector_path.exists():
        raise HTTPException(status_code=404, detail="event_vectors.parquet not found")

    rows = pq.read_table(token_path).to_pylist()
    valid_k_values = sorted({int(row["k"]) for row in rows})
    if not valid_k_values:
        raise HTTPException(
            status_code=404, detail="event_tokens.parquet contains no token rows"
        )
    selected_k = int(k) if k is not None else valid_k_values[0]
    if selected_k not in valid_k_values:
        raise HTTPException(
            status_code=422,
            detail=f"k={selected_k} is not available for this event encoder job",
        )

    selected_rows = sorted(
        (row for row in rows if int(row["k"]) == selected_k),
        key=lambda row: (
            str(row["source_sequence_key"]),
            float(row["start_timestamp"]),
            float(row["end_timestamp"]),
            str(row["event_id"]),
        ),
    )
    vector_values = _load_event_vectors(vector_path)
    matched: list[tuple[dict, list[float]]] = []
    for row in selected_rows:
        vector = vector_values.get(_event_vector_key(row))
        if vector is not None:
            matched.append((row, vector))
    if not matched:
        raise HTTPException(
            status_code=404,
            detail="event_vectors.parquet contains no vectors for selected tokens",
        )

    try:
        matrix = np.asarray([vector for _, vector in matched], dtype=np.float32)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail="event_vectors.parquet contains inconsistent event_vector dimensions",
        ) from exc

    coords = reduce_projection_2d(
        matrix,
        method=method,
        random_state=int(job.random_seed),
    )
    x_axis_label = "UMAP 1" if method == "umap" else "PC 1"
    y_axis_label = "UMAP 2" if method == "umap" else "PC 2"

    return EventEncoderProjectionResponse(
        job_id=job.id,
        selected_k=selected_k,
        valid_k_values=valid_k_values,
        method=method,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        points=[
            EventEncoderProjectionPoint(
                event_id=str(row["event_id"]),
                region_id=str(row["region_id"]),
                source_sequence_key=str(row["source_sequence_key"]),
                sequence_index=int(row["sequence_index"]),
                start_timestamp=float(row["start_timestamp"]),
                end_timestamp=float(row["end_timestamp"]),
                token_id=int(row["token_id"]),
                token_label=str(row["token_label"]),
                token_confidence=float(row["token_confidence"]),
                distance_to_centroid=float(row["distance_to_centroid"]),
                second_centroid_distance=(
                    None
                    if row.get("second_centroid_distance") is None
                    else float(row["second_centroid_distance"])
                ),
                x=float(coords[index, 0]),
                y=float(coords[index, 1]),
            )
            for index, (row, _) in enumerate(matched)
        ],
    )


@router.get("/event-encoders/{job_id}/notes-status")
async def get_piano_roll_notes_status(
    job_id: str,
    session: SessionDep,
) -> PianoRollNotesStatusResponse:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    return await _notes_status_for(session, job.id)


@router.post("/event-encoders/{job_id}/notes-jobs")
async def create_piano_roll_notes_job(
    job_id: str,
    body: PianoRollNotesJobCreateRequest,
    session: SessionDep,
    response: Response,
) -> PianoRollNotesJobRead:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    if job.status != JobStatus.complete.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"event encoder job status is {job.status!r}, not 'complete' — "
                "piano roll notes require a completed encoder job"
            ),
        )

    try:
        row, created = await enqueue_piano_roll_notes_job(
            session,
            event_encoder_job_id=job.id,
            extractor_version=body.extractor_version,
            params=body.params,
        )
    except PianoRollNotesJobConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    response.status_code = 201 if created else 200
    return PianoRollNotesJobRead.model_validate(row)


@router.get("/event-encoders/{job_id}/notes")
async def get_piano_roll_notes(
    job_id: str,
    session: SessionDep,
    start_utc: Optional[float] = Query(default=None),
    end_utc: Optional[float] = Query(default=None),
    event_ids: Optional[list[str]] = Query(default=None),
    extractor_version: Optional[str] = Query(default=None),
) -> PianoRollNotesResponse:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    if extractor_version is not None:
        latest = await complete_for_encoder_job_version(
            session,
            event_encoder_job_id=job.id,
            extractor_version=extractor_version,
        )
        if latest is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"piano roll notes for extractor_version={extractor_version!r} "
                    "are not available"
                ),
            )
    else:
        latest = await latest_for_encoder_job(session, event_encoder_job_id=job.id)
        if latest is None or latest.status != JobStatus.complete.value:
            raise HTTPException(
                status_code=404,
                detail="no completed piano roll notes job for this event encoder",
            )
    if not latest.notes_path:
        raise HTTPException(status_code=404, detail="notes parquet not recorded")

    notes_path = Path(latest.notes_path)
    if not notes_path.exists():
        raise HTTPException(status_code=404, detail="notes parquet not found on disk")

    if start_utc is not None and end_utc is not None and end_utc < start_utc:
        raise HTTPException(status_code=422, detail="end_utc must be >= start_utc")

    rows = pq.read_table(notes_path).to_pylist()
    event_filter = set(event_ids) if event_ids else None
    filtered: list[PianoRollNote] = []
    for row in rows:
        note_start = float(row["start_utc"])
        duration = float(row["duration_s"])
        note_end = note_start + duration
        if start_utc is not None and note_end < start_utc:
            continue
        if end_utc is not None and note_start > end_utc:
            continue
        if event_filter is not None and str(row["event_id"]) not in event_filter:
            continue
        note_uid = row.get("note_uid")
        f0_track_id = row.get("f0_track_id")
        contour_frame_count = row.get("contour_frame_count")
        filtered.append(
            PianoRollNote(
                event_id=str(row["event_id"]),
                event_token=int(row["event_token"]),
                partial_index=int(row["partial_index"]),
                midi_pitch=int(row["midi_pitch"]),
                start_utc=note_start,
                start_offset_s=float(row["start_offset_s"]),
                duration_s=duration,
                velocity=int(row["velocity"]),
                peak_magnitude=float(row["peak_magnitude"]),
                track_id=int(row["track_id"]),
                note_uid=str(note_uid) if note_uid is not None else None,
                f0_track_id=int(f0_track_id) if f0_track_id is not None else None,
                contour_frame_count=(
                    int(contour_frame_count)
                    if contour_frame_count is not None
                    else None
                ),
            )
        )

    return PianoRollNotesResponse(
        job_id=job.id,
        extractor_version=latest.extractor_version,
        n_notes=len(filtered),
        notes=filtered,
    )


_MAX_CONTOUR_NOTE_UIDS = 2000


@router.post("/event-encoders/{job_id}/notes/contours")
async def get_piano_roll_note_contours(
    job_id: str,
    body: PianoRollNoteContourRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> PianoRollNoteContourResponse:
    """Return per-frame contour rows for the requested ``note_uid``s.

    POST so the ``note_uids`` list (UUIDs at ~48 bytes each in URL form)
    rides in the JSON body and stays clear of the dev server's HTTP
    header limit. The endpoint always reads the v3 contour sidecar so it
    returns 422 when no v3 sidecar exists yet. Requests above
    ``_MAX_CONTOUR_NOTE_UIDS`` return 413; unknown ``note_uid``s in an
    otherwise valid request are dropped from the response.
    """
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    if len(body.note_uids) > _MAX_CONTOUR_NOTE_UIDS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"note_uids cap is {_MAX_CONTOUR_NOTE_UIDS};"
                f" received {len(body.note_uids)}"
            ),
        )

    if body.extractor_version is not None:
        latest = await complete_for_encoder_job_version(
            session,
            event_encoder_job_id=job.id,
            extractor_version=body.extractor_version,
        )
    else:
        latest = await latest_for_encoder_job(session, event_encoder_job_id=job.id)
        if latest is not None and latest.status != JobStatus.complete.value:
            latest = None

    if latest is None:
        raise HTTPException(
            status_code=422,
            detail="no v3 piano roll notes contour sidecar for this event encoder",
        )

    contours_path = event_encoder_note_contours_path(
        settings.storage_root, job.id, latest.extractor_version
    )
    if not contours_path.exists():
        raise HTTPException(
            status_code=422,
            detail=(
                "contour sidecar is not available for "
                f"extractor_version={latest.extractor_version!r}"
            ),
        )

    contours: dict[str, list[PianoRollNoteContourFrame]] = {}
    if body.note_uids:
        table = pq.read_table(contours_path)
        # Mirror the worker's push-down pattern: deduplicate uids and let
        # PyArrow filter the table column-wise rather than materializing
        # the rows in Python and probing a set per row.
        unique_uids = list({uid for uid in body.note_uids})
        mask = pc.is_in(  # type: ignore[attr-defined]
            table.column("note_uid"),
            value_set=pa.array(unique_uids, type=pa.string()),
        )
        filtered = table.filter(mask)
        for row in filtered.to_pylist():
            uid = str(row["note_uid"])
            contours.setdefault(uid, []).append(
                PianoRollNoteContourFrame(
                    frame_index=int(row["frame_index"]),
                    time_offset_s=float(row["time_offset_s"]),
                    cents_from_pitch=float(row["cents_from_pitch"]),
                    harmonic_strength=float(row["harmonic_strength"]),
                    subharmonic_octave=int(row["subharmonic_octave"]),
                )
            )
        for uid in contours:
            contours[uid].sort(key=lambda c: c.frame_index)

    return PianoRollNoteContourResponse(
        job_id=job.id,
        extractor_version=latest.extractor_version,
        n_notes=len(contours),
        contours=contours,
    )


@router.get("/event-encoders/{job_id}/midi-export-status")
async def get_piano_roll_midi_export_status(
    job_id: str,
    session: SessionDep,
    extractor_version: Optional[str] = Query(default=None),
) -> PianoRollMidiExportStatusResponse:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    if extractor_version is not None:
        pinned = await midi_export_complete_for_encoder_job_version(
            session,
            event_encoder_job_id=job.id,
            extractor_version=extractor_version,
        )
        if pinned is None:
            return PianoRollMidiExportStatusAbsent()
        return PianoRollMidiExportRead.model_validate(pinned)

    latest = await midi_export_latest_for_encoder_job(
        session, event_encoder_job_id=job.id
    )
    if latest is None:
        return PianoRollMidiExportStatusAbsent()
    return PianoRollMidiExportRead.model_validate(latest)


@router.post("/event-encoders/{job_id}/midi-exports")
async def create_piano_roll_midi_export(
    job_id: str,
    body: PianoRollMidiExportCreateRequest,
    session: SessionDep,
    response: Response,
) -> PianoRollMidiExportRead:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    try:
        await _validate_export_window_overlap(
            session,
            job,
            body.window_start_utc,
            body.window_end_utc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    try:
        row, created = await enqueue_piano_roll_midi_export(
            session,
            event_encoder_job_id=job.id,
            window_start_utc=body.window_start_utc,
            window_end_utc=body.window_end_utc,
            extractor_version=body.extractor_version,
            params=body.params,
            force=body.force,
        )
    except PianoRollMidiExportConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    response.status_code = 201 if created else 200
    return PianoRollMidiExportRead.model_validate(row)


async def _validate_export_window_overlap(
    session: SessionDep,
    encoder_job,
    window_start_utc: float,
    window_end_utc: float,
) -> None:
    """Ensure the requested window overlaps the encoder's source data range.

    Resolves the chain ``EventEncoderJob → EventSegmentationJob →
    RegionDetectionJob`` and uses the region's ``start_timestamp`` /
    ``end_timestamp`` as the authoritative bounds. Raises ``ValueError``
    when the window misses that range entirely.
    """
    from humpback.models.call_parsing import (
        EventSegmentationJob,
        RegionDetectionJob,
    )

    seg_job = await session.get(
        EventSegmentationJob, encoder_job.event_segmentation_job_id
    )
    if seg_job is None:
        raise ValueError("event_segmentation_job not found for this encoder")
    region_job = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if region_job is None:
        raise ValueError("region_detection_job not found for this encoder")
    job_start = region_job.start_timestamp
    job_end = region_job.end_timestamp
    if job_start is None or job_end is None:
        raise ValueError(
            "export window cannot be validated: the encoder's source region "
            "detection job is missing start_timestamp / end_timestamp"
        )
    if window_end_utc <= float(job_start) or window_start_utc >= float(job_end):
        raise ValueError(
            "export window does not overlap the job's data range "
            f"[{float(job_start):.6f}, {float(job_end):.6f}]"
        )


@router.get("/event-encoders/{job_id}/midi-export")
async def download_piano_roll_midi_export(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    extractor_version: Optional[str] = Query(default=None),
) -> Response:
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    if extractor_version is not None:
        row = await midi_export_complete_for_encoder_job_version(
            session,
            event_encoder_job_id=job.id,
            extractor_version=extractor_version,
        )
    else:
        row = await midi_export_latest_for_encoder_job(
            session, event_encoder_job_id=job.id
        )
        if row is not None and row.status != JobStatus.complete.value:
            row = None
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="no completed MIDI export for this event encoder",
        )

    midi_file = event_encoder_midi_export_path(
        settings.storage_root, job.id, row.extractor_version
    )
    if not midi_file.exists():
        raise HTTPException(
            status_code=404, detail="MIDI export file not found on disk"
        )

    filename = f"event_encoder_{job.id}_notes_{row.extractor_version}.mid"
    return Response(
        content=midi_file.read_bytes(),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/event-encoders/{job_id}/audio-export")
async def download_piano_roll_audio_export(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    extractor_version: Optional[str] = Query(default=None),
) -> Response:
    """Stream the FLAC clip co-exported with the windowed MIDI artifact."""
    job = await get_event_encoder_job(session, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")

    if extractor_version is not None:
        row = await midi_export_complete_for_encoder_job_version(
            session,
            event_encoder_job_id=job.id,
            extractor_version=extractor_version,
        )
    else:
        row = await midi_export_latest_for_encoder_job(
            session, event_encoder_job_id=job.id
        )
        if row is not None and row.status != JobStatus.complete.value:
            row = None
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No completed audio export for this job.",
        )

    audio_file = event_encoder_audio_export_path(
        settings.storage_root, job.id, row.extractor_version
    )
    if not audio_file.exists():
        raise HTTPException(
            status_code=404, detail="audio export file not found on disk"
        )

    start = float(row.window_start_utc)
    end = float(row.window_end_utc)
    filename = (
        f"event_encoder_{job.id}_{row.extractor_version}_{start:.3f}_{end:.3f}.flac"
    )
    return Response(
        content=audio_file.read_bytes(),
        media_type="audio/flac",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/event-encoders/{job_id}/cancel")
async def cancel_event_encoder(
    job_id: str,
    session: SessionDep,
) -> EventEncoderJobOut:
    try:
        job = await cancel_event_encoder_job(session, job_id)
    except CancelEventEncoderTerminalJobError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if job is None:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    return _to_event_encoder_out(job)


@router.delete("/event-encoders/{job_id}", status_code=204)
async def delete_event_encoder(job_id: str, session: SessionDep, settings: SettingsDep):
    deleted = await delete_event_encoder_job(session, job_id, settings)
    if not deleted:
        raise HTTPException(status_code=404, detail="event encoder job not found")
    return None


async def _notes_status_for(
    session: SessionDep, event_encoder_job_id: str
) -> PianoRollNotesStatusResponse:
    latest = await latest_for_encoder_job(
        session, event_encoder_job_id=event_encoder_job_id
    )
    if latest is None:
        return PianoRollNotesStatusAbsent()
    return PianoRollNotesJobRead.model_validate(latest)


def _load_json_sidecar(path_value: Optional[str]) -> Optional[dict]:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _descriptor_feature_names(
    manifest: Optional[dict],
    token_schema_names: set[str],
) -> list[str]:
    if manifest is not None:
        names = manifest.get("descriptor_feature_names")
        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            return list(names)
    inferred = [name for name in DESCRIPTOR_ORDER if name in token_schema_names]
    return inferred or list(DESCRIPTOR_ORDER)


def _load_descriptor_vector_values(
    event_vectors_path: Optional[str],
    descriptor_feature_names: list[str],
) -> dict[tuple[str, int, str], dict[str, float]]:
    if not event_vectors_path:
        return {}
    path = Path(event_vectors_path)
    if not path.exists():
        return {}
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception:
        return {}

    values: dict[tuple[str, int, str], dict[str, float]] = {}
    for row in rows:
        vector = row.get("descriptor_vector")
        if not isinstance(vector, list):
            continue
        keyed_values = {
            name: float(vector[index])
            for index, name in enumerate(descriptor_feature_names)
            if index < len(vector) and _is_number(vector[index])
        }
        values[_event_vector_key(row)] = keyed_values
    return values


def _load_event_vectors(path: Path) -> dict[tuple[str, int, str], list[float]]:
    try:
        rows = pq.read_table(path).to_pylist()
    except Exception:
        return {}

    values: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        vector = _numeric_vector(row.get("event_vector"))
        if vector is not None:
            values[_event_vector_key(row)] = vector
    return values


def _numeric_vector(value) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[float] = []
    for item in value:
        if not _is_number(item):
            return None
        result.append(float(item))
    return result


def _descriptor_values(
    row: dict, descriptor_feature_names: list[str]
) -> dict[str, float]:
    return {
        name: float(row[name])
        for name in descriptor_feature_names
        if name in row and _is_number(row[name])
    }


def _event_vector_key(row: dict) -> tuple[str, int, str]:
    return (
        str(row.get("source_sequence_key", "")),
        int(row.get("sequence_index") or 0),
        str(row.get("event_id", "")),
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
