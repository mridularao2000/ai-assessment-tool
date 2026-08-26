from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_curriculum_upload_service
from app.exceptions import CurriculumUploadValidationError, InvalidStateError, NotFoundError
from app.schemas.curriculum_upload import (
    CurriculumEntrySummary,
    CurriculumUploadDetailResponse,
    CurriculumUploadResponse,
    GPAResponse,
    PendingResourcesResponse,
    PendingResourcesUpdate,
    TranscriptChapterGroupResponse,
    TranscriptResponse,
    TranscriptRowResponse,
)
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.gpa_service import compute_gpa
from app.services.transcript_service import compute_transcript

router = APIRouter()


@router.post("/", response_model=CurriculumUploadResponse, status_code=201)
def upload_curriculum(
    curriculum_upload_svc: Annotated[CurriculumUploadService, Depends(get_curriculum_upload_service)],
    file: Annotated[UploadFile, File()],
) -> CurriculumUploadResponse:
    try:
        raw = json.loads(file.file.read())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid JSON file: {exc}")
    if not isinstance(raw, dict):
        raise HTTPException(status_code=422, detail="Upload file must be a JSON object.")

    try:
        upload = curriculum_upload_svc.ingest(raw, source_filename=file.filename or "upload.json")
    except CurriculumUploadValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return CurriculumUploadResponse(
        upload_id=upload.id,
        entry_count=len(upload.entries),
        syllabus_email_sent=upload.syllabus_email_sent_at is not None,
    )


@router.get("/{upload_id}", response_model=CurriculumUploadDetailResponse)
def get_curriculum_upload(
    upload_id: str,
    curriculum_upload_svc: Annotated[CurriculumUploadService, Depends(get_curriculum_upload_service)],
) -> CurriculumUploadDetailResponse:
    try:
        upload = curriculum_upload_svc.get_upload(upload_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return CurriculumUploadDetailResponse(
        upload_id=upload.id,
        source_filename=upload.source_filename,
        uploaded_at=upload.uploaded_at,
        entries=[
            CurriculumEntrySummary(
                id=e.id,
                topic=e.topic,
                entry_type=e.entry_type.value if e.entry_type else "",
                chapter_label=e.chapter_label or "",
                completion_date=e.target_completion_date,
                max_marks=e.max_marks or 0.0,
                resources_hold=e.resources_hold,
            )
            for e in upload.entries
        ],
    )


@router.get("/{upload_id}/gpa", response_model=GPAResponse)
def get_gpa(
    upload_id: str,
    curriculum_upload_svc: Annotated[CurriculumUploadService, Depends(get_curriculum_upload_service)],
) -> GPAResponse:
    try:
        curriculum_upload_svc.get_upload(upload_id)  # 404 if upload doesn't exist
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    summary = compute_gpa(curriculum_upload_svc.db, upload_id)
    return GPAResponse(
        upload_id=upload_id,
        total_earned=summary.total_earned,
        total_max=summary.total_max,
        gpa=summary.gpa,
        graded_count=summary.graded_count,
        missed_count=summary.missed_count,
    )


@router.get("/{upload_id}/transcript", response_model=TranscriptResponse)
def get_transcript(
    upload_id: str,
    curriculum_upload_svc: Annotated[CurriculumUploadService, Depends(get_curriculum_upload_service)],
) -> TranscriptResponse:
    try:
        content = compute_transcript(curriculum_upload_svc.db, upload_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return TranscriptResponse(
        upload_id=content.upload_id,
        source_filename=content.source_filename,
        chapter_groups=[
            TranscriptChapterGroupResponse(
                chapter_label=g.chapter_label,
                rows=[
                    TranscriptRowResponse(
                        row_id=r.row_id, topic=r.topic, chapter_number=r.chapter_number,
                        max_marks=r.max_marks, status_label=r.status_label,
                        points=r.points, retake_note=r.retake_note,
                    )
                    for r in g.rows
                ],
            )
            for g in content.chapter_groups
        ],
        resolved_count=content.resolved_count,
        total_entry_count=content.total_entry_count,
        graded_count=content.graded_count,
        total_credits=content.total_credits,
        total_points=content.total_points,
        gpa=content.gpa,
        course_material_captured_at=content.course_material_captured_at,
    )


@router.patch(
    "/entries/{curriculum_id}/pending-resources",
    response_model=PendingResourcesResponse,
)
def fill_pending_resources(
    curriculum_id: str,
    body: PendingResourcesUpdate,
    curriculum_upload_svc: Annotated[CurriculumUploadService, Depends(get_curriculum_upload_service)],
) -> PendingResourcesResponse:
    try:
        curriculum = curriculum_upload_svc.fill_pending_resources(curriculum_id, body.values)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PendingResourcesResponse(
        curriculum_id=curriculum.id,
        resources_hold=curriculum.resources_hold,
        pending_completion_slots=curriculum.midterm_detail.pending_completion_slots,
    )
