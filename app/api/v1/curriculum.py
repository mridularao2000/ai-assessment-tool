from __future__ import annotations

from datetime import date
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_assessment_service, get_curriculum_service, get_scheduler_service
from app.exceptions import IngestionError, InvalidStateError, NotFoundError
from app.schemas.curriculum import CurriculumCreateResponse, CurriculumResponse
from app.services.assessment_service import AssessmentService
from app.services.curriculum_service import CurriculumService
from app.services.scheduler_service import SchedulerService

router = APIRouter()


@router.post("/", response_model=CurriculumCreateResponse, status_code=201)
def create_curriculum(
    topic: Annotated[str, Form()],
    target_completion_date: Annotated[date, Form()],
    links: Annotated[List[str], Form()] = [],
    github_repos: Annotated[List[str], Form()] = [],
    notes: Annotated[Optional[str], Form()] = None,
    files: Annotated[List[UploadFile], File()] = [],
    curriculum_svc: CurriculumService = Depends(get_curriculum_service),
    assessment_svc: AssessmentService = Depends(get_assessment_service),
    scheduler_svc: SchedulerService = Depends(get_scheduler_service),
) -> CurriculumCreateResponse:
    uploaded_files = [(f.filename or "upload", f.file.read()) for f in files]
    try:
        curriculum = curriculum_svc.create(
            topic=topic,
            target_completion_date=target_completion_date,
            links=links,
            github_repos=github_repos,
            notes=notes,
            uploaded_files=uploaded_files,
        )
        assessment = assessment_svc.create_for_curriculum(curriculum.id)
        scheduler_svc.schedule_assessment_jobs(
            assessment_id=assessment.id,
            scheduled_at=assessment.scheduled_at,
            reminder_at=assessment.reminder_at,
            due_date=assessment.due_date,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return CurriculumCreateResponse(curriculum_id=curriculum.id)


@router.get("/{curriculum_id}", response_model=CurriculumResponse)
def get_curriculum(
    curriculum_id: str,
    curriculum_svc: CurriculumService = Depends(get_curriculum_service),
) -> CurriculumResponse:
    try:
        curriculum = curriculum_svc.get(curriculum_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return CurriculumResponse.model_validate(curriculum)
