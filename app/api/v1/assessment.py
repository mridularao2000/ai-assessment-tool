from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_assessment_service, get_scheduler_service
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.schemas.assessment import AssessmentDetailResponse, AssessmentSummary
from app.services.assessment_service import AssessmentService
from app.services.scheduler_service import SchedulerService

router = APIRouter()


@router.post("/{curriculum_id}", response_model=AssessmentSummary, status_code=201)
def create_assessment(
    curriculum_id: str,
    assessment_svc: AssessmentService = Depends(get_assessment_service),
    scheduler_svc: SchedulerService = Depends(get_scheduler_service),
) -> AssessmentSummary:
    try:
        assessment = assessment_svc.create_for_curriculum(curriculum_id)
        scheduler_svc.schedule_assessment_jobs(
            assessment_id=assessment.id,
            scheduled_at=assessment.scheduled_at,
            reminder_at=assessment.reminder_at,
            due_date=assessment.due_date,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return AssessmentSummary.model_validate(assessment)


@router.get("/{assessment_id}", response_model=AssessmentDetailResponse)
def get_assessment(
    assessment_id: str,
    token: str = Query(...),
    assessment_svc: AssessmentService = Depends(get_assessment_service),
) -> AssessmentDetailResponse:
    try:
        assessment = assessment_svc.get_by_id_and_token(assessment_id, token)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTokenError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    return AssessmentDetailResponse(
        assessment_id=assessment.id,
        topic=assessment.curriculum.topic,
        assessment_text=assessment.assessment_text,
        duration_minutes=assessment.duration_minutes,
        scheduled_at=assessment.scheduled_at,
        due_date=assessment.due_date,
        status=assessment.status,
    )
