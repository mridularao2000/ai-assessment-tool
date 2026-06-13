from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_grading_service
from app.exceptions import NotFoundError
from app.schemas.grade import GradeResponse
from app.services.grading_service import GradingService

router = APIRouter()


@router.get("/{submission_id}/results", response_model=GradeResponse)
def get_results(
    submission_id: str,
    grading_svc: GradingService = Depends(get_grading_service),
) -> GradeResponse:
    try:
        return grading_svc.get_results(submission_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
