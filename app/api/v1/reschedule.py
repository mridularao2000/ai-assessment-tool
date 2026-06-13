from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_reschedule_service
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.schemas.reschedule import RescheduleRequestCreate, RescheduleResponse
from app.services.reschedule_service import RescheduleService

router = APIRouter()


@router.post("/{assessment_id}/reschedule", response_model=RescheduleResponse)
def request_reschedule(
    assessment_id: str,
    body: RescheduleRequestCreate,
    reschedule_svc: RescheduleService = Depends(get_reschedule_service),
) -> RescheduleResponse:
    try:
        result, updated_assessment = reschedule_svc.request_reschedule(
            assessment_id=assessment_id,
            token=body.token,
            reason=body.reason,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTokenError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return RescheduleResponse(
        approved=result.approved or False,
        reasoning=result.category_reasoning or "",
        new_scheduled_at=updated_assessment.scheduled_at if updated_assessment else None,
        new_due_date=updated_assessment.due_date if updated_assessment else None,
    )
