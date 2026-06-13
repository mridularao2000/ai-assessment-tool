from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RescheduleRequestCreate(BaseModel):
    """Request body for POST /api/v1/assessments/{id}/reschedule.

    The token identifies and authenticates the requester — same HMAC token
    that was embedded in the original assessment email link.
    """

    token: str = Field(..., min_length=1)
    reason: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        strip_whitespace=True,
        description=(
            "Plain-text explanation for the reschedule request. "
            "Must be at least 20 characters. "
            "Claude will classify this into a category; "
            "the application determines approval from the category."
        ),
    )


class RescheduleResponse(BaseModel):
    """Response returned after a reschedule request is classified.

    `approved` is set by application logic from RescheduleService.APPROVED_CATEGORIES,
    not by Claude directly.  `reasoning` is Claude's classification explanation.
    """

    approved: bool
    reasoning: str
    # Populated only when approved=True
    new_scheduled_at: Optional[datetime] = None
    new_due_date: Optional[datetime] = None
