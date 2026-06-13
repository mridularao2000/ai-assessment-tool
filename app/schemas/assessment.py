from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models.assessment import AssessmentStatus


class AssessmentSummary(BaseModel):
    """Lightweight assessment record embedded in curriculum responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    attempt_number: int
    status: AssessmentStatus
    scheduled_at: datetime
    due_date: datetime
    duration_minutes: Optional[int] = None


class AssessmentDetailResponse(BaseModel):
    """Full assessment view returned when a user opens their submission link.

    Fields intentionally excluded from this response:
      - rubric           — grading reference only; must never reach the user
      - submission_token — HMAC credential; must never be echoed back in JSON
      - scheduled_job_ids — internal APScheduler state
      - curriculum_id    — not needed; topic is included directly
    """

    # Manually constructed by the route — not populated from ORM directly.
    # assessment_id and topic are explicit kwargs because:
    #   assessment_id: renamed from ORM field `id`
    #   topic:         lives on the related Curriculum, not on Assessment

    assessment_id: str
    topic: str
    assessment_text: Optional[str] = None
    duration_minutes: Optional[int] = None
    scheduled_at: datetime
    due_date: datetime
    status: AssessmentStatus
