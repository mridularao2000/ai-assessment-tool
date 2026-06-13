from __future__ import annotations

from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.config import get_settings
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.interfaces.llm import LLMInterface, RescheduleClassificationRequest
from app.interfaces.scheduler import AssessmentJobIds, SchedulerInterface
from app.models.assessment import Assessment, AssessmentStatus
from app.models.prompt_template import PromptTemplate
from app.models.reschedule_request import RescheduleRequest
from app.utils import token_auth

# Application-level decision: Claude returns a category, this set determines approval.
# Denied categories: procrastination, lack_of_preparation, missed_schedule.
APPROVED_CATEGORIES: frozenset[str] = frozenset({
    "interview",
    "medical",
    "emergency",
    "work_escalation",
})


class RescheduleService:
    """Classifies reschedule reasons and reschedules approved assessments.

    Claude provides only a category and reasoning.
    This service makes the approval decision based on APPROVED_CATEGORIES.

    Depends on:
      db                — SQLAlchemy session for all persistence
      llm               — LLMInterface for excuse classification
      scheduler_service — SchedulerInterface to cancel and recreate APScheduler jobs
    """

    def __init__(
        self,
        db: Session,
        llm: LLMInterface,
        scheduler_service: SchedulerInterface,
    ) -> None:
        self.db = db
        self.llm = llm
        self.scheduler_service = scheduler_service

    # ── Public methods ─────────────────────────────────────────────────────────

    def request_reschedule(
        self,
        assessment_id: str,
        token: str,
        reason: str,
    ) -> tuple[RescheduleRequest, Optional[Assessment]]:
        """Classify a reschedule reason and reschedule the assessment if approved.

        Steps:
          1. Load Assessment by assessment_id.
             Raise NotFoundError if missing.
          2. Verify token via token_auth.verify_submission_token.
             Raise InvalidTokenError if verification fails.
          3. Verify Assessment.status not in {completed, expired}.
             Raise InvalidStateError otherwise.
          4. Fetch active PromptTemplate where slug='reschedule_classification'.
             Raise NotFoundError if missing.
          5. Call llm.classify_reschedule_request() with reason and template body.
          6. Persist RescheduleRequest with reason_text, classification_category,
             category_reasoning, and approved.
          7. If approved:
               - Recalculate scheduled_at via AssessmentService._calculate_scheduled_at.
               - Derive reminder_at and due_date from new scheduled_at.
               - Call scheduler_service.reschedule_assessment() to swap APScheduler jobs.
               - Write new dates and new job IDs back to Assessment.
          8. Commit and return (RescheduleRequest, Assessment | None).
             Assessment is the refreshed, updated Assessment if approved; None if denied.
             Callers must not traverse ORM relationships on the returned objects.

        Raises:
            NotFoundError: if assessment_id or the prompt template does not exist.
            InvalidTokenError: if the token does not match.
            InvalidStateError: if the assessment is completed or expired, or if
                               approved but no stored job IDs exist to cancel.
        """
        # ── 1. Load Assessment ─────────────────────────────────────────────────
        assessment = self.db.get(Assessment, assessment_id)
        if assessment is None:
            raise NotFoundError(f"Assessment {assessment_id!r} not found.")

        # ── 2. Verify token ────────────────────────────────────────────────────
        if not token_auth.verify_submission_token(assessment_id, token):
            raise InvalidTokenError(
                f"Invalid token for assessment {assessment_id!r}."
            )

        # ── 3. Enforce state ───────────────────────────────────────────────────
        _non_reschedulable = {AssessmentStatus.completed, AssessmentStatus.expired}
        if assessment.status in _non_reschedulable:
            raise InvalidStateError(
                f"Assessment {assessment_id!r} cannot be rescheduled: "
                f"status is {assessment.status.value!r}."
            )

        # ── 4. Fetch active prompt template ────────────────────────────────────
        prompt_template = (
            self.db.query(PromptTemplate)
            .filter(
                PromptTemplate.slug == "reschedule_classification",
                PromptTemplate.is_active.is_(True),
            )
            .first()
        )
        if prompt_template is None:
            raise NotFoundError(
                "No active 'reschedule_classification' prompt template found."
            )

        # ── 5. Classify via LLM ────────────────────────────────────────────────
        classification = self.llm.classify_reschedule_request(
            RescheduleClassificationRequest(
                reason=reason,
                prompt_template_body=prompt_template.body,
            )
        )

        # ── 6. Determine approval ──────────────────────────────────────────────
        approved = classification.category in APPROVED_CATEGORIES

        # ── 7. Persist RescheduleRequest ───────────────────────────────────────
        reschedule_request = RescheduleRequest(
            assessment_id=assessment_id,
            reason_text=reason,
            classification_category=classification.category,
            category_reasoning=classification.reasoning,
            approved=approved,
        )
        self.db.add(reschedule_request)
        self.db.flush()

        # ── 8. Reschedule if approved ──────────────────────────────────────────
        updated_assessment: Optional[Assessment] = None

        if approved:
            if not assessment.scheduled_job_ids:
                raise InvalidStateError(
                    f"Assessment {assessment_id!r} is approved for reschedule "
                    "but has no stored job IDs to cancel."
                )

            # Load curriculum explicitly to avoid lazy load on a potentially
            # expired object after commit.
            from sqlalchemy.orm import joinedload
            assessment = (
                self.db.query(Assessment)
                .options(joinedload(Assessment.curriculum))
                .filter(Assessment.id == assessment_id)
                .first()
            )

            # Delegate schedule calculation to AssessmentService to keep the
            # window logic in one place.
            from app.services.assessment_service import AssessmentService
            new_scheduled_at = AssessmentService(
                self.db, self.llm
            )._calculate_scheduled_at(assessment.curriculum.target_completion_date)

            settings = get_settings()
            new_reminder_at = new_scheduled_at - timedelta(
                hours=settings.reminder_hours_before
            )
            new_due_date = new_scheduled_at + timedelta(
                days=settings.assessment_due_days
            )

            new_job_ids: AssessmentJobIds = self.scheduler_service.reschedule_assessment(
                assessment_id=assessment_id,
                new_scheduled_at=new_scheduled_at,
                new_reminder_at=new_reminder_at,
                new_due_date=new_due_date,
                existing_job_ids=AssessmentJobIds(**assessment.scheduled_job_ids),
            )

            assessment.scheduled_at = new_scheduled_at
            assessment.reminder_at = new_reminder_at
            assessment.due_date = new_due_date
            assessment.scheduled_job_ids = {
                "send_reminder": new_job_ids.send_reminder,
                "send_assessment": new_job_ids.send_assessment,
                "expire": new_job_ids.expire,
            }
            updated_assessment = assessment

        self.db.commit()
        self.db.refresh(reschedule_request)
        if updated_assessment is not None:
            self.db.refresh(updated_assessment)

        return reschedule_request, updated_assessment
