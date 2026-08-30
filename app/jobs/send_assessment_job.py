import logging
from datetime import datetime

from app.database import SessionLocal
from app.dependencies import _email, _llm
from app.interfaces.llm import LLMToolBudgetExceededError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.services.assessment_service import AssessmentService
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def send_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send assessment email and activate assessment.

    For curriculum-upload entries, content isn't generated until this exact
    moment (assessment_text and part1_text are both still None) — deferring
    generation to send-time keeps resource-grounding maximally current and
    avoids spending LLM calls on exams that were scheduled weeks/months
    earlier. Standalone assessments already have their content populated
    eagerly at creation (today's unmodified behavior), so this branch is a
    no-op for them.

    Claims the assessment atomically before doing any work — a discovered-
    in-production race (see Assessment.send_job_claimed_at) where a manual
    retry racing a still-in-flight scheduled execution both generated
    content and both sent an assessment-ready email for the same row.
    """
    logger.info("Starting job: assessment_%s", assessment_id)
    db = SessionLocal()
    try:
        claimed = (
            db.query(Assessment)
            .filter(Assessment.id == assessment_id, Assessment.send_job_claimed_at.is_(None))
            .update({"send_job_claimed_at": datetime.utcnow()}, synchronize_session=False)
        )
        db.commit()
        if claimed == 0:
            logger.info(
                "assessment_%s already claimed by another execution of send_assessment_job — skipping",
                assessment_id,
            )
            return

        try:
            assessment = db.get(Assessment, assessment_id)
            if assessment is None:
                return

            if assessment.assessment_text is None and assessment.part1_text is None:
                service = AssessmentService(db, _llm)
                if assessment.curriculum.entry_type == CurriculumEntryType.midterm:
                    service.generate_midterm_content(assessment)
                else:
                    service.generate_assessment_content(assessment)
                db.commit()

            EmailService(db, _email).send_assessment_email(assessment_id)

            assessment = db.get(Assessment, assessment_id)
            if assessment and assessment.status == AssessmentStatus.scheduled:
                assessment.status = AssessmentStatus.active
                db.commit()
        except LLMToolBudgetExceededError as exc:
            # Distinct from the generic failure below: this generation is
            # expensive and has already failed the same way
            # _TOOL_PATH_MAX_ATTEMPTS times in a row within one call to
            # this job — a 15-minute-interval automatic retry (see
            # recheck_stuck_assessments_job) would just re-pay for the same
            # failure, possibly for hours before giving up. Mark it
            # explicitly instead: the sweep's query only matches
            # status == scheduled, so needs_manual_diagnosis is never
            # picked up automatically. A human resolves the real cause and
            # uses POST /api/v1/assessments/{id}/resend.
            curriculum_id = assessment.curriculum_id if assessment else "?"
            db.rollback()
            db.query(Assessment).filter(Assessment.id == assessment_id).update(
                {
                    "status": AssessmentStatus.needs_manual_diagnosis,
                    "send_job_claimed_at": None,
                },
                synchronize_session=False,
            )
            db.commit()
            logger.error(
                "Assessment %s (curriculum %s) exhausted its tool-call budget "
                "during content generation — %s token(s) spent across %s "
                "attempt(s), ceiling %s — and needs manual diagnosis, not "
                "automatic retry. Marked needs_manual_diagnosis; "
                "recheck_stuck_assessments_job will NOT retry it. Resolve the "
                "underlying cause, then use POST "
                "/api/v1/assessments/%s/resend to recover it manually. "
                "Diagnostic: %s",
                assessment_id, curriculum_id,
                exc.tokens_spent, exc.attempts_made, exc.ceiling,
                assessment_id, exc,
            )
            raise
        except Exception:
            # Release the claim so a genuine retry after a real failure
            # (not a race) isn't permanently locked out.
            db.rollback()
            db.query(Assessment).filter(Assessment.id == assessment_id).update(
                {"send_job_claimed_at": None}, synchronize_session=False
            )
            db.commit()
            raise
    finally:
        db.close()
