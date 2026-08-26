import logging

from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import _email, _llm, get_scheduler_adapter
from app.models.submission import Submission
from app.services.assessment_service import AssessmentService
from app.services.curriculum_service import CurriculumService
from app.services.email_service import EmailService
from app.services.grading_service import GradingService
from app.services.scheduler_service import SchedulerService

logger = logging.getLogger(__name__)


def grade_submission_job(submission_id: str) -> None:
    """Scheduler entrypoint: grade a submission, then either mark mastery or
    generate a retest — capped at 2 total attempts (original + 1 retake),
    regardless of score, and regardless of whether the graded attempt was
    on-time or late-token-covered (attempt_number increments identically
    either way, so no separate branch is needed for that).
    """
    logger.info("Starting job: grade_%s", submission_id)
    db = SessionLocal()
    try:
        grade = GradingService(db, _llm).grade(submission_id)
        submission = db.get(Submission, submission_id)
        assessment = submission.assessment
        settings = get_settings()
        scheduler_service = SchedulerService(db, get_scheduler_adapter())

        if grade.mastery_score >= settings.mastery_threshold:
            CurriculumService(db, _llm, scheduler_service).mark_mastery(
                assessment.curriculum_id
            )
        elif assessment.attempt_number < 2:
            retest = AssessmentService(db, _llm).create_retest(
                assessment.curriculum_id, previous_grade_id=grade.id
            )
            scheduler_service.schedule_assessment_jobs(
                assessment_id=retest.id,
                scheduled_at=retest.scheduled_at,
                reminder_at=retest.reminder_at,
                due_date=retest.due_date,
            )
        # else: attempt_number == 2 and still below threshold — cap reached,
        # terminal, no further action.

        # Grade is committed. Send results email; failure is non-fatal since
        # the grade is already persisted and the user can check manually.
        try:
            EmailService(db, _email).send_results_email(submission_id)
        except Exception:
            logger.exception("Results email failed for submission %s", submission_id)

        # Flow (e): transcript regeneration, entries only, immediately after
        # (d) — non-fatal for the same reason as the results email above.
        if assessment.curriculum.upload_id is not None:
            try:
                EmailService(db, _email).send_transcript_email(assessment.curriculum.upload_id)
            except Exception:
                logger.exception(
                    "Transcript email failed for upload %s", assessment.curriculum.upload_id
                )
    finally:
        db.close()
