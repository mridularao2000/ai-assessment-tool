import logging

from app.database import SessionLocal
from app.dependencies import _email, _llm
from app.services.email_service import EmailService
from app.services.grading_service import GradingService

logger = logging.getLogger(__name__)


def grade_submission_job(submission_id: str) -> None:
    """Scheduler entrypoint: grade a submitted assessment and email results."""
    logger.info("Starting job: grade_%s", submission_id)
    db = SessionLocal()
    try:
        GradingService(db, _llm).grade(submission_id)
        # Grade is committed. Send results email; failure is non-fatal since
        # the grade is already persisted and the user can check manually.
        try:
            EmailService(db, _email).send_results_email(submission_id)
        except Exception:
            logger.exception("Results email failed for submission %s", submission_id)
    finally:
        db.close()
