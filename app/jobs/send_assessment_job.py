import logging

from app.database import SessionLocal
from app.dependencies import _email
from app.models.assessment import Assessment, AssessmentStatus
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def send_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send assessment email and activate assessment."""
    logger.info("Starting job: assessment_%s", assessment_id)
    db = SessionLocal()
    try:
        EmailService(db, _email).send_assessment_email(assessment_id)

        assessment = db.get(Assessment, assessment_id)
        if assessment and assessment.status == AssessmentStatus.scheduled:
            assessment.status = AssessmentStatus.active
            db.commit()
    finally:
        db.close()
