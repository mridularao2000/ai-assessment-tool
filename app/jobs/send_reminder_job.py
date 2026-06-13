import logging

from app.database import SessionLocal
from app.dependencies import _email
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def send_reminder_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send the pre-assessment reminder email."""
    logger.info("Starting job: send_reminder_%s", assessment_id)
    db = SessionLocal()
    try:
        EmailService(db, _email).send_reminder_email(assessment_id)
    finally:
        db.close()
