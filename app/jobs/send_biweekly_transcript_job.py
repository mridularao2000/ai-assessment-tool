import logging

from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import _email
from app.models.curriculum_upload import CurriculumUpload
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def send_biweekly_transcript_job() -> None:
    """Scheduler entrypoint: runs every transcript_secondary_interval_days
    (default 14), independent of grading events.

    Sends the CURRENT transcript (same content EmailService.send_transcript_email
    computes for the primary, per-event send — not a different digest format)
    for every existing, non-closed CurriculumUpload to the secondary
    recipient. A closed curriculum already got its one final transcript at
    close time and must stay silent forever after — see
    CurriculumUploadService.close_upload(). No-ops entirely if no secondary
    recipient is configured yet.
    """
    logger.info("Starting job: send_biweekly_transcript")
    settings = get_settings()
    if not settings.transcript_secondary_recipient_email:
        logger.info("No transcript_secondary_recipient_email configured — skipping.")
        return

    db = SessionLocal()
    try:
        email_service = EmailService(db, _email)
        upload_ids = [
            u.id for u in db.query(CurriculumUpload)
            .filter(CurriculumUpload.closed_at.is_(None))
            .all()
        ]
        for upload_id in upload_ids:
            try:
                email_service.send_transcript_email(
                    upload_id,
                    recipient_emails=[settings.transcript_secondary_recipient_email],
                )
            except Exception:
                logger.exception(
                    "Biweekly transcript send failed for upload %s", upload_id
                )
    finally:
        db.close()
