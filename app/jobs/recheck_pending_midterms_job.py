import logging
from datetime import timedelta

from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import _email, get_scheduler_adapter
from app.models._utils import utcnow
from app.models.curriculum import Curriculum
from app.models.curriculum_upload import CurriculumUpload
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.email_service import EmailService
from app.services.scheduler_service import SchedulerService

logger = logging.getLogger(__name__)


def recheck_pending_midterms_job() -> None:
    """Scheduler entrypoint: runs daily.

    For every Midterm currently held for missing resources, clears the hold
    if all pending_completion slots are now filled; otherwise re-sends the
    hold reminder only if the throttle interval has elapsed. Re-checking
    itself always runs daily — only the reminder EMAIL is throttled.
    """
    logger.info("Starting job: recheck_pending_midterms")
    db = SessionLocal()
    try:
        scheduler_service = SchedulerService(db, get_scheduler_adapter())
        service = CurriculumUploadService(db, _email, scheduler_service)
        email_service = EmailService(db, _email)
        settings = get_settings()
        interval = timedelta(days=settings.pending_hold_reminder_interval_days)

        held = (
            db.query(Curriculum)
            .outerjoin(CurriculumUpload, Curriculum.upload_id == CurriculumUpload.id)
            .filter(
                Curriculum.resources_hold.is_(True),
                # A held entry always belongs to an upload in practice (the
                # hold state only exists for curriculum-upload Midterms),
                # but outerjoin + "upload is None OR not closed" keeps this
                # correct even if that ever changes.
                (CurriculumUpload.id.is_(None)) | (CurriculumUpload.closed_at.is_(None)),
            )
            .all()
        )
        for curriculum in held:
            if service.check_and_clear_hold(curriculum):
                continue
            due = (
                curriculum.last_hold_reminder_at is None
                or (utcnow() - curriculum.last_hold_reminder_at) >= interval
            )
            if due:
                service.send_hold_reminder(curriculum, email_service)
    finally:
        db.close()
