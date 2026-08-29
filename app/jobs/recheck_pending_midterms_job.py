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
from app.services.transcript_service import MISSED_NO_SCORE, display_status

logger = logging.getLogger(__name__)


def recheck_pending_midterms_job() -> None:
    """Scheduler entrypoint: runs daily (CronTrigger, absolute calendar
    time — see apscheduler_adapter._schedule_pending_midterm_recheck).

    Does two independent things on this same daily tick, both driven by
    real elapsed wall-clock time rather than scheduler-registration time,
    which is what makes it safe to re-register on every process restart:

    1. For every Midterm currently held for missing resources, clears the
       hold if all pending_completion slots are now filled; otherwise
       re-sends the hold reminder only if the throttle interval has
       elapsed. Re-checking itself always runs daily — only the reminder
       EMAIL is throttled.
    2. For every open (non-closed) curriculum upload, sends the periodic
       secondary-recipient transcript copy if its configured interval has
       elapsed since the last one. This used to be its own
       IntervalTrigger-scheduled job (send_biweekly_transcript_job); folded
       in here because IntervalTrigger recomputes next_run_time relative to
       *registration* time, so re-registering it on every restart (as
       start() did, with replace_existing=True) silently reset its
       countdown — see CurriculumUploadService.send_periodic_transcript_if_due
       for the fix (a persisted per-upload timestamp compared against wall
       clock, immune to how many restarts happen between checks).
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
            if display_status(db, curriculum) == MISSED_NO_SCORE:
                # Permanently past its grace month — no token can recover
                # it (see check_and_clear_hold), so there's nothing left
                # to remind about. Leave resources_hold as-is; the
                # transcript/GPA already read this state correctly via
                # display_status without needing it flipped.
                continue
            due = (
                curriculum.last_hold_reminder_at is None
                or (utcnow() - curriculum.last_hold_reminder_at) >= interval
            )
            if due:
                service.send_hold_reminder(curriculum, email_service)

        open_uploads = db.query(CurriculumUpload).filter(CurriculumUpload.closed_at.is_(None)).all()
        for upload in open_uploads:
            service.send_periodic_transcript_if_due(upload, email_service)
    finally:
        db.close()
