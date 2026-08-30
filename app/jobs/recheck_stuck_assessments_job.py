import logging
from datetime import datetime, timedelta

from app.config import get_settings
from app.database import SessionLocal
from app.models.assessment import Assessment, AssessmentStatus

logger = logging.getLogger(__name__)


def recheck_stuck_assessments_job() -> None:
    """Scheduler entrypoint: runs every 15 minutes (CronTrigger — see
    apscheduler_adapter._schedule_stuck_assessment_recheck) to catch and
    retry any assessment stuck in `scheduled` status past its scheduled_at.

    send_assessment_job is registered as a one-shot APScheduler `date`
    trigger. If its single execution raised (an LLM call failure, an email
    send failure, the process being killed mid-flight, ...),
    Assessment.send_job_claimed_at is released so a manual retry CAN
    succeed, but nothing retries it automatically — the job itself is
    already consumed from the jobstore the moment it ran once, regardless
    of outcome. Without this sweep, such a row sits stuck indefinitely: the
    UI shows "Not Yet Due" (display_status only reads stored status, never
    compares scheduled_at against now — see transcript_service.py), no
    exam is ever delivered, and it isn't reachable through the late-send
    recovery flow either, since that requires status == expired, which a
    row stuck in `scheduled` never reaches on its own.

    Deliberately recurring rather than a one-shot per-assessment retry: a
    single retry only helps a transient failure. If the underlying cause is
    persistent (broken credentials, a bad prompt template, a structural
    bug), a one-shot retry fails exactly the same way and then gives up
    forever, same as the original job. Running every 15 minutes means the
    system keeps trying — the moment a persistent cause is fixed, the very
    next tick recovers every affected row automatically, with no one
    needing to notice and manually intervene.

    A grace period (settings.stuck_assessment_grace_minutes) avoids racing
    a job that's legitimately still mid-flight (e.g. a slow LLM call) —
    only rows well past their scheduled_at get swept.

    Bounded, not infinite: a retry only helps if the underlying cause is
    transient. If it's persistent (bad credentials, insufficient API
    credit, a broken prompt template), every 15-minute retry is a real,
    billed LLM call that fails for the same reason, forever, with no one
    necessarily noticing. Past settings.stuck_assessment_max_auto_retry_hours,
    this stops calling the LLM for that row automatically and just logs
    loudly instead — a human must resolve the real cause and use the manual
    /resend endpoint once it's fixed.
    """
    logger.info("Starting job: recheck_stuck_assessments")
    db = SessionLocal()
    try:
        settings = get_settings()
        now = datetime.utcnow()
        grace_cutoff = now - timedelta(minutes=settings.stuck_assessment_grace_minutes)
        give_up_cutoff = now - timedelta(hours=settings.stuck_assessment_max_auto_retry_hours)

        stuck = (
            db.query(Assessment.id, Assessment.scheduled_at)
            .filter(
                Assessment.status == AssessmentStatus.scheduled,
                Assessment.scheduled_at < grace_cutoff,
            )
            .all()
        )
        retry_ids = [a_id for a_id, scheduled_at in stuck if scheduled_at >= give_up_cutoff]
        gave_up_ids = [a_id for a_id, scheduled_at in stuck if scheduled_at < give_up_cutoff]
    finally:
        db.close()

    for assessment_id in gave_up_ids:
        logger.error(
            "Assessment %s has been stuck in 'scheduled' for over %s hours — giving up "
            "automatic retries to avoid repeatedly spending real API credit on a likely-"
            "persistent failure. Resolve the underlying cause, then use POST "
            "/api/v1/assessments/%s/resend to recover it manually.",
            assessment_id, settings.stuck_assessment_max_auto_retry_hours, assessment_id,
        )

    if not retry_ids:
        return

    from app.jobs.send_assessment_job import send_assessment_job

    for assessment_id in retry_ids:
        logger.warning(
            "Assessment %s stuck in 'scheduled' more than %s minutes past its "
            "scheduled_at — retrying send",
            assessment_id, settings.stuck_assessment_grace_minutes,
        )
        try:
            send_assessment_job(assessment_id)
        except Exception:
            logger.exception("Retry send failed again for assessment %s", assessment_id)
