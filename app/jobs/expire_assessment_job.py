import logging

from app.database import SessionLocal
from app.models.assessment import Assessment, AssessmentStatus

logger = logging.getLogger(__name__)


def expire_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: mark an assessment expired if due_date has
    passed with no submission on record.

    Checks status in (scheduled, active), not just active. Under normal
    scheduling scheduled_at and due_date are always 2+ real days apart, so
    by the time this fires send_assessment_job has long since flipped
    status to active. But when both were registered already in the past
    (e.g. a stale-date bug elsewhere computing scheduled_at/due_date from
    an outdated reference date), APScheduler's misfire handling fires both
    jobs at nearly the same instant instead of days apart — and
    send_assessment_job is slow (LLM generation + email) while this check
    is near-instant, so this job can win the race and run while status is
    still `scheduled`. Since this is a one-shot date-triggered job,
    checking only `active` would silently no-op and vanish forever in that
    case (a real production incident), permanently stuck at whatever
    status send_assessment_job leaves it in, with no way left to ever mark
    it expired. Accepting `scheduled` too makes the outcome correct
    regardless of which job wins the race: send_assessment_job's own
    activation check already only flips scheduled/needs_manual_diagnosis
    to active, so it won't overwrite an expired status set here first.
    """
    logger.info("Starting job: expire_%s", assessment_id)
    db = SessionLocal()
    try:
        assessment = db.get(Assessment, assessment_id)
        if assessment is not None and assessment.status in (
            AssessmentStatus.scheduled, AssessmentStatus.active
        ):
            assessment.status = AssessmentStatus.expired
            db.commit()
    finally:
        db.close()
