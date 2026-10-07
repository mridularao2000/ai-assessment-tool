import logging
from datetime import datetime, timedelta

from app.config import get_settings
from app.database import SessionLocal
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum
from app.models.curriculum_upload import CurriculumUpload

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

    No time-based give-up ceiling: there used to be one
    (stuck_assessment_max_auto_retry_hours), added to stop repeated billed
    LLM calls against a persistent failure. It's gone now because the
    actually-expensive persistent-failure mode — a tool-enabled generation
    that burns its full attempt budget every time — no longer reaches this
    sweep at all: send_assessment_job catches LLMToolBudgetExceededError
    and moves the row straight to needs_manual_diagnosis (excluded by the
    status == scheduled filter below) before this job would ever retry it
    again. Any failure that *does* stay `scheduled` is, by construction,
    one that doesn't hit that budget ceiling, so retrying it every 15
    minutes forever is bounded, not runaway — and a stuck row always
    self-heals the moment its cause is fixed, with no silent permanent
    give-up and no manual step required.

    Excludes any assessment whose curriculum belongs to a CLOSED upload
    (CurriculumUpload.closed_at is not None) — a real production incident
    (2026-10-07): CurriculumUploadService.close_upload() cancels the three
    APScheduler jobs for every entry but deliberately never touches
    Assessment.status (closing is a jobstore-level action, not a DB-state
    rewrite — see its docstring). A row already past its scheduled_at at
    close time is therefore left sitting at status=scheduled forever,
    indistinguishable from a genuinely stuck row to this sweep's query —
    which then "recovered" it, paying for a real (unwanted) LLM generation
    and likely sending its exam email, for a curriculum the user had
    already archived. The outer join to CurriculumUpload naturally covers
    standalone assessments too (upload_id is None -> no matching row ->
    closed_at reads NULL -> IS NULL -> still eligible), so this changes
    nothing for the standalone/non-upload case.
    """
    logger.info("Starting job: recheck_stuck_assessments")
    db = SessionLocal()
    try:
        settings = get_settings()
        grace_cutoff = datetime.utcnow() - timedelta(minutes=settings.stuck_assessment_grace_minutes)

        retry_ids = [
            row[0]
            for row in db.query(Assessment.id)
            .join(Curriculum, Assessment.curriculum_id == Curriculum.id)
            .outerjoin(CurriculumUpload, Curriculum.upload_id == CurriculumUpload.id)
            .filter(
                Assessment.status == AssessmentStatus.scheduled,
                Assessment.scheduled_at < grace_cutoff,
                CurriculumUpload.closed_at.is_(None),
            )
            .all()
        ]
    finally:
        db.close()

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
