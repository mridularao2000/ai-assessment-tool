from __future__ import annotations

from datetime import datetime, timedelta, timezone

from apscheduler.jobstores.base import JobLookupError as APSJobLookupError
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from app.config import get_settings
from app.interfaces.scheduler import (
    AssessmentJobIds,
    SchedulerNotRunningError,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class APSchedulerAdapter:
    """APScheduler 3.x implementation of SchedulerInterface.

    Uses SQLAlchemyJobStore so scheduled jobs survive process restarts.
    start() must be called once (FastAPI lifespan startup) before any
    scheduling method is used. All methods are synchronous.

    Job ID naming:
      send_reminder_{assessment_id}
      assessment_{assessment_id}
      expire_{assessment_id}
      grade_{submission_id}
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._scheduler = BackgroundScheduler(
            jobstores={"default": SQLAlchemyJobStore(url=settings.database_url)},
            job_defaults={
                # Fire jobs that were missed while the server was down,
                # no matter how late, so emails are never silently skipped.
                "misfire_grace_time": None,
                # If multiple missed firings stack up (e.g. after a long outage),
                # run each job once instead of replaying every missed tick.
                "coalesce": True,
            },
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler. Idempotent — safe to call more than once."""
        if not self._scheduler.running:
            self._scheduler.start()
        self._schedule_monthly_token_grant()
        self._schedule_pending_midterm_recheck()
        self._schedule_biweekly_transcript()

    def _schedule_monthly_token_grant(self) -> None:
        """Register the recurring monthly late-submission token grant.

        Fixed job ID + replace_existing=True keeps this idempotent across
        restarts, since jobs persist in the SQLAlchemyJobStore.
        """
        from app.jobs.grant_late_tokens_job import grant_late_tokens_job

        self._scheduler.add_job(
            grant_late_tokens_job,
            trigger="cron",
            day=1,
            hour=0,
            minute=0,
            id="grant_late_tokens_monthly",
            replace_existing=True,
        )

    def _schedule_pending_midterm_recheck(self) -> None:
        """Register the recurring daily pending-resources recheck.

        Same shape as the monthly token grant — global, argument-less,
        fixed-id cron — rather than a per-entry one-shot job, since Midterms
        enter/leave the held state at unpredictable times and a single
        poll-all-held-rows job survives future uploads with no new
        scheduling code.
        """
        from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job

        self._scheduler.add_job(
            recheck_pending_midterms_job,
            trigger="cron",
            hour=6,
            minute=0,
            id="recheck_pending_midterms_daily",
            replace_existing=True,
        )

    def _schedule_biweekly_transcript(self) -> None:
        """Register the recurring biweekly transcript send to the secondary
        recipient — global, argument-less, fixed-id, same shape as the
        monthly token grant and daily recheck.

        Uses trigger="interval" rather than "cron": cron is calendar-based
        (day-of-month / day-of-week) and can't express "every N days from
        an arbitrary anchor" the way a fixed 14-day cadence needs — interval
        is APScheduler's mechanism for that, everything else about the
        registration (fixed id, replace_existing, called from start()) is
        identical to the other two recurring jobs.
        """
        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job

        settings = get_settings()
        self._scheduler.add_job(
            send_biweekly_transcript_job,
            trigger="interval",
            days=settings.transcript_secondary_interval_days,
            id="send_biweekly_transcript",
            replace_existing=True,
        )

    def shutdown(self) -> None:
        """Gracefully stop the scheduler, letting in-progress jobs finish."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=True)

    # ── Internal guard ─────────────────────────────────────────────────────────

    def _require_running(self) -> None:
        if not self._scheduler.running:
            raise SchedulerNotRunningError(
                "APSchedulerAdapter.start() has not been called."
            )

    # ── SchedulerInterface methods ─────────────────────────────────────────────

    def schedule_assessment_jobs(
        self,
        assessment_id: str,
        scheduled_at: datetime,
        reminder_at: datetime,
        due_date: datetime,
    ) -> AssessmentJobIds:
        """Schedule send_reminder, send_assessment, and expire jobs for one assessment."""
        self._require_running()

        from app.jobs.send_reminder_job import send_reminder_job
        from app.jobs.send_assessment_job import send_assessment_job
        from app.jobs.expire_assessment_job import expire_assessment_job

        reminder_id = f"send_reminder_{assessment_id}"
        assessment_job_id = f"assessment_{assessment_id}"
        expire_id = f"expire_{assessment_id}"

        self._scheduler.add_job(
            send_reminder_job,
            trigger="date",
            run_date=reminder_at,
            id=reminder_id,
            args=[assessment_id],
            replace_existing=True,
        )
        self._scheduler.add_job(
            send_assessment_job,
            trigger="date",
            run_date=scheduled_at,
            id=assessment_job_id,
            args=[assessment_id],
            replace_existing=True,
        )
        self._scheduler.add_job(
            expire_assessment_job,
            trigger="date",
            run_date=due_date,
            id=expire_id,
            args=[assessment_id],
            replace_existing=True,
        )

        return AssessmentJobIds(
            send_reminder=reminder_id,
            send_assessment=assessment_job_id,
            expire=expire_id,
        )

    def schedule_grade_job(self, submission_id: str) -> str:
        """Schedule an immediate grade_submission_job. Returns the APScheduler job ID."""
        self._require_running()

        from app.jobs.grade_submission_job import grade_submission_job

        job_id = f"grade_{submission_id}"
        self._scheduler.add_job(
            grade_submission_job,
            trigger="date",
            run_date=_utcnow(),
            id=job_id,
            args=[submission_id],
            replace_existing=True,
        )
        return job_id

    def cancel_jobs_for_assessment(self, job_ids: AssessmentJobIds) -> None:
        """Cancel the three assessment lifecycle jobs. Silently skips missing jobs."""
        self._require_running()

        for jid in (job_ids.send_reminder, job_ids.send_assessment, job_ids.expire):
            try:
                self._scheduler.remove_job(jid)
            except APSJobLookupError:
                pass

    def reschedule_assessment(
        self,
        assessment_id: str,
        new_scheduled_at: datetime,
        new_reminder_at: datetime,
        new_due_date: datetime,
        existing_job_ids: AssessmentJobIds,
    ) -> AssessmentJobIds:
        """Cancel existing jobs and schedule new ones for the updated dates."""
        self._require_running()

        self.cancel_jobs_for_assessment(existing_job_ids)
        return self.schedule_assessment_jobs(
            assessment_id=assessment_id,
            scheduled_at=new_scheduled_at,
            reminder_at=new_reminder_at,
            due_date=new_due_date,
        )
