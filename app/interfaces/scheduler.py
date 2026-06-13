"""Scheduler interface contract and associated data types.

The implementing class (e.g. APSchedulerAdapter) is responsible for:
  - Registering jobs with APScheduler using SQLAlchemyJobStore
  - Persisting job IDs so they can be cancelled on reschedule
  - Enforcing the secondary scheduling window guard before creating jobs
    (primary guard is in AssessmentService._calculate_scheduled_at)
  - Raising SchedulerNotRunningError if called before start()
  - Silently ignoring already-completed or missing jobs in cancel operations
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class AssessmentJobIds:
    """APScheduler job ID triplet for the three jobs tied to one assessment.

    Stored as JSON on assessments.scheduled_job_ids so they can be
    retrieved and cancelled when a reschedule is approved.
    """

    send_reminder: str    # fires at Assessment.reminder_at
    send_assessment: str  # fires at Assessment.scheduled_at → sets status=active
    expire: str           # fires at Assessment.due_date → sets status=expired if still active


# ── Exceptions ────────────────────────────────────────────────────────────────


class SchedulerError(Exception):
    """Base class for all scheduler interface errors."""


class SchedulerNotRunningError(SchedulerError):
    """Raised when a scheduling operation is attempted before start()
    has been called (e.g. during application startup ordering issues)."""


class JobNotFoundError(SchedulerError):
    """Raised when a job ID supplied to cancel or reschedule does not exist
    in the scheduler store and silent-ignore is not appropriate."""


# ── Protocol ──────────────────────────────────────────────────────────────────


class SchedulerInterface(Protocol):
    """Structural interface for APScheduler job management.

    All methods are synchronous.  The scheduler must be started via start()
    (called once in the FastAPI lifespan) before any other method is used.

    Future implementing class: APSchedulerAdapter
      Located at: app/adapters/apscheduler_adapter.py
      Dependencies: apscheduler, SQLAlchemyJobStore, app.config.get_settings
    """

    def start(self) -> None:
        """Start the scheduler.

        Called once during FastAPI lifespan startup.
        Must be idempotent — calling twice should not raise.
        """
        ...

    def shutdown(self) -> None:
        """Gracefully stop the scheduler, allowing in-progress jobs to finish.

        Called during FastAPI lifespan teardown.
        """
        ...

    def schedule_assessment_jobs(
        self,
        assessment_id: str,
        scheduled_at: datetime,
        reminder_at: datetime,
        due_date: datetime,
    ) -> AssessmentJobIds:
        """Schedule the three lifecycle jobs for one assessment.

        Jobs created:
          send_reminder_job(assessment_id)    at reminder_at
          send_assessment_job(assessment_id)  at scheduled_at
          expire_assessment_job(assessment_id) at due_date

        Secondary scheduling window guard fires here:
          if not (target_completion_date + min_days <= scheduled_at
                                         <= target_completion_date + max_days):
              raise ValueError

        Returns the three job IDs for storage in assessments.scheduled_job_ids.
        """
        ...

    def schedule_grade_job(self, submission_id: str) -> str:
        """Schedule an immediate grade_submission_job for a received submission.

        The job fires as soon as the scheduler processes it (run_date=now).
        Returns the APScheduler job ID.
        """
        ...

    def cancel_jobs_for_assessment(self, job_ids: AssessmentJobIds) -> None:
        """Cancel the three assessment lifecycle jobs.

        Called when a reschedule request is approved.
        Silently skips jobs that have already fired or been removed.
        """
        ...

    def reschedule_assessment(
        self,
        assessment_id: str,
        new_scheduled_at: datetime,
        new_reminder_at: datetime,
        new_due_date: datetime,
        existing_job_ids: AssessmentJobIds,
    ) -> AssessmentJobIds:
        """Cancel existing jobs and schedule new ones for updated dates.

        Combines cancel_jobs_for_assessment + schedule_assessment_jobs
        as a single logical operation.

        Returns the new AssessmentJobIds for storage in the database.
        """
        ...
