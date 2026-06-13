from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.interfaces.scheduler import AssessmentJobIds, SchedulerInterface
from app.models.assessment import Assessment


class SchedulerService:
    """Thin orchestration layer over SchedulerInterface with DB persistence.

    Depends on:
      db        — SQLAlchemy session for persisting job IDs to Assessment rows
      scheduler — SchedulerInterface implementation (e.g. APSchedulerAdapter)

    Contains no business rules, scheduling calculations, or LLM usage.
    """

    def __init__(self, db: Session, scheduler: SchedulerInterface) -> None:
        self.db = db
        self.scheduler = scheduler

    def schedule_assessment_jobs(
        self,
        assessment_id: str,
        scheduled_at: datetime,
        reminder_at: datetime,
        due_date: datetime,
    ) -> AssessmentJobIds:
        """Schedule the three assessment lifecycle jobs and persist their IDs.

        Calls scheduler.schedule_assessment_jobs, writes the returned job IDs
        to Assessment.scheduled_job_ids, and commits.
        """
        job_ids = self.scheduler.schedule_assessment_jobs(
            assessment_id=assessment_id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
        )

        assessment = self.db.get(Assessment, assessment_id)
        assessment.scheduled_job_ids = {
            "send_reminder": job_ids.send_reminder,
            "send_assessment": job_ids.send_assessment,
            "expire": job_ids.expire,
        }
        self.db.commit()

        return job_ids

    def schedule_grade_job(self, submission_id: str) -> str:
        """Schedule an immediate grade job for submission_id.

        No DB writes — the job ID is returned to the caller for optional use.
        """
        return self.scheduler.schedule_grade_job(submission_id)

    def cancel_jobs_for_assessment(self, job_ids: AssessmentJobIds) -> None:
        """Cancel the three assessment lifecycle jobs.

        No DB writes — callers that update the assessment record handle
        the scheduled_job_ids field themselves.
        """
        self.scheduler.cancel_jobs_for_assessment(job_ids)

    def reschedule_assessment(
        self,
        assessment_id: str,
        new_scheduled_at: datetime,
        new_reminder_at: datetime,
        new_due_date: datetime,
        existing_job_ids: AssessmentJobIds,
    ) -> AssessmentJobIds:
        """Cancel existing jobs, schedule new ones, and update the Assessment row.

        Calls scheduler.reschedule_assessment, writes the new job IDs and
        updated dates to the Assessment, and commits.
        """
        new_job_ids = self.scheduler.reschedule_assessment(
            assessment_id=assessment_id,
            new_scheduled_at=new_scheduled_at,
            new_reminder_at=new_reminder_at,
            new_due_date=new_due_date,
            existing_job_ids=existing_job_ids,
        )

        assessment = self.db.get(Assessment, assessment_id)
        assessment.scheduled_at = new_scheduled_at
        assessment.reminder_at = new_reminder_at
        assessment.due_date = new_due_date
        assessment.scheduled_job_ids = {
            "send_reminder": new_job_ids.send_reminder,
            "send_assessment": new_job_ids.send_assessment,
            "expire": new_job_ids.expire,
        }
        self.db.commit()

        return new_job_ids
