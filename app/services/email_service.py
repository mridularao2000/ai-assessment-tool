from __future__ import annotations

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import NotFoundError
from app.interfaces.email import (
    AssessmentEmailData,
    EmailInterface,
    ReminderEmailData,
    ResultsEmailData,
)
from app.models.assessment import Assessment
from app.models.submission import Submission


class EmailService:
    """Sends transactional emails for assessment lifecycle events."""

    def __init__(self, db: Session, email: EmailInterface) -> None:
        self.db = db
        self.email = email

    def _submission_link(self, assessment_id: str, token: str) -> str:
        base = get_settings().app_base_url.rstrip("/")
        return f"{base}?assessment_id={assessment_id}&token={token}"

    def send_assessment_email(self, assessment_id: str) -> None:
        assessment = (
            self.db.query(Assessment)
            .options(joinedload(Assessment.curriculum))
            .filter(Assessment.id == assessment_id)
            .first()
        )
        if not assessment:
            raise NotFoundError(f"Assessment {assessment_id!r} not found")

        self.email.send_assessment_email(AssessmentEmailData(
            recipient_email=get_settings().user_email,
            assessment_id=assessment_id,
            topic=assessment.curriculum.topic,
            assessment_text=assessment.assessment_text or "",
            duration_minutes=assessment.duration_minutes,
            scheduled_at=assessment.scheduled_at,
            due_date=assessment.due_date,
            submission_link=self._submission_link(
                assessment_id, assessment.submission_token
            ),
        ))

    @staticmethod
    def _parse_key_topics(extracted_content: str | None) -> list[str]:
        """Extract the Key Topics line written by the curriculum analysis step."""
        if not extracted_content:
            return []
        for line in extracted_content.splitlines():
            if line.startswith("Key Topics:"):
                raw = line[len("Key Topics:"):].strip()
                return [t.strip() for t in raw.split(",") if t.strip()]
        return []

    def send_reminder_email(self, assessment_id: str) -> None:
        assessment = (
            self.db.query(Assessment)
            .options(joinedload(Assessment.curriculum))
            .filter(Assessment.id == assessment_id)
            .first()
        )
        if not assessment:
            raise NotFoundError(f"Assessment {assessment_id!r} not found")

        self.email.send_reminder_email(ReminderEmailData(
            recipient_email=get_settings().user_email,
            topic=assessment.curriculum.topic,
            scheduled_at=assessment.scheduled_at,
            expire_date=assessment.due_date,
            key_topics=self._parse_key_topics(assessment.curriculum.extracted_content),
        ))

    def send_results_email(self, submission_id: str) -> None:
        submission = (
            self.db.query(Submission)
            .options(
                joinedload(Submission.assessment).joinedload(Assessment.curriculum),
                joinedload(Submission.grade),
            )
            .filter(Submission.id == submission_id)
            .first()
        )
        if not submission or not submission.grade:
            raise NotFoundError(f"Graded submission {submission_id!r} not found")

        settings = get_settings()
        grade = submission.grade
        passed = grade.mastery_score >= settings.mastery_threshold

        self.email.send_results_email(ResultsEmailData(
            recipient_email=settings.user_email,
            topic=submission.assessment.curriculum.topic,
            attempt_number=submission.assessment.attempt_number,
            mastery_score=grade.mastery_score,
            passed=passed,
            overall_feedback=grade.overall_feedback,
            weak_areas=grade.weak_areas or [],
        ))
