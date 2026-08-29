from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import NotFoundError
from app.interfaces.email import (
    AssessmentEmailData,
    EmailInterface,
    MidtermHoldReminderEmailData,
    ReminderEmailData,
    ResultsEmailData,
    SyllabusEmailData,
    TranscriptEmailData,
)
from app.models.assessment import Assessment
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.submission import Submission
from app.services.syllabus_builder import build_syllabus, serialize_syllabus_content
from app.services.transcript_service import compute_transcript


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

        settings = get_settings()
        is_entry = assessment.curriculum.entry_type is not None
        is_midterm = assessment.curriculum.entry_type == CurriculumEntryType.midterm
        recipients = (
            [settings.exam_recipient_email or settings.user_email]
            if is_entry else [settings.user_email]
        )
        self.email.send_assessment_email(AssessmentEmailData(
            recipient_emails=recipients,
            assessment_id=assessment_id,
            topic=assessment.curriculum.topic,
            assessment_text=(assessment.part1_text if is_midterm else assessment.assessment_text) or "",
            part2_text=assessment.part2_text if is_midterm else None,
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

        settings = get_settings()
        is_entry = assessment.curriculum.entry_type is not None
        recipients = (
            [settings.exam_recipient_email or settings.user_email]
            if is_entry else [settings.user_email]
        )
        self.email.send_reminder_email(ReminderEmailData(
            recipient_emails=recipients,
            topic=assessment.curriculum.topic,
            scheduled_at=assessment.scheduled_at,
            expire_date=assessment.due_date,
            key_topics=self._parse_key_topics(assessment.curriculum.extracted_content),
            is_pre_deadline=is_entry,
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
        is_entry = submission.assessment.curriculum.entry_type is not None
        recipients = settings.results_recipient_emails if is_entry else [settings.user_email]

        self.email.send_results_email(ResultsEmailData(
            recipient_emails=recipients,
            topic=submission.assessment.curriculum.topic,
            attempt_number=submission.assessment.attempt_number,
            mastery_score=grade.mastery_score,
            passed=passed,
            overall_feedback=grade.overall_feedback,
            weak_areas=grade.weak_areas or [],
        ))

    def send_syllabus_email(self, upload_id: str) -> None:
        upload = (
            self.db.query(CurriculumUpload)
            .options(
                joinedload(CurriculumUpload.entries).joinedload(Curriculum.resources),
                joinedload(CurriculumUpload.entries).joinedload(Curriculum.midterm_detail),
            )
            .filter(CurriculumUpload.id == upload_id)
            .first()
        )
        if not upload:
            raise NotFoundError(f"CurriculumUpload {upload_id!r} not found")

        content = build_syllabus(upload, list(upload.entries))
        settings = get_settings()
        recipients = [settings.syllabus_recipient_email or settings.user_email]

        if upload.course_material_snapshot is None:
            from app.models._utils import utcnow
            upload.course_material_snapshot = serialize_syllabus_content(content)
            upload.course_material_captured_at = utcnow()
            self.db.commit()

        self.email.send_syllabus_email(SyllabusEmailData(
            recipient_emails=recipients,
            upload_id=upload.id,
            source_filename=content.source_filename,
            chapters=content.chapters,
            midterms=content.midterms,
        ))

    def send_transcript_email(
        self, upload_id: str, recipient_emails: Optional[list[str]] = None
    ) -> None:
        """Send the current transcript for one upload.

        recipient_emails is None for the per-event trigger (grade_submission_job)
        — sends to the primary (user_email) only. The periodic secondary-copy
        check (CurriculumUploadService.send_periodic_transcript_if_due, run
        from recheck_pending_midterms_job) passes recipient_emails explicitly
        so the secondary recipient is never touched by the per-event path.
        """
        content = compute_transcript(self.db, upload_id)
        settings = get_settings()
        recipients = recipient_emails if recipient_emails is not None else [settings.user_email]

        self.email.send_transcript_email(TranscriptEmailData(
            recipient_emails=recipients,
            upload_id=content.upload_id,
            source_filename=content.source_filename,
            entry_groups=content.chapter_groups,
            resolved_count=content.resolved_count,
            total_entry_count=content.total_entry_count,
            graded_count=content.graded_count,
            total_credits=content.total_credits,
            total_points=content.total_points,
            gpa=content.gpa,
            course_material=content.course_material,
            course_material_captured_at=content.course_material_captured_at,
        ))

    def send_midterm_hold_reminder_email(self, curriculum_id: str) -> None:
        curriculum = self.db.get(Curriculum, curriculum_id)
        if not curriculum or not curriculum.midterm_detail:
            raise NotFoundError(f"Midterm curriculum entry {curriculum_id!r} not found")

        detail = curriculum.midterm_detail
        missing = [
            label
            for slug, label in detail.pending_completion_labels.items()
            if detail.pending_completion_slots.get(slug) is None
        ]
        settings = get_settings()
        recipients = [settings.syllabus_recipient_email or settings.user_email]

        self.email.send_midterm_hold_reminder_email(MidtermHoldReminderEmailData(
            recipient_emails=recipients,
            topic=curriculum.topic,
            completion_date=curriculum.target_completion_date,
            missing_labels=missing,
        ))
