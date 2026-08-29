"""Email interface contract and associated data types.

The implementing class (GmailSMTPEmailAdapter — the live send path; also
ResendEmailAdapter, kept as an unused rollback path) is responsible for:
  - Composing the HTML/text body from the provided data
  - Sending it through the underlying provider
  - Raising EmailDeliveryError on non-recoverable send failures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, Protocol

# ── Request DTOs ──────────────────────────────────────────────────────────────


@dataclass
class AssessmentEmailData:
    """Data required to send the assessment delivery email.

    Per spec, the email must include:
      assessment_id, topic, duration_minutes, due_date, submission_link
    """

    recipient_emails: list[str]
    assessment_id: str
    topic: str
    # For Midterms: Part 1 content. For everything else: the whole exam.
    assessment_text: str
    duration_minutes: Optional[int]
    scheduled_at: datetime
    due_date: datetime
    # Full signed URL: {app_base_url}/submit?token={submission_token}
    # Constructed by the service from settings.app_base_url + the HMAC token.
    submission_link: str
    # Set only for Midterms — Part 2 content, rendered as a second section
    # in the same email rather than a separate method, since the
    # difference is purely presentational (one exam, two parts).
    part2_text: Optional[str] = None


@dataclass
class ReminderEmailData:
    """Data required to send the pre-assessment reminder email.

    For standalone (is_pre_deadline=False, default): fires 1 day BEFORE the
    assessment email is sent — a "your exam arrives tomorrow" heads-up. It
    does NOT include a submission link — that is in the assessment email.

    For curriculum-upload entries (is_pre_deadline=True): fires counting
    back from due_date (the deadline) instead, AFTER the exam was already
    sent — a "your deadline is approaching" nudge. The copy must differ
    accordingly; see ResendEmailAdapter.send_reminder_email().
    """

    recipient_emails: list[str]
    topic: str
    scheduled_at: datetime   # when the assessment email will be delivered
    expire_date: datetime    # submission deadline; assessment expires after this
    key_topics: list[str]    # parsed from curriculum analysis; may be empty
    is_pre_deadline: bool = False


@dataclass
class ResultsEmailData:
    """Data required to send the grading results email."""

    recipient_emails: list[str]
    topic: str
    attempt_number: int
    mastery_score: float
    passed: bool
    overall_feedback: str
    weak_areas: list[str]


@dataclass
class SyllabusEmailData:
    """Data required to send the full chapter-organized syllabus email.

    chapters/midterms are the SyllabusChapterSection/SyllabusMidtermRow
    dataclasses from app.services.syllabus_builder — passed through as-is
    so the adapter renders exactly what was built, verbatim, with no
    re-summarization at the email layer.
    """

    recipient_emails: list[str]
    upload_id: str
    source_filename: str
    chapters: list  # list[SyllabusChapterSection]
    midterms: list  # list[SyllabusMidtermRow]


@dataclass
class TranscriptEmailData:
    """Data required to send the transcript email (flow e) — the transcript
    table is rendered first, the frozen "Course Material" section after it.

    entry_groups/course_material are passed through as-is from
    app.services.transcript_service (TranscriptChapterGroup list and the
    raw course_material snapshot dict) — no re-summarization at the email
    layer, same discipline as SyllabusEmailData.
    """

    recipient_emails: list[str]
    upload_id: str
    source_filename: str
    entry_groups: list          # list[TranscriptChapterGroup]
    resolved_count: int
    total_entry_count: int
    graded_count: int
    total_credits: float
    total_points: float
    gpa: float
    course_material: Optional[dict]
    course_material_captured_at: Optional[datetime]


@dataclass
class MidtermHoldReminderEmailData:
    """Data required to send the "resources still missing" reminder for a
    Midterm whose completion_date has passed with pending_completion slots
    still unfilled."""

    recipient_emails: list[str]
    topic: str
    completion_date: date
    missing_labels: list[str]   # verbatim labels of still-unfilled slots


# ── Exceptions ────────────────────────────────────────────────────────────────


class EmailError(Exception):
    """Base class for all email interface errors."""


class EmailDeliveryError(EmailError):
    """Raised when an email cannot be delivered after the provider's
    own retry logic has been exhausted."""


# ── Protocol ──────────────────────────────────────────────────────────────────


class EmailInterface(Protocol):
    """Structural interface for sending transactional emails.

    Implementing classes:
      GmailSMTPEmailAdapter (live send path)
        Located at: app/adapters/gmail_smtp_email.py
        Dependencies: smtplib (stdlib), app.config.get_settings
      ResendEmailAdapter (rollback path, unused by default)
        Located at: app/adapters/resend_email.py
        Dependencies: resend SDK, app.config.get_settings
    """

    def send_assessment_email(self, data: AssessmentEmailData) -> None: ...

    def send_reminder_email(self, data: ReminderEmailData) -> None: ...

    def send_results_email(self, data: ResultsEmailData) -> None: ...

    def send_syllabus_email(self, data: SyllabusEmailData) -> None: ...

    def send_transcript_email(self, data: TranscriptEmailData) -> None: ...

    def send_midterm_hold_reminder_email(self, data: MidtermHoldReminderEmailData) -> None: ...
