"""Email interface contract and associated data types.

The implementing class (e.g. ResendEmailAdapter) is responsible for:
  - Composing the HTML/text body from the provided data
  - Calling the Resend API
  - Raising EmailDeliveryError on non-recoverable send failures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Protocol

# ── Request DTOs ──────────────────────────────────────────────────────────────


@dataclass
class AssessmentEmailData:
    """Data required to send the assessment delivery email.

    Per spec, the email must include:
      assessment_id, topic, duration_minutes, due_date, submission_link
    """

    recipient_email: str
    assessment_id: str
    topic: str
    assessment_text: str
    duration_minutes: Optional[int]
    scheduled_at: datetime
    due_date: datetime
    # Full signed URL: {app_base_url}/submit?token={submission_token}
    # Constructed by the service from settings.app_base_url + the HMAC token.
    submission_link: str


@dataclass
class ReminderEmailData:
    """Data required to send the pre-assessment reminder email.

    The reminder fires 1 day before the assessment email is sent.
    It does NOT include a submission link — that is in the assessment email.
    """

    recipient_email: str
    topic: str
    scheduled_at: datetime   # when the assessment email will be delivered
    expire_date: datetime    # submission deadline; assessment expires after this
    key_topics: list[str]    # parsed from curriculum analysis; may be empty


@dataclass
class ResultsEmailData:
    """Data required to send the grading results email."""

    recipient_email: str
    topic: str
    attempt_number: int
    mastery_score: float
    passed: bool
    overall_feedback: str
    weak_areas: list[str]


# ── Exceptions ────────────────────────────────────────────────────────────────


class EmailError(Exception):
    """Base class for all email interface errors."""


class EmailDeliveryError(EmailError):
    """Raised when an email cannot be delivered after the provider's
    own retry logic has been exhausted."""


# ── Protocol ──────────────────────────────────────────────────────────────────


class EmailInterface(Protocol):
    """Structural interface for sending transactional emails.

    Future implementing class: ResendEmailAdapter
      Located at: app/adapters/resend_email.py
      Dependencies: resend SDK, app.config.get_settings
    """

    def send_assessment_email(self, data: AssessmentEmailData) -> None: ...

    def send_reminder_email(self, data: ReminderEmailData) -> None: ...

    def send_results_email(self, data: ResultsEmailData) -> None: ...
