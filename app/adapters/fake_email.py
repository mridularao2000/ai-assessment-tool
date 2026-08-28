"""Test-mode email adapter — safe no-op, no network call.

Selected by app.dependencies._build_email() only when Settings.test_mode
is true. Production never sets that flag, so this adapter is never
constructed outside an isolated-verification harness — app.adapters.
resend_email (the real Resend-backed adapter) is untouched and remains
the only adapter production code paths can reach.

Distinct from StubEmailAdapter (app.dependencies): that one raises
NotImplementedError to fail loudly when no RESEND_API_KEY is configured
at all — a misconfiguration signal for a server that was never meant to
send email. This one is the opposite: a deliberate, safe no-op for a
server that WILL have real credentials in its environment (inherited
from .env) but must not use them, because the point of running it was to
verify something in isolation, not to actually send mail. This is the
gap that let a real syllabus email go out during a UI-verification run
with an isolated DB but a live RESEND_API_KEY still in scope — test_mode
covers this and the equivalent LLM case with one flag instead of two
independently-rememberable checks.
"""
from __future__ import annotations

import logging

from app.interfaces.email import (
    AssessmentEmailData,
    MidtermHoldReminderEmailData,
    ReminderEmailData,
    ResultsEmailData,
    SyllabusEmailData,
    TranscriptEmailData,
)

logger = logging.getLogger(__name__)


class FakeEmailAdapter:
    """No-op EmailInterface implementation for test mode. See module docstring."""

    def send_assessment_email(self, data: AssessmentEmailData) -> None:
        logger.info("[FAKE_EMAIL] suppressed send_assessment_email to %s", data.recipient_emails)

    def send_reminder_email(self, data: ReminderEmailData) -> None:
        logger.info("[FAKE_EMAIL] suppressed send_reminder_email to %s", data.recipient_emails)

    def send_results_email(self, data: ResultsEmailData) -> None:
        logger.info("[FAKE_EMAIL] suppressed send_results_email to %s", data.recipient_emails)

    def send_syllabus_email(self, data: SyllabusEmailData) -> None:
        logger.info("[FAKE_EMAIL] suppressed send_syllabus_email to %s", data.recipient_emails)

    def send_transcript_email(self, data: TranscriptEmailData) -> None:
        logger.info("[FAKE_EMAIL] suppressed send_transcript_email to %s", data.recipient_emails)

    def send_midterm_hold_reminder_email(self, data: MidtermHoldReminderEmailData) -> None:
        logger.info(
            "[FAKE_EMAIL] suppressed send_midterm_hold_reminder_email to %s", data.recipient_emails
        )
