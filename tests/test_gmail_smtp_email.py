"""Gmail SMTP migration: GmailSMTPEmailAdapter's transport behavior, and
app.dependencies._build_email()'s provider-selection order (Gmail is now
the live path; Resend is a kept-but-unused rollback path; RESEND_API_KEY
must never be read once Gmail credentials are configured).

Every test here mocks smtplib — zero real network calls, same discipline
as the rest of the suite.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.config import get_settings
from app.interfaces.email import EmailDeliveryError, ReminderEmailData


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestGmailSMTPAdapterSend:
    def test_send_logs_in_and_sends_via_smtp_ssl(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "sender@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd efgh ijkl mnop")
        get_settings.cache_clear()

        from app.adapters.gmail_smtp_email import GmailSMTPEmailAdapter

        server = MagicMock()
        server.__enter__.return_value = server
        smtp_ssl = MagicMock(return_value=server)
        monkeypatch.setattr("smtplib.SMTP_SSL", smtp_ssl)

        adapter = GmailSMTPEmailAdapter()
        adapter.send_reminder_email(ReminderEmailData(
            recipient_emails=["someone.else@example.com"],
            topic="Async/Await",
            scheduled_at=datetime(2026, 9, 1),
            expire_date=datetime(2026, 9, 3),
            key_topics=[],
            is_pre_deadline=False,
        ))

        smtp_ssl.assert_called_once_with("smtp.gmail.com", 465, timeout=30)
        server.login.assert_called_once_with("sender@gmail.com", "abcd efgh ijkl mnop")

        send_args = server.sendmail.call_args.args
        assert send_args[0] == "sender@gmail.com"
        assert send_args[1] == ["someone.else@example.com"]
        # The specific thing that's been broken: the recipient in the sent
        # message is genuinely a different address from the sender's own.
        assert "someone.else@example.com" != "sender@gmail.com"
        assert "Async/Await" in send_args[2]

    def test_smtp_failure_raises_email_delivery_error(self, monkeypatch):
        import smtplib

        monkeypatch.setenv("GMAIL_ADDRESS", "sender@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "bad-password")
        get_settings.cache_clear()

        from app.adapters.gmail_smtp_email import GmailSMTPEmailAdapter

        def _raise(*a, **k):
            raise smtplib.SMTPAuthenticationError(535, b"Authentication failed")

        monkeypatch.setattr("smtplib.SMTP_SSL", _raise)

        adapter = GmailSMTPEmailAdapter()
        with pytest.raises(EmailDeliveryError):
            adapter.send_reminder_email(ReminderEmailData(
                recipient_emails=["someone.else@example.com"],
                topic="Async/Await",
                scheduled_at=datetime(2026, 9, 1),
                expire_date=datetime(2026, 9, 3),
                key_topics=[],
                is_pre_deadline=False,
            ))


class TestBuildEmailProviderSelection:
    """app.dependencies._build_email() — Gmail first, Resend as rollback,
    Stub if neither is configured. test_mode still short-circuits to
    FakeEmailAdapter regardless (unchanged, not re-tested here)."""

    def test_gmail_selected_when_configured(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "sender@gmail.com")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "app-password")
        monkeypatch.setenv("RESEND_API_KEY", "re_should_be_ignored")
        get_settings.cache_clear()

        from app.dependencies import _build_email
        from app.adapters.gmail_smtp_email import GmailSMTPEmailAdapter

        assert isinstance(_build_email(), GmailSMTPEmailAdapter)

    def test_resend_selected_as_rollback_when_gmail_unset(self, monkeypatch):
        # setenv("", ...) rather than delenv: a real .env file may itself
        # define these (e.g. a developer's live Gmail credentials), and a
        # bare os.environ deletion doesn't beat a value pydantic-settings
        # would otherwise load from that file — an explicit empty string
        # in the process env does.
        monkeypatch.setenv("GMAIL_ADDRESS", "")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "")
        monkeypatch.setenv("RESEND_API_KEY", "re_test_key")
        get_settings.cache_clear()

        from app.dependencies import _build_email
        from app.adapters.resend_email import ResendEmailAdapter

        assert isinstance(_build_email(), ResendEmailAdapter)

    def test_stub_when_neither_configured(self, monkeypatch):
        monkeypatch.setenv("GMAIL_ADDRESS", "")
        monkeypatch.setenv("GMAIL_APP_PASSWORD", "")
        monkeypatch.setenv("RESEND_API_KEY", "")
        get_settings.cache_clear()

        from app.dependencies import _build_email, StubEmailAdapter

        assert isinstance(_build_email(), StubEmailAdapter)
