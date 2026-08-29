"""Resend email adapter implementing EmailInterface.

Kept in the codebase unused-but-present after the migration to
GmailSMTPEmailAdapter (see app/adapters/gmail_smtp_email.py) — purely as a
rollback path. See app.dependencies._build_email() for the current
provider-selection order.
"""
from __future__ import annotations

import resend

from app.adapters.html_email_bodies import HtmlEmailBodyMixin
from app.config import get_settings
from app.interfaces.email import EmailDeliveryError


class ResendEmailAdapter(HtmlEmailBodyMixin):
    """EmailInterface implementation using the Resend transactional email API."""

    def __init__(self) -> None:
        settings = get_settings()
        resend.api_key = settings.resend_api_key
        self._from = f"{settings.resend_from_name} <{settings.resend_from_email}>"

    def _send(self, to: list[str], subject: str, body_html: str) -> None:
        try:
            resend.Emails.send({
                "from": self._from,
                "to": to,
                "subject": subject,
                "html": body_html,
            })
        except Exception as exc:
            raise EmailDeliveryError(f"Resend failed: {exc}") from exc
