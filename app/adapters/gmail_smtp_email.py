"""Gmail SMTP email adapter implementing EmailInterface.

Replaces ResendEmailAdapter as the live send path (see
app.dependencies._build_email()). Uses a Gmail account's App Password —
generated from Google Account → Security → 2-Step Verification → App
Passwords, not the account's regular login password — over authenticated
SMTPS. The sending account does not need to be the real inbox: sending
and receiving addresses are independent, so a dedicated free Gmail
account works fine as the "From".
"""
from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.adapters.html_email_bodies import HtmlEmailBodyMixin
from app.config import get_settings
from app.interfaces.email import EmailDeliveryError

_SMTP_HOST = "smtp.gmail.com"
_SMTP_PORT = 465  # SMTPS (implicit TLS) — Gmail's recommended port for App Passwords.


class GmailSMTPEmailAdapter(HtmlEmailBodyMixin):
    """EmailInterface implementation sending via Gmail SMTP with an App Password."""

    def __init__(self) -> None:
        settings = get_settings()
        self._address = settings.gmail_address
        self._app_password = settings.gmail_app_password
        self._from = f"{settings.resend_from_name} <{settings.gmail_address}>"

    def _send(self, to: list[str], subject: str, body_html: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from
        msg["To"] = ", ".join(to)
        msg.attach(MIMEText(body_html, "html"))

        try:
            with smtplib.SMTP_SSL(_SMTP_HOST, _SMTP_PORT, timeout=30) as server:
                server.login(self._address, self._app_password)
                server.sendmail(self._address, to, msg.as_string())
        except (smtplib.SMTPException, OSError) as exc:
            raise EmailDeliveryError(f"Gmail SMTP failed: {exc}") from exc
