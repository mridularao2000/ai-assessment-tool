"""Resend email adapter implementing EmailInterface."""
from __future__ import annotations

import html
from datetime import datetime

import resend

from app.config import get_settings
from app.interfaces.email import (
    AssessmentEmailData,
    EmailDeliveryError,
    ReminderEmailData,
    ResultsEmailData,
)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%A, %d %B %Y at %H:%M UTC")


def _e(text: str) -> str:
    return html.escape(str(text))


class ResendEmailAdapter:
    """EmailInterface implementation using the Resend transactional email API."""

    def __init__(self) -> None:
        settings = get_settings()
        resend.api_key = settings.resend_api_key
        self._from = f"{settings.resend_from_name} <{settings.resend_from_email}>"

    def _send(self, to: str, subject: str, body_html: str) -> None:
        try:
            resend.Emails.send({
                "from": self._from,
                "to": [to],
                "subject": subject,
                "html": body_html,
            })
        except Exception as exc:
            raise EmailDeliveryError(f"Resend failed: {exc}") from exc

    # ── EmailInterface ────────────────────────────────────────────────────────

    def send_assessment_email(self, data: AssessmentEmailData) -> None:
        duration = f"{data.duration_minutes} minutes" if data.duration_minutes else "unspecified"
        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:620px;margin:0 auto;padding:24px">
  <h2 style="color:#0d6efd;margin-top:0">Your {_e(data.topic)} Assessment</h2>
  <p>Your assessment is ready. Please submit your answer by the due date below.</p>

  <table style="width:100%;border-collapse:collapse;margin:16px 0">
    <tr>
      <td style="padding:6px 12px;background:#f8f9fa;font-weight:600;width:40%">Scheduled</td>
      <td style="padding:6px 12px">{_e(_fmt_dt(data.scheduled_at))}</td>
    </tr>
    <tr>
      <td style="padding:6px 12px;background:#f8f9fa;font-weight:600">Expires</td>
      <td style="padding:6px 12px;color:#dc3545"><strong>{_e(_fmt_dt(data.due_date))}</strong></td>
    </tr>
    <tr>
      <td style="padding:6px 12px;background:#f8f9fa;font-weight:600">Duration</td>
      <td style="padding:6px 12px">{_e(duration)}</td>
    </tr>
  </table>

  <div style="background:#f1f3f5;border-left:4px solid #0d6efd;padding:16px 20px;
              border-radius:4px;margin:20px 0">
    <h3 style="margin-top:0;color:#0d6efd">Assessment</h3>
    <div style="white-space:pre-wrap;line-height:1.6">{_e(data.assessment_text)}</div>
  </div>

  <p style="text-align:center;margin:28px 0">
    <a href="{_e(data.submission_link)}"
       style="background:#0d6efd;color:#fff;padding:14px 28px;text-decoration:none;
              border-radius:6px;font-weight:600;display:inline-block">
      Open Control Panel →
    </a>
  </p>

  <hr style="border:none;border-top:1px solid #dee2e6;margin:24px 0">
  <p style="color:#6c757d;font-size:0.82rem;line-height:1.8">
    <strong>Assessment ID:</strong> {_e(data.assessment_id)}<br>
    <strong>Token:</strong> {_e(data.submission_link.split("token=")[-1] if "token=" in data.submission_link else "—")}
  </p>
</div>"""
        self._send(data.recipient_email, f"Your {data.topic} Assessment is Ready", body)

    def send_reminder_email(self, data: ReminderEmailData) -> None:
        topics_section = ""
        if data.key_topics:
            items = "".join(
                f'<li style="padding:3px 0">{_e(t)}</li>' for t in data.key_topics
            )
            topics_section = f"""
  <h3 style="margin-bottom:8px">Concepts to review</h3>
  <ul style="margin:0 0 20px 0;padding-left:20px;line-height:1.8;
             background:#f8f9fa;padding:12px 12px 12px 32px;border-radius:4px">
    {items}
  </ul>"""

        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:620px;margin:0 auto;padding:24px">
  <h2 style="color:#fd7e14;margin-top:0">⏰ Assessment Reminder: {_e(data.topic)}</h2>
  <p>Your assessment is scheduled for tomorrow. Use today to review the concepts below.</p>
  {topics_section}
  <table style="width:100%;border-collapse:collapse;margin:16px 0">
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600;width:42%">Assessment sent</td>
      <td style="padding:8px 12px"><strong>{_e(_fmt_dt(data.scheduled_at))}</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600">Expires</td>
      <td style="padding:8px 12px;color:#dc3545"><strong>{_e(_fmt_dt(data.expire_date))}</strong></td>
    </tr>
  </table>

  <p style="color:#6c757d;font-size:0.85rem;margin-top:20px">
    The assessment questions and submission link will arrive in a separate email
    at the time shown above.
  </p>
</div>"""
        self._send(data.recipient_email, f"Reminder: {data.topic} Assessment Tomorrow", body)

    def send_results_email(self, data: ResultsEmailData) -> None:
        passed_color = "#198754" if data.passed else "#dc3545"
        passed_label = "PASSED ✓" if data.passed else "FAILED ✗"
        score_bar_width = max(4, int(data.mastery_score))

        weak_section = ""
        if data.weak_areas:
            items = "".join(f"<li>{_e(w)}</li>" for w in data.weak_areas)
            weak_section = f"""
  <h4 style="color:#dc3545">Areas to Improve</h4>
  <ul style="line-height:1.8">{items}</ul>"""

        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:620px;margin:0 auto;padding:24px">
  <h2 style="margin-top:0">Results: {_e(data.topic)}</h2>
  <p style="font-size:1.1rem">Attempt #{_e(str(data.attempt_number))}</p>

  <div style="background:{passed_color};color:#fff;padding:16px 24px;border-radius:8px;
              text-align:center;margin:20px 0">
    <div style="font-size:2rem;font-weight:700">{_e(passed_label)}</div>
    <div style="font-size:1.5rem;margin-top:4px">{data.mastery_score:.1f}%</div>
  </div>

  <div style="background:#f8f9fa;border-radius:4px;height:12px;margin:16px 0;overflow:hidden">
    <div style="background:{passed_color};height:100%;width:{score_bar_width}%"></div>
  </div>

  <h4>Feedback</h4>
  <p style="line-height:1.7;background:#f8f9fa;padding:12px 16px;border-radius:4px">
    {_e(data.overall_feedback)}
  </p>
  {weak_section}
</div>"""
        verdict = "Passed" if data.passed else "Failed"
        self._send(
            data.recipient_email,
            f"[{verdict}] {data.topic} — {data.mastery_score:.1f}%",
            body,
        )
