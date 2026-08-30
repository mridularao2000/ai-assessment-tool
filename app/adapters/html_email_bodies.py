"""Shared HTML email body-building logic for every EmailInterface adapter
that renders transactional email as inline-styled HTML.

Extracted out of ResendEmailAdapter so the Gmail SMTP migration could add a
new transport underneath these six methods without touching (or
duplicating, and risking divergence in) a single line of their HTML —
every adapter that mixes this in sends byte-identical content; only the
concrete `_send(to, subject, body_html)` implementation differs.
"""
from __future__ import annotations

import html
from collections import defaultdict
from datetime import date, datetime

from app.interfaces.email import (
    AssessmentEmailData,
    MidtermHoldReminderEmailData,
    ReminderEmailData,
    ResultsEmailData,
    SyllabusEmailData,
    TranscriptEmailData,
)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%A, %d %B %Y at %H:%M UTC")


def _fmt_date(d: date) -> str:
    return d.strftime("%A, %d %B %Y")


def _e(text: str) -> str:
    return html.escape(str(text))


class HtmlEmailBodyMixin:
    """Implements the six EmailInterface methods by rendering inline-styled
    HTML and handing it to `self._send(to, subject, body_html)`, which the
    concrete adapter (Resend, Gmail SMTP, ...) must implement."""

    def _send(self, to: list[str], subject: str, body_html: str) -> None:
        raise NotImplementedError

    # ── EmailInterface ────────────────────────────────────────────────────────

    def send_assessment_email(self, data: AssessmentEmailData) -> None:
        duration = f"{data.duration_minutes} minutes" if data.duration_minutes else "unspecified"
        part1_heading = "Part 1 — Assignment" if data.part2_text else "Assessment"
        part2_section = ""
        if data.part2_text:
            part2_section = f"""
  <div style="background:#f1f3f5;border-left:4px solid #6610f2;padding:16px 20px;
              border-radius:4px;margin:20px 0">
    <h3 style="margin-top:0;color:#6610f2">Part 2 — Project Submission</h3>
    <div style="white-space:pre-wrap;line-height:1.6">{_e(data.part2_text)}</div>
  </div>"""
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
    <h3 style="margin-top:0;color:#0d6efd">{_e(part1_heading)}</h3>
    <div style="white-space:pre-wrap;line-height:1.6">{_e(data.assessment_text)}</div>
  </div>
  {part2_section}

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
        self._send(data.recipient_emails, f"Your {data.topic} Assessment is Ready", body)

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

        if data.is_pre_deadline:
            heading = f"⏰ Deadline Approaching: {_e(data.topic)}"
            intro = "Your submission deadline is approaching. Don't forget to submit your answer."
            rows = f"""
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600;width:42%">Sent</td>
      <td style="padding:8px 12px">{_e(_fmt_dt(data.scheduled_at))}</td>
    </tr>
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600">Deadline</td>
      <td style="padding:8px 12px;color:#dc3545"><strong>{_e(_fmt_dt(data.expire_date))}</strong></td>
    </tr>"""
        else:
            heading = f"⏰ Assessment Reminder: {_e(data.topic)}"
            intro = "Your assessment is scheduled for tomorrow. Use today to review the concepts below."
            rows = f"""
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600;width:42%">Assessment sent</td>
      <td style="padding:8px 12px"><strong>{_e(_fmt_dt(data.scheduled_at))}</strong></td>
    </tr>
    <tr>
      <td style="padding:8px 12px;background:#fff3cd;font-weight:600">Expires</td>
      <td style="padding:8px 12px;color:#dc3545"><strong>{_e(_fmt_dt(data.expire_date))}</strong></td>
    </tr>"""

        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:620px;margin:0 auto;padding:24px">
  <h2 style="color:#fd7e14;margin-top:0">{heading}</h2>
  <p>{intro}</p>
  {topics_section}
  <table style="width:100%;border-collapse:collapse;margin:16px 0">
    {rows}
  </table>

  <p style="color:#6c757d;font-size:0.85rem;margin-top:20px">
    {"Use the submission link from your assessment email to submit before the deadline above." if data.is_pre_deadline else "The assessment questions and submission link will arrive in a separate email at the time shown above."}
  </p>
</div>"""
        subject = (
            f"Reminder: {data.topic} deadline approaching"
            if data.is_pre_deadline
            else f"Reminder: {data.topic} Assessment Tomorrow"
        )
        self._send(data.recipient_emails, subject, body)

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
            data.recipient_emails,
            f"[{verdict}] {data.topic} — {data.mastery_score:.1f}%",
            body,
        )

    def send_syllabus_email(self, data: SyllabusEmailData) -> None:
        """Renders a chronological, month-sectioned reading of the upload —
        every Assessment and Midterm sorted by completion_date ascending
        and bucketed under the month it falls in — rather than a chapter-
        by-chapter data dump. Chapter grouping (SyllabusChapterSection)
        stays the underlying data shape only because the transcript email's
        frozen course_material_snapshot (serialize_syllabus_content) still
        needs it; this method simply reads across chapters/midterms and
        re-sorts for display, without touching that shared data.
        """
        due_badge = (
            "background:#fff3cd;color:#7a5b00;font-size:0.78rem;font-weight:700;"
            "padding:3px 10px;border-radius:4px;white-space:nowrap"
        )
        window_badge = (
            "background:#e6f9f0;color:#0f7b4f;font-size:0.78rem;font-weight:700;"
            "padding:3px 10px;border-radius:4px;white-space:nowrap"
        )

        no_standalone = [
            (chapter.chapter_label, chapter.no_standalone_note)
            for chapter in data.chapters
            if not chapter.assessments
        ]

        dated_cards: list[tuple[date, str]] = []

        for chapter in data.chapters:
            for a in chapter.assessments:
                resources_html = "".join(f"<li>{_e(r)}</li>" for r in a.resources)
                card = f"""
    <div style="margin:12px 0;padding:12px 16px;background:#f8f9fa;border-left:4px solid #0d6efd;
                border-radius:4px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
        <strong>{_e(a.topic)}</strong>
        <span style="background:#e7f1ff;color:#0d6efd;font-size:0.72rem;font-weight:700;
                     padding:2px 9px;border-radius:12px;white-space:nowrap">{_e(chapter.chapter_label)}</span>
      </div>
      <div style="margin:9px 0 6px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="{due_badge}">Due {_e(_fmt_date(a.completion_date))}</span>
        <span style="{window_badge}">Exam window: {_e(_fmt_date(a.window_start))} – {_e(_fmt_date(a.window_end))}</span>
      </div>
      <div style="color:#868e96;font-size:0.72rem;font-family:ui-monospace,'Cascadia Code',monospace;margin:4px 0 8px">
        ID: {_e(a.id)}
      </div>
      <ul style="margin:6px 0 0 0;padding-left:20px;line-height:1.6">{resources_html}</ul>
    </div>"""
                dated_cards.append((a.completion_date, card))

        for m in data.midterms:
            known_now_html = "".join(f"<li>{_e(r)}</li>" for r in m.known_now)

            def _pending_row(label: str, filled: bool) -> str:
                status = (
                    '<strong style="color:#198754">provided</strong>'
                    if filled
                    else '<strong style="color:#dc3545">pending</strong>'
                )
                return f"<li>{_e(label)} — {status}</li>"

            pending_html = "".join(_pending_row(label, filled) for label, filled in m.pending_status)
            hold_banner = (
                """<div style="background:#f8d7da;color:#58151c;padding:8px 12px;
                    border-radius:4px;margin-bottom:8px">Held — awaiting the resources above.</div>"""
                if m.resources_hold else ""
            )
            special_case_html = (
                f'<p style="color:#6c757d;font-size:0.85rem;font-style:italic">{_e(m.special_case)}</p>'
                if m.special_case else ""
            )
            probe_html = (
                f"<p><strong>Probe focus:</strong> {_e(m.probe_focus)}</p>"
                if m.probe_focus else ""
            )
            card = f"""
    <div style="margin:12px 0;padding:14px 18px;background:#f8f9fa;border-left:4px solid #6610f2;
                border-radius:4px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px">
        <strong>{_e(m.topic)}</strong>
        <span style="background:#f1e6ff;color:#6610f2;font-size:0.72rem;font-weight:700;
                     padding:2px 9px;border-radius:12px;white-space:nowrap">{_e(m.chapter_label)} · Midterm</span>
      </div>
      <div style="margin:9px 0 6px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="{due_badge}">Due {_e(_fmt_date(m.completion_date))}</span>
      </div>
      <div style="color:#868e96;font-size:0.72rem;font-family:ui-monospace,'Cascadia Code',monospace;margin:4px 0 8px">
        ID: {_e(m.id)}
      </div>
      {hold_banner}
      <strong>Known now:</strong>
      <ul style="margin:4px 0 10px 0;padding-left:20px">{known_now_html}</ul>
      <strong>Project resources:</strong>
      <ul style="margin:4px 0 10px 0;padding-left:20px">{pending_html}</ul>
      {probe_html}
      {special_case_html}
    </div>"""
            dated_cards.append((m.completion_date, card))

        dated_cards.sort(key=lambda pair: pair[0])

        months: dict[tuple[int, int], list[str]] = defaultdict(list)
        month_order: list[tuple[int, int]] = []
        for d, card in dated_cards:
            key = (d.year, d.month)
            if key not in months:
                month_order.append(key)
            months[key].append(card)

        timeline_html = ""
        for key in month_order:
            year, month = key
            label = date(year, month, 1).strftime("%B %Y")
            timeline_html += f"""
  <h3 style="color:#495057;margin:26px 0 10px;padding-bottom:6px;border-bottom:2px solid #dee2e6;
             text-transform:uppercase;letter-spacing:0.04em;font-size:0.9rem">{_e(label)}</h3>
  {''.join(months[key])}"""

        no_standalone_html = ""
        if no_standalone:
            items = "".join(
                f"<li><strong>{_e(label)}</strong>{f' — {_e(note)}' if note else ''}</li>"
                for label, note in no_standalone
            )
            no_standalone_html = f"""
  <div style="margin:16px 0;padding:10px 14px;background:#fff3cd;border-radius:4px">
    <strong>No standalone assessment:</strong>
    <ul style="margin:6px 0 0 0;padding-left:20px;line-height:1.6">{items}</ul>
  </div>"""

        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:680px;margin:0 auto;padding:24px">
  <h2 style="margin-top:0">Your Curriculum: {_e(data.source_filename)}</h2>
  <p style="color:#6c757d">Uploaded curriculum, arranged chronologically by due date.</p>
  <p style="color:#868e96;font-size:0.75rem;font-family:ui-monospace,'Cascadia Code',monospace;margin:0 0 16px">
    Upload ID: {_e(data.upload_id)}
  </p>
  {no_standalone_html}
  {timeline_html}
</div>"""
        self._send(data.recipient_emails, f"Your Curriculum: {data.source_filename}", body)

    def send_transcript_email(self, data: TranscriptEmailData) -> None:
        """Black-and-white, Times New Roman / Space Mono transcript design
        (approved mockup: "React Engineering Transcript"). Inline styles
        throughout, matching every other email in this adapter — email
        clients don't reliably honor <style> blocks.
        """
        serif = "'Times New Roman', Times, Georgia, serif"
        mono = ("'Space Mono', ui-monospace, 'SF Mono', 'Cascadia Mono', "
                "'Segoe UI Mono', Consolas, 'Courier New', monospace")
        ink, rule, rule_soft = "#000000", "#000000", "#B7B7B7"

        def _fmt_iso_date(iso: str) -> str:
            return _fmt_date(date.fromisoformat(iso))

        def _pts(v) -> str:
            return f"{v:.2f}" if v is not None else "—"

        group_rows = ""
        for group in data.entry_groups:
            group_rows += (
                f'<tr><td colspan="6" style="padding:10px 6px 4px;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.02em;'
                f'border-bottom:1px solid {rule_soft};font-family:{mono}">'
                f'{_e(group.chapter_label)}</td></tr>'
            )
            for row in group.rows:
                title = _e(row.topic)
                if row.retake_note:
                    title += f" ({_e(row.retake_note)})"
                ch = str(row.chapter_number) if row.chapter_number is not None else "—"
                group_rows += f"""
    <tr>
      <td style="padding:5px 6px;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{_e(row.row_id)}</td>
      <td style="padding:5px 6px;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{title}</td>
      <td style="padding:5px 6px;text-align:right;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{ch}</td>
      <td style="padding:5px 6px;text-align:right;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{row.max_marks:.2f}</td>
      <td style="padding:5px 6px;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{_e(row.status_label)}</td>
      <td style="padding:5px 6px;text-align:right;border-bottom:1px solid {rule_soft};font-family:{mono};font-size:12px">{_pts(row.points)}</td>
    </tr>"""

        omitted = data.total_entry_count - data.resolved_count
        omitted_note = (
            f'<p style="font-size:12px;font-style:italic;color:#545454;margin:8px 0 0">'
            f'{omitted} entr{"y is" if omitted == 1 else "ies are"} not yet resolved — '
            f'they will appear once graded or their late-submission window closes.</p>'
            if omitted > 0 else ""
        )

        table_html = f"""
  <p style="font-family:{serif};font-size:13px;font-weight:bold;text-transform:uppercase;
            letter-spacing:0.03em;margin:0 0 8px">Record of Entries — Resolved</p>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;table-layout:fixed;font-family:{mono};font-size:12px">
    <colgroup>
      <col style="width:9%"><col style="width:38%"><col style="width:7%">
      <col style="width:11%"><col style="width:25%"><col style="width:10%">
    </colgroup>
    <thead>
      <tr>
        <th style="text-align:left;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">No.</th>
        <th style="text-align:left;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">Course Title</th>
        <th style="text-align:right;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">Ch</th>
        <th style="text-align:right;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">Cred</th>
        <th style="text-align:left;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">Status</th>
        <th style="text-align:right;padding:5px 6px;border-bottom:2px solid {rule};font-weight:700">Pts</th>
      </tr>
    </thead>
    <tbody>{group_rows}</tbody>
    <tfoot>
      <tr>
        <td colspan="2" style="border-top:2px solid {rule};font-weight:700;padding-top:8px">
          CUMULATIVE — {data.resolved_count} OF {data.total_entry_count} RESOLVED, {data.graded_count} GRADED</td>
        <td style="border-top:2px solid {rule};padding-top:8px"></td>
        <td style="text-align:right;border-top:2px solid {rule};font-weight:700;padding-top:8px">{data.total_credits:.2f}</td>
        <td style="border-top:2px solid {rule};font-weight:700;padding-top:8px">GPA {data.gpa:.1f}</td>
        <td style="text-align:right;border-top:2px solid {rule};font-weight:700;padding-top:8px">{data.total_points:.2f}</td>
      </tr>
    </tfoot>
  </table>
  </div>
  {omitted_note}"""

        material = data.course_material or {"chapters": [], "midterms": []}
        chapter_blocks = ""
        for ch in material.get("chapters", []):
            entries_html = ""
            for a in ch.get("assessments", []):
                resources_html = "".join(f"<li>{_e(r)}</li>" for r in a.get("resources", []))
                id_html = (
                    f'<span style="display:block;font-size:11px;color:#8a8a8a;font-family:'
                    f'ui-monospace,\'Cascadia Code\',monospace">ID: {_e(a["id"])}</span>'
                    if a.get("id") else ""
                )
                entries_html += f"""
    <span style="font-style:italic;font-size:12px">{_e(a['topic'])}</span>
    {id_html}
    <ul style="margin:2px 0 8px;padding-left:20px;font-size:12px;line-height:1.55">{resources_html}</ul>"""
            if not entries_html and ch.get("no_standalone_note"):
                entries_html = f'<p style="font-size:12px;font-style:italic;color:#545454">{_e(ch["no_standalone_note"])}</p>'
            chapter_blocks += f"""
  <div style="margin-bottom:14px">
    <h3 style="font-family:{serif};font-size:13px;font-weight:bold;margin:0 0 3px">{_e(ch['chapter_label'])}</h3>
    {entries_html}
  </div>"""

        midterms = material.get("midterms", [])
        if midterms:
            midterm_entries = ""
            for m in midterms:
                pending = [label for label, filled in m.get("pending_status", []) if not filled]
                span = f"Due {_e(_fmt_iso_date(m['completion_date']))}"
                if m.get("known_now"):
                    span += f" — known now: {_e(', '.join(m['known_now']))}"
                if pending:
                    span += f" — pending: {_e(', '.join(pending))}"
                id_html = (
                    f'<span style="display:block;font-size:11px;color:#8a8a8a;font-family:'
                    f'ui-monospace,\'Cascadia Code\',monospace">ID: {_e(m["id"])}</span>'
                    if m.get("id") else ""
                )
                midterm_entries += f"""
    <div style="margin:0 0 8px;font-size:12px">
      <span style="display:block;font-style:italic">{_e(m['topic'])}</span>
      {id_html}
      <span style="font-size:11px;color:#545454">{span}</span>
    </div>"""
            chapter_blocks += f"""
  <div style="margin-bottom:14px">
    <h3 style="font-family:{serif};font-size:13px;font-weight:bold;margin:0 0 3px">Midterms</h3>
    {midterm_entries}
  </div>"""

        captured_note = (
            f'<p style="font-size:11px;font-style:italic;color:#545454;margin:0 0 14px">'
            f'Captured once at upload ({_e(_fmt_dt(data.course_material_captured_at))}) — '
            f'reused unchanged in every transcript email, not regenerated per send.</p>'
            if data.course_material_captured_at else ""
        )

        material_html = f"""
  <hr style="border:none;border-top:1px solid {rule};margin:22px 0 12px">
  <p style="font-family:{serif};font-size:13px;font-weight:bold;text-transform:uppercase;
            letter-spacing:0.03em;margin:0 0 8px">Course Material</p>
  {captured_note}
  {chapter_blocks}"""

        body = f"""
<div style="max-width:760px;margin:0 auto;padding:44px 20px 60px;background:#ffffff;color:{ink};
            font-family:{serif};font-size:13px">
  <div style="text-align:center;margin-bottom:4px">
    <h1 style="font-size:22px;font-weight:bold;letter-spacing:0.01em;margin:0 0 3px">{_e(data.source_filename)}</h1>
    <p style="font-size:13px;font-style:italic;color:#2B2B2B;margin:0 0 12px">
      {data.total_entry_count} entries — {data.resolved_count} resolved</p>
  </div>
  <hr style="border:none;border-top:3px double {rule};margin:0 0 18px">
  {table_html}
  {material_html}
  <p style="text-align:center;font-size:11px;font-style:italic;color:#545454;margin-top:24px">
    Regenerated after every grading event.</p>
</div>"""

        self._send(data.recipient_emails, f"Transcript: {data.source_filename}", body)

    def send_midterm_hold_reminder_email(self, data: MidtermHoldReminderEmailData) -> None:
        missing_html = "".join(f"<li>{_e(label)}</li>" for label in data.missing_labels)
        body = f"""
<div style="font-family:sans-serif;color:#212529;max-width:620px;margin:0 auto;padding:24px">
  <h2 style="color:#fd7e14;margin-top:0">⏸ Midterm Held: {_e(data.topic)}</h2>
  <p>
    This Midterm's completion date ({_e(_fmt_date(data.completion_date))}) has arrived, but
    the project resources below are still missing — its exam window won't open until
    they're provided.
  </p>
  <ul style="line-height:1.8;background:#fff3cd;padding:12px 12px 12px 32px;border-radius:4px">
    {missing_html}
  </ul>
  <p style="color:#6c757d;font-size:0.85rem">
    This reminder repeats periodically until every resource above is filled in.
  </p>
</div>"""
        self._send(data.recipient_emails, f"Midterm Held: {data.topic} — resources needed", body)
