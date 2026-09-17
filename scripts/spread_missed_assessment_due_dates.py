"""One-time admin fix, replaces push_due_date_to_sept18.py (never run).

That script pinned all 4 assessments to the identical fixed due_date of
2026-09-18 09:00 — reproducing, for a second time, the exact "everything
due the same day" problem this whole correction chain exists to fix.

This version uses the app's own due-date rule verbatim, but keeps the
resend itself immediate — two separate concerns that must not be
conflated:

  - The RESEND (email delivery) should go out now, today, with the
    already-generated content — same as both prior scripts. Delaying
    the send itself by the +1..+3 window would leave these 4 students
    with no email at all for another 1-3 days, which is a new problem,
    not a fix.
  - The DUE DATE should follow the app's real rule
    (calculate_scheduled_at + build_assessment_dates' 2-day window,
    both from app/services/assessment_service.py) applied against
    *today* as the effective completion date, each target getting its
    own independent random draw — so due dates land across
    today+3..today+5 instead of all pinned to one fixed day.

Content already exists on each row, so the immediate resend skips
generation entirely and just re-sends the existing exam with the
corrected due_date rendered into the email body.

send_job_claimed_at is explicitly cleared — see push_due_date_to_sept18.py's
docstring (same reasoning): it's still set from the first (same-day) send,
and without resetting it the imminent resend would silently no-op.

Run once, from the repo root:
    python3 scripts/spread_missed_assessment_due_dates.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.dependencies import get_scheduler_adapter
from app.interfaces.scheduler import AssessmentJobIds
from app.models.assessment import Assessment, AssessmentStatus
from app.services.assessment_service import calculate_scheduled_at
from app.services.curriculum_upload_service import _build_entry_dates
from app.services.scheduler_service import SchedulerService

TARGETS = [
    # (curriculum_id, assessment_id, label) — same 4 as reschedule_missed_to_sept17.py
    ("1ef92d23-4148-4c89-b776-c9c060111d69", "572c3e30-48ef-4589-9e15-00a23a6198d3", "State Management — Redux, Normalized State"),
    ("c0bfc1cd-71a8-4afe-95f3-829b495713c7", "c4527778-cbc8-408d-8d0b-1c6d38ba7893", "Frontend Architecture & Patterns"),
    ("82ca295e-ba69-4363-812a-77616815c803", "20006383-a43d-46ba-ada5-2e2110254a3f", "Performance Optimization I — Tree-Shaking, Event Bubbling, Synthetic Events"),
    ("9a4a87ae-50cc-46e9-936e-0ec11cbd1a5f", "a72c2a33-34a7-4985-a18b-c10ca359fc40", "Browser Internals — Rendering Path, Reflow/Repaint"),
]

db = SessionLocal()
try:
    adapter = get_scheduler_adapter()
    adapter.start()  # idempotent; required so this stand-alone process's adapter is attached to the live jobstore
    scheduler_service = SchedulerService(db, adapter)

    today = datetime.utcnow().date()

    for curriculum_id, assessment_id, label in TARGETS:
        assessment = db.get(Assessment, assessment_id)
        assert assessment is not None, f"{label}: assessment {assessment_id} not found"
        assert assessment.curriculum_id == curriculum_id, f"{label}: curriculum_id mismatch"
        assert assessment.status == AssessmentStatus.active, (
            f"{label}: expected status active (from the Sept 17 resend), got {assessment.status.value!r}"
        )
        assert assessment.assessment_text is not None or assessment.part1_text is not None, (
            f"{label}: no existing content to re-deliver — refusing to touch it"
        )
        assert assessment.scheduled_job_ids, f"{label}: no scheduled_job_ids on record — refusing to touch it"

        # Nominal date used only to derive due_date/reminder_at via the app's
        # real rule — NOT the actual send time, so the resend isn't delayed.
        nominal_scheduled_at = calculate_scheduled_at(today)
        new_reminder_at, new_due_date = _build_entry_dates(nominal_scheduled_at)
        real_send_at = datetime.utcnow() + timedelta(minutes=1)

        existing_job_ids = AssessmentJobIds(**assessment.scheduled_job_ids)
        scheduler_service.reschedule_assessment(
            assessment_id=assessment.id,
            new_scheduled_at=real_send_at,
            new_reminder_at=new_reminder_at,
            new_due_date=new_due_date,
            existing_job_ids=existing_job_ids,
        )

        assessment = db.get(Assessment, assessment_id)
        assessment.send_job_claimed_at = None
        db.commit()
        print(
            f"{label}: resend at {real_send_at}, due_date={new_due_date}, "
            f"reminder_at={new_reminder_at}"
        )
finally:
    db.close()
