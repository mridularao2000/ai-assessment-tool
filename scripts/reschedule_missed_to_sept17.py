"""One-time admin reschedule: re-deliver 4 already-generated assessments
with a fresh due date of 2026-09-17, without regenerating any content.

Two of these (State Management — Redux, Normalized State; Frontend
Architecture & Patterns) are stuck at status=active with a due_date
already in the past — a direct consequence of the create_retest()
stale-date bug (fixed in a prior commit) racing expire_assessment_job
(also fixed in a prior commit). Being stuck at `active`, not `expired`,
they are NOT eligible for the existing POST .../entries/bulk-reschedule
endpoint (which requires status == expired), and that endpoint also
can't hit an exact date anyway — it recomputes scheduled_at via a random
1-3 day offset from a shifted completion_date. This script sets exact
dates directly instead.

The other two (Performance Optimization I; Browser Internals — Rendering
Path, Reflow/Repaint) are genuinely `expired` and technically could go
through bulk-reschedule, but are included here too so all four land on
the identical, exact due date and go through one consistent path.

Mechanism (same as CurriculumUploadService.bulk_reschedule_entries):
content already exists on each row, so re-triggering send_assessment_job
at the new scheduled_at will skip generation entirely (see
send_assessment_job.py's `if assessment_text is None and part1_text is
None: generate()` guard) and go straight to re-sending the existing
content by email — zero LLM cost.

scheduled_at is set to ~1 minute from whenever this script actually runs
(so the resend fires almost immediately), NOT due_date - 2 days like the
normal pipeline — there's only ~1 day of runway to due_date, and the
standard 2-day gap would itself land scheduled_at in the past. due_date
is fixed at 2026-09-17 09:00 UTC, matching this app's 9am-UTC convention.
reminder_at follows the same reminder-anchored-to-due_date rule as any
other curriculum-upload entry (settings.entry_reminder_hours_before_deadline
hours before due_date) — it may itself land in the past given the short
runway here, in which case it simply fires alongside the resend rather
than days ahead; this is expected and harmless (worst case: a slightly
redundant reminder email momentarily before/alongside the resend).

Run once, from the repo root:
    python3 scripts/reschedule_missed_to_sept17.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import get_scheduler_adapter
from app.interfaces.scheduler import AssessmentJobIds
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum
from app.services.scheduler_service import SchedulerService

TARGETS = [
    # (curriculum_id, assessment_id, label) — verified via a prior read-only check
    ("1ef92d23-4148-4c89-b776-c9c060111d69", "572c3e30-48ef-4589-9e15-00a23a6198d3", "State Management — Redux, Normalized State"),
    ("c0bfc1cd-71a8-4afe-95f3-829b495713c7", "c4527778-cbc8-408d-8d0b-1c6d38ba7893", "Frontend Architecture & Patterns"),
    ("82ca295e-ba69-4363-812a-77616815c803", "20006383-a43d-46ba-ada5-2e2110254a3f", "Performance Optimization I — Tree-Shaking, Event Bubbling, Synthetic Events"),
    ("9a4a87ae-50cc-46e9-936e-0ec11cbd1a5f", "a72c2a33-34a7-4985-a18b-c10ca359fc40", "Browser Internals — Rendering Path, Reflow/Repaint"),
]

NEW_DUE_DATE = datetime(2026, 9, 17, 9, 0, 0)
NEW_COMPLETION_DATE = NEW_DUE_DATE.date()

db = SessionLocal()
try:
    settings = get_settings()
    adapter = get_scheduler_adapter()
    adapter.start()  # idempotent; required so this stand-alone process's adapter is attached to the live jobstore
    scheduler_service = SchedulerService(db, adapter)
    scheduled_at = datetime.utcnow() + timedelta(minutes=1)
    reminder_at = NEW_DUE_DATE - timedelta(hours=settings.entry_reminder_hours_before_deadline)

    for curriculum_id, assessment_id, label in TARGETS:
        assessment = db.get(Assessment, assessment_id)
        assert assessment is not None, f"{label}: assessment {assessment_id} not found"
        assert assessment.curriculum_id == curriculum_id, f"{label}: curriculum_id mismatch"
        assert assessment.status in (AssessmentStatus.active, AssessmentStatus.expired), (
            f"{label}: expected status active/expired, got {assessment.status.value!r}"
        )
        assert assessment.assessment_text is not None or assessment.part1_text is not None, (
            f"{label}: no existing content to re-deliver — refusing to touch it"
        )

        curriculum = db.get(Curriculum, curriculum_id)
        assert curriculum is not None, f"{label}: curriculum not found"

        curriculum.target_completion_date = NEW_COMPLETION_DATE

        existing_job_ids = (
            AssessmentJobIds(**assessment.scheduled_job_ids) if assessment.scheduled_job_ids else None
        )
        if existing_job_ids is not None:
            scheduler_service.reschedule_assessment(
                assessment_id=assessment.id,
                new_scheduled_at=scheduled_at,
                new_reminder_at=reminder_at,
                new_due_date=NEW_DUE_DATE,
                existing_job_ids=existing_job_ids,
            )
        else:
            assessment.scheduled_at = scheduled_at
            assessment.reminder_at = reminder_at
            assessment.due_date = NEW_DUE_DATE
            scheduler_service.schedule_assessment_jobs(
                assessment_id=assessment.id,
                scheduled_at=scheduled_at,
                reminder_at=reminder_at,
                due_date=NEW_DUE_DATE,
            )

        assessment.status = AssessmentStatus.scheduled
        assessment.send_job_claimed_at = None
        db.commit()
        print(f"{label}: rescheduled — scheduled_at={scheduled_at}, due_date={NEW_DUE_DATE} (resend imminent)")
finally:
    db.close()
