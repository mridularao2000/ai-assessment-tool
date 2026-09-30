"""One-time admin fix: set due_date = 2026-10-04 09:00 UTC for the same 7
assessments set_due_date_sept30.py targeted, re-delivering existing
content again (no regeneration). Third date change for these — same
guard discipline as before: each target is checked and updated
independently in its own try/except, and 007985f3 (status=completed as
of the last run) is expected to be SKIPPED again unless its state
changed since.

Run once, from the repo root:
    python3 scripts/set_due_date_oct4.py
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
from app.services.scheduler_service import SchedulerService

ASSESSMENT_IDS = [
    "9563a766-03d3-447a-8230-2d09e19af894",
    "007985f3-4721-4434-b4a0-b75c0cb76d15",
    "ac5f5e2d-2778-47fc-bd83-ea0b350fa551",
    "572c3e30-48ef-4589-9e15-00a23a6198d3",
    "a72c2a33-34a7-4985-a18b-c10ca359fc40",
    "c4527778-cbc8-408d-8d0b-1c6d38ba7893",
    "20006383-a43d-46ba-ada5-2e2110254a3f",
]

SAFE_STATUSES = (AssessmentStatus.scheduled, AssessmentStatus.active, AssessmentStatus.expired)

NEW_DUE_DATE = datetime(2026, 10, 4, 9, 0, 0)

db = SessionLocal()
try:
    settings = get_settings()
    adapter = get_scheduler_adapter()
    adapter.start()  # idempotent; required so this stand-alone process's adapter is attached to the live jobstore
    scheduler_service = SchedulerService(db, adapter)

    new_reminder_at = NEW_DUE_DATE - timedelta(hours=settings.entry_reminder_hours_before_deadline)

    for assessment_id in ASSESSMENT_IDS:
        try:
            assessment = db.get(Assessment, assessment_id)
            if assessment is None:
                print(f"{assessment_id}: SKIPPED — not found")
                continue
            if assessment.status not in SAFE_STATUSES:
                print(
                    f"{assessment_id}: SKIPPED — status={assessment.status.value!r} "
                    f"not in {[s.value for s in SAFE_STATUSES]}; refusing to touch"
                )
                continue
            if assessment.assessment_text is None and assessment.part1_text is None:
                print(f"{assessment_id}: SKIPPED — no existing content to re-deliver")
                continue

            real_send_at = datetime.utcnow() + timedelta(minutes=1)

            if assessment.scheduled_job_ids:
                existing_job_ids = AssessmentJobIds(**assessment.scheduled_job_ids)
                scheduler_service.reschedule_assessment(
                    assessment_id=assessment.id,
                    new_scheduled_at=real_send_at,
                    new_reminder_at=new_reminder_at,
                    new_due_date=NEW_DUE_DATE,
                    existing_job_ids=existing_job_ids,
                )
            else:
                assessment.scheduled_at = real_send_at
                assessment.reminder_at = new_reminder_at
                assessment.due_date = NEW_DUE_DATE
                scheduler_service.schedule_assessment_jobs(
                    assessment_id=assessment.id,
                    scheduled_at=real_send_at,
                    reminder_at=new_reminder_at,
                    due_date=NEW_DUE_DATE,
                )

            assessment = db.get(Assessment, assessment_id)
            assessment.send_job_claimed_at = None
            db.commit()
            print(
                f"{assessment_id}: OK — resend at {real_send_at}, "
                f"due_date={NEW_DUE_DATE}, reminder_at={new_reminder_at}"
            )
        except Exception as exc:
            db.rollback()
            print(f"{assessment_id}: FAILED — {exc!r}")
finally:
    db.close()
