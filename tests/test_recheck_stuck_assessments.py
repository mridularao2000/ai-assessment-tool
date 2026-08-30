"""recheck_stuck_assessments_job — the self-healing sweep for
send_assessment_job's one-shot-job failure mode.

Real incident this exists for: an assessment's send_assessment_job is a
one-shot APScheduler `date` trigger. If its single execution fails (or
never actually runs), the row sits forever in `scheduled` status past its
scheduled_at with no exam ever delivered, no automatic retry, and
"Not Yet Due" shown in the UI (display_status never compares scheduled_at
against now). This job sweeps for exactly that state and retries delivery,
repeatedly (every 15 minutes) rather than once — a single retry only
recovers a transient failure; a persistent cause (broken credentials, a
bad prompt template) needs the system to keep trying so it self-heals the
moment the underlying cause is fixed, with no one needing to notice.

Fully mocked — FakeLLM + the autouse Noop `_email` patch on
send_assessment_job — zero real Anthropic/email calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from app.jobs.recheck_stuck_assessments_job import recheck_stuck_assessments_job
from app.models.assessment import Assessment, AssessmentStatus
from app.utils.token_auth import generate_submission_token
from tests.conftest import FakeLLM, TestSessionLocal, make_curriculum, seed_prompt_templates


def _make_scheduled_assessment(db, curriculum, *, scheduled_at: datetime) -> Assessment:
    assessment_id = str(uuid.uuid4())
    assessment = Assessment(
        id=assessment_id,
        curriculum_id=curriculum.id,
        attempt_number=1,
        assessment_text=None,
        rubric=None,
        duration_minutes=None,
        scheduled_at=scheduled_at,
        reminder_at=scheduled_at - timedelta(hours=24),
        due_date=scheduled_at + timedelta(days=2),
        status=AssessmentStatus.scheduled,
        submission_token=generate_submission_token(assessment_id),
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment


def _patch_job_infra(monkeypatch):
    monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
    monkeypatch.setattr("app.jobs.recheck_stuck_assessments_job.SessionLocal", TestSessionLocal)


class TestRecheckStuckAssessments:
    def test_sweeps_and_resends_a_stuck_assessment(self, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        well_overdue = datetime.utcnow() - timedelta(hours=2)
        assessment = _make_scheduled_assessment(db, curriculum, scheduled_at=well_overdue)

        recheck_stuck_assessments_job()

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.active
        assert refreshed.assessment_text is not None

    def test_leaves_a_recently_overdue_assessment_within_grace_period_alone(self, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        just_overdue = datetime.utcnow() - timedelta(minutes=5)
        assessment = _make_scheduled_assessment(db, curriculum, scheduled_at=just_overdue)

        recheck_stuck_assessments_job()

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.scheduled
        assert refreshed.assessment_text is None

    def test_leaves_a_future_scheduled_assessment_untouched(self, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        future = datetime.utcnow() + timedelta(days=1)
        assessment = _make_scheduled_assessment(db, curriculum, scheduled_at=future)

        recheck_stuck_assessments_job()

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.scheduled

    def test_sweeps_multiple_stuck_assessments_in_one_pass(self, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        overdue = datetime.utcnow() - timedelta(hours=2)
        assessments = [
            _make_scheduled_assessment(
                db, make_curriculum(db, entry_type="assessment", topic=f"Topic {i}"),
                scheduled_at=overdue,
            )
            for i in range(3)
        ]

        recheck_stuck_assessments_job()

        db.expire_all()
        for a in assessments:
            assert db.get(Assessment, a.id).status == AssessmentStatus.active

    def test_recovers_after_a_persistent_failure_is_fixed(self, db, monkeypatch):
        """The property a one-shot per-assessment retry can't offer: if the
        underlying cause is persistent (not transient), repeated sweeps keep
        failing harmlessly until the cause is fixed, then the very next
        tick recovers automatically — no manual intervention needed."""
        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_stuck_assessments_job.SessionLocal", TestSessionLocal)
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(hours=2)
        )

        class _BrokenLLM(FakeLLM):
            def generate_assessment(self, req):
                raise RuntimeError("persistent misconfiguration")

        monkeypatch.setattr("app.jobs.send_assessment_job._llm", _BrokenLLM())
        recheck_stuck_assessments_job()  # first tick: still fails, stays scheduled

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.scheduled

        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())  # "fixed"
        recheck_stuck_assessments_job()  # next tick: recovers with no manual step

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.active

    def test_stops_auto_retrying_past_the_give_up_window(self, db, monkeypatch):
        """The bound this exists for: once a row has been stuck long enough
        that a persistent (not transient) cause is the likely explanation,
        the sweep must stop spending real API credit on it automatically —
        no LLM call at all past that point, not even a failing one."""
        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.recheck_stuck_assessments_job.SessionLocal", TestSessionLocal)
        curriculum = make_curriculum(db, entry_type="assessment")
        # Default give-up window is 3 hours — this row has been stuck for 4.
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(hours=4)
        )

        def _poison(*args, **kwargs):
            raise AssertionError("must not call the LLM past the give-up window")

        monkeypatch.setattr("app.jobs.send_assessment_job.send_assessment_job", _poison)

        recheck_stuck_assessments_job()  # must not raise, must not touch send_assessment_job

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.scheduled
