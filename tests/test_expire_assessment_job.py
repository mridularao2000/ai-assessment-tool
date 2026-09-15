"""expire_assessment_job — the send/expire race condition fix.

Real incident this exists for: a retest whose scheduled_at/due_date were
both computed from a stale reference date (see
AssessmentService.create_retest's date.today() fix) ended up registered
already in the past. With misfire_grace_time=None, APScheduler fired both
the slow send_assessment_job (LLM generation + email) and the near-instant
expire_assessment_job at nearly the same instant instead of their normal
2+ real days apart. expire_assessment_job won the race, found status
still `scheduled` (send hadn't flipped it to `active` yet), no-op'd under
the old `status == active`-only check, and — being a one-shot
date-triggered job — was removed from the jobstore right there. No job
was ever left to expire that assessment again; it stayed `active` forever
with a due_date in the past.

Fully mocked — TestSessionLocal, zero real Anthropic/email calls.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from app.jobs.expire_assessment_job import expire_assessment_job
from app.models.assessment import Assessment, AssessmentStatus
from tests.conftest import TestSessionLocal, make_curriculum, seed_prompt_templates
from tests.test_send_assessment_job import _make_scheduled_assessment


class TestExpireAssessmentJob:
    def test_active_assessment_past_due_date_expires(self, db, monkeypatch):
        """Baseline: the ordinary, non-race case still works exactly as before."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(days=4)
        )
        assessment.status = AssessmentStatus.active
        db.commit()
        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)

        expire_assessment_job(assessment.id)

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.expired

    def test_scheduled_assessment_past_due_date_also_expires(self, db, monkeypatch):
        """Regression test: expire_assessment_job winning the race against a
        still-in-flight send_assessment_job (status still `scheduled`, not
        yet `active`) must still correctly mark the assessment expired,
        not silently no-op and vanish forever as a one-shot job.
        """
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(days=4)
        )
        assert assessment.status == AssessmentStatus.scheduled
        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)

        expire_assessment_job(assessment.id)

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.expired

    def test_completed_assessment_is_left_alone(self, db, monkeypatch):
        """A late-firing expire job must never downgrade a real outcome
        (already graded) back to expired."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(days=4)
        )
        assessment.status = AssessmentStatus.completed
        db.commit()
        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)

        expire_assessment_job(assessment.id)

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.completed

    def test_needs_manual_diagnosis_assessment_is_left_alone(self, db, monkeypatch):
        """A generation failure that needs a human's attention must not be
        silently reclassified as an ordinary missed deadline."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(days=4)
        )
        assessment.status = AssessmentStatus.needs_manual_diagnosis
        db.commit()
        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)

        expire_assessment_job(assessment.id)

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.needs_manual_diagnosis
