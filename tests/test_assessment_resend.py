"""POST /api/v1/assessments/{assessment_id}/resend — manual retry for an
assessment stuck in `scheduled` status past its scheduled_at.

Real production incident this covers: send_assessment_job is a one-shot
APScheduler `date`-trigger job. If that single execution raises (LLM call
failure, email send failure, transient Render/network issue, ...), the row's
send_job_claimed_at is released so a retry CAN succeed, but nothing retries
it automatically — the job itself is already consumed from the jobstore.
Without this endpoint, such an entry sits forever showing "Not Yet Due" in
the UI (display_status() only looks at Assessment.status, not whether
scheduled_at has already passed) with no exam ever delivered and no way to
recover it via the existing "Send late exam" flow either, since that
requires status == expired, not scheduled.

Fully mocked — FakeLLM + the autouse Noop `_email` patch — zero real
Anthropic/email calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from app.models.assessment import Assessment, AssessmentStatus
from app.utils.token_auth import generate_submission_token
from tests.conftest import FakeLLM, TestSessionLocal, make_curriculum, seed_prompt_templates


def _make_scheduled_assessment(db, curriculum, *, scheduled_at: datetime) -> Assessment:
    """A curriculum-upload-style entry: content deliberately ungenerated
    (assessment_text=None, part1_text=None) — mirrors how send_assessment_job
    defers generation to send-time for entry-type curricula."""
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


class TestResendStuckAssessment:
    def test_overdue_scheduled_assessment_gets_sent(self, client, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        overdue = datetime.utcnow() - timedelta(days=1)
        assessment = _make_scheduled_assessment(db, curriculum, scheduled_at=overdue)

        response = client.post(f"/api/v1/assessments/{assessment.id}/resend")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "active"

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.active
        assert refreshed.assessment_text is not None  # FakeLLM-generated

    def test_refuses_to_send_early(self, client, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        not_yet = datetime.utcnow() + timedelta(days=1)
        assessment = _make_scheduled_assessment(db, curriculum, scheduled_at=not_yet)

        response = client.post(f"/api/v1/assessments/{assessment.id}/resend")

        assert response.status_code == 409
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.scheduled

    def test_wrong_status_is_409(self, client, db, monkeypatch):
        seed_prompt_templates(db)
        _patch_job_infra(monkeypatch)
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(days=1)
        )
        assessment.status = AssessmentStatus.active
        db.commit()

        response = client.post(f"/api/v1/assessments/{assessment.id}/resend")
        assert response.status_code == 409

    def test_unknown_assessment_is_404(self, client):
        response = client.post("/api/v1/assessments/does-not-exist/resend")
        assert response.status_code == 404
