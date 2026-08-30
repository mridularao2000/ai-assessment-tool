"""send_assessment_job — specifically the needs_manual_diagnosis branch
added for the 2026-08-30 tool-budget-exhaustion incident.

Real incident this exists for: a tool-enabled generation that exhausts its
attempt budget (LLMToolBudgetExceededError) used to fall into the same
generic `except Exception` handling as any other failure — the row stayed
`scheduled`, so recheck_stuck_assessments_job's 15-minute sweep would keep
retrying it, re-paying for the same expensive, deterministic failure for up
to stuck_assessment_max_auto_retry_hours before giving up. This asserts the
row is instead moved to a distinct status the sweep never picks up.

Fully mocked — FakeLLM subclass + TestSessionLocal — zero real
Anthropic/email calls.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from app.interfaces.llm import LLMToolBudgetExceededError
from app.jobs.send_assessment_job import send_assessment_job
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


class _BudgetExhaustedLLM(FakeLLM):
    def generate_assessment(self, req):
        raise LLMToolBudgetExceededError(
            "Tool-enabled generation exhausted all 2 attempt(s) without "
            "producing valid output — 316322 token(s) spent (ceiling "
            "200000). No text block in response content after 51 block(s) "
            "(25 tool-use round(s): {'code_execution': 19, 'web_search': 2, "
            "'web_fetch': 3, 'text_editor_code_execution': 1}).",
            tokens_spent=316322,
            attempts_made=2,
            ceiling=200000,
        )


class TestSendAssessmentJobToolBudgetExceeded:
    def test_marks_needs_manual_diagnosis_not_scheduled(self, db, monkeypatch):
        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", _BudgetExhaustedLLM())
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(minutes=5)
        )

        with pytest.raises(LLMToolBudgetExceededError):
            send_assessment_job(assessment.id)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.needs_manual_diagnosis
        assert refreshed.assessment_text is None

    def test_releases_the_send_job_claim_for_manual_resend(self, db, monkeypatch):
        """A human resolving the cause and calling POST /resend must not be
        blocked by a stale claim left over from the failed attempt."""
        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", _BudgetExhaustedLLM())
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(minutes=5)
        )

        with pytest.raises(LLMToolBudgetExceededError):
            send_assessment_job(assessment.id)

        db.expire_all()
        assert db.get(Assessment, assessment.id).send_job_claimed_at is None

    def test_recheck_sweep_never_picks_up_a_needs_manual_diagnosis_row(self, db, monkeypatch):
        """The actual "do not auto-retry" guarantee: recheck_stuck_
        assessments_job's query only matches status == scheduled, so once
        moved to needs_manual_diagnosis, the 15-minute sweep leaves it
        alone — proven here by pointing the sweep at an LLM that would
        raise AssertionError if it were ever called again."""
        from app.jobs.recheck_stuck_assessments_job import recheck_stuck_assessments_job

        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.recheck_stuck_assessments_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", _BudgetExhaustedLLM())
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(hours=2)
        )

        with pytest.raises(LLMToolBudgetExceededError):
            send_assessment_job(assessment.id)
        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.needs_manual_diagnosis

        def _poison(*args, **kwargs):
            raise AssertionError("recheck sweep must not retry a needs_manual_diagnosis row")

        monkeypatch.setattr("app.jobs.send_assessment_job.send_assessment_job", _poison)

        recheck_stuck_assessments_job()  # must not raise, must not touch the poisoned job

        db.expire_all()
        assert db.get(Assessment, assessment.id).status == AssessmentStatus.needs_manual_diagnosis

    def test_ordinary_failure_still_stays_scheduled_for_auto_retry(self, db, monkeypatch):
        """Scoping check: an ordinary (non-budget) failure must be
        unaffected — it stays `scheduled` so the normal self-healing sweep
        still retries it, exactly as before this change."""
        seed_prompt_templates(db)
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)

        class _OrdinaryFailureLLM(FakeLLM):
            def generate_assessment(self, req):
                raise RuntimeError("transient failure, unrelated to tool budget")

        monkeypatch.setattr("app.jobs.send_assessment_job._llm", _OrdinaryFailureLLM())
        curriculum = make_curriculum(db, entry_type="assessment")
        assessment = _make_scheduled_assessment(
            db, curriculum, scheduled_at=datetime.utcnow() - timedelta(minutes=5)
        )

        with pytest.raises(RuntimeError):
            send_assessment_job(assessment.id)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.scheduled
        assert refreshed.send_job_claimed_at is None
