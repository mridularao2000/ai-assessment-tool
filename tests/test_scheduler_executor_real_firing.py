"""Proves a job fired through APScheduler's real ThreadPoolExecutor actually
completes (content generated, status flipped, email sent) — not just that
the job function works when called directly.

Discovered via a live dry-run: APScheduler logs "Removed job X" for a
one-shot ("date"-trigger) job the instant it's handed to the executor
(schedulers/base.py::_process_jobs — executor.submit_job() is
fire-and-forget; the job is removed from the jobstore in the same loop
iteration because a one-shot trigger has no next fire time). That log line
is jobstore bookkeeping, not a completion signal — the real completion
signal is EVENT_JOB_EXECUTED / EVENT_JOB_ERROR (or the executor's own
"executed successfully" / "raised an exception" log lines), fired only
after the job function actually returns. Every existing scheduler test for
send_assessment_job called the job function directly, so nothing in the
suite would have caught a real defect in the executor path — this closes
that gap.
"""
from __future__ import annotations

import threading
from datetime import datetime

import pytest

from app.models.assessment import AssessmentStatus
from tests.conftest import (
    FakeLLM,
    RecordingEmailAdapter,
    TestSessionLocal,
    make_assessment,
    make_curriculum,
    seed_prompt_templates,
)


class TestSendAssessmentJobThroughRealExecutor:

    def test_completes_correctly_when_fired_through_real_apscheduler_executor(
        self, db, monkeypatch, tmp_path
    ):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        assessment.assessment_text = None
        db.commit()
        assessment_id = assessment.id

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/executor_test.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        from app.jobs.send_assessment_job import send_assessment_job

        adapter = APSchedulerAdapter()
        done = threading.Event()
        outcome: dict = {}

        def _on_event(event):
            outcome["exception"] = getattr(event, "exception", None)
            done.set()

        adapter._scheduler.add_listener(_on_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        try:
            adapter._scheduler.start()
            adapter._scheduler.add_job(
                send_assessment_job,
                trigger="date",
                run_date=datetime.utcnow(),
                id=f"assessment_{assessment_id}",
                args=[assessment_id],
            )
            # Deliberately NOT asserting on get_job()/"Removed" here — that
            # bookkeeping fires the instant the job is handed to the
            # executor, regardless of whether the job function has even
            # started. Only EVENT_JOB_EXECUTED/ERROR proves it finished.
            fired = done.wait(timeout=10)
            assert fired, "job did not complete through the real executor within timeout"
            assert outcome["exception"] is None, f"job raised inside the executor: {outcome['exception']!r}"
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

        db.refresh(assessment)
        assert assessment.assessment_text is not None
        assert assessment.status == AssessmentStatus.active
        assert len(recording.assessment_calls) == 1

    def test_job_removed_from_jobstore_does_not_mean_job_finished(
        self, db, monkeypatch, tmp_path
    ):
        """The specific misreading from the dry run, pinned down directly:
        get_job() returning None (removed) can happen while the job function
        is still genuinely running in the executor thread."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        assessment.assessment_text = None
        db.commit()
        assessment_id = assessment.id

        release_llm = threading.Event()
        started = threading.Event()

        class SlowFakeLLM(FakeLLM):
            def generate_assessment(self, request):
                started.set()
                release_llm.wait(timeout=10)
                return super().generate_assessment(request)

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", SlowFakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/executor_test2.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        from app.jobs.send_assessment_job import send_assessment_job

        adapter = APSchedulerAdapter()
        job_id = f"assessment_{assessment_id}"
        try:
            adapter._scheduler.start()
            adapter._scheduler.add_job(
                send_assessment_job,
                trigger="date",
                run_date=datetime.utcnow(),
                id=job_id,
                args=[assessment_id],
            )
            assert started.wait(timeout=10), "job never started"

            # The job is mid-flight (blocked inside the LLM call) — yet the
            # one-shot trigger has already been removed from the jobstore.
            assert adapter._scheduler.get_job(job_id) is None
            db.refresh(assessment)
            assert assessment.status == AssessmentStatus.scheduled  # not yet flipped
            assert assessment.assessment_text is None  # not yet written

            release_llm.set()
            # Now let it actually finish before asserting completion.
            for _ in range(100):
                db.refresh(assessment)
                if assessment.status == AssessmentStatus.active:
                    break
                threading.Event().wait(0.05)
            assert assessment.status == AssessmentStatus.active
            assert assessment.assessment_text is not None
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()


class TestSendAssessmentJobTOCTOUGuard:
    """The actual incident this guard fixes: a manual retry raced a
    still-in-flight scheduled execution of send_assessment_job for the same
    assessment_id, and BOTH generated content and BOTH sent an
    assessment-ready email. Assessment.send_job_claimed_at (an atomic
    UPDATE...WHERE claim at the top of the job) closes that race."""

    def test_second_invocation_while_first_still_in_flight_is_rejected(
        self, db, monkeypatch
    ):
        """Reproduces the actual incident directly: a second invocation
        fired while the first has already claimed the row and is still
        mid-generation (exactly what a manual retry racing a still-running
        scheduled execution looks like) must be rejected, not race it."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        assessment.assessment_text = None
        db.commit()
        assessment_id = assessment.id

        started = threading.Event()
        release = threading.Event()

        class SlowFakeLLM(FakeLLM):
            def generate_assessment(self, request):
                started.set()
                release.wait(timeout=10)
                return super().generate_assessment(request)

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", SlowFakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)

        from app.jobs.send_assessment_job import send_assessment_job

        first = threading.Thread(target=send_assessment_job, args=(assessment_id,))
        first.start()
        assert started.wait(timeout=10), "first execution never reached generation"

        # The claim happens before the LLM is ever called, so by now the
        # first execution has already committed it and is simply blocked
        # (paused in Python, holding no transaction) waiting on `release`.
        # A second invocation right now must see the claim taken and return
        # immediately — no LLM call, no email.
        send_assessment_job(assessment_id)
        assert recording.assessment_calls == [], (
            "the second (racing) invocation must not send an email while "
            "the first is still in flight"
        )

        release.set()
        first.join(timeout=10)

        assert len(recording.assessment_calls) == 1
        db.refresh(assessment)
        assert assessment.status == AssessmentStatus.active
        assert assessment.assessment_text is not None
        assert assessment.send_job_claimed_at is not None

    def test_claim_is_released_on_failure_so_a_real_retry_still_works(
        self, db, monkeypatch
    ):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        assessment.assessment_text = None
        db.commit()
        assessment_id = assessment.id

        class FailingLLM(FakeLLM):
            def generate_assessment(self, request):
                raise RuntimeError("simulated transient failure")

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FailingLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)

        from app.jobs.send_assessment_job import send_assessment_job

        with pytest.raises(RuntimeError, match="simulated transient failure"):
            send_assessment_job(assessment_id)

        db.refresh(assessment)
        assert assessment.send_job_claimed_at is None, (
            "a genuine failure (not a race) must release the claim, or every "
            "future retry would be permanently locked out"
        )
        assert assessment.assessment_text is None

        # A real retry (e.g. an operator re-running the job, or the
        # scheduler's own retry path) must now proceed normally.
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        send_assessment_job(assessment_id)

        db.refresh(assessment)
        assert assessment.status == AssessmentStatus.active
        assert assessment.assessment_text is not None
        assert len(recording.assessment_calls) == 1
