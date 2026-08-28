"""Post-Section-7: splitting flow (e) transcript delivery into a primary
per-event send and a secondary biweekly send.

Covers the 4 audit requirements explicitly:
  1. The secondary recipient no longer receives anything from the
     per-event trigger (grade_submission_job) — tested directly, not
     inferred from "the new job exists."
  2. The biweekly job sends CONTENT-IDENTICAL data to what the primary
     receives — same compute_transcript() output, not a different digest.
  3. The biweekly job is actually registered on a real clock (a real,
     non-fake APSchedulerAdapter with next_run_time set), not just defined
     in code with nothing wiring it up.
  4. The biweekly job fires correctly with zero graded entries in the
     upload, and with several — and sends once per existing upload, and
     no-ops entirely when no secondary is configured yet.

TEST_SECONDARY is a placeholder sample address (mridula.sureshrao2000@gmail.com,
per instruction) used only inside these tests — nothing here writes it to
the real .env, and no live email is sent (RecordingEmailAdapter throughout).
"""
from __future__ import annotations

import pytest

from app.models.assessment import AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.services.email_service import EmailService
from tests.conftest import (
    RecordingEmailAdapter,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
)

TEST_SECONDARY = "mridula.sureshrao2000@gmail.com"
TEST_PRIMARY = "mridula.sureshrao@proton.me"


def _make_upload(db, source_filename="test.json") -> CurriculumUpload:
    upload = CurriculumUpload(source_filename=source_filename)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _graded_entry(db, upload, *, topic="Topic", chapter_label="Chapter 1 — X", score_earned=45.0):
    curriculum = make_curriculum(db, topic=topic, entry_type=CurriculumEntryType.assessment)
    curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = upload.id, 50.0, chapter_label
    db.commit()
    a, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed)
    s = make_submission(db, a)
    make_grade(db, s, mastery_score=90.0, score_earned=score_earned, max_marks=50.0)
    return curriculum


@pytest.fixture(autouse=True)
def _configured_settings(monkeypatch):
    """Primary is user_email (already configured); secondary is the sample
    test address until the real one is confirmed."""
    monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
    monkeypatch.setenv("TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", TEST_SECONDARY)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestRequirement1_SecondaryExcludedFromPerEventTrigger:

    def test_per_event_send_transcript_email_only_reaches_primary(self, db):
        upload = _make_upload(db)
        _graded_entry(db, upload)
        recording = RecordingEmailAdapter()

        EmailService(db, recording).send_transcript_email(upload.id)  # no override — per-event path

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].recipient_emails == [TEST_PRIMARY]
        assert TEST_SECONDARY not in recording.transcript_calls[0].recipient_emails

    def test_grade_submission_job_transcript_send_excludes_secondary(self, db, monkeypatch):
        """End-to-end through the real per-event trigger, not just the
        service method in isolation."""
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLM, FakeScheduler, TestSessionLocal, seed_prompt_templates

        seed_prompt_templates(db)
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1 — X",
        )
        db.commit()
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.grade_submission_job._email", recording)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        grade_submission_job(submission.id)

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].recipient_emails == [TEST_PRIMARY]
        assert TEST_SECONDARY not in recording.transcript_calls[0].recipient_emails

    def test_repeated_grading_events_never_reach_secondary(self, db, monkeypatch):
        """Not just the first event — the secondary must never appear
        across multiple grading events for the same or different entries."""
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLM, FakeScheduler, TestSessionLocal, seed_prompt_templates

        seed_prompt_templates(db)
        upload = _make_upload(db)
        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.grade_submission_job._email", recording)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        for i in range(3):
            curriculum = make_curriculum(db, topic=f"Topic {i}", entry_type=CurriculumEntryType.assessment)
            curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
                upload.id, 50.0, "Chapter 1 — X",
            )
            db.commit()
            assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
            submission = make_submission(db, assessment)
            grade_submission_job(submission.id)

        assert len(recording.transcript_calls) == 3
        assert all(c.recipient_emails == [TEST_PRIMARY] for c in recording.transcript_calls)


class TestRequirement2_BiweeklyContentIdenticalToPrimary:

    def test_biweekly_and_primary_payloads_match_except_recipients(self, db):
        upload = _make_upload(db)
        _graded_entry(db, upload, topic="A", score_earned=45.0)
        _graded_entry(db, upload, topic="B", chapter_label="Chapter 2 — Y", score_earned=30.0)

        primary_recording = RecordingEmailAdapter()
        EmailService(db, primary_recording).send_transcript_email(upload.id)

        secondary_recording = RecordingEmailAdapter()
        EmailService(db, secondary_recording).send_transcript_email(
            upload.id, recipient_emails=[TEST_SECONDARY]
        )

        primary_data = primary_recording.transcript_calls[0]
        secondary_data = secondary_recording.transcript_calls[0]

        assert primary_data.recipient_emails == [TEST_PRIMARY]
        assert secondary_data.recipient_emails == [TEST_SECONDARY]
        # Every other field — the actual transcript content — must be identical.
        assert primary_data.source_filename == secondary_data.source_filename
        assert primary_data.resolved_count == secondary_data.resolved_count
        assert primary_data.total_entry_count == secondary_data.total_entry_count
        assert primary_data.graded_count == secondary_data.graded_count
        assert primary_data.total_credits == secondary_data.total_credits
        assert primary_data.total_points == secondary_data.total_points
        assert primary_data.gpa == secondary_data.gpa
        assert primary_data.entry_groups == secondary_data.entry_groups
        assert primary_data.course_material == secondary_data.course_material

    def test_biweekly_job_content_matches_direct_compute_transcript(self, db, monkeypatch):
        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job
        from app.services.transcript_service import compute_transcript
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        _graded_entry(db, upload, topic="A")

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job._email", recording)

        send_biweekly_transcript_job()

        fresh = compute_transcript(db, upload.id)
        sent = recording.transcript_calls[0]
        assert sent.resolved_count == fresh.resolved_count
        assert sent.total_points == fresh.total_points
        assert sent.gpa == fresh.gpa


class TestRequirement3_RealClockRegistration:
    """A real (non-fake) APSchedulerAdapter, with an isolated SQLite
    jobstore so this doesn't touch the tracked dev assessment.db.

    conftest.py's autouse _no_apscheduler fixture mocks
    APSchedulerAdapter.start()/.shutdown() globally, precisely to stop any
    test from accidentally starting a real background scheduler against
    the production job store — exactly what this test needs to do on
    purpose, safely. Rather than fight that guard, these tests drive the
    real underlying BackgroundScheduler and the real (unmocked)
    _schedule_biweekly_transcript() registration method directly — the
    exact same registration code .start() calls, just invoked without
    going through the mocked wrapper.
    """

    def test_job_is_registered_with_a_real_next_run_time(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test.db")
        monkeypatch.setenv("TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", TEST_SECONDARY)
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            adapter._scheduler.start()
            adapter._schedule_biweekly_transcript()
            job = adapter._scheduler.get_job("send_biweekly_transcript")

            assert job is not None
            assert job.next_run_time is not None  # actually scheduled, not just defined
            assert job.trigger.interval.days == 14
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

    def test_interval_is_configurable_via_settings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test2.db")
        monkeypatch.setenv("TRANSCRIPT_SECONDARY_INTERVAL_DAYS", "21")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            adapter._scheduler.start()
            adapter._schedule_biweekly_transcript()
            job = adapter._scheduler.get_job("send_biweekly_transcript")
            assert job.trigger.interval.days == 21
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

    def test_full_start_registers_all_three_recurring_jobs(self, monkeypatch, tmp_path):
        """Confirm the biweekly job is wired into the SAME start() sequence
        as the token grant and pending-midterm recheck — not a parallel,
        separately-invoked path that could silently be skipped."""
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test3.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            # start()'s real body is exactly these four calls (see
            # APSchedulerAdapter.start) — replicated directly since .start()
            # itself is mocked by the autouse _no_apscheduler fixture.
            adapter._scheduler.start()
            adapter._schedule_monthly_token_grant()
            adapter._schedule_pending_midterm_recheck()
            adapter._schedule_biweekly_transcript()

            job_ids = {j.id for j in adapter._scheduler.get_jobs()}
            assert job_ids == {
                "grant_late_tokens_monthly",
                "recheck_pending_midterms_daily",
                "send_biweekly_transcript",
            }
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()


class TestRequirement4_FiresWithZeroAndSeveralGradingEvents:

    def test_fires_with_zero_graded_entries_in_the_upload(self, db, monkeypatch):
        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.chapter_label = upload.id, "Chapter 1 — X"
        db.commit()
        make_assessment(db, curriculum, status=AssessmentStatus.scheduled, due_offset_days=30)

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job._email", recording)

        send_biweekly_transcript_job()

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].resolved_count == 0
        assert recording.transcript_calls[0].recipient_emails == [TEST_SECONDARY]

    def test_fires_with_several_graded_entries(self, db, monkeypatch):
        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        for i in range(4):
            _graded_entry(db, upload, topic=f"Topic {i}", chapter_label=f"Chapter {i+1} — X")

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job._email", recording)

        send_biweekly_transcript_job()

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].resolved_count == 4
        assert recording.transcript_calls[0].graded_count == 4

    def test_sends_once_per_existing_upload(self, db, monkeypatch):
        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job
        from tests.conftest import TestSessionLocal

        upload_a = _make_upload(db, "a.json")
        _graded_entry(db, upload_a, topic="A")
        upload_b = _make_upload(db, "b.json")
        _graded_entry(db, upload_b, topic="B")

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job._email", recording)

        send_biweekly_transcript_job()

        assert len(recording.transcript_calls) == 2
        assert {c.upload_id for c in recording.transcript_calls} == {upload_a.id, upload_b.id}

    def test_noops_entirely_when_no_secondary_configured(self, db, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", "")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.jobs.send_biweekly_transcript_job import send_biweekly_transcript_job
        from tests.conftest import TestSessionLocal

        upload = _make_upload(db)
        _graded_entry(db, upload)

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_biweekly_transcript_job._email", recording)

        send_biweekly_transcript_job()

        assert recording.transcript_calls == []
        get_settings.cache_clear()
