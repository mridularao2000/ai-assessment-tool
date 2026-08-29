"""Post-Section-7: splitting flow (e) transcript delivery into a primary
per-event send and a secondary periodic send.

Covers the 4 audit requirements explicitly:
  1. The secondary recipient no longer receives anything from the
     per-event trigger (grade_submission_job) — tested directly, not
     inferred from "the new job exists."
  2. The periodic send is CONTENT-IDENTICAL to what the primary receives —
     same compute_transcript() output, not a different digest.
  3. The periodic send is folded into the existing daily
     recheck_pending_midterms_daily CronTrigger — an absolute-calendar
     trigger, not a separate IntervalTrigger job — since an IntervalTrigger's
     next_run_time is recomputed relative to *registration* time on every
     add_job() call. That job is re-registered on every process restart
     (start() calls it with replace_existing=True), so restarts silently
     reset an interval trigger's countdown; that this is a real production
     risk here — not a hypothetical — see how often this server has
     already restarted during this build. CronTrigger anchors to an
     absolute calendar instant instead, so restarts don't move it.
  4. The periodic check fires correctly with zero graded entries in the
     upload, and with several — sends once per open upload, no-ops when no
     secondary is configured, and — the actual point of the fix — is
     immune to however many restarts happen between two daily ticks,
     because "is it due" is decided from a persisted per-upload timestamp
     compared against wall-clock time, not from scheduler state.

TEST_SECONDARY is a placeholder sample address (mridula.sureshrao2000@gmail.com,
per instruction) used only inside these tests — nothing here writes it to
the real .env, and no live email is sent (RecordingEmailAdapter throughout).
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from app.models._utils import utcnow
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


def _run_recheck_job(monkeypatch, recording):
    """Drives the real daily job entrypoint (the same one .start() puts on
    the CronTrigger), with SessionLocal/_email/get_scheduler_adapter
    patched the same way every other job test in this suite patches its
    target job."""
    from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job
    from tests.conftest import FakeScheduler, TestSessionLocal

    monkeypatch.setattr("app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording)
    monkeypatch.setattr(
        "app.jobs.recheck_pending_midterms_job.get_scheduler_adapter", lambda: FakeScheduler()
    )
    recheck_pending_midterms_job()


@pytest.fixture(autouse=True)
def _configured_settings(monkeypatch):
    """Primary is user_email (already configured); secondary is the sample
    test address until the real one is confirmed."""
    monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
    monkeypatch.setenv("TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", TEST_SECONDARY)
    monkeypatch.setenv("TRANSCRIPT_SECONDARY_INTERVAL_DAYS", "15")
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


class TestRequirement2_PeriodicContentIdenticalToPrimary:

    def test_periodic_and_primary_payloads_match_except_recipients(self, db):
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

    def test_periodic_check_content_matches_direct_compute_transcript(self, db, monkeypatch):
        from app.services.transcript_service import compute_transcript

        upload = _make_upload(db)
        _graded_entry(db, upload, topic="A")
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        fresh = compute_transcript(db, upload.id)
        sent = recording.transcript_calls[0]
        assert sent.resolved_count == fresh.resolved_count
        assert sent.total_points == fresh.total_points
        assert sent.gpa == fresh.gpa


class TestRequirement3_FoldedIntoAbsoluteCalendarCron:
    """No separate scheduler job for the secondary transcript exists
    anymore — it is folded into recheck_pending_midterms_daily, an
    absolute-calendar CronTrigger (hour=6), which is immune to the
    restart-resets-the-countdown problem an IntervalTrigger has.

    These tests drive the real underlying BackgroundScheduler directly,
    the same registration code .start() calls, since conftest's autouse
    _no_apscheduler fixture mocks APSchedulerAdapter.start()/.shutdown()
    globally to stop any test from starting a real background scheduler
    against the production job store — exactly what these tests need to
    do on purpose, safely.
    """

    def test_no_separate_interval_job_exists_anymore(self, monkeypatch, tmp_path):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            adapter._scheduler.start()
            adapter._schedule_monthly_token_grant()
            adapter._schedule_pending_midterm_recheck()

            assert adapter._scheduler.get_job("send_biweekly_transcript") is None
            # The registration method itself is gone, not just unregistered.
            assert not hasattr(adapter, "_schedule_biweekly_transcript")
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

    def test_daily_recheck_uses_absolute_cron_trigger_immune_to_restarts(self, monkeypatch, tmp_path):
        """The actual fix: an absolute CronTrigger's next fire time is a
        fixed calendar instant, unaffected by when add_job() is called —
        unlike IntervalTrigger, which computes next_run_time relative to
        registration time. Re-registering it repeatedly (exactly what
        start() does on every process restart) must not move next_run_time
        at all."""
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test2.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from apscheduler.triggers.cron import CronTrigger
        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            adapter._scheduler.start()
            adapter._schedule_pending_midterm_recheck()
            job = adapter._scheduler.get_job("recheck_pending_midterms_daily")

            assert job is not None
            assert job.next_run_time is not None
            assert isinstance(job.trigger, CronTrigger)
            first_next_run = job.next_run_time

            for _ in range(5):  # simulate 5 restarts
                adapter._schedule_pending_midterm_recheck()

            assert adapter._scheduler.get_job("recheck_pending_midterms_daily").next_run_time == first_next_run
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

    def test_full_start_registers_exactly_the_two_recurring_jobs(self, monkeypatch, tmp_path):
        """Confirm the periodic transcript check rides along inside
        recheck_pending_midterms_daily rather than adding a third,
        separately-invoked job that could silently be skipped."""
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/scheduler_test3.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        try:
            adapter._scheduler.start()
            adapter._schedule_monthly_token_grant()
            adapter._schedule_pending_midterm_recheck()

            job_ids = {j.id for j in adapter._scheduler.get_jobs()}
            assert job_ids == {"grant_late_tokens_monthly", "recheck_pending_midterms_daily"}
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()


class TestRequirement4_PeriodicCheckFiresCorrectlyAndSurvivesRestarts:

    def test_fires_with_zero_graded_entries_in_the_upload(self, db, monkeypatch):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.chapter_label = upload.id, "Chapter 1 — X"
        db.commit()
        make_assessment(db, curriculum, status=AssessmentStatus.scheduled, due_offset_days=30)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].resolved_count == 0
        assert recording.transcript_calls[0].recipient_emails == [TEST_SECONDARY]

    def test_fires_with_several_graded_entries(self, db, monkeypatch):
        upload = _make_upload(db)
        for i in range(4):
            _graded_entry(db, upload, topic=f"Topic {i}", chapter_label=f"Chapter {i+1} — X")
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].resolved_count == 4
        assert recording.transcript_calls[0].graded_count == 4

    def test_sends_once_per_open_upload(self, db, monkeypatch):
        upload_a = _make_upload(db, "a.json")
        _graded_entry(db, upload_a, topic="A")
        upload_a.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        upload_b = _make_upload(db, "b.json")
        _graded_entry(db, upload_b, topic="B")
        upload_b.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 2
        assert {c.upload_id for c in recording.transcript_calls} == {upload_a.id, upload_b.id}

    def test_noops_entirely_when_no_secondary_configured(self, db, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", "")
        from app.config import get_settings
        get_settings.cache_clear()

        upload = _make_upload(db)
        _graded_entry(db, upload)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert recording.transcript_calls == []
        get_settings.cache_clear()

    def test_not_yet_due_when_interval_has_not_elapsed(self, db, monkeypatch):
        upload = _make_upload(db)
        _graded_entry(db, upload)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=5)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert recording.transcript_calls == []

    def test_never_sent_uses_upload_time_as_anchor_not_yet_due(self, db, monkeypatch):
        """A brand-new upload (last_secondary_transcript_sent_at still
        None) must not fire the moment the daily job next ticks — it
        anchors on uploaded_at, same as a real send would."""
        upload = _make_upload(db)  # uploaded_at defaults to just now
        _graded_entry(db, upload)

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert recording.transcript_calls == []

    def test_never_sent_fires_once_upload_itself_is_old_enough(self, db, monkeypatch):
        upload = _make_upload(db)
        upload.uploaded_at = utcnow() - timedelta(days=16)
        db.commit()
        _graded_entry(db, upload)

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 1

    def test_does_not_resend_before_interval_elapses_again(self, db, monkeypatch):
        """Three consecutive daily ticks right after a send must produce
        exactly one send total — the second and third find the anchor too
        recent."""
        upload = _make_upload(db)
        _graded_entry(db, upload)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)
        _run_recheck_job(monkeypatch, recording)
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 1

    def test_survives_multiple_restarts_within_the_interval_window(self, db, monkeypatch, tmp_path):
        """Simulates the exact failure mode reported: many server restarts
        happening inside one interval window. Under the old IntervalTrigger
        design, each restart re-registered the job and reset its
        next_run_time to (restart_time + interval) — frequent restarts
        could push the send arbitrarily far into the future, and a server
        that restarts more often than the interval would never send it at
        all. Restarts are simulated here by repeatedly re-registering the
        daily cron job — exactly what start() does on every process boot.
        The gate is driven by last_secondary_transcript_sent_at vs wall
        clock, not by scheduler registration, so restart count must have
        zero effect on the outcome.
        """
        upload = _make_upload(db)
        _graded_entry(db, upload)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/restart_sim.db")
        from app.config import get_settings
        get_settings.cache_clear()

        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        adapter._scheduler.start()
        try:
            for _ in range(5):  # 5 simulated restarts before the daily tick fires
                adapter._schedule_pending_midterm_recheck()

            recording = RecordingEmailAdapter()
            _run_recheck_job(monkeypatch, recording)

            assert len(recording.transcript_calls) == 1
            assert recording.transcript_calls[0].recipient_emails == [TEST_SECONDARY]
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()

    def test_control_zero_restarts_same_anchor_sends_identically(self, db, monkeypatch):
        """Control for the restart-simulation test above: the same anchor
        and interval, with no restart simulation at all, must produce the
        identical outcome — one send. Proves the restart simulation isn't
        itself what caused the send in the test above."""
        upload = _make_upload(db)
        _graded_entry(db, upload)
        upload.last_secondary_transcript_sent_at = utcnow() - timedelta(days=16)
        db.commit()

        recording = RecordingEmailAdapter()
        _run_recheck_job(monkeypatch, recording)

        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].recipient_emails == [TEST_SECONDARY]
