"""MODIFY (add/edit entries, protected once an attempt exists) and
DELETE/CLOSE (soft-delete/archive, final transcript, permanent silence)
for curriculum_upload entries.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from app.exceptions import InvalidStateError, NotFoundError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeLLM,
    FakeScheduler,
    NoopEmailAdapter,
    RecordingEmailAdapter,
    TestSessionLocal,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
    seed_prompt_templates,
)

TEST_PRIMARY = "mridula.sureshrao@proton.me"
FIXED_NOW = datetime(2026, 8, 20, 12, 0, 0)
TARGET_COMPLETION_DATE = date(2026, 8, 10)


class _FixedDateTime(datetime):
    @classmethod
    def utcnow(cls):
        return FIXED_NOW


def _make_upload(db, source_filename="modify_test.json") -> CurriculumUpload:
    upload = CurriculumUpload(source_filename=source_filename)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


class TestAddEntry:

    def test_add_entry_to_open_upload_schedules_it(self, db):
        seed_prompt_templates(db)
        upload = _make_upload(db)
        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )

        curriculum = upload_service.add_entry(upload.id, {
            "topic": "Added Later", "type": "assessment", "chapter": "Chapter 3",
            "resources": ["https://example.com/doc"],
            "completion_date": "2026-09-01", "max_marks": 50,
        })

        assert curriculum.upload_id == upload.id
        assert curriculum.topic == "Added Later"
        assert len(curriculum.assessments) == 1
        assert curriculum.assessments[0].status == AssessmentStatus.scheduled

    def test_add_entry_to_closed_upload_is_refused(self, db):
        seed_prompt_templates(db)
        upload = _make_upload(db)
        upload_service = CurriculumUploadService(
            db, RecordingEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        # Close an empty upload — final transcript is empty but still sent.
        upload_service.close_upload(upload.id)

        try:
            upload_service.add_entry(upload.id, {
                "topic": "Too Late", "type": "assessment", "chapter": "X",
                "resources": [], "completion_date": "2026-09-01", "max_marks": 50,
            })
            assert False, "expected InvalidStateError"
        except InvalidStateError:
            pass


class TestUpdateEntry:

    def test_entry_with_no_attempt_yet_is_editable(self, db):
        seed_prompt_templates(db)
        upload = _make_upload(db)
        # A real future date (not the shared past-oriented TARGET_COMPLETION_DATE
        # constant, which other tests pair with a mocked "now") -- this test
        # doesn't mock the clock, and needs schedule_ready_entries() to take
        # the normal "future window" path, not the retroactive-expired one.
        future_date = date.today().replace(year=date.today().year + 1)
        curriculum = make_curriculum(
            db, topic="Original Topic", entry_type=CurriculumEntryType.assessment,
            target_completion_date=future_date,
        )
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1",
        )
        db.commit()

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        # Not-yet-scheduled entry (no Assessment row at all) is editable.
        # update_entry() calls schedule_ready_entries() internally, so this
        # single call also auto-schedules the entry's first Assessment row.
        updated = upload_service.update_entry(curriculum.id, {"topic": "Renamed Topic"})
        assert updated.topic == "Renamed Topic"
        assert len(updated.assessments) == 1
        first_assessment = updated.assessments[0]
        assert first_assessment.status == AssessmentStatus.scheduled

        # Still editable once scheduled but not yet sent (status=scheduled).
        updated2 = upload_service.update_entry(curriculum.id, {"max_marks": 75.0})
        assert updated2.max_marks == 75.0
        # Editing a scheduled-but-unsent entry replaces its stale Assessment
        # row with a fresh one rather than leaving it pointing at old data.
        assert len(updated2.assessments) == 1
        assert updated2.assessments[0].id != first_assessment.id
        assert updated2.assessments[0].status == AssessmentStatus.scheduled

    def test_entry_stuck_in_needs_manual_diagnosis_is_editable(self, db):
        """Real incident this covers: Performance Optimization I,
        2026-08-31 — generation exhausted its tool-call budget with no
        content ever produced (needs_manual_diagnosis), yet the entry was
        refused for editing with the same "already has an attempt or
        grade on record" message used for entries with REAL content —
        there was nothing to corrupt, and no way to fix the resource that
        was likely causing the runaway tool loop. This status must be
        editable, same as a fresh 'scheduled' row."""
        seed_prompt_templates(db)
        upload = _make_upload(db)
        curriculum = make_curriculum(
            db, topic="Stuck Topic", entry_type=CurriculumEntryType.assessment,
            target_completion_date=date.today().replace(year=date.today().year + 1),
        )
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 6",
        )
        db.commit()
        stuck, _ = make_assessment(db, curriculum, status=AssessmentStatus.needs_manual_diagnosis)
        assert stuck.assessment_text is not None  # make_assessment's default; irrelevant here —
        # the real-world row has assessment_text=None, but editability must not depend on that.

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        updated = upload_service.update_entry(curriculum.id, {"topic": "Fixed Topic"})

        assert updated.topic == "Fixed Topic"
        # The broken row was deleted and a fresh one scheduled in its place.
        assert len(updated.assessments) == 1
        assert updated.assessments[0].id != stuck.id
        assert updated.assessments[0].status == AssessmentStatus.scheduled

    def test_entry_with_an_attempt_cannot_be_silently_modified(self, db):
        """The explicit design choice: an entry with any attempt or grade
        on record is protected from modification, not silently allowed —
        changing resources/dates after an exam was already generated from
        them would corrupt what's already on record."""
        seed_prompt_templates(db)
        upload = _make_upload(db)
        curriculum = make_curriculum(
            db, topic="Already Sent Topic", entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1",
        )
        db.commit()
        # active = exam already sent -- an attempt window is open.
        make_assessment(db, curriculum, status=AssessmentStatus.active)

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        try:
            upload_service.update_entry(curriculum.id, {"topic": "Sneaky Rename"})
            assert False, "expected InvalidStateError"
        except InvalidStateError as exc:
            assert "already has an attempt" in str(exc)

        db.refresh(curriculum)
        assert curriculum.topic == "Already Sent Topic"  # untouched

    def test_entry_in_closed_upload_cannot_be_modified(self, db):
        seed_prompt_templates(db)
        upload = _make_upload(db)
        curriculum = make_curriculum(
            db, topic="Closed Upload Entry", entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1",
        )
        db.commit()

        upload_service = CurriculumUploadService(
            db, RecordingEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        upload_service.close_upload(upload.id)

        try:
            upload_service.update_entry(curriculum.id, {"topic": "New"})
            assert False, "expected InvalidStateError"
        except InvalidStateError:
            pass


class TestCloseCurriculum:

    def test_close_sends_final_transcript_then_permanently_silences_jobs(
        self, db, monkeypatch, tmp_path
    ):
        """Create a curriculum, generate real history on it (scheduled ->
        exam sent -> graded), close it, confirm the final transcript
        arrives with the real final state, then prove — via a REAL
        APScheduler instance, not a mock — that every job tied to its
        entries is actually gone (not just "should be cancelled"), and
        that the recurring biweekly job skips it from then on.
        """
        monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr("app.services.transcript_service.datetime", _FixedDateTime)
        monkeypatch.setattr("app.services.submission_service.utcnow", lambda: FIXED_NOW)

        seed_prompt_templates(db)
        upload = _make_upload(db, "close_lifecycle_test.json")
        curriculum = make_curriculum(
            db, topic="Close Lifecycle Topic", entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1 — Close Test",
        )
        db.commit()

        recording = RecordingEmailAdapter()

        # ── Generate real history via FakeScheduler (same proven pattern
        # as test_full_entry_lifecycle_integration.py) -- job REGISTRATION
        # is exercised separately below, against a real scheduler, with
        # dates far enough in the future that nothing fires prematurely
        # while this history is being built. ──
        fake_scheduler_service = SchedulerService(db, FakeScheduler())
        upload_service = CurriculumUploadService(db, recording, fake_scheduler_service)
        assessment = upload_service._schedule_entry_assessment(
            curriculum, curriculum.target_completion_date
        )

        from app.jobs.send_assessment_job import send_assessment_job
        from app.jobs.expire_assessment_job import expire_assessment_job
        from app.jobs.grade_submission_job import grade_submission_job
        from app.services.submission_service import SubmissionService
        from app.services.late_token_service import LateTokenService
        from app.models.submission import SubmissionType

        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)
        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.grade_submission_job._email", recording)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        send_assessment_job(assessment.id)
        expire_assessment_job(assessment.id)

        token_svc = LateTokenService(db)
        token_svc.grant_monthly(upload.id)
        db.expire_all()
        assessment = db.get(Assessment, assessment.id)

        submission_svc = SubmissionService(db, FakeScheduler(), token_svc)
        submission = submission_svc.create(
            assessment_id=assessment.id,
            token=assessment.submission_token,
            submission_type=SubmissionType.text,
            text_content="Late but real answer.",
        )
        grade_submission_job(submission.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.completed
        recording.transcript_calls.clear()  # ignore the per-grading-event send

        # ── Register this assessment's job triplet against a REAL
        # APScheduler instance, far enough in the future that nothing
        # fires on its own -- purely to prove close_upload() actually
        # removes them, not a mock standing in for that proof. ──
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/close_test_scheduler.db")
        get_settings.cache_clear()
        from app.adapters.apscheduler_adapter import APSchedulerAdapter
        adapter = APSchedulerAdapter()
        adapter._scheduler.start()
        real_scheduler_service = SchedulerService(db, adapter)

        try:
            far_future = datetime.utcnow() + timedelta(days=400)
            job_ids = real_scheduler_service.schedule_assessment_jobs(
                assessment_id=assessment.id,
                scheduled_at=far_future,
                reminder_at=far_future,
                due_date=far_future + timedelta(days=2),
            )
            assert adapter._scheduler.get_job(job_ids.send_reminder) is not None
            assert adapter._scheduler.get_job(job_ids.send_assessment) is not None
            assert adapter._scheduler.get_job(job_ids.expire) is not None

            # ── Close it -- via a service instance using this same real
            # scheduler, so close_upload()'s cancellation call hits it. ──
            upload_service_real = CurriculumUploadService(db, recording, real_scheduler_service)
            closed_upload = upload_service_real.close_upload(upload.id)
            assert closed_upload.closed_at is not None

            # The ONE final transcript snapshot, capturing the exact final
            # state -- graded, late.
            assert len(recording.transcript_calls) == 1
            final = recording.transcript_calls[0]
            assert final.upload_id == upload.id
            assert final.recipient_emails == [TEST_PRIMARY]
            final_row = final.entry_groups[0].rows[0]
            assert final_row.topic == "Close Lifecycle Topic"
            assert final_row.was_late is True
            assert "GRADED" in final_row.status_label

            # ── Every job tied to this entry is actually gone from the
            # real scheduler -- not "should be cancelled", verifiably gone. ──
            assert adapter._scheduler.get_job(job_ids.send_reminder) is None
            assert adapter._scheduler.get_job(job_ids.send_assessment) is None
            assert adapter._scheduler.get_job(job_ids.expire) is None

            # ── Closing again is refused -- a closed curriculum stays
            # silent by construction, not by coincidence. ──
            try:
                upload_service_real.close_upload(upload.id)
                assert False, "expected InvalidStateError on double-close"
            except InvalidStateError:
                pass

            # ── The recurring daily job's periodic-transcript check (a
            # real, still-running recurring job -- unrelated to whether
            # THIS upload is closed) must skip this closed upload's content
            # entirely from now on. ──
            monkeypatch.setenv(
                "TRANSCRIPT_SECONDARY_RECIPIENT_EMAIL", "mridula.sureshrao2000@gmail.com"
            )
            get_settings.cache_clear()
            from app.jobs.recheck_pending_midterms_job import recheck_pending_midterms_job

            recording.transcript_calls.clear()
            monkeypatch.setattr(
                "app.jobs.recheck_pending_midterms_job.SessionLocal", TestSessionLocal
            )
            monkeypatch.setattr("app.jobs.recheck_pending_midterms_job._email", recording)
            monkeypatch.setattr(
                "app.jobs.recheck_pending_midterms_job.get_scheduler_adapter",
                lambda: adapter,
            )
            recheck_pending_midterms_job()

            assert recording.transcript_calls == [], (
                "the daily job's periodic-transcript check fired but must "
                "not send anything for a closed curriculum, ever, after "
                "its final transcript"
            )
        finally:
            adapter._scheduler.shutdown(wait=True)
            get_settings.cache_clear()
