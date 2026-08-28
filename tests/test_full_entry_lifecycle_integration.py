"""One synthetic curriculum-upload entry, walked through the full
real-world lifecycle in a single sequence, asserting state at every
transition:

  scheduled -> exam window opens -> exam sent (1 recipient) ->
  window closes with no attempt -> Missed, late-eligible ->
  late attempt taken, token consumed -> graded, result sent (2 recipients)
  -> GPA reflects the score correctly weighted -> transcript shows the
  full history (both the miss and the late grade, not just the final state)

This is deliberately different from every other test file in this suite:
those prove each section works in isolation (with FakeLLM/FakeScheduler/
NoopEmailAdapter shortcuts, hand-built fixtures). This one proves the
HANDOFFS between sections work — real service calls in the real order,
one entry, one continuous story. Nothing here should duplicate what a
per-section test already covers; it exists to catch what only shows up
when the pieces are chained.

Real-time independence: expiry/late-eligibility both depend on "is
due_date in the current calendar month" (transcript_service.display_status
and submission_service.create()), so this test pins both to a fixed,
consistent "now" rather than deriving anything from the real wall clock —
otherwise it would be flaky depending on what day of the real month it's
run on (this bit an earlier test in this suite for exactly this reason).
"""
from __future__ import annotations

from datetime import date, datetime

from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.late_submission_token import LateSubmissionToken
from app.models.resource import Resource, ResourceType
from app.models.submission import SubmissionType
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.gpa_service import compute_gpa
from app.services.late_token_service import LateTokenService
from app.services.scheduler_service import SchedulerService
from app.services.submission_service import SubmissionService
from app.services.transcript_service import compute_transcript, display_status
from tests.conftest import (
    FakeLLM,
    FakeLLMBelowThreshold,
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
TEST_SECONDARY_PLACEHOLDER = "mridula.sureshrao2000@gmail.com"

# Fixed "now" for the whole test — due_date must land before this, in the
# same calendar month, regardless of the real date the suite is run on.
FIXED_NOW = datetime(2026, 8, 20, 12, 0, 0)
TARGET_COMPLETION_DATE = date(2026, 8, 10)  # scheduled_at in [11,13], due_date in [13,15] Aug


class _FixedDateTime(datetime):
    @classmethod
    def utcnow(cls):
        return FIXED_NOW


class TestFullEntryLifecycleIntegration:

    def test_scheduled_through_missed_late_grade_gpa_and_transcript(self, db, monkeypatch):
        monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
        monkeypatch.setenv(
            "RESULTS_RECIPIENT_EMAILS_RAW", f"{TEST_PRIMARY},{TEST_SECONDARY_PLACEHOLDER}"
        )
        from app.config import get_settings
        get_settings.cache_clear()

        # Calendar-month checks in both transcript_service and
        # submission_service must agree with each other and with due_date —
        # pin both to the same fixed instant.
        monkeypatch.setattr("app.services.transcript_service.datetime", _FixedDateTime)
        monkeypatch.setattr(
            "app.services.submission_service.utcnow", lambda: FIXED_NOW
        )

        seed_prompt_templates(db)

        # ── Setup: one synthetic entry in one upload, exactly as real
        # ingestion would create it (chapter_label, max_marks, upload_id). ──
        upload = CurriculumUpload(source_filename="lifecycle_test.json")
        db.add(upload)
        db.commit()
        db.refresh(upload)

        curriculum = make_curriculum(
            db,
            topic="Lifecycle Test Topic",
            entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        curriculum.upload_id = upload.id
        curriculum.max_marks = 50.0
        curriculum.chapter_label = "Chapter 1 — Lifecycle"
        db.add(Resource(
            curriculum_id=curriculum.id, type=ResourceType.note,
            source_ref="lifecycle-test-resource.dev", raw_content=None,
        ))
        db.commit()

        # Two unused late tokens available, as if the monthly grant already ran.
        db.add_all([
            LateSubmissionToken(curriculum_upload_id=upload.id),
            LateSubmissionToken(curriculum_upload_id=upload.id),
        ])
        db.commit()

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )

        # ── 1. scheduled ─────────────────────────────────────────────────
        assessment = upload_service._schedule_entry_assessment(
            curriculum, curriculum.target_completion_date
        )
        assert assessment.status == AssessmentStatus.scheduled
        assert assessment.assessment_text is None  # deferred generation

        # ── 2. exam window opens -> exam sent (1 recipient) ─────────────
        from app.jobs.send_assessment_job import send_assessment_job

        recording = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", recording)

        send_assessment_job(assessment.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.active
        assert assessment.assessment_text is not None
        assert len(recording.assessment_calls) == 1
        assert len(recording.assessment_calls[0].recipient_emails) == 1

        # ── 3. window closes with no attempt -> Missed, late-eligible ───
        from app.jobs.expire_assessment_job import expire_assessment_job

        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)
        expire_assessment_job(assessment.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        curriculum_refreshed = assessment.curriculum
        assert assessment.status == AssessmentStatus.expired

        status_at_miss = display_status(db, curriculum_refreshed)
        assert status_at_miss.startswith("Missed — Late-Eligible")
        assert "(2 left)" in status_at_miss  # nothing spent yet — balance untouched
        assert LateTokenService(db).get_balance(upload.id) == 2

        # ── 4. late attempt taken, token consumed ───────────────────────
        submission_svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission = submission_svc.create(
            assessment_id=assessment.id,
            token=assessment.submission_token,
            submission_type=SubmissionType.text,
            text_content="Late answer for the lifecycle test.",
        )

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.late_submitted
        assert LateTokenService(db).get_balance(upload.id) == 1  # decremented by exactly 1

        spent_token = (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id == assessment.id)
            .first()
        )
        assert spent_token is not None
        assert spent_token.used_at is not None

        # ── 5. graded, result sent (2 recipients) ────────────────────────
        from app.jobs.grade_submission_job import grade_submission_job

        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.grade_submission_job._email", recording)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
        )

        grade_submission_job(submission.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.completed

        grade = assessment.submission.grade
        # FakeLLM grades mastery_score=90.0; score_earned is the WEIGHTED
        # value (90% of this entry's 50.0 max_marks), not the raw
        # percentage — proves the weighting transform actually ran, not
        # just a pass-through.
        assert grade.mastery_score == 90.0
        assert grade.score_earned == 45.0
        assert grade.max_marks == 50.0

        assert len(recording.results_calls) == 1
        assert len(recording.results_calls[0].recipient_emails) == 2
        assert TEST_PRIMARY in recording.results_calls[0].recipient_emails
        assert TEST_SECONDARY_PLACEHOLDER in recording.results_calls[0].recipient_emails

        # Flow (e) transcript per-event send: primary only — proving the
        # secondary-recipient split from the prior turn holds even inside
        # this full chained lifecycle, not just in its own isolated tests.
        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].recipient_emails == [TEST_PRIMARY]

        # ── 6. GPA reflects the score correctly weighted ─────────────────
        gpa_summary = compute_gpa(db, upload.id)
        assert gpa_summary.total_earned == 45.0
        assert gpa_summary.total_max == 50.0
        assert gpa_summary.gpa == 90.0
        assert gpa_summary.graded_count == 1
        assert gpa_summary.missed_count == 0

        # ── 7. transcript shows the full history, not just final state ──
        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        assert len(content.chapter_groups) == 1
        row = content.chapter_groups[0].rows[0]

        # Final state alone ("GRADED") would be indistinguishable from a
        # same-day grade — the transcript must retain that this was
        # missed first, then late-graded, not just show the end state.
        assert row.was_late is True
        assert "GRADED" in row.status_label
        assert "LATE" in row.status_label
        assert row.points == 45.0

        # The underlying trace of the miss (the spent late token) is still
        # queryable after the fact — the history isn't overwritten by the
        # final `completed` assessment status.
        assert (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id == assessment.id)
            .first()
            is not None
        )

    def test_token_used_on_failed_attempt_then_free_retake_passes_still_shows_late(
        self, db, monkeypatch
    ):
        """A token is spent to get access past a CLOSED window, not for a
        normal fail-then-retry. Sequence:

          window missed -> token used, attempt (1) fails ->
          free retake (attempt 2, no token) submitted ON TIME against its
          own new window -> retake passes

        The token is recorded against attempt 1's assessment_id; the FINAL
        graded record is attempt 2's, a DIFFERENT assessment_id. The
        transcript's lateness lookup must still find the token by checking
        every attempt in the curriculum's history, not just the final one —
        otherwise a late-then-recovered entry would be indistinguishable
        from one that was never late at all.
        """
        monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
        from app.config import get_settings
        get_settings.cache_clear()

        monkeypatch.setattr("app.services.transcript_service.datetime", _FixedDateTime)
        monkeypatch.setattr("app.services.submission_service.utcnow", lambda: FIXED_NOW)

        seed_prompt_templates(db)

        upload = CurriculumUpload(source_filename="retake_lateness_test.json")
        db.add(upload)
        db.commit()
        db.refresh(upload)

        curriculum = make_curriculum(
            db,
            topic="Retake Lateness Topic",
            entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        curriculum.upload_id = upload.id
        curriculum.max_marks = 50.0
        curriculum.chapter_label = "Chapter 2 — Retake Lateness"
        db.commit()

        db.add_all([
            LateSubmissionToken(curriculum_upload_id=upload.id),
            LateSubmissionToken(curriculum_upload_id=upload.id),
        ])
        db.commit()

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )

        # ── Attempt 1: scheduled -> active -> expired (window missed) ───
        attempt1 = upload_service._schedule_entry_assessment(
            curriculum, curriculum.target_completion_date
        )

        from app.jobs.send_assessment_job import send_assessment_job
        from app.jobs.expire_assessment_job import expire_assessment_job

        monkeypatch.setattr("app.jobs.send_assessment_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.send_assessment_job._llm", FakeLLM())
        monkeypatch.setattr("app.jobs.send_assessment_job._email", NoopEmailAdapter())
        send_assessment_job(attempt1.id)

        monkeypatch.setattr("app.jobs.expire_assessment_job.SessionLocal", TestSessionLocal)
        expire_assessment_job(attempt1.id)

        db.expire_all()
        attempt1 = db.get(Assessment, attempt1.id)
        assert attempt1.status == AssessmentStatus.expired

        # ── Attempt 1: late submission (token spent) -> fails grading ────
        submission_svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission1 = submission_svc.create(
            assessment_id=attempt1.id,
            token=attempt1.submission_token,
            submission_type=SubmissionType.text,
            text_content="Late but incorrect answer.",
        )
        db.expire_all()
        attempt1 = db.get(Assessment, attempt1.id)
        assert attempt1.status == AssessmentStatus.late_submitted
        assert LateTokenService(db).get_balance(upload.id) == 1  # one token spent

        token_used_on_attempt1 = (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id == attempt1.id)
            .first()
        )
        assert token_used_on_attempt1 is not None

        from app.jobs.grade_submission_job import grade_submission_job

        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLMBelowThreshold())
        monkeypatch.setattr("app.jobs.grade_submission_job._email", NoopEmailAdapter())
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
        )
        grade_submission_job(submission1.id)

        db.expire_all()
        attempt1 = db.get(Assessment, attempt1.id)
        assert attempt1.status == AssessmentStatus.completed
        assert attempt1.submission.grade.mastery_score < get_settings().mastery_threshold

        # A FREE retake must have been created — attempt_number 2, no token
        # spent for it (cap is attempt-number-based, not token-based).
        db.expire_all()
        all_attempts = (
            db.query(Assessment)
            .filter_by(curriculum_id=curriculum.id)
            .order_by(Assessment.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in all_attempts] == [1, 2]
        attempt2 = all_attempts[1]
        assert attempt2.assessment_text is not None  # retest content generated eagerly
        assert LateTokenService(db).get_balance(upload.id) == 1  # unchanged — retake is free

        # ── Attempt 2: on-time submission against its OWN new window ─────
        attempt2.status = AssessmentStatus.active
        db.commit()

        submission2 = submission_svc.create(
            assessment_id=attempt2.id,
            token=attempt2.submission_token,
            submission_type=SubmissionType.text,
            text_content="Correct answer, submitted on time this attempt.",
        )
        db.expire_all()
        attempt2 = db.get(Assessment, attempt2.id)
        assert attempt2.status == AssessmentStatus.submitted  # on-time, not late
        assert LateTokenService(db).get_balance(upload.id) == 1  # still unchanged

        # ── Attempt 2: passes grading ─────────────────────────────────────
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        grade_submission_job(submission2.id)

        db.expire_all()
        attempt2 = db.get(Assessment, attempt2.id)
        assert attempt2.status == AssessmentStatus.completed
        assert attempt2.submission.grade.mastery_score >= get_settings().mastery_threshold

        # No third attempt — attempt 2 passed.
        db.expire_all()
        assert (
            db.query(Assessment).filter_by(curriculum_id=curriculum.id).count() == 2
        )

        # ── Transcript: the FINAL graded record is attempt 2's — a
        # DIFFERENT assessment_id than the one the token was spent
        # against. The lateness lookup must still find it. ────────────────
        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        row = content.chapter_groups[0].rows[0]

        assert row.was_late is True
        assert "LATE" in row.status_label
        assert "GRADED" in row.status_label
        assert row.retake_note is not None and row.retake_note.startswith("retake, was")
        assert row.points == attempt2.submission.grade.score_earned

        get_settings.cache_clear()

    def test_token_used_on_different_entry_does_not_mark_this_entry_late(
        self, db, monkeypatch
    ):
        """Scoping confirmation (carried over): attempt_ids in
        transcript_service._row_for() must be scoped to THIS curriculum
        entry's own attempts only -- never another entry's, even one in the
        same curriculum_upload. Curriculum.assessments is FK'd to the
        specific Curriculum.id (one entry), not to CurriculumUpload.id, so
        this is structurally guaranteed by the schema -- this test proves
        it holds at the transcript-output level, not just by reading the
        relationship definition.

        Two entries, same upload: entry A has a token spent against its
        attempt; entry B is graded with no token anywhere near it. Entry
        B's transcript row must NOT show (LATE).
        """
        monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
        from app.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setattr("app.services.transcript_service.datetime", _FixedDateTime)

        seed_prompt_templates(db)

        upload = CurriculumUpload(source_filename="cross_entry_scoping_test.json")
        db.add(upload)
        db.commit()
        db.refresh(upload)

        entry_a = make_curriculum(
            db, topic="Entry A", entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        entry_a.upload_id, entry_a.max_marks, entry_a.chapter_label = (
            upload.id, 50.0, "Chapter 1 — A",
        )
        db.commit()
        assessment_a, _ = make_assessment(db, entry_a, status=AssessmentStatus.completed)
        submission_a = make_submission(db, assessment_a)
        make_grade(db, submission_a, mastery_score=90.0, score_earned=45.0, max_marks=50.0)
        db.add(LateSubmissionToken(used_at=FIXED_NOW, used_by_assessment_id=assessment_a.id))
        db.commit()

        entry_b = make_curriculum(
            db, topic="Entry B", entry_type=CurriculumEntryType.assessment,
            target_completion_date=TARGET_COMPLETION_DATE,
        )
        entry_b.upload_id, entry_b.max_marks, entry_b.chapter_label = (
            upload.id, 50.0, "Chapter 2 — B",
        )
        db.commit()
        assessment_b, _ = make_assessment(db, entry_b, status=AssessmentStatus.completed)
        submission_b = make_submission(db, assessment_b)
        make_grade(db, submission_b, mastery_score=88.0, score_earned=44.0, max_marks=50.0)

        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 2

        rows_by_topic = {
            row.topic: row for group in content.chapter_groups for row in group.rows
        }
        row_a, row_b = rows_by_topic["Entry A"], rows_by_topic["Entry B"]

        assert row_a.was_late is True
        assert "LATE" in row_a.status_label

        assert row_b.was_late is False, (
            "entry B must not inherit entry A's token -- attempt_ids in "
            "_row_for() must scope to this curriculum entry's own attempts, "
            "not the whole upload"
        )
        assert "LATE" not in row_b.status_label

        get_settings.cache_clear()
