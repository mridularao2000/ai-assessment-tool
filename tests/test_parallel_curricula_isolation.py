"""Two real curriculum_uploads, coexisting with deliberately overlapping
completion dates, walked through interleaved lifecycle events. Proves zero
cross-contamination in: token balances, GPA, transcript contents, and every
email type's content — not inferred from the single-curriculum suite, but
demonstrated with two curricula actually coexisting in one test.

Curriculum One: window missed -> late token spent -> graded (mirrors the
single-curriculum lifecycle test's path).
Curriculum Two: on-time exam -> graded, no token ever touched.

Both entries share the exact same target_completion_date, so their
scheduling windows genuinely overlap in time, not just in theory.
"""
from __future__ import annotations

from datetime import date, datetime

from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.late_submission_token import LateSubmissionToken
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.gpa_service import compute_gpa
from app.services.late_token_service import LateTokenService
from app.services.scheduler_service import SchedulerService
from app.models.submission import SubmissionType
from app.services.submission_service import SubmissionService
from app.services.transcript_service import compute_transcript
from tests.conftest import (
    FakeLLM,
    FakeScheduler,
    NoopEmailAdapter,
    RecordingEmailAdapter,
    TestSessionLocal,
    make_curriculum,
    seed_prompt_templates,
)

TEST_PRIMARY = "mridula.sureshrao@proton.me"
FIXED_NOW = datetime(2026, 8, 20, 12, 0, 0)
SHARED_COMPLETION_DATE = date(2026, 8, 10)  # both curricula's window overlaps


class _FixedDateTime(datetime):
    @classmethod
    def utcnow(cls):
        return FIXED_NOW


def _make_entry(db, upload, topic, chapter_label):
    curriculum = make_curriculum(
        db, topic=topic, entry_type=CurriculumEntryType.assessment,
        target_completion_date=SHARED_COMPLETION_DATE,
    )
    curriculum.upload_id = upload.id
    curriculum.max_marks = 50.0
    curriculum.chapter_label = chapter_label
    db.commit()
    return curriculum


class TestParallelCurriculaIsolation:

    def test_two_curricula_coexisting_zero_cross_contamination(self, db, monkeypatch):
        monkeypatch.setenv("USER_EMAIL", TEST_PRIMARY)
        from app.config import get_settings
        get_settings.cache_clear()

        monkeypatch.setattr("app.services.transcript_service.datetime", _FixedDateTime)
        monkeypatch.setattr("app.services.submission_service.utcnow", lambda: FIXED_NOW)

        seed_prompt_templates(db)

        upload_one = CurriculumUpload(source_filename="curriculum_one.json")
        upload_two = CurriculumUpload(source_filename="curriculum_two.json")
        db.add_all([upload_one, upload_two])
        db.commit()
        db.refresh(upload_one)
        db.refresh(upload_two)

        entry_one = _make_entry(db, upload_one, "Curriculum One Topic", "Chapter 1 — One")
        entry_two = _make_entry(db, upload_two, "Curriculum Two Topic", "Chapter 1 — Two")

        # Each curriculum's OWN independent pool, granted separately —
        # proves grant_monthly's per-pool iteration, not a shared top-up.
        token_svc = LateTokenService(db)
        token_svc.grant_monthly(upload_one.id)
        token_svc.grant_monthly(upload_two.id)
        assert token_svc.get_balance(upload_one.id) == 2
        assert token_svc.get_balance(upload_two.id) == 2

        upload_service = CurriculumUploadService(
            db, NoopEmailAdapter(), SchedulerService(db, FakeScheduler())
        )
        assessment_one = upload_service._schedule_entry_assessment(
            entry_one, entry_one.target_completion_date
        )
        assessment_two = upload_service._schedule_entry_assessment(
            entry_two, entry_two.target_completion_date
        )

        recording = RecordingEmailAdapter()
        from app.jobs.send_assessment_job import send_assessment_job
        from app.jobs.expire_assessment_job import expire_assessment_job
        from app.jobs.grade_submission_job import grade_submission_job

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

        # Interleaved on purpose: send both exams, expire only One's, so any
        # shared/global state would show up as cross-talk between them.
        send_assessment_job(assessment_one.id)
        send_assessment_job(assessment_two.id)
        expire_assessment_job(assessment_one.id)

        db.expire_all()
        assessment_one = db.get(Assessment, assessment_one.id)
        assessment_two = db.get(Assessment, assessment_two.id)
        assert assessment_one.status == AssessmentStatus.expired
        assert assessment_two.status == AssessmentStatus.active

        # ── Curriculum One: late submission, spends from ITS pool only ──
        submission_svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission_one = submission_svc.create(
            assessment_id=assessment_one.id,
            token=assessment_one.submission_token,
            submission_type=SubmissionType.text,
            text_content="Late answer, curriculum one.",
        )
        db.expire_all()
        assert token_svc.get_balance(upload_one.id) == 1  # One's pool: spent
        assert token_svc.get_balance(upload_two.id) == 2  # Two's pool: untouched

        # ── Curriculum Two: on-time submission, never touches any token ──
        submission_two = submission_svc.create(
            assessment_id=assessment_two.id,
            token=assessment_two.submission_token,
            submission_type=SubmissionType.text,
            text_content="On-time answer, curriculum two.",
        )
        db.expire_all()
        assert token_svc.get_balance(upload_one.id) == 1  # unchanged by Two's activity
        assert token_svc.get_balance(upload_two.id) == 2  # unchanged — no token needed

        grade_submission_job(submission_one.id)
        grade_submission_job(submission_two.id)

        # ── Token balances: still fully isolated after grading ──────────
        assert token_svc.get_balance(upload_one.id) == 1
        assert token_svc.get_balance(upload_two.id) == 2
        spent_token = (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id == assessment_one.id)
            .first()
        )
        assert spent_token is not None
        assert spent_token.curriculum_upload_id == upload_one.id
        assert (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id == assessment_two.id)
            .first()
            is None
        )

        # ── GPA: each upload's own score only, no bleed ──────────────────
        gpa_one = compute_gpa(db, upload_one.id)
        gpa_two = compute_gpa(db, upload_two.id)
        assert gpa_one.total_earned == 45.0  # FakeLLM: 90% of 50.0
        assert gpa_one.total_max == 50.0
        assert gpa_one.graded_count == 1
        assert gpa_two.total_earned == 45.0
        assert gpa_two.total_max == 50.0
        assert gpa_two.graded_count == 1
        # Same numeric values by coincidence (same FakeLLM score) -- the
        # real proof is each is computed from ONLY its own upload's rows.
        assert compute_gpa(db, upload_one.id).total_max == 50.0  # not 100.0 (both entries)

        # ── Transcript: each upload shows only its own entry ─────────────
        content_one = compute_transcript(db, upload_one.id)
        content_two = compute_transcript(db, upload_two.id)
        assert content_one.resolved_count == 1
        assert content_two.resolved_count == 1

        topics_one = {r.topic for g in content_one.chapter_groups for r in g.rows}
        topics_two = {r.topic for g in content_two.chapter_groups for r in g.rows}
        assert topics_one == {"Curriculum One Topic"}
        assert topics_two == {"Curriculum Two Topic"}
        assert "Curriculum Two Topic" not in topics_one
        assert "Curriculum One Topic" not in topics_two

        row_one = content_one.chapter_groups[0].rows[0]
        row_two = content_two.chapter_groups[0].rows[0]
        assert row_one.was_late is True    # One really was late
        assert row_two.was_late is False   # Two was never late

        # ── Every email captured: content never crosses over ─────────────
        assessment_topics = {c.topic for c in recording.assessment_calls}
        results_topics = {c.topic for c in recording.results_calls}
        assert assessment_topics == {"Curriculum One Topic", "Curriculum Two Topic"}
        assert results_topics == {"Curriculum One Topic", "Curriculum Two Topic"}

        transcript_calls_by_upload = {c.upload_id: c for c in recording.transcript_calls}
        assert set(transcript_calls_by_upload) == {upload_one.id, upload_two.id}
        for upload_id, call in transcript_calls_by_upload.items():
            all_topics_in_call = {
                r.topic for g in call.entry_groups for r in g.rows
            }
            if upload_id == upload_one.id:
                assert all_topics_in_call == {"Curriculum One Topic"}
            else:
                assert all_topics_in_call == {"Curriculum Two Topic"}

        get_settings.cache_clear()
