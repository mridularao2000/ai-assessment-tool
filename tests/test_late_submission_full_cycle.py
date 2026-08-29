"""Full late-submission-token lifecycle, end to end, fully mocked (FakeLLM +
FakeScheduler + Recording/NoopEmailAdapter — zero real Anthropic/email calls
anywhere in this file).

Unlike tests/test_late_send.py (which exercises AssessmentService.
trigger_late_send() in isolation, one state transition at a time) and
tests/test_full_entry_lifecycle_integration.py's existing
test_token_used_on_failed_attempt_then_free_retake_passes_still_shows_late
(which proves the transcript's attempt_ids/token-lookup fix by hand-inserting
LateSubmissionToken rows and calling SubmissionService.create() directly),
this file drives the WHOLE sequence through the real, user-facing entry
point — trigger_late_send() — then a real submission, real (fake-graded)
failure, real free retake, real (fake-graded) pass — for both an
Assessment-type entry and a Midterm-type one, to prove the retake-cap +
token-accounting + transcript-lateness mechanism is generic across entry
types, not just re-proven in isolation.

Also covers the permanently-missed terminal state (due-date month fully
elapsed, token never spent) for an Assessment-type entry.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.config import get_settings
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.models.submission import SubmissionType
from app.services.assessment_service import AssessmentService
from app.services.email_service import EmailService
from app.services.gpa_service import compute_gpa
from app.services.late_token_service import LateTokenService
from app.services.submission_service import SubmissionService
from app.services.transcript_service import MISSED_NO_SCORE, compute_transcript, display_status
from tests.conftest import (
    FakeLLM,
    FakeLLMBelowThreshold,
    FakeScheduler,
    RecordingEmailAdapter,
    TestSessionLocal,
    make_assessment,
    make_curriculum,
    seed_prompt_templates,
)


def _make_upload(db, filename: str) -> CurriculumUpload:
    upload = CurriculumUpload(source_filename=filename)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _patch_grading_job(monkeypatch, llm):
    monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
    monkeypatch.setattr("app.jobs.grade_submission_job._llm", llm)
    monkeypatch.setattr(
        "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: FakeScheduler()
    )


class TestAssessmentFullLateCycle:
    """Items 1-3: a synthetic Assessment-type entry, late-triggered via
    token, failed, freely retaken, and passed — with the transcript
    correctly showing (LATE) at the end."""

    def _make_entry(self, db):
        upload = _make_upload(db, "late_cycle_assessment.json")
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id = upload.id
        curriculum.max_marks = 50.0
        curriculum.chapter_label = "Chapter 2 — Late Cycle"
        db.commit()
        # due 2 days ago, current calendar month — token-eligible.
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2,
        )
        # No content yet — the retroactive shape trigger_late_send() must
        # fill in via FakeLLM before anything can be submitted against it.
        assessment.assessment_text = None
        assessment.rubric = None
        db.commit()
        return upload, curriculum, assessment, token

    def test_trigger_fail_retake_pass_full_cycle(self, db, monkeypatch):
        seed_prompt_templates(db)
        upload, curriculum, assessment, token = self._make_entry(db)
        LateTokenService(db).grant_monthly(upload.id)
        balance_before = LateTokenService(db).get_balance(upload.id)
        assert balance_before == 2

        # ── Item 1: trigger late send, then actually submit ─────────────
        result = AssessmentService(db, FakeLLM()).trigger_late_send(
            curriculum.id, EmailService(db, RecordingEmailAdapter())
        )
        assert result.assessment_text is not None  # fake-generated content
        # The trigger itself must not spend anything — only a real
        # submission does (see AssessmentService.trigger_late_send).
        assert LateTokenService(db).get_balance(upload.id) == balance_before

        submission_svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission1 = submission_svc.create(
            assessment_id=assessment.id,
            token=token,
            submission_type=SubmissionType.text,
            text_content="Late but incorrect answer.",
        )

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.late_submitted
        assert LateTokenService(db).get_balance(upload.id) == balance_before - 1

        # ── Item 2: fails grading -> free retake, no 2nd token spent ────
        _patch_grading_job(monkeypatch, FakeLLMBelowThreshold())
        from app.jobs.grade_submission_job import grade_submission_job

        grade_submission_job(submission1.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.completed
        assert assessment.submission.grade.mastery_score < get_settings().mastery_threshold

        attempts = (
            db.query(Assessment)
            .filter_by(curriculum_id=curriculum.id)
            .order_by(Assessment.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in attempts] == [1, 2]
        attempt2 = attempts[1]
        assert LateTokenService(db).get_balance(upload.id) == balance_before - 1  # unchanged

        # ── Item 3: passes the retake -> transcript shows (LATE) ────────
        attempt2.status = AssessmentStatus.active
        db.commit()
        submission2 = submission_svc.create(
            assessment_id=attempt2.id,
            token=attempt2.submission_token,
            submission_type=SubmissionType.text,
            text_content="Correct answer, submitted on time this attempt.",
        )
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        grade_submission_job(submission2.id)

        db.expire_all()
        attempt2 = db.get(Assessment, attempt2.id)
        assert attempt2.status == AssessmentStatus.completed
        assert attempt2.submission.grade.mastery_score >= get_settings().mastery_threshold
        assert (
            db.query(Assessment).filter_by(curriculum_id=curriculum.id).count() == 2
        )  # cap reached, no 3rd attempt

        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        row = content.chapter_groups[0].rows[0]
        assert row.was_late is True
        assert "LATE" in row.status_label
        assert "GRADED" in row.status_label
        assert row.retake_note is not None and row.retake_note.startswith("retake, was")
        assert row.points == attempt2.submission.grade.score_earned

    def test_permanently_missed_once_due_date_month_elapses(self, db):
        """A separate synthetic entry: due-date month fully elapses with the
        token never used. Must resolve to a clean terminal state on its
        own — no job needs to run to "finalize" it, since display_status()
        recomputes live from the current date every time it's read."""
        upload = _make_upload(db, "missed_forever_assessment.json")
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id = upload.id
        curriculum.max_marks = 50.0
        curriculum.chapter_label = "Chapter 9 — Missed Forever"
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2,
        )
        # Force due_date into an unambiguously elapsed previous calendar month.
        assessment.due_date = datetime.utcnow().replace(day=1) - timedelta(days=1)
        db.commit()
        # No grant_monthly() call — token never used, never even available.

        assert display_status(db, curriculum) == MISSED_NO_SCORE

        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        row = content.chapter_groups[0].rows[0]
        assert row.status_label == "MISSED"
        assert row.points is None

        summary = compute_gpa(db, upload.id)
        assert summary.total_max == pytest.approx(50.0)
        assert summary.total_earned == 0.0
        assert summary.missed_count == 1


class TestMidtermFullLateCycle:
    """Repeats items 1-4 for a synthetic Midterm whose Assessment row
    already exists (i.e. its resources were filled and it was scheduled
    normally, then the exam itself went unsubmitted past its window) —
    this shares the EXACT SAME due-date/calendar-month mechanism as an
    Assessment-type entry (display_status()'s `AssessmentStatus.expired`
    branch doesn't distinguish entry_type), and trigger_late_send()
    already branches for midterm content generation, so the whole
    sequence below is the same code path, just two-part content.

    NOT covered here (a genuinely different mechanism, already covered in
    tests/test_midterm_monthly_deadline.py): a midterm that never got an
    Assessment row at all because its project resources
    (pending_completion) were never filled in time. That path is gated by
    target_completion_date's month via
    CurriculumUploadService.check_and_clear_hold(), not by
    Assessment.due_date via trigger_late_send()/SubmissionService — a
    midterm stuck in resources_hold never reaches trigger_late_send() in
    the first place, since no Assessment exists yet to trigger against.
    """

    def _make_entry(self, db):
        upload = _make_upload(db, "late_cycle_midterm.json")
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.upload_id = upload.id
        curriculum.max_marks = 100.0
        curriculum.chapter_label = "Midterm X — Late Cycle"
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["cumulative design principles"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2,
        )
        # Not yet generated — the retroactive-midterm shape.
        assessment.assessment_text = None
        assessment.rubric = None
        assessment.part1_text = None
        assessment.part1_rubric = None
        assessment.part2_text = None
        assessment.part2_rubric = None
        db.commit()
        return upload, curriculum, assessment, token

    def test_trigger_fail_retake_pass_full_cycle(self, db, monkeypatch):
        seed_prompt_templates(db)
        upload, curriculum, assessment, token = self._make_entry(db)
        LateTokenService(db).grant_monthly(upload.id)
        balance_before = LateTokenService(db).get_balance(upload.id)
        assert balance_before == 2

        # ── Item 1: trigger late send (midterm branch), then submit ────
        result = AssessmentService(db, FakeLLM()).trigger_late_send(
            curriculum.id, EmailService(db, RecordingEmailAdapter())
        )
        assert result.part1_text is not None  # fake-generated, two-part
        assert result.part2_text is not None
        assert LateTokenService(db).get_balance(upload.id) == balance_before

        submission_svc = SubmissionService(db, FakeScheduler(), LateTokenService(db))
        submission1 = submission_svc.create(
            assessment_id=assessment.id,
            token=token,
            submission_type=SubmissionType.text,
            text_content="Late project submission, incomplete.",
            part1_text_content="Late Part 1 answer.",
        )

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.late_submitted
        assert LateTokenService(db).get_balance(upload.id) == balance_before - 1

        # ── Item 2: fails grading -> free two-part retake, no 2nd token ─
        _patch_grading_job(monkeypatch, FakeLLMBelowThreshold())
        from app.jobs.grade_submission_job import grade_submission_job

        grade_submission_job(submission1.id)

        db.expire_all()
        assessment = db.get(Assessment, assessment.id)
        assert assessment.status == AssessmentStatus.completed
        assert assessment.submission.grade.mastery_score < get_settings().mastery_threshold

        attempts = (
            db.query(Assessment)
            .filter_by(curriculum_id=curriculum.id)
            .order_by(Assessment.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in attempts] == [1, 2]
        attempt2 = attempts[1]
        assert attempt2.part1_text is not None  # regenerated two-part retest
        assert attempt2.part2_text is not None
        assert LateTokenService(db).get_balance(upload.id) == balance_before - 1  # unchanged

        # ── Item 3: passes the retake -> transcript shows (LATE) ────────
        attempt2.status = AssessmentStatus.active
        db.commit()
        submission2 = submission_svc.create(
            assessment_id=attempt2.id,
            token=attempt2.submission_token,
            submission_type=SubmissionType.text,
            text_content="Complete, correct project submission.",
            part1_text_content="Correct Part 1 answer this time.",
        )
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", FakeLLM())
        grade_submission_job(submission2.id)

        db.expire_all()
        attempt2 = db.get(Assessment, attempt2.id)
        assert attempt2.status == AssessmentStatus.completed
        assert attempt2.submission.grade.mastery_score >= get_settings().mastery_threshold
        assert db.query(Assessment).filter_by(curriculum_id=curriculum.id).count() == 2

        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        row = content.chapter_groups[0].rows[0]
        assert row.was_late is True
        assert "LATE" in row.status_label
        assert "GRADED" in row.status_label
        assert row.points == attempt2.submission.grade.score_earned

    def test_permanently_missed_once_due_date_month_elapses(self, db):
        """Item 4 for a midterm whose Assessment row already exists — same
        due-date/calendar-month mechanism as the Assessment-type case."""
        upload = _make_upload(db, "missed_forever_midterm.json")
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.upload_id = upload.id
        curriculum.max_marks = 100.0
        curriculum.chapter_label = "Midterm Y — Missed Forever"
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["cumulative design principles"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-2,
        )
        assessment.due_date = datetime.utcnow().replace(day=1) - timedelta(days=1)
        db.commit()

        assert display_status(db, curriculum) == MISSED_NO_SCORE

        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 1
        row = content.chapter_groups[0].rows[0]
        assert row.status_label == "MISSED"
        assert row.points is None

        summary = compute_gpa(db, upload.id)
        assert summary.total_max == pytest.approx(100.0)
        assert summary.total_earned == 0.0
        assert summary.missed_count == 1
