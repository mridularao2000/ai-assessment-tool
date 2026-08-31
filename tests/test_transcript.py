"""Section 7 — Transcript.

Covers:
  - Only resolved entries (Graded / Missed, either flavor) appear — the
    explicit reversal from "never omit any entry" to "only show due-date-
    passed entries"; Not Yet Due / Held / Exam Sent are excluded
  - Chapter grouping + row numbering (CH{N}-{letter} for Assessments,
    MT-{letter} for Midterms, whose chapter_label isn't a "Chapter N"
    pattern), mirroring syllabus_builder's own chapter ordering
  - A retake shows "(retake, was X.XX)" using the FIRST attempt's points,
    and only the final attempt's points/credit count
  - Footer totals (credits/points/GPA) match gpa_service.compute_gpa exactly
  - "Course Material" is captured once at syllabus-send time and stays
    frozen even after pending_completion_slots are filled in later
  - Flow (e): transcript email fires after results email, entries only,
    non-fatal, via grade_submission_job
"""
from __future__ import annotations

import pytest

from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.services.email_service import EmailService
from app.services.transcript_service import NEEDS_MANUAL_DIAGNOSIS, compute_transcript, display_status
from tests.conftest import (
    RecordingEmailAdapter,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
)


def _make_upload(db, source_filename="test.json") -> CurriculumUpload:
    upload = CurriculumUpload(source_filename=source_filename)
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _graded_entry(db, upload, *, topic, chapter_label, max_marks=50.0, score_earned=45.0):
    curriculum = make_curriculum(db, topic=topic, entry_type=CurriculumEntryType.assessment)
    curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
        upload.id, max_marks, chapter_label,
    )
    db.commit()
    a, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed)
    s = make_submission(db, a)
    make_grade(db, s, mastery_score=90.0, score_earned=score_earned, max_marks=max_marks)
    return curriculum


class TestOnlyResolvedEntriesShown:

    def test_not_yet_due_and_held_are_excluded(self, db):
        upload = _make_upload(db)

        future = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        future.upload_id, future.max_marks, future.chapter_label = upload.id, 50.0, "Chapter 1 — X"
        db.commit()
        make_assessment(db, future, status=AssessmentStatus.scheduled, due_offset_days=30)

        held = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        held.upload_id, held.max_marks, held.resources_hold, held.chapter_label = (
            upload.id, 100.0, True, "Chapters 1-3 — Capstone",
        )
        db.commit()

        content = compute_transcript(db, upload.id)

        assert content.chapter_groups == []
        assert content.resolved_count == 0
        assert content.total_entry_count == 2

    def test_needs_manual_diagnosis_is_distinct_from_exam_sent_and_excluded(self, db):
        """Real incident this covers: needs_manual_diagnosis used to fall
        through display_status()'s catch-all into EXAM_SENT — showing
        "Exam Sent" for a row where nothing was ever generated or sent at
        all (Performance Optimization I, 2026-08-31). It must get its own
        distinct label, and — same as Exam Sent — stay excluded from the
        resolved-only transcript, since it isn't a final outcome."""
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 6 — X",
        )
        db.commit()
        make_assessment(db, curriculum, status=AssessmentStatus.needs_manual_diagnosis)

        assert display_status(db, curriculum) == NEEDS_MANUAL_DIAGNOSIS
        content = compute_transcript(db, upload.id)
        assert content.resolved_count == 0
        assert content.chapter_groups == []

    def test_graded_entry_is_shown(self, db):
        upload = _make_upload(db)
        _graded_entry(db, upload, topic="JS Internals", chapter_label="Chapter 1 — JS")

        content = compute_transcript(db, upload.id)

        assert content.resolved_count == 1
        assert content.chapter_groups[0].rows[0].status_label == "GRADED"

    def test_missed_late_eligible_is_shown_with_count(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 1 — X",
        )
        db.commit()
        from datetime import datetime
        due_date = datetime.utcnow().replace(day=1, hour=0, minute=1)
        a, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired)
        a.due_date = due_date
        db.commit()

        content = compute_transcript(db, upload.id)

        assert content.resolved_count == 1
        assert content.chapter_groups[0].rows[0].status_label.startswith("MISSED–LATE")
        assert content.chapter_groups[0].rows[0].points is None


class TestChapterGroupingAndRowNumbering:

    def test_rows_are_lettered_within_chapter_in_date_order(self, db):
        upload = _make_upload(db)
        first = _graded_entry(db, upload, topic="A Topic", chapter_label="Chapter 1 — X",
                               score_earned=40.0)
        first.target_completion_date = first.target_completion_date.replace(day=1)
        second = _graded_entry(db, upload, topic="B Topic", chapter_label="Chapter 1 — X",
                                score_earned=45.0)
        second.target_completion_date = second.target_completion_date.replace(day=5)
        db.commit()

        content = compute_transcript(db, upload.id)

        assert len(content.chapter_groups) == 1
        row_ids = [r.row_id for r in content.chapter_groups[0].rows]
        assert row_ids == ["CH1-A", "CH1-B"]

    def test_chapters_ordered_numerically_then_unparseable_last(self, db):
        upload = _make_upload(db)
        _graded_entry(db, upload, topic="Ch3 topic", chapter_label="Chapter 3 — Y")
        _graded_entry(db, upload, topic="Ch1 topic", chapter_label="Chapter 1 — X")

        content = compute_transcript(db, upload.id)

        labels = [g.chapter_label for g in content.chapter_groups]
        assert labels == ["Chapter 1 — X", "Chapter 3 — Y"]

    def test_midterm_row_id_uses_mt_prefix(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        curriculum.chapter_label = "Chapters 1-6 (cumulative) — Capstone"
        db.add(MidtermDetail(
            curriculum_id=curriculum.id, known_now=[], pending_completion_labels={},
            pending_completion_slots={}, probe_focus=None,
            part1_max_marks=30.0, part2_max_marks=70.0,
        ))
        db.commit()
        a, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.completed,
            part1_text="p1", part1_rubric="r1", part2_text="p2", part2_rubric="r2",
        )
        s = make_submission(db, a, part1_text_content="answer")
        make_grade(db, s, mastery_score=90.0, part1_score=27.0, part2_score=63.0,
                   score_earned=90.0, max_marks=100.0)

        content = compute_transcript(db, upload.id)

        assert content.chapter_groups[0].rows[0].row_id == "MT-A"


class TestRetakeAndTotals:

    def test_retake_shows_prior_score_and_only_final_counts(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks, curriculum.chapter_label = (
            upload.id, 50.0, "Chapter 6 — Perf",
        )
        db.commit()

        a1, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed, attempt_number=1)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=76.0, score_earned=38.0, max_marks=50.0)

        a2, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed, attempt_number=2)
        s2 = make_submission(db, a2)
        make_grade(db, s2, mastery_score=92.0, score_earned=46.0, max_marks=50.0)

        content = compute_transcript(db, upload.id)

        row = content.chapter_groups[0].rows[0]
        assert row.retake_note == "retake, was 38.00"
        assert row.points == pytest.approx(46.0)
        assert content.total_credits == pytest.approx(50.0)  # not double-counted
        assert content.total_points == pytest.approx(46.0)

    def test_footer_totals_match_compute_gpa(self, db):
        from app.services.gpa_service import compute_gpa

        upload = _make_upload(db)
        _graded_entry(db, upload, topic="A", chapter_label="Chapter 1 — X", score_earned=45.0)
        _graded_entry(db, upload, topic="B", chapter_label="Chapter 2 — Y", score_earned=30.0)

        content = compute_transcript(db, upload.id)
        gpa_summary = compute_gpa(db, upload.id)

        assert content.total_credits == gpa_summary.total_max
        assert content.total_points == gpa_summary.total_earned
        assert content.gpa == gpa_summary.gpa
        assert content.graded_count == 2


class TestCourseMaterialFrozenSnapshot:

    def test_snapshot_captured_once_and_not_regenerated(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        curriculum.chapter_label = "Chapters 1-3 — Capstone"
        detail = MidtermDetail(
            curriculum_id=curriculum.id, known_now=["design doc"],
            pending_completion_labels={"repo_url": "repo URL"},
            pending_completion_slots={"repo_url": None},
            probe_focus=None, part1_max_marks=30.0, part2_max_marks=70.0,
        )
        db.add(detail)
        db.commit()

        # First syllabus send captures the snapshot — slot is still pending.
        EmailService(db, RecordingEmailAdapter()).send_syllabus_email(upload.id)

        db.expire_all()
        upload_after_first = db.get(CurriculumUpload, upload.id)
        assert upload_after_first.course_material_snapshot is not None
        first_pending = upload_after_first.course_material_snapshot["midterms"][0]["pending_status"]
        assert first_pending == [["repo URL", False]]

        # Now the slot gets filled in — a real transcript build afterward
        # must NOT reflect this, per the frozen-snapshot design.
        curriculum.midterm_detail.pending_completion_slots = {"repo_url": "https://github.com/x/y"}
        db.commit()

        # Sending the syllabus email again must NOT overwrite the snapshot.
        EmailService(db, RecordingEmailAdapter()).send_syllabus_email(upload.id)
        db.expire_all()

        content = compute_transcript(db, upload.id)
        still_pending = content.course_material["midterms"][0]["pending_status"]
        assert still_pending == [["repo URL", False]]  # unchanged — frozen


class TestTranscriptEmailFlow:

    def test_transcript_email_fires_after_results_for_entries(self, db, monkeypatch):
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

        assert len(recording.results_calls) == 1
        assert len(recording.transcript_calls) == 1
        assert recording.transcript_calls[0].upload_id == upload.id

    def test_transcript_email_not_sent_for_standalone(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLM, FakeScheduler, TestSessionLocal, seed_prompt_templates

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)  # entry_type=None
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

        assert len(recording.results_calls) == 1
        assert recording.transcript_calls == []


class TestTranscriptEmailRendering:
    """Direct check against the real Resend adapter's HTML output — not
    inferred from method call order — for the explicit spec requirement
    'make the transcript the first item in the email'."""

    def test_transcript_table_renders_before_course_material_section(self, monkeypatch):
        from app.adapters.resend_email import ResendEmailAdapter
        from app.interfaces.email import TranscriptEmailData
        from app.services.transcript_service import TranscriptChapterGroup, TranscriptEntryRow

        monkeypatch.setenv("RESEND_API_KEY", "test-key")
        from app.config import get_settings
        get_settings.cache_clear()
        adapter = ResendEmailAdapter()
        sent = {}
        monkeypatch.setattr("resend.Emails.send", lambda params: sent.update(params))

        adapter.send_transcript_email(TranscriptEmailData(
            recipient_emails=["a@example.com"],
            upload_id="u1",
            source_filename="curriculum.json",
            entry_groups=[
                TranscriptChapterGroup(
                    chapter_label="Chapter 1 — X",
                    rows=[TranscriptEntryRow(
                        row_id="CH1-A", topic="JS Internals", chapter_number=1,
                        max_marks=50.0, status_label="GRADED", points=47.0,
                    )],
                )
            ],
            resolved_count=1, total_entry_count=1, graded_count=1,
            total_credits=50.0, total_points=47.0, gpa=94.0,
            course_material={"chapters": [
                {"chapter_label": "Chapter 1 — X", "no_standalone_note": None,
                 "assessments": [{"topic": "JS Internals", "resources": ["mdn.org"]}]}
            ], "midterms": []},
            course_material_captured_at=None,
        ))
        get_settings.cache_clear()

        html = sent["html"]
        assert "Record of Entries" in html
        assert "Course Material" in html
        assert html.index("Record of Entries") < html.index("Course Material")
        assert "JS Internals" in html  # transcript row rendered
        assert "mdn.org" in html       # course-material resource rendered


class TestTranscriptRoute:

    def test_transcript_endpoint_returns_content(self, client, db):
        upload = _make_upload(db)
        _graded_entry(db, upload, topic="A", chapter_label="Chapter 1 — X")

        response = client.get(f"/api/v1/curriculum-uploads/{upload.id}/transcript")

        assert response.status_code == 200
        data = response.json()
        assert data["resolved_count"] == 1
        assert data["chapter_groups"][0]["rows"][0]["status_label"] == "GRADED"

    def test_transcript_endpoint_404_for_unknown_upload(self, client, db):
        response = client.get("/api/v1/curriculum-uploads/does-not-exist/transcript")
        assert response.status_code == 404
