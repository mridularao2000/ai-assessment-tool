"""Section 6 — Weighted GPA.

Covers:
  - GPA is weighted by max_marks (total_earned/total_max), not a simple
    average of per-entry percentages
  - Missed — No Score entries count as 0/max_marks in the denominator
  - Missed — Late-Eligible entries (still within this calendar month) are
    NOT counted yet — the outcome isn't final
  - Held and not-yet-due entries are excluded entirely
  - Only the FINAL attempt counts when an entry was retaken
  - Assessment-type and Midterm-type entries both feed the same GPA
"""
from __future__ import annotations

from datetime import datetime

import pytest

from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.services.gpa_service import compute_gpa
from tests.conftest import make_assessment, make_curriculum, make_grade, make_submission


def _make_upload(db) -> CurriculumUpload:
    upload = CurriculumUpload(source_filename="test.json")
    db.add(upload)
    db.commit()
    db.refresh(upload)
    return upload


def _attach(db, curriculum, upload):
    curriculum.upload_id = upload.id
    db.commit()


class TestWeightedFormula:

    def test_gpa_weighted_by_max_marks_not_averaged(self, db):
        """A 100/100 entry and a 50/50 entry average to 100% either way, so
        use unequal max_marks to distinguish weighting from averaging:
        60/100 and 45/50 average to (60%+90%)/2=75%, but weighted by marks
        it's (60+45)/(100+50)=70%."""
        upload = _make_upload(db)

        c1 = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        c1.upload_id, c1.max_marks = upload.id, 100.0
        a1, _ = make_assessment(db, c1, status=AssessmentStatus.completed)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=60.0, score_earned=60.0, max_marks=100.0)

        c2 = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        c2.upload_id, c2.max_marks = upload.id, 50.0
        a2, _ = make_assessment(db, c2, status=AssessmentStatus.completed)
        s2 = make_submission(db, a2)
        make_grade(db, s2, mastery_score=90.0, score_earned=45.0, max_marks=50.0)
        db.commit()

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == pytest.approx(105.0)
        assert summary.total_max == pytest.approx(150.0)
        assert summary.gpa == pytest.approx(70.0)
        assert summary.gpa != pytest.approx(75.0)  # would be the wrong, averaged answer
        assert summary.graded_count == 2


class TestMissedHandling:

    def test_missed_no_score_counts_as_zero_in_denominator(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        db.commit()

        # due_date over a year ago — unambiguously a past calendar month.
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.expired, due_offset_days=-400
        )

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == 0.0
        assert summary.total_max == pytest.approx(100.0)
        assert summary.gpa == 0.0
        assert summary.missed_count == 1
        assert summary.graded_count == 0

    def test_missed_late_eligible_not_counted_yet(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        db.commit()

        now = datetime.utcnow()
        due_date = now.replace(day=1, hour=0, minute=1, second=0, microsecond=0)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.expired)
        assessment.due_date = due_date
        db.commit()

        summary = compute_gpa(db, upload.id)

        assert summary.total_max == 0.0  # not counted — still a live outcome
        assert summary.missed_count == 0
        assert summary.graded_count == 0

    def test_held_and_not_yet_due_entries_excluded(self, db):
        upload = _make_upload(db)

        held = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        held.upload_id, held.max_marks, held.resources_hold = upload.id, 100.0, True
        db.commit()

        not_yet_due = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        not_yet_due.upload_id, not_yet_due.max_marks = upload.id, 100.0
        db.commit()
        make_assessment(db, not_yet_due, status=AssessmentStatus.scheduled, due_offset_days=30)

        summary = compute_gpa(db, upload.id)

        assert summary.total_max == 0.0
        assert summary.graded_count == 0
        assert summary.missed_count == 0


class TestFinalAttemptOnly:

    def test_only_final_attempt_counts_after_retake(self, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        db.commit()

        # Attempt 1: failed, 40/100.
        a1, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed, attempt_number=1)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=40.0, score_earned=40.0, max_marks=100.0)

        # Attempt 2 (retake): passed, 85/100 — this is the one that should count.
        a2, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed, attempt_number=2)
        s2 = make_submission(db, a2)
        make_grade(db, s2, mastery_score=85.0, score_earned=85.0, max_marks=100.0)

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == pytest.approx(85.0)
        assert summary.total_max == pytest.approx(100.0)  # not double-counted
        assert summary.graded_count == 1

    def test_unresolved_retake_excludes_stale_earlier_grade(self, db):
        """Regression test: an entry whose retake exists but hasn't been
        graded yet must not be counted at all — and specifically must not
        silently fall back to attempt 1's stale failing grade. Before the
        fix, compute_gpa() picked the latest attempt among GRADED ones only,
        so a fresh, still-ungraded attempt 2 was invisible to it and it
        counted attempt 1's grade instead — inconsistent with the
        transcript, which correctly excludes this entry entirely via
        _row_for()'s same "not yet resolved" rule."""
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        db.commit()

        # Attempt 1: failed, 40/100 — graded.
        a1, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed, attempt_number=1)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=40.0, score_earned=40.0, max_marks=100.0)

        # Attempt 2 (retake): generated, scheduled, NOT yet submitted/graded.
        make_assessment(db, curriculum, status=AssessmentStatus.active, attempt_number=2)

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == 0.0
        assert summary.total_max == 0.0  # NOT 100.0 — not counted yet, not attempt 1's stale grade
        assert summary.graded_count == 0


class TestMixedEntryTypes:

    def test_gpa_includes_both_assessment_and_midterm_entries(self, db):
        upload = _make_upload(db)

        assessment_c = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment_c.upload_id, assessment_c.max_marks = upload.id, 100.0
        a1, _ = make_assessment(db, assessment_c, status=AssessmentStatus.completed)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=80.0, score_earned=80.0, max_marks=100.0)

        midterm_c = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        midterm_c.upload_id, midterm_c.max_marks = upload.id, 100.0
        db.add(MidtermDetail(
            curriculum_id=midterm_c.id, known_now=[], pending_completion_labels={},
            pending_completion_slots={}, probe_focus=None,
            part1_max_marks=30.0, part2_max_marks=70.0,
        ))
        db.commit()
        a2, _ = make_assessment(db, midterm_c, status=AssessmentStatus.completed)
        s2 = make_submission(db, a2, part1_text_content="answer")
        make_grade(db, s2, mastery_score=90.0, part1_score=27.0, part2_score=63.0,
                   score_earned=90.0, max_marks=100.0)

        summary = compute_gpa(db, upload.id)

        assert summary.total_earned == pytest.approx(170.0)
        assert summary.total_max == pytest.approx(200.0)
        assert summary.gpa == pytest.approx(85.0)
        assert summary.graded_count == 2


class TestGPARoute:

    def test_gpa_endpoint_returns_summary(self, client, db):
        upload = _make_upload(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        curriculum.upload_id, curriculum.max_marks = upload.id, 100.0
        a1, _ = make_assessment(db, curriculum, status=AssessmentStatus.completed)
        s1 = make_submission(db, a1)
        make_grade(db, s1, mastery_score=75.0, score_earned=75.0, max_marks=100.0)

        response = client.get(f"/api/v1/curriculum-uploads/{upload.id}/gpa")

        assert response.status_code == 200
        data = response.json()
        assert data["gpa"] == pytest.approx(75.0)
        assert data["graded_count"] == 1

    def test_gpa_endpoint_404_for_unknown_upload(self, client, db):
        response = client.get("/api/v1/curriculum-uploads/does-not-exist/gpa")
        assert response.status_code == 404
