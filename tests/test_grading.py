"""C + F: Grading tests and database state verification.

Covers:
  - Text submission grading via service (happy path)
  - File submission grading via service (reads from disk)
  - GitHub URL submission handling (IngestionError path)
  - GradingService.grade() persists Grade and sets assessment.status=completed
  - GradingService.get_results() via API (200 / 404)
  - passed=True when mastery_score >= threshold (default 85.0)
  - passed=False when mastery_score < threshold
  - Grade not found → 404
  - Grading a non-submitted assessment → InvalidStateError
"""

import io

import pytest

from app.exceptions import InvalidStateError, NotFoundError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.grade import Grade
from app.models.submission import Submission, SubmissionType
from app.services.grading_service import GradingService
from tests.conftest import (
    FakeLLM,
    FakeLLMBelowThreshold,
    TestSessionLocal,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
    seed_prompt_templates,
)


class TestGradingService:

    def test_grade_text_submission_returns_grade(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        grade = GradingService(db, fake_llm).grade(submission.id)

        assert grade.mastery_score == 90.0
        assert grade.overall_feedback == "Excellent understanding demonstrated."
        assert isinstance(grade.weak_areas, list)

    def test_grade_persists_grade_row(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        GradingService(db, fake_llm).grade(submission.id)

        db.expire_all()
        grade = db.query(Grade).filter_by(submission_id=submission.id).first()
        assert grade is not None
        assert grade.mastery_score == 90.0

    def test_grade_sets_assessment_status_completed(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        GradingService(db, fake_llm).grade(submission.id)

        db.expire_all()
        refreshed = db.get(Assessment, assessment.id)
        assert refreshed.status == AssessmentStatus.completed

    def test_grade_non_submitted_assessment_raises_invalid_state(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        # Assessment status is 'active', not 'submitted'
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        # Create submission row but leave status as active (bypass service)
        from app.models.submission import Submission
        submission = Submission(
            id="manual-submission-id",
            assessment_id=assessment.id,
            submission_type=SubmissionType.text,
            text_content="Test content.",
        )
        db.add(submission)
        db.commit()

        with pytest.raises(InvalidStateError):
            GradingService(db, fake_llm).grade(submission.id)

    def test_grade_missing_submission_raises_not_found(self, db, fake_llm):
        seed_prompt_templates(db)

        with pytest.raises(NotFoundError):
            GradingService(db, fake_llm).grade("does-not-exist")

    def test_grade_below_threshold_recorded_correctly(self, db):
        seed_prompt_templates(db)
        llm = FakeLLMBelowThreshold()
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        grade = GradingService(db, llm).grade(submission.id)

        assert grade.mastery_score == 70.0
        assert "event loop internals" in grade.weak_areas

    def test_grade_stores_weak_areas_for_retest(self, db):
        seed_prompt_templates(db)
        llm = FakeLLMBelowThreshold()
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        GradingService(db, llm).grade(submission.id)

        db.expire_all()
        grade = db.query(Grade).filter_by(submission_id=submission.id).first()
        assert grade.weak_areas == ["event loop internals", "coroutine lifecycle"]

    def test_grade_file_submission(self, db, tmp_path, monkeypatch, fake_llm):
        from app.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("UPLOADS_DIR", str(tmp_path))

        try:
            seed_prompt_templates(db)
            curriculum = make_curriculum(db)
            assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)

            # Write a file to the tmp uploads dir
            filename = "submission.py"
            (tmp_path / filename).write_text("def solve(): return 42", encoding="utf-8")

            # Create submission with file_path pointing to that file
            submission = Submission(
                id="file-submission-id",
                assessment_id=assessment.id,
                submission_type=SubmissionType.file,
                file_path=filename,
            )
            assessment.status = AssessmentStatus.submitted
            db.add(submission)
            db.commit()

            grade = GradingService(db, fake_llm).grade(submission.id)

            assert grade.mastery_score == 90.0
        finally:
            get_settings.cache_clear()

    def test_grade_github_url_submission_raises_ingestion_error(self, db, fake_llm):
        from app.exceptions import IngestionError

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)

        submission = Submission(
            id="github-submission-id",
            assessment_id=assessment.id,
            submission_type=SubmissionType.github_url,
            github_url="https://github.com/example/nonexistent-repo",
        )
        assessment.status = AssessmentStatus.submitted
        db.add(submission)
        db.commit()

        # github_ingestor doesn't exist / can't fetch → IngestionError
        with pytest.raises(IngestionError):
            GradingService(db, fake_llm).grade(submission.id)

    def test_grade_uses_grading_prompt_template(self, db, fake_llm):
        """GradingService fetches the 'grading' PromptTemplate — missing it raises NotFoundError."""
        # Deliberately omit 'grading' template
        from app.models.prompt_template import PromptTemplate
        import uuid
        for slug in ("assessment_generation", "retest_generation", "reschedule_classification"):
            db.add(PromptTemplate(id=str(uuid.uuid4()), slug=slug, version="1.0",
                                  body=f"body for {slug}", is_active=True))
        db.commit()

        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        with pytest.raises(NotFoundError, match="grading"):
            GradingService(db, fake_llm).grade(submission.id)


class TestGradingResults:

    def test_get_results_returns_grade_data(self, client, db):
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=90.0)

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        assert response.status_code == 200
        data = response.json()
        assert data["mastery_score"] == 90.0
        assert data["overall_feedback"] == "Well done."

    def test_get_results_passed_true_when_above_threshold(self, client, db):
        """Default mastery_threshold=85.0; score 90.0 → passed=True."""
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=90.0)

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        assert response.json()["passed"] is True

    def test_get_results_passed_false_when_below_threshold(self, client, db):
        """Score 70.0 < threshold 85.0 → passed=False."""
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=70.0, weak_areas=["event loop"])

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        data = response.json()
        assert data["passed"] is False
        assert data["weak_areas"] == ["event loop"]

    def test_get_results_at_exact_threshold_passes(self, client, db):
        """Score equal to threshold (85.0) → passed=True (>=)."""
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        make_grade(db, submission, mastery_score=85.0)

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        assert response.json()["passed"] is True

    def test_get_results_missing_grade_returns_404(self, client, db):
        response = client.get("/api/v1/submissions/does-not-exist/results")

        assert response.status_code == 404

    def test_get_results_no_grade_yet_returns_404(self, client, db):
        """Submission exists but grading hasn't run yet → 404 from get_results."""
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        # Deliberately do NOT create a grade

        response = client.get(f"/api/v1/submissions/{submission.id}/results")

        assert response.status_code == 404


class TestGradingDatabaseState:

    def test_one_grade_per_submission(self, db, fake_llm):
        """Grade table has UNIQUE constraint on submission_id."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        GradingService(db, fake_llm).grade(submission.id)

        db.expire_all()
        count = db.query(Grade).filter_by(submission_id=submission.id).count()
        assert count == 1

    def test_graded_assessment_id_matches_submission(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        grade = GradingService(db, fake_llm).grade(submission.id)

        db.expire_all()
        persisted = db.get(Grade, grade.id)
        assert persisted.submission_id == submission.id

    def test_grading_result_records_prompt_template_id(self, db, fake_llm):
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        grade = GradingService(db, fake_llm).grade(submission.id)

        db.expire_all()
        persisted = db.get(Grade, grade.id)
        # grading_prompt_id should reference the 'grading' template
        assert persisted.grading_prompt_id is not None
