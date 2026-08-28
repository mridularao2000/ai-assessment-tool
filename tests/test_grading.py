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
from app.models.curriculum import Curriculum, CurriculumEntryType, CurriculumStatus
from app.models.grade import Grade
from app.models.submission import Submission, SubmissionType
from app.services.grading_service import GradingService
from tests.conftest import (
    FakeLLM,
    FakeLLMBelowThreshold,
    FakeLLMRetestFails,
    FakeScheduler,
    RecordingEmailAdapter,
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


class TestRetakeCap:
    """grade_submission_job: cap of 2 total attempts (original + 1 retake),
    regardless of score. Both create_retest() and mark_mastery() existed but
    had zero callers before this wiring — these tests exercise the job
    function directly (SessionLocal/_llm/get_scheduler_adapter monkeypatched
    to the test DB/fakes, since the job opens its own session rather than
    going through FastAPI's dependency-injected one).
    """

    def _patch_job(self, monkeypatch, llm, fake_scheduler):
        # _email is already forced to a no-op by the autouse _no_real_email_in_jobs
        # fixture in conftest.py — see that fixture's docstring for why this matters.
        monkeypatch.setattr("app.jobs.grade_submission_job.SessionLocal", TestSessionLocal)
        monkeypatch.setattr("app.jobs.grade_submission_job._llm", llm)
        monkeypatch.setattr(
            "app.jobs.grade_submission_job.get_scheduler_adapter", lambda: fake_scheduler
        )

    def test_below_threshold_first_attempt_schedules_retest(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), fake_scheduler)

        grade_submission_job(submission.id)

        db.expire_all()
        assessments = (
            db.query(Assessment)
            .filter_by(curriculum_id=curriculum.id)
            .order_by(Assessment.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in assessments] == [1, 2]
        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1

    def test_retest_generation_failure_does_not_block_results_email(self, db, monkeypatch):
        """Regression test: create_retest() raising (e.g. the model exhausting
        its token budget on tool calls, surfaced as LLMValidationError) must
        not crash the whole job. Before the fix, this exception propagated
        out of grade_submission_job uncaught, skipping the results/transcript
        email sends below it and leaving no retest AND no notification —
        the grade was recorded but everything downstream silently vanished.
        """
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMRetestFails(), fake_scheduler)
        email = RecordingEmailAdapter()
        monkeypatch.setattr("app.jobs.grade_submission_job._email", email)

        grade_submission_job(submission.id)  # must not raise

        db.expire_all()
        # Grade is still recorded despite the downstream retest failure.
        grade = db.query(Grade).filter_by(submission_id=submission.id).first()
        assert grade is not None
        assert grade.mastery_score == 70.0
        # No retest was created and no job was scheduled for one.
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1
        assert fake_scheduler.schedule_assessment_jobs_calls == []
        # The results email still fired — this is the actual bug fix.
        assert len(email.results_calls) == 1

    def test_below_threshold_second_attempt_reaches_cap_no_third(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        assessment.attempt_number = 2
        db.commit()
        submission = make_submission(db, assessment)
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), fake_scheduler)

        grade_submission_job(submission.id)

        db.expire_all()
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1  # no attempt 3 created
        assert fake_scheduler.schedule_assessment_jobs_calls == []

    def test_passing_first_attempt_marks_mastery_no_retest(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLM(), fake_scheduler)  # FakeLLM grades 90.0, passes

        grade_submission_job(submission.id)

        db.expire_all()
        refreshed = db.get(Curriculum, curriculum.id)
        assert refreshed.mastery_achieved is True
        assert refreshed.status == CurriculumStatus.complete
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1
        assert fake_scheduler.schedule_assessment_jobs_calls == []

    def test_passing_first_attempt_marks_mastery_for_assessment_type_entry(self, db, monkeypatch):
        """mark_mastery() is generic (no entry_type branch) and correct for
        entries by construction, but was never actually exercised end-to-end
        with an entry-type curriculum until now — only the standalone path
        above was covered."""
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.assessment)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLM(), fake_scheduler)  # FakeLLM grades 90.0, passes

        grade_submission_job(submission.id)

        db.expire_all()
        refreshed = db.get(Curriculum, curriculum.id)
        assert refreshed.mastery_achieved is True
        assert refreshed.status == CurriculumStatus.complete
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1
        assert fake_scheduler.schedule_assessment_jobs_calls == []

    def _make_midterm_curriculum(self, db):
        from app.models.midterm_detail import MidtermDetail

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.max_marks = 100.0
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["design doc"],
            pending_completion_labels={},
            pending_completion_slots={},
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        db.refresh(curriculum)
        return curriculum

    def test_below_threshold_midterm_first_attempt_schedules_two_part_retest(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLMBelowThreshold

        seed_prompt_templates(db)
        curriculum = self._make_midterm_curriculum(db)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), fake_scheduler)

        grade_submission_job(submission.id)

        db.expire_all()
        assessments = (
            db.query(Assessment)
            .filter_by(curriculum_id=curriculum.id)
            .order_by(Assessment.attempt_number)
            .all()
        )
        assert [a.attempt_number for a in assessments] == [1, 2]
        retest = assessments[1]
        assert retest.part1_text is not None
        assert retest.part2_text is not None
        assert retest.assessment_text is None
        assert len(fake_scheduler.schedule_assessment_jobs_calls) == 1

    def test_passing_midterm_first_attempt_marks_mastery_no_retest(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = self._make_midterm_curriculum(db)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLM(), fake_scheduler)  # 90% of each part, passes

        grade_submission_job(submission.id)

        db.expire_all()
        refreshed = db.get(Curriculum, curriculum.id)
        assert refreshed.mastery_achieved is True
        assert refreshed.status == CurriculumStatus.complete
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1
        assert fake_scheduler.schedule_assessment_jobs_calls == []

    def test_midterm_second_attempt_reaches_cap_no_third(self, db, monkeypatch):
        from app.jobs.grade_submission_job import grade_submission_job
        from tests.conftest import FakeLLMBelowThreshold

        seed_prompt_templates(db)
        curriculum = self._make_midterm_curriculum(db)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active, attempt_number=2,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), fake_scheduler)

        grade_submission_job(submission.id)

        db.expire_all()
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1  # no attempt 3 created
        assert fake_scheduler.schedule_assessment_jobs_calls == []

    def test_late_attempt_counts_toward_same_cap(self, db, monkeypatch):
        """A late-token-covered attempt increments attempt_number identically
        to an on-time one, so the cap applies uniformly."""
        from app.jobs.grade_submission_job import grade_submission_job

        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        assessment.attempt_number = 2
        assessment.status = AssessmentStatus.late_submitted
        db.commit()
        submission = Submission(
            id="late-retake-submission",
            assessment_id=assessment.id,
            submission_type=SubmissionType.text,
            text_content="Late retake answer.",
        )
        db.add(submission)
        db.commit()
        fake_scheduler = FakeScheduler()
        self._patch_job(monkeypatch, FakeLLMBelowThreshold(), fake_scheduler)

        grade_submission_job(submission.id)

        db.expire_all()
        assessments = db.query(Assessment).filter_by(curriculum_id=curriculum.id).all()
        assert len(assessments) == 1  # still capped at 2 total, no attempt 3


class _CapturingMidtermLLM(FakeLLM):
    """Records the MidtermGradingRequest it received, so a test can assert
    on exactly what grading_service._grade_midterm() built and sent —
    still returns a valid canned result via FakeLLM, no real API call."""

    def __init__(self):
        self.last_request = None

    def grade_midterm_submission(self, req):
        self.last_request = req
        return super().grade_midterm_submission(req)


class TestMidtermGradingReadmeSeparation:
    """Regression test for the README/file-path spot-check wiring
    (grading_service.py::_grade_midterm): a pending_completion slot whose
    LABEL contains "readme" must be routed to
    MidtermGradingRequest.readme_content, not into `resources` — that list
    is treated as fetchable URLs/labels via build_resource_guidance
    (web_search + web_fetch instructions per entry), which is wrong for
    free-text prose like a submitted design writeup.

    This only proves the WIRING is correct — it can't prove (and doesn't
    try to prove) that the spot-check actually changes a real grade; that
    requires a real LLM call and was verified separately, once, outside
    the test suite (see the session notes / PR description).
    """

    def test_readme_labeled_slot_goes_to_readme_content_not_resources(self, db):
        from app.models.midterm_detail import MidtermDetail

        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.max_marks = 100.0
        db.add(MidtermDetail(
            curriculum_id=curriculum.id,
            known_now=["general OOP principles"],
            pending_completion_labels={
                "repo_url": "Repo URL",
                "readme_with_design_decisions": "README with design decisions",
            },
            pending_completion_slots={
                "repo_url": "https://github.com/octocat/Hello-World",
                "readme_with_design_decisions": "See src/auth/token.py, function verify_token().",
            },
            probe_focus="defend the design",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
        ))
        db.commit()
        db.refresh(curriculum)

        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part1_text="Part 1 exam", part1_rubric="Part 1 rubric",
            part2_text="Part 2 exam", part2_rubric="Part 2 rubric",
        )
        submission = make_submission(db, assessment, part1_text_content="Part 1 answer")
        seed_prompt_templates(db)

        capturing_llm = _CapturingMidtermLLM()
        GradingService(db, capturing_llm).grade(submission.id)

        req = capturing_llm.last_request
        assert req is not None
        assert req.readme_content == "See src/auth/token.py, function verify_token()."
        assert req.resources == ["general OOP principles", "https://github.com/octocat/Hello-World"]
        # The README's content must never leak into `resources`.
        assert not any("verify_token" in r for r in req.resources)
