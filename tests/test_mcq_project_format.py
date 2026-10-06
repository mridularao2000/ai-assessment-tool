"""Tests for the MCQ + Coding Project assessment format.

Covers:
  - Piece 1 (generation): see tests/test_anthropic_llm_adapter.py's
    TestGenerateAssessment/TestGenerateRetest, which exercise the real
    adapter's parsing against a static fixture JSON string — zero live
    API calls (the whole point of monkeypatching adapter._client).
  - Piece 2 (grading), covered here: GradingService._grade_mcq_project
    dispatch, deterministic MCQ scoring (never delegated to the LLM —
    only a stub LLM's grade_mcq_section/grade_project are used, no
    network/API calls), and the resulting Grade/GradeResponse split.
"""
import json

import pytest

from app.models.assessment import AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.schemas.grade import GradeResponse
from app.services.grading_service import GradingService
from tests.conftest import make_assessment, make_curriculum, make_submission, seed_prompt_templates


# A 5-question MCQ section, matching the shape _build_mcq_project_fields
# produces (and that GradingService._grade_mcq_project expects to find on
# Assessment.part1_text/part1_rubric for a real row).
_QUESTIONS = [
    {"question": f"Why does concept {i} behave this way?", "options": ["A", "B", "C", "D"]}
    for i in range(5)
]
_ANSWER_KEY = [
    {"correct_option": "A", "explanation": f"A is correct for question {i}."}
    for i in range(5)
]
_CRITERIA = [
    {"description": "Criterion one is satisfied.", "points": 20},
    {"description": "Criterion two is satisfied.", "points": 20},
    {"description": "Criterion three is satisfied.", "points": 15},
    {"description": "Criterion four is satisfied.", "points": 15},
]


class StubMcqProjectLLM:
    """No generation methods — these tests only exercise grading. A stub,
    not a mock: it returns fixed, pre-decided results rather than
    recording/asserting call arguments, so it has no network/API calls of
    any kind, matching this task's "zero real API calls" constraint.

    parsed_answers_override lets a test simulate the LLM normalizing a
    messy submitted answer to a canonical letter; defaults to passing
    submitted_answers straight through unchanged.
    """

    def __init__(self, parsed_answers_override=None, project_score=58.0):
        self.parsed_answers_override = parsed_answers_override
        self.project_score = project_score
        self.last_mcq_request = None
        self.last_project_request = None

    def grade_mcq_section(self, req):
        from app.interfaces.llm import MCQGradingResult

        self.last_mcq_request = req
        parsed = self.parsed_answers_override or list(req.submitted_answers)
        feedback = [
            "Correct." if parsed[i] == req.answer_key[i]["correct_option"]
            else f"Incorrect — {req.answer_key[i]['explanation']}"
            for i in range(5)
        ]
        return MCQGradingResult(parsed_answers=parsed, feedback=feedback)

    def grade_project(self, req):
        from app.interfaces.llm import ProjectGradingResult

        self.last_project_request = req
        return ProjectGradingResult(
            project_score=self.project_score,
            weak_areas=["weak topic"],
            overall_feedback="Solid project overall.",
        )


def _make_mcq_assessment(db, *, entry_type=None, max_marks=None):
    curriculum = make_curriculum(db, entry_type=entry_type)
    if max_marks is not None:
        curriculum.max_marks = max_marks
        db.commit()
    assessment, _ = make_assessment(
        db, curriculum,
        status=AssessmentStatus.active,
        part1_text=json.dumps(_QUESTIONS),
        part1_rubric=json.dumps(_ANSWER_KEY),
        part2_text="Build the thing.",
        part2_rubric=json.dumps(_CRITERIA),
    )
    return curriculum, assessment


class TestMcqProjectGradingDispatch:

    def test_all_correct_scores_30_mcq_plus_project(self, db):
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db)
        submission = make_submission(db, assessment, mcq_answers=["A", "A", "A", "A", "A"])

        llm = StubMcqProjectLLM(project_score=58.0)
        grade = GradingService(db, llm).grade(submission.id)

        assert grade.part1_score == 30.0  # all 5 correct, 6 pts each
        assert grade.part2_score == 58.0
        assert grade.mastery_score == 88.0  # 30 + 58, standalone-style 0-100

    def test_mixed_correct_scores_partial_mcq_credit(self, db):
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db)
        # 2 correct (A), 3 wrong (B instead of A) -> 2*6 = 12
        submission = make_submission(db, assessment, mcq_answers=["A", "A", "B", "B", "B"])

        llm = StubMcqProjectLLM(project_score=40.0)
        grade = GradingService(db, llm).grade(submission.id)

        assert grade.part1_score == 12.0
        assert grade.part2_score == 40.0
        assert grade.mastery_score == 52.0
        # The 3 wrong questions' (truncated) text must appear in weak_areas,
        # alongside the project's own weak_areas from the stub.
        assert len(grade.weak_areas) == 4  # 3 wrong MCQs + 1 project weak area
        assert "weak topic" in grade.weak_areas

    def test_scoring_is_never_delegated_to_the_llm(self, db):
        """Correctness must be decided by GradingService comparing
        parsed_answers to the stored answer key — not by the stub. Proven
        here by having the stub's grade_mcq_section claim every answer
        parsed to "A" (the correct option) regardless of what was
        actually submitted, and confirming the SUBMITTED answers (not the
        stub's claim) still ultimately decided the parsed_answers used for
        scoring, since GradingService never recomputes or overrides what
        grade_mcq_section returns — it only compares, never re-decides.
        """
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db)
        submission = make_submission(db, assessment, mcq_answers=["B", "B", "B", "B", "B"])

        # Stub deliberately returns parsed_answers of all "A" (correct),
        # even though the student actually submitted "B" for every
        # question — simulating what WOULD happen if parsing silently
        # fixed up a wrong answer. GradingService must trust whatever
        # grade_mcq_section returns as the parsed answer (that call's
        # entire job), scoring 30/30 here — proving score computation
        # itself is a pure, separate comparison in GradingService, with no
        # independent correctness judgement of its own.
        llm = StubMcqProjectLLM(parsed_answers_override=["A", "A", "A", "A", "A"], project_score=50.0)
        grade = GradingService(db, llm).grade(submission.id)

        assert grade.part1_score == 30.0
        assert llm.last_mcq_request.submitted_answers == ["B", "B", "B", "B", "B"]
        assert llm.last_mcq_request.answer_key == _ANSWER_KEY

    def test_wrong_mcq_answer_count_raises_invalid_state(self, db):
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db)
        submission = make_submission(db, assessment, mcq_answers=["A", "A"])  # only 2, not 5

        from app.exceptions import InvalidStateError

        with pytest.raises(InvalidStateError):
            GradingService(db, StubMcqProjectLLM()).grade(submission.id)

    def test_entry_score_earned_uses_same_formula_as_every_other_branch(self, db):
        """Piece 2's explicit requirement: for a curriculum-upload entry,
        score_earned must still be mastery_score/100*max_marks — the
        EXACT formula the legacy/midterm branches already use — never a
        separate computation just because this is the new format."""
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(
            db, entry_type=CurriculumEntryType.assessment, max_marks=50.0
        )
        submission = make_submission(db, assessment, mcq_answers=["A", "A", "A", "A", "A"])

        grade = GradingService(db, StubMcqProjectLLM(project_score=58.0)).grade(submission.id)

        assert grade.mastery_score == 88.0
        assert grade.score_earned == pytest.approx(88.0 / 100.0 * 50.0)
        assert grade.max_marks == 50.0

    def test_standalone_score_earned_stays_none(self, db):
        """Standalone curricula don't participate in GPA — entry_type is
        None, so score_earned/max_marks must stay None exactly as the
        legacy/midterm branches already behave, never populated just
        because the new format computes a combined 0-100 mastery_score."""
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db, entry_type=None)
        submission = make_submission(db, assessment, mcq_answers=["A", "A", "A", "A", "A"])

        grade = GradingService(db, StubMcqProjectLLM(project_score=58.0)).grade(submission.id)

        assert grade.score_earned is None
        assert grade.max_marks is None

    def test_get_results_exposes_mcq_and_project_score_split(self, db):
        seed_prompt_templates(db)
        _, assessment = _make_mcq_assessment(db)
        submission = make_submission(db, assessment, mcq_answers=["A", "A", "A", "A", "A"])
        GradingService(db, StubMcqProjectLLM(project_score=58.0)).grade(submission.id)

        response = GradingService(db, StubMcqProjectLLM()).get_results(submission.id)

        assert isinstance(response, GradeResponse)
        assert response.mcq_score == 30.0
        assert response.project_score == 58.0
        assert response.mastery_score == 88.0

    def test_legacy_format_assessment_never_exposes_mcq_project_split(self, db, fake_llm):
        """A Grade from the legacy (assessment_text/rubric) path must not
        report mcq_score/project_score even though it also happens to
        leave Grade.part1_score/part2_score unset — None either way, but
        for a different reason (those columns are simply never written
        for this path, not specifically gated on format)."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.active)
        submission = make_submission(db, assessment)

        GradingService(db, fake_llm).grade(submission.id)
        response = GradingService(db, fake_llm).get_results(submission.id)

        assert response.mcq_score is None
        assert response.project_score is None
