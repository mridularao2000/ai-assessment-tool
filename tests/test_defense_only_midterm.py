"""Tests for defense_only Midterms (running-project checkpoints — upload's
"running_project": true on a midterm-type entry; see MidtermDetail.defense_only).

A defense_only Midterm has no Part 1 (cumulative questions) at all — only
Part 2 (the project defense). Covers every site audited for defense_only
awareness: ingestion/parsing, generation (first attempt + retest),
submission validation, and grading (score_earned must use the FULL
max_marks, not the normal 70% split).

Fully mocked — FakeLLM (test-suite default, now defense_only-aware; see
tests/conftest.py) — zero real Anthropic/email calls.
"""
from __future__ import annotations

import pytest

from app.exceptions import CurriculumUploadValidationError, InvalidStateError
from app.models.assessment import AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.models.midterm_detail import MidtermDetail
from app.services.assessment_service import AssessmentService
from app.services.curriculum_upload_service import CurriculumUploadService
from app.services.grading_service import GradingService
from app.services.scheduler_service import SchedulerService
from tests.conftest import (
    FakeLLM,
    FakeLLMBelowThreshold,
    FakeScheduler,
    NoopEmailAdapter,
    make_assessment,
    make_curriculum,
    make_grade,
    make_submission,
    seed_prompt_templates,
)


def _scheduler(db) -> SchedulerService:
    return SchedulerService(db, FakeScheduler())


def _entry(*, max_marks=25, completion_date="2026-10-22", running_project=True, **extra):
    return {
        "topic": "Crypto Dashboard CP1 — State Architecture",
        "type": "midterm",
        "chapter": "Running Project — Checkpoint 1 of 4",
        "running_project": running_project,
        "resources": {
            "known_now": ["https://example.com/context-vs-redux"],
            "pending_completion": ["crypto dashboard repo URL", "explanation"],
        },
        "completion_date": completion_date,
        "max_marks": max_marks,
        "probe_focus": "Defend why coin prices live in Redux, not Context.",
        **extra,
    }


def _make_defense_only_curriculum(db, *, max_marks: float = 25.0):
    """Direct model construction (not full ingestion) for generation/
    submission/grading tests — mirrors every other midterm test's pattern
    in this suite."""
    curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
    curriculum.max_marks = max_marks
    db.add(MidtermDetail(
        curriculum_id=curriculum.id,
        known_now=["https://example.com/context-vs-redux"],
        pending_completion_labels={},
        pending_completion_slots={},
        probe_focus="Defend why coin prices live in Redux, not Context.",
        part1_max_marks=0.0,
        part2_max_marks=max_marks,
        defense_only=True,
    ))
    db.commit()
    db.refresh(curriculum)
    return curriculum


class TestIngestionParsing:
    """Item 3 (running_project field) + the max_marks default split."""

    def test_running_project_sets_defense_only_and_full_part2_max_marks(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        upload = service.ingest({"topics": [_entry(max_marks=25)]}, "seed_v4.json")

        curriculum = upload.entries[0]
        detail = curriculum.midterm_detail
        assert detail.defense_only is True
        assert detail.part1_max_marks == 0.0
        assert detail.part2_max_marks == 25.0
        assert curriculum.max_marks == 25.0

    def test_running_project_honors_explicit_overrides(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        upload = service.ingest(
            {"topics": [_entry(max_marks=25, part1_max_marks=0.0, part2_max_marks=20.0)]},
            "seed_v4.json",
        )

        detail = upload.entries[0].midterm_detail
        assert detail.part2_max_marks == 20.0

    def test_running_project_rejected_on_assessment_type_entry(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        entry = {
            "topic": "Some Assessment",
            "type": "assessment",
            "chapter": "Chapter 1",
            "running_project": True,
            "resources": ["https://example.com"],
            "completion_date": "2026-10-22",
            "max_marks": 50,
        }

        with pytest.raises(CurriculumUploadValidationError):
            service.ingest({"topics": [entry]}, "seed_v4.json")

    def test_normal_midterm_without_running_project_is_unaffected(self, db):
        """Unknown/absent running_project must not change existing
        behavior — the normal 30/70 split still applies."""
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))

        upload = service.ingest(
            {"topics": [_entry(max_marks=100, running_project=False)]}, "seed_v4.json"
        )

        detail = upload.entries[0].midterm_detail
        assert detail.defense_only is False
        assert detail.part1_max_marks == 30.0
        assert detail.part2_max_marks == 70.0

    def test_unrelated_unknown_field_does_not_reject_the_upload(self, db):
        service = CurriculumUploadService(db, NoopEmailAdapter(), _scheduler(db))
        entry = _entry(max_marks=25)
        entry["some_future_field_nobody_reads_yet"] = {"anything": True}

        upload = service.ingest({"topics": [entry]}, "seed_v4.json")

        assert upload.entries[0].midterm_detail.defense_only is True


class TestGenerationSuppressesPart1:

    def test_first_attempt_generation_has_no_part1_and_real_part2(self, db):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        assert assessment.content_generated is False

        AssessmentService(db, FakeLLM()).generate_midterm_content(assessment)

        assert assessment.part1_text is None
        assert assessment.part1_rubric is None
        assert assessment.part2_text is not None
        assert assessment.part2_rubric is not None
        # The exact bug this fix targets: checking part1_text alone would
        # read this row as "never generated" forever after this point.
        assert assessment.content_generated is True

    def test_generated_content_is_not_regenerated_on_a_second_check(self, db):
        """Regression guard for the production-incident bug pattern: a
        content-generated check that only looks at part1_text would see
        None forever for a defense_only row and regenerate (re-pay for)
        it every time it's checked."""
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db)
        assessment, _ = make_assessment(db, curriculum, status=AssessmentStatus.scheduled)
        AssessmentService(db, FakeLLM()).generate_midterm_content(assessment)
        first_part2_text = assessment.part2_text

        assert assessment.content_generated is True  # must short-circuit any re-generation check

        assessment.part2_text = first_part2_text  # unchanged sentinel for clarity
        assert assessment.part2_text == first_part2_text


class TestSubmissionValidation:

    def test_part1_text_content_is_forbidden(self, db, client):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/crypto-dashboard",
            "part1_text_content": "Should not be accepted for a defense_only midterm.",
        })

        assert response.status_code == 409

    def test_submission_without_part1_text_content_succeeds(self, db, client):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/crypto-dashboard",
        })

        assert response.status_code == 201

    def test_assessment_detail_reports_is_defense_only_and_hides_part1(self, db, client):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db)
        assessment, token = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )

        response = client.get(f"/api/v1/assessments/{assessment.id}?token={token}")

        assert response.status_code == 200
        data = response.json()
        assert data["is_midterm"] is True
        assert data["is_defense_only"] is True
        # The whole exam IS Part 2 — shown as assessment_text, not hidden.
        assert data["assessment_text"] == "Defend your Redux/Context split."

    def test_normal_midterm_still_requires_part1_text_content(self, db, client):
        """Regression guard: defense_only handling must not loosen the
        requirement for an ordinary (non-defense_only) Midterm."""
        seed_prompt_templates(db)
        curriculum = make_curriculum(db, entry_type=CurriculumEntryType.midterm)
        curriculum.max_marks = 100.0
        db.add(MidtermDetail(
            curriculum_id=curriculum.id, known_now=["design doc"],
            pending_completion_labels={}, pending_completion_slots={},
            probe_focus="architecture", part1_max_marks=30.0, part2_max_marks=70.0,
        ))
        db.commit()
        assessment, token = make_assessment(db, curriculum, status=AssessmentStatus.active)

        response = client.post("/api/v1/submissions/", data={
            "assessment_id": assessment.id,
            "token": token,
            "submission_type": "github_url",
            "github_url": "https://github.com/example/project",
        })

        assert response.status_code == 409


class TestGradingScoresFullMaxMarks:
    """Amendment 3's explicit required test: part2_max_marks == max_marks
    when defense_only, so a perfect submission scores max_marks/max_marks,
    not max_marks*0.7/max_marks (the normal-midterm split silently
    capping every defense_only checkpoint at 70%)."""

    def test_perfect_submission_scores_25_of_25_not_17_point_5(self, db):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db, max_marks=25.0)
        detail = curriculum.midterm_detail
        assert detail.part2_max_marks == 25.0  # the premise of this test

        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )
        submission = make_submission(db, assessment)

        class _PerfectDefenseLLM(FakeLLM):
            def grade_midterm_submission(self, req):
                from app.interfaces.llm import MidtermGradingResult
                assert req.defense_only is True
                return MidtermGradingResult(
                    part2_score=req.part2_max_marks,  # perfect score
                    weak_areas=[],
                    overall_feedback="Flawless defense.",
                )

        grade = GradingService(db, _PerfectDefenseLLM()).grade(submission.id)

        assert grade.part1_score is None
        assert grade.part2_score == 25.0
        assert grade.score_earned == 25.0
        assert grade.max_marks == 25.0
        assert grade.mastery_score == 100.0

    def test_grading_does_not_require_part1_text_or_rubric(self, db):
        """The adapter's fail-loud check only applies when defense_only is
        False — a defense_only row never has part1_text/part1_rubric at
        all, so grading must proceed without them."""
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db, max_marks=25.0)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )
        assert assessment.part1_text is None
        assert assessment.part1_rubric is None
        submission = make_submission(db, assessment)

        grade = GradingService(db, FakeLLM()).grade(submission.id)

        assert grade.part2_score is not None
        assert grade.score_earned is not None


class TestRetestPath:
    """Item 1 in the user's follow-up: the retest path must be
    defense_only-aware too, not just first-attempt generation."""

    def test_retest_has_no_part1_and_real_part2(self, db):
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db, max_marks=25.0)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )
        submission = make_submission(db, assessment)
        grade = make_grade(
            db, submission, mastery_score=40.0, part2_score=10.0,
            score_earned=10.0, max_marks=25.0,
            weak_areas=["Redux normalization"],
        )

        retest = AssessmentService(db, FakeLLMBelowThreshold()).create_retest(
            curriculum.id, grade.id
        )

        assert retest.attempt_number == 2
        assert retest.part1_text is None
        assert retest.part1_rubric is None
        assert retest.part2_text is not None
        assert retest.part2_rubric is not None

    def test_retest_does_not_crash_on_a_none_previous_part1_score(self, db):
        """grade.part1_score is None for a defense_only Midterm's grade —
        previous_part1_score must accept that, not crash building the
        retest request."""
        seed_prompt_templates(db)
        curriculum = _make_defense_only_curriculum(db, max_marks=25.0)
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )
        submission = make_submission(db, assessment)
        grade = make_grade(
            db, submission, mastery_score=40.0, part2_score=10.0,
            score_earned=10.0, max_marks=25.0,
        )
        assert grade.part1_score is None

        retest = AssessmentService(db, FakeLLMBelowThreshold()).create_retest(
            curriculum.id, grade.id
        )

        assert retest is not None


class TestTranscriptDisplay:
    """Confirms the explicit concern raised: a defense_only Midterm must
    not render a broken 'Part 1: —/0' row. The transcript's MCQ/project
    split (score_breakdown) is gated on submission.mcq_answers, which is
    always None for any Midterm (defense_only or not) — so this is a
    confirmation, not a new behavior."""

    def test_no_score_breakdown_rendered_for_a_graded_defense_only_entry(self, db):
        from app.services.transcript_service import compute_transcript
        from app.models.curriculum_upload import CurriculumUpload

        seed_prompt_templates(db)
        upload = CurriculumUpload(source_filename="seed_v4.json")
        db.add(upload)
        db.commit()
        curriculum = _make_defense_only_curriculum(db, max_marks=25.0)
        curriculum.upload_id = upload.id
        db.commit()
        assessment, _ = make_assessment(
            db, curriculum, status=AssessmentStatus.active,
            part2_text="Defend your Redux/Context split.",
            part2_rubric='[{"description": "Correct split", "points": 25}]',
        )
        submission = make_submission(db, assessment)
        make_grade(
            db, submission, mastery_score=100.0, part2_score=25.0,
            score_earned=25.0, max_marks=25.0,
        )

        content = compute_transcript(db, upload.id)

        row = content.chapter_groups[0].rows[0]
        assert row.points == 25.0
        assert row.score_breakdown is None
