from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import IngestionError, InvalidStateError, NotFoundError
from app.interfaces.llm import (
    GradingRequest,
    LLMInterface,
    MCQGradingRequest,
    MidtermGradingRequest,
    ProjectGradingRequest,
    llm_log_context,
)
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.grade import Grade
from app.models.prompt_template import PromptTemplate
from app.models.submission import Submission, SubmissionType
from app.schemas.grade import GradeResponse


class GradingService:
    """Grades a submission and persists the result.

    Depends on:
      db  — SQLAlchemy session for all persistence
      llm — LLMInterface for grading

    Does NOT decide what happens after grading (mastery marking or retest
    scheduling). The caller — grade_submission_job — receives the Grade,
    checks mastery_score against settings.mastery_threshold, and then
    calls CurriculumService.mark_mastery() or AssessmentService.create_retest()
    as appropriate.
    """

    def __init__(self, db: Session, llm: LLMInterface) -> None:
        self.db = db
        self.llm = llm

    def grade(self, submission_id: str) -> Grade:
        """Resolve submission content, grade via LLM, and persist the result.

        Steps:
          1. Load Submission with its Assessment and the Assessment's Curriculum.
             Verify Assessment.status is submitted or late_submitted.
          2. Resolve submission_content from Submission.submission_type:
               github_url → github_ingestor.fetch_repo_content() (README +
                            any file paths referenced in text_content),
                            combined with text_content itself (the
                            student's explanation) when both are present —
                            see SubmissionType.github_url branch below.
               text       → submission.text_content (used directly)
               file       → read file bytes from submission.file_path on disk
             For a Midterm, this is Part 2's content — Part 1's is
             submission.part1_text_content directly (always plain text).
          3. Midterm-type curricula branch to _grade_midterm() (fetches the
             'midterm_grading' template, calls llm.grade_midterm_submission()
             with resources so Part 2 claims can be checked against the
             project's real artifacts). Everything else uses the single-part
             'grading' template + llm.grade_submission(), as before.
          4. Persist Grade — mastery_score/weak_areas/overall_feedback always;
             score_earned/max_marks additionally for entries (null for
             standalone, which doesn't participate in GPA); part1_score/
             part2_score additionally for Midterms.
          5. Update Assessment.status → completed.
          6. Return the Grade.

        Raises:
            NotFoundError: if submission_id does not exist.
            InvalidStateError: if Assessment.status is not submitted or late_submitted.
            IngestionError: if a github_url or file cannot be fetched.
            LLMValidationError: if grading fails after all retries.
        """
        # ── 1. Load submission → assessment → curriculum ───────────────────────
        submission = (
            self.db.query(Submission)
            .options(
                joinedload(Submission.assessment).joinedload(Assessment.curriculum)
            )
            .filter(Submission.id == submission_id)
            .first()
        )
        if submission is None:
            raise NotFoundError(f"Submission {submission_id!r} not found.")

        assessment = submission.assessment
        if assessment.status not in (
            AssessmentStatus.submitted,
            AssessmentStatus.late_submitted,
        ):
            raise InvalidStateError(
                f"Assessment {assessment.id!r} is not in submitted state "
                f"(current: {assessment.status.value!r})."
            )

        curriculum = assessment.curriculum

        # ── 2. Resolve submission content ──────────────────────────────────────
        # For a Midterm this is Part 2's content; for everything else it's
        # the whole submission.
        if submission.submission_type == SubmissionType.github_url:
            from app.ingestors.github_ingestor import fetch_repo_content

            fetched_evidence = fetch_repo_content(
                submission.github_url, self.llm, text_content=submission.text_content
            )
            if submission.text_content:
                # Combined text + github_url submission (see
                # SubmissionCreate.validate_exactly_one_content): the
                # student's own explanation is the answer, the fetched
                # repo content is the evidence to check it against.
                submission_content = (
                    f"Student's written explanation:\n{submission.text_content}\n\n"
                    f"Fetched repository content (evidence):\n{fetched_evidence}"
                )
            else:
                submission_content = fetched_evidence
        elif submission.submission_type == SubmissionType.text:
            submission_content = submission.text_content or ""
        else:
            submission_content = self._read_file(submission.file_path)

        # ── 3-4. Grade + build Grade fields (branches by entry_type / format) ──
        if curriculum.entry_type == CurriculumEntryType.midterm:
            grade = self._grade_midterm(submission, assessment, curriculum, submission_content)
        elif assessment.part1_text is not None:
            # New MCQ + Coding Project format (see AssessmentGenerationResult) —
            # never true for a Midterm (handled above), so this is
            # unambiguous: any non-Midterm assessment with part1_text
            # populated was generated under this format.
            grade = self._grade_mcq_project(submission, assessment, curriculum, submission_content)
        else:
            prompt_template = (
                self.db.query(PromptTemplate)
                .filter(
                    PromptTemplate.slug == "grading",
                    PromptTemplate.is_active.is_(True),
                )
                .first()
            )
            if prompt_template is None:
                raise NotFoundError("No active 'grading' prompt template found.")

            with llm_log_context(
                f"assessment={assessment.id} curriculum={curriculum.id} "
                f"topic={curriculum.topic!r} (grading)"
            ):
                grading_result = self.llm.grade_submission(
                    GradingRequest(
                        assessment_text=assessment.assessment_text or "",
                        rubric=assessment.rubric or "",
                        curriculum_content=curriculum.extracted_content or "",
                        submission_content=submission_content,
                        prompt_template_body=prompt_template.body,
                        github_url=(
                            submission.github_url
                            if submission.submission_type == SubmissionType.github_url
                            else None
                        ),
                    )
                )

            is_entry = curriculum.entry_type is not None
            score_earned = (
                grading_result.mastery_score / 100.0 * (curriculum.max_marks or 0.0)
                if is_entry else None
            )
            max_marks = curriculum.max_marks if is_entry else None

            grade = Grade(
                submission_id=submission_id,
                mastery_score=grading_result.mastery_score,
                weak_areas=grading_result.weak_areas,
                overall_feedback=grading_result.overall_feedback,
                grading_prompt_id=prompt_template.id,
                score_earned=score_earned,
                max_marks=max_marks,
            )
            self.db.add(grade)
            self.db.flush()

        # ── 5. Transition assessment → completed ───────────────────────────────
        assessment.status = AssessmentStatus.completed
        self.db.commit()
        self.db.refresh(grade)

        return grade

    def _grade_midterm(
        self, submission: Submission, assessment: Assessment, curriculum: Curriculum, part2_content: str
    ) -> Grade:
        """Grade both parts of a Midterm submission and build (but don't
        commit) the resulting Grade row.

        mastery_score is back-derived from score_earned/max_marks purely so
        the existing mastery_threshold pass/fail comparison in
        grade_submission_job works identically for Midterms without a branch.
        """
        prompt_template = (
            self.db.query(PromptTemplate)
            .filter(
                PromptTemplate.slug == "midterm_grading",
                PromptTemplate.is_active.is_(True),
            )
            .first()
        )
        if prompt_template is None:
            raise NotFoundError("No active 'midterm_grading' prompt template found.")

        detail = curriculum.midterm_detail
        resources = list(detail.known_now)
        readme_content = None
        for slug, label in detail.pending_completion_labels.items():
            value = detail.pending_completion_slots.get(slug)
            if not value:
                continue
            if "readme" in label.lower():
                # Prose to spot-check, not a fetchable label — see
                # MidtermGradingRequest.readme_content's docstring.
                readme_content = value
            else:
                resources.append(value)

        with llm_log_context(
            f"assessment={assessment.id} curriculum={curriculum.id} "
            f"topic={curriculum.topic!r} (midterm grading)"
        ):
            grading_result = self.llm.grade_midterm_submission(
                MidtermGradingRequest(
                    part1_text=assessment.part1_text or "",
                    part1_rubric=assessment.part1_rubric or "",
                    part2_text=assessment.part2_text or "",
                    part2_rubric=assessment.part2_rubric or "",
                    part1_max_marks=detail.part1_max_marks,
                    part2_max_marks=detail.part2_max_marks,
                    part1_submission_content=submission.part1_text_content or "",
                    part2_submission_content=part2_content,
                    prompt_template_body=prompt_template.body,
                    resources=resources,
                    readme_content=readme_content,
                )
            )

        max_marks = curriculum.max_marks or (detail.part1_max_marks + detail.part2_max_marks)
        score_earned = grading_result.part1_score + grading_result.part2_score
        mastery_score = (score_earned / max_marks * 100.0) if max_marks else 0.0

        grade = Grade(
            submission_id=submission.id,
            mastery_score=mastery_score,
            weak_areas=grading_result.weak_areas,
            overall_feedback=grading_result.overall_feedback,
            grading_prompt_id=prompt_template.id,
            part1_score=grading_result.part1_score,
            part2_score=grading_result.part2_score,
            score_earned=score_earned,
            max_marks=max_marks,
        )
        self.db.add(grade)
        self.db.flush()
        return grade

    def _grade_mcq_project(
        self, submission: Submission, assessment: Assessment, curriculum: Curriculum, project_content: str
    ) -> Grade:
        """Grade a new-format (MCQ + Coding Project) assessment and build
        (but don't commit) the resulting Grade row.

        Section 1 (MCQs) is scored deterministically in Python — comparing
        each parsed answer against the stored correct_option — never by
        the LLM (see MCQGradingRequest's docstring); the one LLM call for
        this section only normalizes free-form answers and writes
        feedback. Section 2 (the project) is graded by a dedicated
        rubric-based LLM call, scored directly in its own 0-70 point
        space. part1_score/part2_score reuse the exact columns a Midterm
        grade uses for its own two parts — same shape, different
        semantics, disambiguated by Submission.mcq_answers being set.

        mastery_score = part1_score + part2_score is already 0-100 by
        construction (30 + 70), so score_earned for entries uses the
        IDENTICAL mastery_score/100*max_marks formula every other branch
        uses — the MCQ/project split is a display/scoring-mechanism
        decomposition, never a change to how GPA weighting is computed
        (see Grade.score_earned's docstring and compute_gpa()).
        """
        mcq_prompt_template = (
            self.db.query(PromptTemplate)
            .filter(PromptTemplate.slug == "mcq_grading", PromptTemplate.is_active.is_(True))
            .first()
        )
        if mcq_prompt_template is None:
            raise NotFoundError("No active 'mcq_grading' prompt template found.")

        project_prompt_template = (
            self.db.query(PromptTemplate)
            .filter(PromptTemplate.slug == "project_grading", PromptTemplate.is_active.is_(True))
            .first()
        )
        if project_prompt_template is None:
            raise NotFoundError("No active 'project_grading' prompt template found.")

        questions = json.loads(assessment.part1_text)   # [{"question","options"}] x5
        answer_key = json.loads(assessment.part1_rubric)  # [{"correct_option","explanation"}] x5
        submitted_answers = submission.mcq_answers or []
        if len(submitted_answers) != 5:
            raise InvalidStateError(
                f"Submission {submission.id!r} has {len(submitted_answers)} "
                "mcq_answers, expected exactly 5."
            )

        with llm_log_context(
            f"assessment={assessment.id} curriculum={curriculum.id} "
            f"topic={curriculum.topic!r} (mcq grading)"
        ):
            mcq_result = self.llm.grade_mcq_section(
                MCQGradingRequest(
                    questions=questions,
                    answer_key=answer_key,
                    submitted_answers=submitted_answers,
                    prompt_template_body=mcq_prompt_template.body,
                )
            )

        mcq_score = 0.0
        weak_mcq_topics: list[str] = []
        for question, key, parsed in zip(questions, answer_key, mcq_result.parsed_answers):
            if parsed == key["correct_option"]:
                mcq_score += 6.0
            else:
                weak_mcq_topics.append(str(question["question"])[:80])

        criteria = json.loads(assessment.part2_rubric)  # [{"description","points"}] x4-6

        with llm_log_context(
            f"assessment={assessment.id} curriculum={curriculum.id} "
            f"topic={curriculum.topic!r} (project grading)"
        ):
            project_result = self.llm.grade_project(
                ProjectGradingRequest(
                    project_text=assessment.part2_text or "",
                    project_criteria=criteria,
                    curriculum_content=curriculum.extracted_content or "",
                    submission_content=project_content,
                    prompt_template_body=project_prompt_template.body,
                    github_url=(
                        submission.github_url
                        if submission.submission_type == SubmissionType.github_url
                        else None
                    ),
                )
            )

        mastery_score = mcq_score + project_result.project_score  # 0-100 by construction (30 + 70)

        is_entry = curriculum.entry_type is not None
        score_earned = (mastery_score / 100.0 * (curriculum.max_marks or 0.0)) if is_entry else None
        max_marks = curriculum.max_marks if is_entry else None

        overall_feedback = (
            f"MCQ Section ({mcq_score:.0f}/30): " + " ".join(mcq_result.feedback)
            + f"\n\nCoding Project ({project_result.project_score:.1f}/70): "
            + project_result.overall_feedback
        )
        weak_areas = weak_mcq_topics + list(project_result.weak_areas)

        grade = Grade(
            submission_id=submission.id,
            mastery_score=mastery_score,
            weak_areas=weak_areas,
            overall_feedback=overall_feedback,
            grading_prompt_id=project_prompt_template.id,
            part1_score=mcq_score,
            part2_score=project_result.project_score,
            score_earned=score_earned,
            max_marks=max_marks,
        )
        self.db.add(grade)
        self.db.flush()
        return grade

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _read_file(self, file_path: str | None) -> str:
        """Read a submitted file from uploads_dir and return its text content.

        Raises:
            IngestionError: if file_path is None or the file cannot be read.
        """
        if not file_path:
            raise IngestionError("Submission has file type but no file_path recorded.")
        full_path = Path(get_settings().uploads_dir) / file_path
        try:
            return full_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise IngestionError(
                f"Could not read submission file {file_path!r}."
            ) from exc

    def get_results(self, submission_id: str) -> GradeResponse:
        """Return the grading result for a completed submission.

        Loads the Grade by submission_id, loads the related Assessment via
        submission, and computes passed = mastery_score >= mastery_threshold.

        Raises:
            NotFoundError: if no Grade exists for submission_id.
        """
        grade = (
            self.db.query(Grade)
            .filter(Grade.submission_id == submission_id)
            .first()
        )
        if grade is None:
            raise NotFoundError(
                f"No grade found for submission {submission_id!r}."
            )

        # Load assessment through the submission relationship for caller context.
        _ = grade.submission.assessment

        settings = get_settings()
        passed = grade.mastery_score >= settings.mastery_threshold
        is_mcq_format = grade.submission.mcq_answers is not None

        return GradeResponse(
            mastery_score=grade.mastery_score,
            overall_feedback=grade.overall_feedback,
            weak_areas=grade.weak_areas,
            passed=passed,
            mcq_score=grade.part1_score if is_mcq_format else None,
            project_score=grade.part2_score if is_mcq_format else None,
        )