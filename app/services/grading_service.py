from __future__ import annotations

from pathlib import Path

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import IngestionError, InvalidStateError, NotFoundError
from app.interfaces.llm import GradingRequest, LLMInterface
from app.models.assessment import Assessment, AssessmentStatus
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
             Verify Assessment.status == submitted.
          2. Resolve submission_content from Submission.submission_type:
               github_url → github_ingestor.fetch_repo_content(submission.github_url)
               text       → submission.text_content (used directly)
               file       → read file bytes from submission.file_path on disk
          3. Fetch the active PromptTemplate where slug='grading'.
          4. Call llm.grade_submission() with:
               assessment_text, rubric, curriculum_content,
               submission_content, prompt_template_body
             → GradingResult(mastery_score, weak_areas, overall_feedback)
          5. Persist Grade with mastery_score, weak_areas, overall_feedback,
             and grading_prompt_id.
          6. Update Assessment.status → completed.
          7. Return the Grade.

        Raises:
            NotFoundError: if submission_id does not exist.
            InvalidStateError: if Assessment.status is not submitted.
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
        if assessment.status != AssessmentStatus.submitted:
            raise InvalidStateError(
                f"Assessment {assessment.id!r} is not in submitted state "
                f"(current: {assessment.status.value!r})."
            )

        curriculum = assessment.curriculum

        # ── 2. Resolve submission content ──────────────────────────────────────
        if submission.submission_type == SubmissionType.github_url:
            try:
                from app.ingestors.github_ingestor import fetch_repo_content
                submission_content = fetch_repo_content(submission.github_url)
            except Exception as exc:
                raise IngestionError(
                    f"Failed to fetch GitHub repo {submission.github_url!r}."
                ) from exc
        elif submission.submission_type == SubmissionType.text:
            submission_content = submission.text_content or ""
        else:
            submission_content = self._read_file(submission.file_path)

        # ── 3. Fetch active grading prompt template ────────────────────────────
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

        # ── 4. Grade via LLM ───────────────────────────────────────────────────
        grading_result = self.llm.grade_submission(
            GradingRequest(
                assessment_text=assessment.assessment_text or "",
                rubric=assessment.rubric or "",
                curriculum_content=curriculum.extracted_content or "",
                submission_content=submission_content,
                prompt_template_body=prompt_template.body,
            )
        )

        # ── 5. Persist Grade ───────────────────────────────────────────────────
        grade = Grade(
            submission_id=submission_id,
            mastery_score=grading_result.mastery_score,
            weak_areas=grading_result.weak_areas,
            overall_feedback=grading_result.overall_feedback,
            grading_prompt_id=prompt_template.id,
        )
        self.db.add(grade)
        self.db.flush()

        # ── 6. Transition assessment → completed ───────────────────────────────
        assessment.status = AssessmentStatus.completed
        self.db.commit()
        self.db.refresh(grade)

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

        return GradeResponse(
            mastery_score=grade.mastery_score,
            overall_feedback=grade.overall_feedback,
            weak_areas=grade.weak_areas,
            passed=passed,
        )