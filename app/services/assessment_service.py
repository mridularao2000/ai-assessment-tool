from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    LLMInterface,
    RetestGenerationRequest,
)
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumStatus
from app.models.grade import Grade
from app.models.prompt_template import PromptTemplate
from app.utils.token_auth import generate_submission_token, verify_submission_token


class AssessmentService:
    """Generates and persists Assessment records for first attempts and retests.

    Depends on:
      db  — SQLAlchemy session for all persistence
      llm — LLMInterface for assessment and retest generation

    Does NOT schedule jobs. The caller (route handler or job function) is
    responsible for passing the returned Assessment to SchedulerService.
    """

    def __init__(self, db: Session, llm: LLMInterface) -> None:
        self.db = db
        self.llm = llm

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _calculate_scheduled_at(self, target_completion_date: date) -> datetime:
        settings = get_settings()

        offset_days = random.randint(1, 3)

        scheduled_date = target_completion_date + timedelta(days=offset_days)

        return datetime(
            scheduled_date.year,
            scheduled_date.month,
            scheduled_date.day,
            9, 0, 0,  # fixed 9 AM UTC
        )

    def _build_dates(self, scheduled_at: datetime) -> tuple[datetime, datetime]:
        reminder_at = scheduled_at - timedelta(days=1)   # FIXED: 1 day before
        due_date = scheduled_at + timedelta(days=1)      # optional grace window

        return reminder_at, due_date

    def _fetch_prompt(self, slug: str) -> PromptTemplate:
        """Return the active PromptTemplate for slug, or raise NotFoundError."""
        template = (
            self.db.query(PromptTemplate)
            .filter(
                PromptTemplate.slug == slug,
                PromptTemplate.is_active.is_(True),
            )
            .first()
        )
        if template is None:
            raise NotFoundError(
                f"Prompt templates not initialized (missing: {slug!r}). "
                "Please run database seed: python -m app.db.seed"
            )
        return template

    # ── Public methods ─────────────────────────────────────────────────────────

    def create_for_curriculum(self, curriculum: Curriculum) -> Assessment:
        """Build a first-attempt Assessment ORM object. Pure factory — no DB writes.

        Accepts the Curriculum ORM object directly so it works on pending
        (unflushed) curricula; Session.get() would miss them because it only
        searches the identity map (persistent rows).

        Does NOT call db.add(), db.commit(), or db.refresh().
        CurriculumService.create() owns the session and is the sole
        transaction boundary for the curriculum-creation pipeline.

        Steps:
          1. Verify curriculum.status == ready.
          2. Fetch the active PromptTemplate where slug='assessment_generation'.
          3. Call llm.generate_assessment() → AssessmentGenerationResult.
          4. Compute scheduled_at, reminder_at, due_date.
          5. Pre-generate assessment ID; derive submission_token from it.
          6. Construct and return Assessment with status=scheduled.
             Caller adds it to the session and commits.

        Raises:
            InvalidStateError: if curriculum.status is not ready.
            NotFoundError: if the prompt template is missing.
            LLMValidationError: if generation fails after all retries.
        """
        if curriculum.status != CurriculumStatus.ready:
            raise InvalidStateError(
                f"Curriculum {curriculum.id!r} is not ready "
                f"(status: {curriculum.status.value!r})."
            )

        prompt_template = self._fetch_prompt("assessment_generation")

        result = self.llm.generate_assessment(
            AssessmentGenerationRequest(
                topic=curriculum.topic,
                curriculum_content=curriculum.extracted_content or "",
                prompt_template_body=prompt_template.body,
            )
        )

        scheduled_at = self._calculate_scheduled_at(curriculum.target_completion_date)
        reminder_at, due_date = self._build_dates(scheduled_at)

        assessment_id = str(uuid.uuid4())
        return Assessment(
            id=assessment_id,
            curriculum_id=curriculum.id,
            attempt_number=1,
            assessment_text=result.assessment_text,
            rubric=result.rubric,
            duration_minutes=result.duration_minutes,
            generation_prompt_id=prompt_template.id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
            status=AssessmentStatus.scheduled,
            submission_token=generate_submission_token(assessment_id),
        )

    def create_retest(
        self,
        curriculum_id: str,
        previous_grade_id: str,
    ) -> Assessment:
        """Generate a targeted retest and persist it.

        Steps:
          1. Load Curriculum by curriculum_id.
          2. Load Grade by previous_grade_id; traverse to its parent Assessment
             to obtain attempt_number, weak_areas, and mastery_score.
          3. Fetch the active PromptTemplate where slug='retest_generation'.
          4. Call llm.generate_retest() with topic, extracted_content,
             previous_mastery_score, weak_areas, and attempt_number + 1.
          5. Repeat scheduling + token generation logic from create_for_curriculum.
          6. Persist Assessment with attempt_number = previous_attempt + 1.
          7. Return the Assessment.
             Caller must pass it to SchedulerService.schedule_assessment_jobs().

        Raises:
            NotFoundError: if curriculum, grade, or prompt template not found.
            LLMValidationError: if generation fails after all retries.
        """
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")

        grade = self.db.get(Grade, previous_grade_id)
        if grade is None:
            raise NotFoundError(f"Grade {previous_grade_id!r} not found.")

        previous_attempt = grade.submission.assessment.attempt_number

        prompt_template = self._fetch_prompt("retest_generation")

        result = self.llm.generate_retest(
            RetestGenerationRequest(
                topic=curriculum.topic,
                curriculum_content=curriculum.extracted_content or "",
                prompt_template_body=prompt_template.body,
                previous_mastery_score=grade.mastery_score,
                weak_areas=grade.weak_areas or [],
                attempt_number=previous_attempt + 1,
            )
        )

        scheduled_at = self._calculate_scheduled_at(curriculum.target_completion_date)
        reminder_at, due_date = self._build_dates(scheduled_at)

        assessment_id = str(uuid.uuid4())
        assessment = Assessment(
            id=assessment_id,
            curriculum_id=curriculum_id,
            attempt_number=previous_attempt + 1,
            assessment_text=result.assessment_text,
            rubric=result.rubric,
            duration_minutes=result.duration_minutes,
            generation_prompt_id=prompt_template.id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
            status=AssessmentStatus.scheduled,
            submission_token=generate_submission_token(assessment_id),
        )
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)

        return assessment

    def get_by_id_and_token(self, assessment_id: str, token: str) -> Assessment:
        """Load an Assessment by ID and verify its submission token.

        Curriculum is eagerly loaded so callers can access assessment.curriculum.topic
        without triggering a lazy load after this method returns.

        Raises:
            NotFoundError: if assessment_id does not exist.
            InvalidTokenError: if the token does not match.
        """
        assessment = (
            self.db.query(Assessment)
            .options(joinedload(Assessment.curriculum))
            .filter(Assessment.id == assessment_id)
            .first()
        )
        if assessment is None:
            raise NotFoundError(f"Assessment {assessment_id!r} not found.")
        if not verify_submission_token(assessment_id, token):
            raise InvalidTokenError(
                f"Invalid token for assessment {assessment_id!r}."
            )
        return assessment
