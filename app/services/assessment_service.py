from __future__ import annotations

import random
import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    LLMInterface,
    MidtermGenerationRequest,
    MidtermRetestGenerationRequest,
    RetestGenerationRequest,
)
from app.models._utils import utcnow
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType, CurriculumStatus
from app.models.grade import Grade
from app.models.prompt_template import PromptTemplate
from app.services.email_service import EmailService
from app.services.late_token_service import LateTokenService
from app.utils.token_auth import generate_submission_token, verify_submission_token


def calculate_scheduled_at(target_completion_date: date) -> datetime:
    """Pick the single send instant within target_completion_date+1..+3, 9am.

    Free function (not just an AssessmentService method) so callers that
    need the date math without an LLM instance — e.g.
    CurriculumUploadService scheduling upload entries before their content
    is generated — can use the exact same logic.
    """
    offset_days = random.randint(1, 3)
    scheduled_date = target_completion_date + timedelta(days=offset_days)
    return datetime(
        scheduled_date.year,
        scheduled_date.month,
        scheduled_date.day,
        9, 0, 0,  # fixed 9 AM UTC
    )


def build_assessment_dates(scheduled_at: datetime) -> tuple[datetime, datetime]:
    reminder_at = scheduled_at - timedelta(days=1)   # 1 day before
    due_date = scheduled_at + timedelta(days=2)      # 2-day submission window
    return reminder_at, due_date


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
        return calculate_scheduled_at(target_completion_date)

    def _build_dates(self, scheduled_at: datetime) -> tuple[datetime, datetime]:
        return build_assessment_dates(scheduled_at)

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
             previous_mastery_score, weak_areas, and attempt_number + 1 —
             plus resources (Assessment-type entries only, so web_search/
             web_fetch grounding still applies on retry, not just attempt 1).
          5. Repeat scheduling + token generation logic from
             create_for_curriculum (standalone) or the entry-specific date
             math (curriculum.entry_type is not None — same reminder-
             anchored-to-due_date rule as a first attempt, not standalone's).
          6. Persist Assessment with attempt_number = previous_attempt + 1.
          7. Return the Assessment.
             Caller must pass it to SchedulerService.schedule_assessment_jobs().

        Midterm-type curricula take a separate branch: a fresh two-part
        exam (both Part 1 and Part 2 regenerated) via generate_midterm_retest(),
        targeted at the weak areas identified from grading the previous
        attempt's two parts — same eager-generation and 2-attempt-cap
        mechanics as the single-part case, just two-part content.

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
        is_entry = curriculum.entry_type is not None

        if curriculum.entry_type == CurriculumEntryType.midterm:
            assessment = self._create_midterm_retest(curriculum, grade, previous_attempt)
        else:
            prompt_template = self._fetch_prompt("retest_generation")
            resources = [r.source_ref for r in curriculum.resources] if is_entry else None

            result = self.llm.generate_retest(
                RetestGenerationRequest(
                    topic=curriculum.topic,
                    curriculum_content=curriculum.extracted_content or "",
                    prompt_template_body=prompt_template.body,
                    previous_mastery_score=grade.mastery_score,
                    weak_areas=grade.weak_areas or [],
                    attempt_number=previous_attempt + 1,
                    resources=resources,
                )
            )

            scheduled_at = self._calculate_scheduled_at(curriculum.target_completion_date)
            if is_entry:
                from app.services.curriculum_upload_service import _build_entry_dates
                reminder_at, due_date = _build_entry_dates(scheduled_at)
            else:
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

    def _create_midterm_retest(
        self, curriculum: Curriculum, grade: Grade, previous_attempt: int
    ) -> Assessment:
        """Build (but don't commit) a fresh two-part retest Assessment for a
        Midterm-type curriculum, targeted at the previous attempt's weak
        areas. Shares pool/resource assembly with generate_midterm_content()
        via _assemble_midterm_pool().
        """
        from app.services.curriculum_upload_service import _build_entry_dates

        detail = curriculum.midterm_detail
        prompt_template = self._fetch_prompt("midterm_retest_generation")
        cumulative_pool_content, own_resources, readme_content = self._assemble_midterm_pool(curriculum)

        result = self.llm.generate_midterm_retest(
            MidtermRetestGenerationRequest(
                topic=curriculum.topic,
                cumulative_pool_content=cumulative_pool_content,
                own_resources=own_resources,
                probe_focus=detail.probe_focus,
                part1_max_marks=detail.part1_max_marks,
                part2_max_marks=detail.part2_max_marks,
                previous_part1_score=grade.part1_score or 0.0,
                previous_part2_score=grade.part2_score or 0.0,
                weak_areas=grade.weak_areas or [],
                attempt_number=previous_attempt + 1,
                prompt_template_body=prompt_template.body,
                readme_content=readme_content,
            )
        )

        scheduled_at = self._calculate_scheduled_at(curriculum.target_completion_date)
        reminder_at, due_date = _build_entry_dates(scheduled_at)

        assessment_id = str(uuid.uuid4())
        assessment = Assessment(
            id=assessment_id,
            curriculum_id=curriculum.id,
            attempt_number=previous_attempt + 1,
            part1_text=result.part1_text,
            part1_rubric=result.part1_rubric,
            part2_text=result.part2_text,
            part2_rubric=result.part2_rubric,
            duration_minutes=result.duration_minutes,
            generation_prompt_id=prompt_template.id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
            status=AssessmentStatus.scheduled,
            submission_token=generate_submission_token(assessment_id),
        )
        self.db.add(assessment)
        return assessment

    def _assemble_midterm_pool(
        self, curriculum: Curriculum
    ) -> tuple[str, list[str], Optional[str]]:
        """Shared Part 1 pool / Part 2 resource assembly, used by both first
        attempts (generate_midterm_content) and retests (_create_midterm_retest).

        Returns (cumulative_pool_content, own_resources, readme_content).
        A pending_completion slot whose LABEL contains "readme" is kept out
        of own_resources and returned separately — see
        MidtermGenerationRequest.readme_content's docstring for why (it's
        prose to ground Part 2 in, not a URL/label for resource_guidance's
        web_search/web_fetch instructions — the same separation
        GradingService._grade_midterm already applies on the grading side).
        """
        from app.services.curriculum_upload_service import assemble_part1_pool

        detail = curriculum.midterm_detail
        qualifying, used_fallback = assemble_part1_pool(self.db, curriculum)
        if used_fallback:
            cumulative_pool_content = "\n".join(f"- {r}" for r in detail.known_now)
        else:
            cumulative_pool_content = "\n\n".join(
                f"[{e.chapter_label}] {e.topic}\n"
                f"Resources: {', '.join(r.source_ref for r in e.resources)}"
                for e in qualifying
            )

        own_resources = list(detail.known_now)
        readme_content = None
        for slug, label in detail.pending_completion_labels.items():
            value = detail.pending_completion_slots.get(slug)
            if not value:
                continue
            if "readme" in label.lower():
                readme_content = value
            else:
                own_resources.append(value)

        return cumulative_pool_content, own_resources, readme_content

    def generate_assessment_content(self, assessment: Assessment) -> None:
        """Populate assessment_text/rubric/duration_minutes on an
        already-scheduled, curriculum-upload Assessment-type Assessment row.

        Called at send-time (see send_assessment_job), not eagerly at
        scheduling time — keeps resource-grounding maximally current and
        avoids spending LLM calls on exams that are weeks/months away.
        Mutates and flushes; caller commits.

        Raises:
            NotFoundError: if the 'assessment_generation' prompt template
                           is missing.
            LLMValidationError: if generation fails after all retries.
        """
        curriculum = assessment.curriculum
        prompt_template = self._fetch_prompt("assessment_generation")
        resources = [r.source_ref for r in curriculum.resources]

        result = self.llm.generate_assessment(
            AssessmentGenerationRequest(
                topic=curriculum.topic,
                curriculum_content=curriculum.extracted_content or "",
                prompt_template_body=prompt_template.body,
                resources=resources,
            )
        )
        assessment.assessment_text = result.assessment_text
        assessment.rubric = result.rubric
        assessment.duration_minutes = result.duration_minutes
        assessment.generation_prompt_id = prompt_template.id
        self.db.flush()

    def generate_midterm_content(self, assessment: Assessment) -> None:
        """Populate part1_text/part1_rubric/part2_text/part2_rubric on an
        already-scheduled Midterm-type Assessment row. Same send-time
        timing rationale as generate_assessment_content().

        Raises:
            NotFoundError: if the 'midterm_generation' prompt template is
                           missing.
            LLMValidationError: if generation fails after all retries.
        """
        curriculum = assessment.curriculum
        detail = curriculum.midterm_detail
        prompt_template = self._fetch_prompt("midterm_generation")
        cumulative_pool_content, own_resources, readme_content = self._assemble_midterm_pool(curriculum)

        result = self.llm.generate_midterm(
            MidtermGenerationRequest(
                topic=curriculum.topic,
                cumulative_pool_content=cumulative_pool_content,
                own_resources=own_resources,
                probe_focus=detail.probe_focus,
                part1_max_marks=detail.part1_max_marks,
                part2_max_marks=detail.part2_max_marks,
                prompt_template_body=prompt_template.body,
                readme_content=readme_content,
            )
        )
        assessment.part1_text = result.part1_text
        assessment.part1_rubric = result.part1_rubric
        assessment.part2_text = result.part2_text
        assessment.part2_rubric = result.part2_rubric
        assessment.duration_minutes = result.duration_minutes
        assessment.generation_prompt_id = prompt_template.id
        self.db.flush()

    def trigger_late_send(self, curriculum_id: str, email_service: EmailService) -> Assessment:
        """UI-facing entry point for a Missed — Late-Eligible entry: the
        student-facing button/action equivalent of what an on-time exam
        gets automatically from send_assessment_job, just entered from a
        different starting point (the window already closed).

        Generates content now if it hasn't been generated yet — true for
        an entry whose window closed before it was ever scheduled (see
        CurriculumUploadService.schedule_ready_entries /
        _create_retroactive_expired_assessment) — then (re)sends the
        assessment email via the exact same EmailService.send_assessment_email
        used for an on-time send. That email is the only place the
        submission_token is ever delivered; this method never returns it,
        so there's no new way to read a token without owning the inbox it
        was sent to.

        Does NOT spend a late-submission token itself — spending stays at
        actual submission time in SubmissionService.create(), exactly as
        for a normally-expired assessment recovered from the original
        email. This only checks a token is available, so a click doesn't
        burn an LLM call + email send for a submission the student can't
        actually make.

        Raises:
            NotFoundError: curriculum_id doesn't exist, or has no
                            Assessment yet (nothing scheduled at all).
            InvalidStateError: not currently expired-this-month (already
                                graded, still within its normal window,
                                on hold, or missed in an earlier calendar
                                month), or no late-submission tokens remain.
            LLMValidationError: if content generation fails after all retries.
        """
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")

        assessments = sorted(curriculum.assessments, key=lambda a: a.attempt_number)
        if not assessments:
            raise NotFoundError(
                f"Curriculum {curriculum_id!r} has no assessment scheduled yet."
            )
        assessment = assessments[-1]

        if assessment.status != AssessmentStatus.expired:
            raise InvalidStateError(
                f"Assessment {assessment.id!r} is not expired "
                f"(current status: {assessment.status.value!r}) — nothing to late-send."
            )
        now = utcnow()
        if (assessment.due_date.year, assessment.due_date.month) != (now.year, now.month):
            raise InvalidStateError(
                f"Assessment {assessment.id!r} expired in a previous calendar "
                "month — no longer late-eligible."
            )
        if LateTokenService(self.db).get_balance(curriculum.upload_id) <= 0:
            raise InvalidStateError(
                f"No late-submission tokens available for curriculum {curriculum_id!r}."
            )

        is_midterm = curriculum.entry_type == CurriculumEntryType.midterm
        has_content = assessment.part1_text is not None if is_midterm else assessment.assessment_text is not None
        if not has_content:
            if is_midterm:
                self.generate_midterm_content(assessment)
            else:
                self.generate_assessment_content(assessment)
            self.db.commit()

        email_service.send_assessment_email(assessment.id)
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
