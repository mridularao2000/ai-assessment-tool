"""LLM interface contract and associated data types.

The implementing class (e.g. AnthropicLLMAdapter) is responsible for:
  - Rendering the prompt_template_body with the provided request data
  - Calling the LLM API
  - Parsing and validating the structured JSON response against
    the expected output shape
  - Retrying on validation failure (up to implementation-defined max_retries)
  - Raising LLMValidationError when all retries are exhausted
  - Raising LLMUnavailableError on unrecoverable API-level failures
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

# ── Category type ─────────────────────────────────────────────────────────────
# Literal union of all valid reschedule categories.
# Approved categories: application code in RescheduleService maps these to
# approved=True.  Denied categories map to approved=False.
RescheduleCategory = Literal[
    "interview",
    "medical",
    "emergency",
    "work_escalation",
    "procrastination",
    "lack_of_preparation",
    "missed_schedule",
]

# ── Request DTOs ──────────────────────────────────────────────────────────────


@dataclass
class CurriculumAnalysisRequest:
    """Input to analyze_curriculum.

    The service fetches the active 'assessment_generation' prompt template
    and passes its body here.  The LLM implementation renders the template
    with the topic and content.
    """

    topic: str
    curriculum_content: str
    prompt_template_body: str


@dataclass
class AssessmentGenerationRequest:
    """Input to generate_assessment (first attempt).

    resources is None for standalone curricula (today's behavior,
    unchanged) and a verbatim list of Study-Checklist labels for
    curriculum-upload Assessment-type entries — when present, the adapter
    enables web_search/web_fetch and interpolates per-resource guidance
    (search-then-fetch, or general-knowledge-only for known non-fetchable
    resources) into the rendered prompt.
    """

    topic: str
    curriculum_content: str
    prompt_template_body: str
    resources: Optional[list[str]] = None


@dataclass
class RetestGenerationRequest:
    """Input to generate_retest (attempt_number >= 2).

    Carries the weak areas identified by the previous grading so the LLM
    can focus the assessment on those specific topics.

    resources mirrors AssessmentGenerationRequest.resources — None for
    standalone (unchanged), a verbatim list of Study-Checklist labels for
    an Assessment-type entry's retest, so web_search/web_fetch grounding
    still applies on retry, not just the first attempt.
    """

    topic: str
    curriculum_content: str
    prompt_template_body: str
    previous_mastery_score: float
    weak_areas: list[str]
    attempt_number: int
    resources: Optional[list[str]] = None


@dataclass
class GradingRequest:
    """Input to grade_submission.

    submission_content is the resolved content string — plain text, GitHub
    repo text fetched by github_ingestor, or file contents read from disk.
    The service resolves the submission type before calling this method.
    """

    assessment_text: str
    rubric: str
    curriculum_content: str
    submission_content: str
    prompt_template_body: str


@dataclass
class MidtermGenerationRequest:
    """Input to generate_midterm — the two-part Midterm exam type.

    Part 1 draws on cumulative_pool_content (every Assessment completed on
    or before this Midterm's completion_date, or — when none exist yet,
    e.g. the chronologically-first Midterm in an upload — the Midterm's
    own known_now resources as a generic fallback). Part 2 draws on
    own_resources (known_now + any filled pending_completion values,
    verbatim) and probe_focus. own_resources is what gets web_search/
    web_fetch grounding — Part 1's pool references material already
    covered by earlier, already-graded Assessments, not fresh external
    pages to fetch.
    """

    topic: str
    cumulative_pool_content: str
    own_resources: list[str]
    probe_focus: Optional[str]
    part1_max_marks: float
    part2_max_marks: float
    prompt_template_body: str


@dataclass
class MidtermGradingRequest:
    """Input to grade_midterm_submission.

    Part 1 and Part 2 are graded together (they share one overall_feedback)
    but scored independently against their own rubric and max_marks.

    resources mirrors the grounding used at generation time (own_resources
    from generate_midterm/generate_midterm_retest) — Part 2 asks the
    student to defend real decisions made in their project, so grading it
    accurately requires checking those claims against the project's actual
    resources (repo, README, design docs), not just a static rubric.
    Without this, a defense that sounds plausible but misrepresents the
    real project could be scored as correct with no way to catch it.
    """

    part1_text: str
    part1_rubric: str
    part2_text: str
    part2_rubric: str
    part1_max_marks: float
    part2_max_marks: float
    part1_submission_content: str
    part2_submission_content: str
    prompt_template_body: str
    resources: Optional[list[str]] = None


@dataclass
class MidtermRetestGenerationRequest:
    """Input to generate_midterm_retest (attempt_number >= 2).

    Mirrors RetestGenerationRequest's weak-areas-targeting idea, applied to
    the two-part Midterm shape. own_resources still gets web_search/
    web_fetch grounding on retry, matching generate_midterm's first
    attempt — Part 2 is being regenerated against the same real project,
    not a fresh one.
    """

    topic: str
    cumulative_pool_content: str
    own_resources: list[str]
    probe_focus: Optional[str]
    part1_max_marks: float
    part2_max_marks: float
    previous_part1_score: float
    previous_part2_score: float
    weak_areas: list[str]
    attempt_number: int
    prompt_template_body: str


@dataclass
class RescheduleClassificationRequest:
    """Input to classify_reschedule_request.

    The LLM only classifies the reason into a category and provides
    reasoning.  The application (RescheduleService) makes the final
    approval/denial decision based on the returned category.
    """

    reason: str
    prompt_template_body: str


# ── Result DTOs ───────────────────────────────────────────────────────────────


@dataclass
class CurriculumAnalysisResult:
    """Structured summary of a curriculum produced before assessment generation."""

    summary: str
    key_topics: list[str]
    complexity_level: str  # "beginner" | "intermediate" | "advanced"
    estimated_study_hours: float


@dataclass
class AssessmentGenerationResult:
    """Output of both generate_assessment and generate_retest.

    The same structure is returned for initial and retest assessments.
    duration_minutes is determined by the LLM from curriculum complexity.

    Examples:
      JavaScript Concepts        → 60 min
      React Components           → 90 min
      VS Code Extension Arch     → 120 min
    """

    assessment_text: str
    rubric: str
    duration_minutes: int


@dataclass
class MidtermGenerationResult:
    """Output of generate_midterm — two independently-scored parts."""

    part1_text: str
    part1_rubric: str
    part2_text: str
    part2_rubric: str
    duration_minutes: int


@dataclass
class GradingResult:
    """Structured grading output from grade_submission.

    weak_areas is passed to RetestGenerationRequest on subsequent attempts
    so that retests focus on the student's identified gaps.
    """

    mastery_score: float  # 0.0–100.0
    weak_areas: list[str]  # e.g. ["Promises", "Async/Await", "Event Loop"]
    overall_feedback: str


@dataclass
class MidtermGradingResult:
    """Structured grading output from grade_midterm_submission.

    weak_areas is passed to generate_midterm_retest on a failing attempt,
    same role as GradingResult.weak_areas for single-part retests.
    """

    part1_score: float  # 0.0–part1_max_marks
    part2_score: float  # 0.0–part2_max_marks
    weak_areas: list[str]
    overall_feedback: str


@dataclass
class RescheduleClassificationResult:
    """Output of classify_reschedule_request.

    The LLM provides only the category and reasoning.  The application
    determines approval in RescheduleService.APPROVED_CATEGORIES.
    """

    category: RescheduleCategory
    reasoning: str


# ── Exceptions ────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Base class for all LLM interface errors."""


class LLMValidationError(LLMError):
    """Raised when the LLM returns a structurally invalid response that
    cannot be coerced into the expected output type after all retries."""


class LLMUnavailableError(LLMError):
    """Raised when the LLM API is unreachable or returns an unrecoverable
    HTTP-level error (e.g. 500, rate-limit exhaustion)."""


# ── Protocol ──────────────────────────────────────────────────────────────────


class LLMInterface(Protocol):
    """Structural interface for all LLM interactions.

    Future implementing class: AnthropicLLMAdapter
      Located at: app/adapters/anthropic_llm.py
      Dependencies: anthropic SDK, app.config.get_settings
    """

    def analyze_curriculum(
        self, request: CurriculumAnalysisRequest
    ) -> CurriculumAnalysisResult: ...

    def generate_assessment(
        self, request: AssessmentGenerationRequest
    ) -> AssessmentGenerationResult: ...

    def generate_retest(
        self, request: RetestGenerationRequest
    ) -> AssessmentGenerationResult: ...

    def generate_midterm(
        self, request: MidtermGenerationRequest
    ) -> MidtermGenerationResult: ...

    def grade_submission(
        self, request: GradingRequest
    ) -> GradingResult: ...

    def grade_midterm_submission(
        self, request: MidtermGradingRequest
    ) -> MidtermGradingResult: ...

    def generate_midterm_retest(
        self, request: MidtermRetestGenerationRequest
    ) -> MidtermGenerationResult: ...

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult: ...
