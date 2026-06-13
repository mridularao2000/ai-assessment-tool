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
from typing import Literal, Protocol

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
    """Input to generate_assessment (first attempt)."""

    topic: str
    curriculum_content: str
    prompt_template_body: str


@dataclass
class RetestGenerationRequest:
    """Input to generate_retest (attempt_number >= 2).

    Carries the weak areas identified by the previous grading so the LLM
    can focus the assessment on those specific topics.
    """

    topic: str
    curriculum_content: str
    prompt_template_body: str
    previous_mastery_score: float
    weak_areas: list[str]
    attempt_number: int


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
class GradingResult:
    """Structured grading output from grade_submission.

    weak_areas is passed to RetestGenerationRequest on subsequent attempts
    so that retests focus on the student's identified gaps.
    """

    mastery_score: float  # 0.0–100.0
    weak_areas: list[str]  # e.g. ["Promises", "Async/Await", "Event Loop"]
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

    def grade_submission(
        self, request: GradingRequest
    ) -> GradingResult: ...

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult: ...
