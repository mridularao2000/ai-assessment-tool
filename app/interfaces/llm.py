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

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal, Optional, Protocol

# ── Cross-cutting call-logging context ──────────────────────────────────────
# "Which curriculum entry triggered this LLM call" — a contextvar rather
# than threading a new parameter through every Request dataclass and every
# LLMInterface method, since this is pure observability with no effect on
# behavior. Callers (assessment_service.py, grading_service.py,
# curriculum_service.py) wrap an llm.* call with llm_log_context(...); the
# implementing adapter reads current_llm_log_context() when logging.
# Defaults to "(no context set)" so a call site that forgets to set it is
# visibly unlabeled in logs rather than silently blank.
_log_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_llm_log_context", default="(no context set)"
)


@contextmanager
def llm_log_context(label: str) -> Iterator[None]:
    """Tag every LLM call log line made within this block with `label`
    (e.g. f"curriculum={curriculum.id} topic={curriculum.topic!r}")."""
    token = _log_context.set(label)
    try:
        yield
    finally:
        _log_context.reset(token)


def current_llm_log_context() -> str:
    """Read the current tag set by the nearest enclosing llm_log_context()."""
    return _log_context.get()


# ── Resource-label filtering ─────────────────────────────────────────────────
# Labels matching these prefixes describe internal/personal context rather
# than any real resource — external and fetchable, OR public and generally
# known (the model has no general knowledge to fall back on for someone's
# own notes, and a self-referential "cumulative: ..." label just
# redescribes content already supplied separately via
# MidtermGenerationRequest.cumulative_pool_content — it was never meant to
# be an independent resource). Passing either into web_search/web_fetch
# produces a doomed search (confirmed via a curriculum_seed.json structural
# audit, 2026-08-30: "own cyber-sale workflow notes" in the System Design
# Fundamentals entry, and "cumulative: all Assessments with completion_date
# on/before ..." in every midterm past the first). Filtered out entirely
# before a resource list ever reaches a Request dataclass — no "use general
# knowledge instead" framing is offered the way AnthropicLLMAdapter's
# NON_FETCHABLE_NOTES does for a paid book or a known tool, since that
# framing would be actively misleading here (the model has never seen
# "your own notes") or redundant (the cumulative pool is already provided).
_INTERNAL_ONLY_PREFIXES: tuple[str, ...] = (
    "own ",
    "cumulative:",
)


def filter_fetchable_resources(resources: list[str]) -> list[str]:
    """Drop labels describing internal/personal context rather than a real
    resource — see _INTERNAL_ONLY_PREFIXES. Callers apply this before a
    resource list is used to build any *Request dataclass's resources/
    own_resources field, so these labels never trigger a nonsensical
    web_search/web_fetch attempt."""
    return [r for r in resources if not r.strip().lower().startswith(_INTERNAL_ONLY_PREFIXES)]

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
    EXCEPT the README — see readme_content) and probe_focus. own_resources
    is what gets web_search/web_fetch grounding — Part 1's pool references
    material already covered by earlier, already-graded Assessments, not
    fresh external pages to fetch.

    readme_content mirrors MidtermGradingRequest.readme_content: the
    free-text project README/design writeup a student supplied when
    filling this midterm's pending resources, kept OUT of own_resources
    for the same reason it's kept out of MidtermGradingRequest.resources —
    it's prose to ground Part 2's questions in, not a URL/label to run
    through resource_guidance's "web_search then web_fetch this" path.
    None when no pending-resource slot was identified as the README.
    """

    topic: str
    cumulative_pool_content: str
    own_resources: list[str]
    probe_focus: Optional[str]
    part1_max_marks: float
    part2_max_marks: float
    prompt_template_body: str
    readme_content: Optional[str] = None


@dataclass
class MidtermGradingRequest:
    """Input to grade_midterm_submission.

    Part 1 and Part 2 are graded together (they share one overall_feedback)
    but scored independently against their own rubric and max_marks.

    resources are the FETCHABLE grounding artifacts (repo URL, known_now
    labels) — these get web_search/web_fetch tool access so their real,
    current content can be checked against the student's claims.

    readme_content is different in kind, not just another resource: it's
    the free-text project README/design writeup the student supplied when
    filling this midterm's pending resources (see
    CurriculumUploadService.fill_pending_resources), specifically prompted
    to reference exact file paths/functions for each design decision it
    describes. It must NOT be run through the same "treat every string as
    a URL to search for" resource_guidance path resources uses — it's
    prose to be spot-checked (its specific file/function claims verified
    against the fetched resources above), not a label to fetch. None when
    no pending-resource slot on this midterm was identified as the README.
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
    readme_content: Optional[str] = None


@dataclass
class MidtermRetestGenerationRequest:
    """Input to generate_midterm_retest (attempt_number >= 2).

    Mirrors RetestGenerationRequest's weak-areas-targeting idea, applied to
    the two-part Midterm shape. own_resources still gets web_search/
    web_fetch grounding on retry, matching generate_midterm's first
    attempt — Part 2 is being regenerated against the same real project,
    not a fresh one. readme_content mirrors MidtermGenerationRequest's
    field of the same name — see its docstring.
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
    readme_content: Optional[str] = None


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


class LLMToolBudgetExceededError(LLMValidationError):
    """Raised specifically when a tool-enabled generation (one of the 5
    web_search/web_fetch call sites) exhausts every attempt in its
    _TOOL_PATH_MAX_ATTEMPTS budget without producing valid output — see
    AnthropicLLMAdapter._retry(). A subclass of LLMValidationError so any
    existing `except LLMValidationError` handling still catches it, but
    distinct so callers that need to (send_assessment_job) can tell "this
    specific generation is expensive and keeps failing the same way, stop
    retrying it automatically" apart from an ordinary one-off schema
    mismatch that a ordinary retry legitimately recovers from.
    """

    def __init__(self, message: str, *, tokens_spent: int, attempts_made: int, ceiling: int):
        super().__init__(message)
        self.tokens_spent = tokens_spent
        self.attempts_made = attempts_made
        self.ceiling = ceiling


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
