"""Anthropic Claude adapter implementing LLMInterface."""
from __future__ import annotations

import json
from typing import Any

import anthropic

from app.adapters.nonfetchable_resources import build_resource_guidance
from app.config import get_settings
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    GradingRequest,
    GradingResult,
    LLMUnavailableError,
    LLMValidationError,
    MidtermGenerationRequest,
    MidtermGenerationResult,
    MidtermGradingRequest,
    MidtermGradingResult,
    MidtermRetestGenerationRequest,
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RescheduleCategory,
    RetestGenerationRequest,
)

# Server-side tools (see the claude-api skill for the current spec — these
# _20260209 variants support Opus 5/4.8/4.7/4.6, Sonnet 5, and Sonnet 4.6;
# the project's configured model qualifies). No beta header required.
# Only attached to calls that pass resources — never to standalone calls.
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 8,
}
WEB_FETCH_TOOL: dict[str, Any] = {
    "type": "web_fetch_20260209",
    "name": "web_fetch",
    "max_uses": 8,
    "max_content_tokens": 20000,
}

# Retry nudges appended on attempt > 0 (see _retry). The plain one covers
# ordinary schema-mismatch retries. The tool-aware one is for the 5
# tool-enabled call sites specifically: a real, observed failure mode is
# the model spending its whole max_tokens budget on web_search/web_fetch
# rounds before ever emitting the final JSON (LLMValidationError: "No text
# block in response content"). Raising max_tokens (below) reduces how
# often this happens; this nudge is the other half — on a retry, tell the
# model explicitly to stop searching and answer now rather than repeating
# the same multi-round search that likely caused the first failure.
_RETRY_NUDGE_PLAIN = "\n\nReturn ONLY valid JSON with no extra text."
_RETRY_NUDGE_TOOL_AWARE = (
    "\n\nReturn ONLY valid JSON with no extra text. If you were still "
    "gathering information via web_search/web_fetch, STOP calling tools "
    "now and respond immediately with your best answer using whatever "
    "you have already found — you are being retried because a previous "
    "attempt ran out of response budget before producing an answer."
)


class AnthropicLLMAdapter:
    """LLMInterface implementation using Anthropic Claude via the official SDK."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            max_retries=0,  # we manage retries ourselves for LLMValidationError
        )
        self._model = settings.llm_model
        self._max_retries = settings.llm_max_retries

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Call Claude and return the text response, mapping SDK errors.

        tools is opt-in — omitted entirely for standalone call sites, so their
        request shape (and therefore response.content shape) is unchanged.
        """
        try:
            kwargs: dict[str, Any] = dict(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if tools:
                kwargs["tools"] = tools
            message = self._client.messages.create(**kwargs)
            return self._extract_text(message.content)
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise LLMUnavailableError(f"Claude API unreachable: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError(f"Claude rate limit exhausted: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError(f"Claude API error {exc.status_code}: {exc.message}") from exc

    def _extract_text(self, content: list) -> str:
        """Return the concatenated text block(s) from a response.

        With no tools passed, content is always [text_block], same as the
        old `content[0].text`. With web_search/web_fetch enabled, content can
        contain server_tool_use/web_search_tool_result/web_fetch_tool_result
        blocks ahead of the final text block(s), so position 0 can no longer
        be assumed to be text.
        """
        parts = [block.text for block in content if getattr(block, "type", None) == "text"]
        if not parts:
            raise LLMValidationError(f"No text block in response content: {content!r}")
        return "\n".join(parts)

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Extract and parse JSON from a Claude response."""
        # Strip markdown code fences if present
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            # Drop opening fence (```json or ```) and closing fence
            inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            stripped = "\n".join(inner)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise LLMValidationError(f"Response is not valid JSON: {exc}\n\nRaw: {text}") from exc

    def _render(self, template_body: str, **kwargs: Any) -> str:
        """Render a prompt template body with keyword substitution."""
        try:
            return template_body.format(**kwargs)
        except KeyError as exc:
            raise LLMValidationError(f"Prompt template missing key: {exc}") from exc

    def _retry(self, fn: Any, *args: Any) -> Any:
        """Call fn(*args), retrying up to max_retries on LLMValidationError."""
        last_exc: LLMValidationError | None = None
        for attempt in range(self._max_retries):
            try:
                return fn(*args, attempt=attempt)
            except LLMValidationError as exc:
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    # ── LLMInterface methods ──────────────────────────────────────────────────

    def analyze_curriculum(
        self, request: CurriculumAnalysisRequest
    ) -> CurriculumAnalysisResult:
        def _attempt(req: CurriculumAnalysisRequest, attempt: int) -> CurriculumAnalysisResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                curriculum_content=req.curriculum_content,
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_PLAIN
            raw = self._call(prompt)
            data = self._parse_json(raw)
            try:
                return CurriculumAnalysisResult(
                    summary=str(data["summary"]),
                    key_topics=list(data["key_topics"]),
                    complexity_level=str(data["complexity_level"]),
                    estimated_study_hours=float(data["estimated_study_hours"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"analyze_curriculum schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def generate_assessment(
        self, request: AssessmentGenerationRequest
    ) -> AssessmentGenerationResult:
        def _attempt(req: AssessmentGenerationRequest, attempt: int) -> AssessmentGenerationResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                curriculum_content=req.curriculum_content,
                resource_guidance=build_resource_guidance(req.resources or []),
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.resources else None
            raw = self._call(prompt, max_tokens=16000, tools=tools)
            data = self._parse_json(raw)
            try:
                return AssessmentGenerationResult(
                    assessment_text=str(data["assessment_text"]),
                    rubric=str(data["rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_assessment schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def generate_midterm(
        self, request: MidtermGenerationRequest
    ) -> MidtermGenerationResult:
        def _attempt(req: MidtermGenerationRequest, attempt: int) -> MidtermGenerationResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                cumulative_pool_content=req.cumulative_pool_content,
                own_resources_list="\n".join(f"- {r}" for r in req.own_resources) or "(none)",
                resource_guidance=build_resource_guidance(req.own_resources),
                probe_focus=req.probe_focus or "(not specified)",
                part1_max_marks=req.part1_max_marks,
                part2_max_marks=req.part2_max_marks,
                readme_content=req.readme_content
                or "(No project README/design writeup was submitted for this midterm.)",
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.own_resources else None
            raw = self._call(prompt, max_tokens=16000, tools=tools)
            data = self._parse_json(raw)
            try:
                return MidtermGenerationResult(
                    part1_text=str(data["part1_text"]),
                    part1_rubric=str(data["part1_rubric"]),
                    part2_text=str(data["part2_text"]),
                    part2_rubric=str(data["part2_rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_midterm schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def generate_retest(
        self, request: RetestGenerationRequest
    ) -> AssessmentGenerationResult:
        def _attempt(req: RetestGenerationRequest, attempt: int) -> AssessmentGenerationResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                curriculum_content=req.curriculum_content,
                previous_mastery_score=req.previous_mastery_score,
                weak_areas=", ".join(req.weak_areas),
                attempt_number=req.attempt_number,
                resource_guidance=build_resource_guidance(req.resources or []),
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.resources else None
            raw = self._call(prompt, max_tokens=16000, tools=tools)
            data = self._parse_json(raw)
            try:
                return AssessmentGenerationResult(
                    assessment_text=str(data["assessment_text"]),
                    rubric=str(data["rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_retest schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def generate_midterm_retest(
        self, request: MidtermRetestGenerationRequest
    ) -> MidtermGenerationResult:
        def _attempt(req: MidtermRetestGenerationRequest, attempt: int) -> MidtermGenerationResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                cumulative_pool_content=req.cumulative_pool_content,
                own_resources_list="\n".join(f"- {r}" for r in req.own_resources) or "(none)",
                resource_guidance=build_resource_guidance(req.own_resources),
                probe_focus=req.probe_focus or "(not specified)",
                part1_max_marks=req.part1_max_marks,
                part2_max_marks=req.part2_max_marks,
                previous_part1_score=req.previous_part1_score,
                previous_part2_score=req.previous_part2_score,
                weak_areas=", ".join(req.weak_areas),
                attempt_number=req.attempt_number,
                readme_content=req.readme_content
                or "(No project README/design writeup was submitted for this midterm.)",
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.own_resources else None
            raw = self._call(prompt, max_tokens=16000, tools=tools)
            data = self._parse_json(raw)
            try:
                return MidtermGenerationResult(
                    part1_text=str(data["part1_text"]),
                    part1_rubric=str(data["part1_rubric"]),
                    part2_text=str(data["part2_text"]),
                    part2_rubric=str(data["part2_rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_midterm_retest schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def grade_submission(self, request: GradingRequest) -> GradingResult:
        def _attempt(req: GradingRequest, attempt: int) -> GradingResult:
            prompt = self._render(
                req.prompt_template_body,
                assessment_text=req.assessment_text,
                rubric=req.rubric,
                curriculum_content=req.curriculum_content,
                submission_content=req.submission_content,
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_PLAIN
            raw = self._call(prompt, max_tokens=4096)
            data = self._parse_json(raw)
            try:
                mastery_score = float(data["mastery_score"])
                if not (0.0 <= mastery_score <= 100.0):
                    raise ValueError(f"mastery_score {mastery_score} out of range 0–100")
                return GradingResult(
                    mastery_score=mastery_score,
                    weak_areas=list(data["weak_areas"]),
                    overall_feedback=str(data["overall_feedback"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"grade_submission schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def grade_midterm_submission(self, request: MidtermGradingRequest) -> MidtermGradingResult:
        def _attempt(req: MidtermGradingRequest, attempt: int) -> MidtermGradingResult:
            prompt = self._render(
                req.prompt_template_body,
                part1_text=req.part1_text,
                part1_rubric=req.part1_rubric,
                part2_text=req.part2_text,
                part2_rubric=req.part2_rubric,
                part1_max_marks=req.part1_max_marks,
                part2_max_marks=req.part2_max_marks,
                part1_submission_content=req.part1_submission_content,
                part2_submission_content=req.part2_submission_content,
                resource_guidance=build_resource_guidance(req.resources or []),
                readme_content=req.readme_content
                or "(No project README/design writeup was submitted for this midterm.)",
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.resources else None
            raw = self._call(prompt, max_tokens=8000, tools=tools)
            data = self._parse_json(raw)
            try:
                part1_score = float(data["part1_score"])
                part2_score = float(data["part2_score"])
                if not (0.0 <= part1_score <= req.part1_max_marks):
                    raise ValueError(
                        f"part1_score {part1_score} out of range 0-{req.part1_max_marks}"
                    )
                if not (0.0 <= part2_score <= req.part2_max_marks):
                    raise ValueError(
                        f"part2_score {part2_score} out of range 0-{req.part2_max_marks}"
                    )
                return MidtermGradingResult(
                    part1_score=part1_score,
                    part2_score=part2_score,
                    weak_areas=list(data["weak_areas"]),
                    overall_feedback=str(data["overall_feedback"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"grade_midterm_submission schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        _valid_categories: set[str] = {
            "interview", "medical", "emergency", "work_escalation",
            "procrastination", "lack_of_preparation", "missed_schedule",
        }

        def _attempt(req: RescheduleClassificationRequest, attempt: int) -> RescheduleClassificationResult:
            prompt = self._render(
                req.prompt_template_body,
                reason=req.reason,
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_PLAIN
            raw = self._call(prompt, max_tokens=1024)
            data = self._parse_json(raw)
            try:
                category = str(data["category"])
                if category not in _valid_categories:
                    raise ValueError(f"Unknown category: {category!r}")
                return RescheduleClassificationResult(
                    category=category,  # type: ignore[arg-type]
                    reasoning=str(data["reasoning"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"classify_reschedule schema mismatch: {exc}\n\nData: {data}") from exc

        return self._retry(_attempt, request)
