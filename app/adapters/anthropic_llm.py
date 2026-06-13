"""Anthropic Claude adapter implementing LLMInterface."""
from __future__ import annotations

import json
from typing import Any

import anthropic

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
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RescheduleCategory,
    RetestGenerationRequest,
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

    def _call(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Claude and return the text response, mapping SDK errors."""
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise LLMUnavailableError(f"Claude API unreachable: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailableError(f"Claude rate limit exhausted: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailableError(f"Claude API error {exc.status_code}: {exc.message}") from exc

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
                prompt += "\n\nReturn ONLY valid JSON with no extra text."
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
            )
            if attempt > 0:
                prompt += "\n\nReturn ONLY valid JSON with no extra text."
            raw = self._call(prompt, max_tokens=8192)
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
            )
            if attempt > 0:
                prompt += "\n\nReturn ONLY valid JSON with no extra text."
            raw = self._call(prompt, max_tokens=8192)
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
                prompt += "\n\nReturn ONLY valid JSON with no extra text."
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
                prompt += "\n\nReturn ONLY valid JSON with no extra text."
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
