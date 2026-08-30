"""Anthropic Claude adapter implementing LLMInterface."""
from __future__ import annotations

import json
import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

# USD per million tokens — Claude Sonnet 4.6 standard (non-batch) API
# pricing, verified 2026-08-30 against Anthropic's published rates, not
# recalled from training data (see the 2026-08-30 credit-usage report this
# was added for). Cache-write rate is the 5-minute-TTL price; this codebase
# never sets a 1-hour TTL. For an estimate only, logged alongside every
# call — the Anthropic console remains the authoritative source for actual
# billed cost.
_PRICE_PER_MTOK_INPUT = 3.00
_PRICE_PER_MTOK_OUTPUT = 15.00
_PRICE_PER_MTOK_CACHE_WRITE = 3.75
_PRICE_PER_MTOK_CACHE_READ = 0.30


def _estimate_cost_usd(usage: Any) -> float:
    input_tok = getattr(usage, "input_tokens", 0) or 0
    output_tok = getattr(usage, "output_tokens", 0) or 0
    cache_write_tok = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tok = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (
        input_tok / 1_000_000 * _PRICE_PER_MTOK_INPUT
        + output_tok / 1_000_000 * _PRICE_PER_MTOK_OUTPUT
        + cache_write_tok / 1_000_000 * _PRICE_PER_MTOK_CACHE_WRITE
        + cache_read_tok / 1_000_000 * _PRICE_PER_MTOK_CACHE_READ
    )

from app.adapters.nonfetchable_resources import build_resource_guidance
from app.config import get_settings
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    GradingRequest,
    GradingResult,
    LLMToolBudgetExceededError,
    LLMUnavailableError,
    LLMValidationError,
    MidtermGenerationRequest,
    current_llm_log_context,
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
#
# max_uses capped low after a confirmed incident (2026-08-30): a tool-enabled
# call that exhausts its round-trip budget without producing a final answer
# raises LLMValidationError ("No text block in response content"), which
# _retry() retried up to 3 times — each retry re-running the tool-use cycle
# again, each round re-sending accumulated context. Two real incidents both
# burned significant credit this way, and in both, every retry failed
# identically.
#
# allowed_callers=["direct"] (2026-08-30, second incident): confirmed via a
# live, authorized single call that even with the max_uses caps in place, one
# *successful* attempt still cost 316k input tokens. Root cause, confirmed
# from the raw response content, not assumed: these _20260209 tool versions
# use code execution internally (Anthropic's own docs: "the _20260209
# versions of web search and web fetch use code execution internally to
# apply dynamic filters against search results") — by default this lets the
# model call web_search/web_fetch as awaitable functions from inside a
# code_execution sandbox (`await web_search({...})`) rather than only as
# direct top-level tool calls. In the captured incident the model used this
# to repeatedly re-print full fetched page content back into its own context
# across 19 separate code_execution rounds (e.g. printing one ~26,600-char
# page in five sequential 5,000-char slices) — each round resending the
# growing transcript, which is what actually drove the token count, not the
# search/fetch call counts themselves (those were correctly capped at
# exactly 2 and 3 the whole time — max_uses was never bypassed).
# allowed_callers restricts these two tools to direct invocation only,
# closing that avenue at its source rather than trying to separately cap a
# tool (code_execution) that isn't declared here and whose availability is
# an internal side effect of using web_search_20260209/web_fetch_20260209 —
# Anthropic's own docs warn that adding a *separate* standalone
# code_execution tool alongside these versions "creates two execution
# environments, which can confuse the model," so that path was deliberately
# not taken. Not yet confirmed against a second live call (see the
# 2026-08-30 incident report) — the mechanism is well-documented and
# directly matches the observed failure, but treat it as verified only once
# re-tested.
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 2,
    "allowed_callers": ["direct"],
}
WEB_FETCH_TOOL: dict[str, Any] = {
    "type": "web_fetch_20260209",
    "name": "web_fetch",
    # Lowered from 3: the incident's 3rd web_fetch call was a same-URL retry
    # after the model's own code hit a TypeError, not a genuine 3rd distinct
    # resource — one fetch per resource (2 resources) is the legitimate case.
    "max_uses": 2,
    # Lowered from 20000: a secondary, independent backstop that shrinks the
    # raw page content available to be re-injected into context in the first
    # place, regardless of whether allowed_callers fully closes the
    # code_execution avenue above.
    "max_content_tokens": 8000,
    "allowed_callers": ["direct"],
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

# Tool-enabled call sites get a tighter attempt cap than the general
# schema-mismatch retry budget (settings.llm_max_retries, still 3 for
# non-tool paths). Both real incidents showed every retry of a tool-budget
# failure failing identically — a 3rd blind repeat has never once helped
# and just re-pays for the same failure. 2 total attempts (the original +
# one adaptive retry carrying _RETRY_NUDGE_TOOL_AWARE) is what's actually
# justified: enough to recover a genuine one-off stumble, not enough to
# burn a third full tool-use cycle chasing a failure that's already
# repeated once.
_TOOL_PATH_MAX_ATTEMPTS = 2


class _CallBudget:
    """Defense-in-depth circuit breaker, independent of the tool-round-trip
    cap and the reduced attempt count above. Tracks cumulative input+output
    tokens spent across every attempt within one logical generate_*/
    grade_*_submission invocation (i.e. across one _retry() loop, all
    attempts combined) — not just one API call. If a single attempt still
    burns an unreasonable amount despite both caps above (e.g. unusually
    large web_fetch results), this stops the *next* attempt from even
    starting, rather than waiting for max_attempts to run out on its own.
    """

    def __init__(self, ceiling: int) -> None:
        self.ceiling = ceiling
        self.spent = 0

    def charge(self, tokens: int) -> None:
        self.spent += tokens

    def exhausted(self) -> bool:
        return self.spent >= self.ceiling


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
        self._tool_call_budget_tokens = settings.llm_tool_call_budget_tokens

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call(
        self,
        prompt: str,
        max_tokens: int = 4096,
        tools: list[dict[str, Any]] | None = None,
        budget: "_CallBudget | None" = None,
    ) -> str:
        """Call Claude and return the text response, mapping SDK errors.

        tools is opt-in — omitted entirely for standalone call sites, so their
        request shape (and therefore response.content shape) is unchanged.

        budget is opt-in too (only the 5 tool-enabled call sites pass one) —
        see _CallBudget. Checked BEFORE spending anything on this attempt,
        so an already-exhausted budget aborts without making another API
        call at all.
        """
        if budget is not None and budget.exhausted():
            raise LLMValidationError(
                f"Circuit breaker: {budget.spent} tokens already spent across "
                f"prior attempts in this generation (ceiling {budget.ceiling}) "
                "without producing valid output — aborting rather than "
                "spending further on a call that keeps failing the same way."
            )
        try:
            kwargs: dict[str, Any] = dict(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            if tools:
                kwargs["tools"] = tools
            message = self._client.messages.create(**kwargs)
            usage = getattr(message, "usage", None)
            if usage is not None:
                spent = usage.input_tokens + usage.output_tokens
                if budget is not None:
                    budget.charge(spent)
                logger.info(
                    "Anthropic call: context=%s model=%s input_tokens=%s "
                    "output_tokens=%s cache_creation_input_tokens=%s "
                    "cache_read_input_tokens=%s tools=%s est_cost_usd=%.4f",
                    current_llm_log_context(), self._model, usage.input_tokens, usage.output_tokens,
                    getattr(usage, "cache_creation_input_tokens", None),
                    getattr(usage, "cache_read_input_tokens", None),
                    bool(tools), _estimate_cost_usd(usage),
                )
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
            # Bounded, specific diagnostic instead of dumping the full
            # content!r (which, for a tool-heavy failure, can be dozens of
            # blocks including full fetched-page text — unusable as a log
            # line and expensive to even format). Enough to diagnose WHY
            # without re-running anything: how many rounds, which tool(s),
            # what the last thing attempted was.
            tool_round_counts: dict[str, int] = {}
            for block in content:
                if getattr(block, "type", None) == "server_tool_use":
                    name = getattr(block, "name", None) or "?"
                    tool_round_counts[name] = tool_round_counts.get(name, 0) + 1
            last_block = str(content[-1])[:300] if content else "(empty content)"
            raise LLMValidationError(
                f"No text block in response content after {len(content)} block(s) "
                f"({sum(tool_round_counts.values())} tool-use round(s): "
                f"{tool_round_counts}). Last block: {last_block!r}"
            )
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

    def _retry(
        self,
        fn: Any,
        *args: Any,
        max_attempts: int | None = None,
        budget_ceiling: int | None = None,
    ) -> Any:
        """Call fn(*args, attempt=N, budget=...), retrying up to max_attempts
        on LLMValidationError.

        max_attempts defaults to settings.llm_max_retries when omitted; the
        5 tool-enabled call sites pass _TOOL_PATH_MAX_ATTEMPTS explicitly
        instead (see its docstring). budget_ceiling, when given, constructs
        one _CallBudget shared across every attempt in this call — also
        tool-enabled call sites only.

        When budget_ceiling was given (i.e. this is one of the 5
        tool-enabled call sites) and every attempt is exhausted, the final
        raise is LLMToolBudgetExceededError rather than the plain
        LLMValidationError from the last attempt — this is the one shared
        point all 5 tool-enabled methods route through, so callers
        (send_assessment_job) can catch it specifically and mark the row
        for manual diagnosis instead of leaving it silently retryable.
        """
        attempts = max_attempts if max_attempts is not None else self._max_retries
        budget = _CallBudget(budget_ceiling) if budget_ceiling is not None else None
        last_exc: LLMValidationError | None = None
        for attempt in range(attempts):
            try:
                return fn(*args, attempt=attempt, budget=budget)
            except LLMValidationError as exc:
                last_exc = exc
        assert last_exc is not None
        if budget is not None:
            raise LLMToolBudgetExceededError(
                f"Tool-enabled generation exhausted all {attempts} attempt(s) "
                f"without producing valid output — {budget.spent} token(s) "
                f"spent (ceiling {budget.ceiling}). Not retrying further; "
                f"this needs manual diagnosis, not another automatic attempt. "
                f"Last failure: {last_exc}",
                tokens_spent=budget.spent,
                attempts_made=attempts,
                ceiling=budget.ceiling,
            ) from last_exc
        raise last_exc

    # ── LLMInterface methods ──────────────────────────────────────────────────

    def analyze_curriculum(
        self, request: CurriculumAnalysisRequest
    ) -> CurriculumAnalysisResult:
        def _attempt(
            req: CurriculumAnalysisRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> CurriculumAnalysisResult:
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
        def _attempt(
            req: AssessmentGenerationRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> AssessmentGenerationResult:
            prompt = self._render(
                req.prompt_template_body,
                topic=req.topic,
                curriculum_content=req.curriculum_content,
                resource_guidance=build_resource_guidance(req.resources or []),
            )
            if attempt > 0:
                prompt += _RETRY_NUDGE_TOOL_AWARE
            tools = [WEB_SEARCH_TOOL, WEB_FETCH_TOOL] if req.resources else None
            raw = self._call(prompt, max_tokens=16000, tools=tools, budget=budget)
            data = self._parse_json(raw)
            try:
                return AssessmentGenerationResult(
                    assessment_text=str(data["assessment_text"]),
                    rubric=str(data["rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_assessment schema mismatch: {exc}\n\nData: {data}") from exc

        if request.resources:
            return self._retry(
                _attempt, request,
                max_attempts=_TOOL_PATH_MAX_ATTEMPTS, budget_ceiling=self._tool_call_budget_tokens,
            )
        return self._retry(_attempt, request)

    def generate_midterm(
        self, request: MidtermGenerationRequest
    ) -> MidtermGenerationResult:
        def _attempt(
            req: MidtermGenerationRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> MidtermGenerationResult:
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
            raw = self._call(prompt, max_tokens=16000, tools=tools, budget=budget)
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

        if request.own_resources:
            return self._retry(
                _attempt, request,
                max_attempts=_TOOL_PATH_MAX_ATTEMPTS, budget_ceiling=self._tool_call_budget_tokens,
            )
        return self._retry(_attempt, request)

    def generate_retest(
        self, request: RetestGenerationRequest
    ) -> AssessmentGenerationResult:
        def _attempt(
            req: RetestGenerationRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> AssessmentGenerationResult:
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
            raw = self._call(prompt, max_tokens=16000, tools=tools, budget=budget)
            data = self._parse_json(raw)
            try:
                return AssessmentGenerationResult(
                    assessment_text=str(data["assessment_text"]),
                    rubric=str(data["rubric"]),
                    duration_minutes=int(data["duration_minutes"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMValidationError(f"generate_retest schema mismatch: {exc}\n\nData: {data}") from exc

        if request.resources:
            return self._retry(
                _attempt, request,
                max_attempts=_TOOL_PATH_MAX_ATTEMPTS, budget_ceiling=self._tool_call_budget_tokens,
            )
        return self._retry(_attempt, request)

    def generate_midterm_retest(
        self, request: MidtermRetestGenerationRequest
    ) -> MidtermGenerationResult:
        def _attempt(
            req: MidtermRetestGenerationRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> MidtermGenerationResult:
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
            raw = self._call(prompt, max_tokens=16000, tools=tools, budget=budget)
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

        if request.own_resources:
            return self._retry(
                _attempt, request,
                max_attempts=_TOOL_PATH_MAX_ATTEMPTS, budget_ceiling=self._tool_call_budget_tokens,
            )
        return self._retry(_attempt, request)

    def grade_submission(self, request: GradingRequest) -> GradingResult:
        def _attempt(
            req: GradingRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> GradingResult:
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
        def _attempt(
            req: MidtermGradingRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> MidtermGradingResult:
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
            raw = self._call(prompt, max_tokens=8000, tools=tools, budget=budget)
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

        if request.resources:
            return self._retry(
                _attempt, request,
                max_attempts=_TOOL_PATH_MAX_ATTEMPTS, budget_ceiling=self._tool_call_budget_tokens,
            )
        return self._retry(_attempt, request)

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        _valid_categories: set[str] = {
            "interview", "medical", "emergency", "work_escalation",
            "procrastination", "lack_of_preparation", "missed_schedule",
        }

        def _attempt(
            req: RescheduleClassificationRequest, attempt: int, budget: "_CallBudget | None" = None
        ) -> RescheduleClassificationResult:
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
