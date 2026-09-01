"""Unit tests for AnthropicLLMAdapter.

All Anthropic API calls are monkeypatched — no network traffic.
Tests cover:
  - _parse_json: valid JSON, code-fenced JSON, invalid JSON, narration
    prefacing a fenced block (the tool-path fence-detection regression)
  - _render: template substitution, missing keys
  - _call: all SDK error types mapped to LLMUnavailableError
  - _retry: exhaustion, recovery, call counts, retry-nudge injection
  - All 5 public LLMInterface methods: happy paths, schema failures,
    boundary conditions, recovery after one bad response
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import anthropic
import httpx
import pytest

from app.adapters.anthropic_llm import (
    WEB_FETCH_TOOL,
    WEB_SEARCH_TOOL,
    AnthropicLLMAdapter,
    _estimate_cost_usd,
)
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    CurriculumAnalysisRequest,
    GradingRequest,
    LLMToolBudgetExceededError,
    LLMUnavailableError,
    LLMValidationError,
    MidtermGenerationRequest,
    MidtermGradingRequest,
    MidtermRetestGenerationRequest,
    RescheduleClassificationRequest,
    RetestGenerationRequest,
    filter_fetchable_resources,
    llm_log_context,
)


# ── Shared httpx stubs for constructing anthropic SDK exceptions ──────────────

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
_RESPONSE_429 = httpx.Response(429, request=_REQUEST)
_RESPONSE_500 = httpx.Response(500, request=_REQUEST)
_RESPONSE_503 = httpx.Response(503, request=_REQUEST)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def adapter(monkeypatch):
    """AnthropicLLMAdapter initialised with a fake API key (no network calls)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    monkeypatch.setenv("LLM_MAX_RETRIES", "3")
    from app.config import get_settings
    get_settings.cache_clear()
    instance = AnthropicLLMAdapter()
    yield instance
    get_settings.cache_clear()


def _fake_response(text: str, input_tokens: int = 100, output_tokens: int = 200) -> MagicMock:
    """Build a fake anthropic.Message with one text content block and real
    integer usage figures — needed since _CallBudget does real arithmetic
    on usage.input_tokens/output_tokens (a bare MagicMock default there
    would silently break its int comparisons rather than emulate real
    SDK behavior, where usage is always present and numeric)."""
    msg = MagicMock()
    msg.content = [MagicMock(type="text", text=text)]
    msg.usage = MagicMock(
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_creation_input_tokens=0, cache_read_input_tokens=0,
    )
    return msg


def _patch_create(adapter: AnthropicLLMAdapter, *responses) -> MagicMock:
    """Replace adapter._client.messages.create with a mock.

    Each positional arg is either:
      - a str → return _fake_response(str) on that call
      - an Exception instance → raise it on that call
    """
    effects = []
    for r in responses:
        if isinstance(r, BaseException):
            effects.append(r)
        else:
            effects.append(_fake_response(r))

    mock = MagicMock(side_effect=effects)
    adapter._client.messages.create = mock
    return mock


# ── _parse_json ───────────────────────────────────────────────────────────────


class TestParseJson:

    def test_plain_json_object(self, adapter):
        result = adapter._parse_json('{"key": "value", "n": 42}')
        assert result == {"key": "value", "n": 42}

    def test_json_with_whitespace(self, adapter):
        result = adapter._parse_json('  \n{"a": 1}\n  ')
        assert result == {"a": 1}

    def test_json_fenced_with_language_marker(self, adapter):
        text = '```json\n{"summary": "ok", "score": 90}\n```'
        result = adapter._parse_json(text)
        assert result == {"summary": "ok", "score": 90}

    def test_json_fenced_without_language_marker(self, adapter):
        text = '```\n{"category": "medical"}\n```'
        result = adapter._parse_json(text)
        assert result == {"category": "medical"}

    def test_json_fenced_without_closing_fence(self, adapter):
        # Claude sometimes omits the closing fence — should still parse content
        text = '```json\n{"key": "value"}'
        result = adapter._parse_json(text)
        assert result == {"key": "value"}

    def test_invalid_json_raises_validation_error(self, adapter):
        with pytest.raises(LLMValidationError, match="not valid JSON"):
            adapter._parse_json("This is just plain text, not JSON.")

    def test_truncated_json_raises_validation_error(self, adapter):
        with pytest.raises(LLMValidationError):
            adapter._parse_json('{"key": "val')

    def test_empty_string_raises_validation_error(self, adapter):
        with pytest.raises(LLMValidationError):
            adapter._parse_json("")


_FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParseJsonToolPathNarration:
    """Regression coverage for the tool-path fence-detection bug: on the
    tool-enabled generation path, Claude reliably prefaces its JSON answer
    with narration ("I'll fetch all the curriculum resources...") before
    the fenced JSON block, even when the prompt explicitly says to return
    only JSON. The old `if stripped.startswith("```")` check only stripped
    a fence when it was the literal first characters of the response, so
    any narration prefix meant json.loads ran directly on prose, producing
    "Expecting value: line 1 column 1 (char 0)" — this was the real,
    confirmed cause of PerfOpt I's needs_manual_diagnosis trip in production.
    """

    def test_narration_before_fenced_json_no_longer_fails_at_position_zero(self, adapter):
        """Uses the EXACT raw response text captured from a real, live
        Anthropic API call made while diagnosing PerfOpt I's failure
        (tokens_spent=83442/200000 across 2 attempts) — not an approximation.

        This exact real response turns out to carry a SECOND, independent
        defect beyond the narration prefix: the model's own JSON has an
        unescaped `"` inside a string value ( called "tree-shaking" ),
        making it genuinely invalid JSON regardless of fence-detection.
        That was missed by the earlier diagnosis, which read the content
        for plausibility but never actually ran it through json.loads.

        So this fixture can't prove a full successful parse end-to-end —
        but it does prove the specific bug this fix targets is gone: the
        failure is no longer "Expecting value: line 1 column 1 (char 0)"
        (fence never found, json.loads called on raw narration). It now
        correctly locates and enters the fenced JSON block, and fails
        *inside* it with a normal, specific JSONDecodeError pointing at the
        real syntax problem — proof the fence was found and stripped this
        time, not that everything about this response is fine.
        """
        text = (_FIXTURES_DIR / "perfopt_narration_before_json_response.txt").read_text()
        with pytest.raises(LLMValidationError) as exc_info:
            adapter._parse_json(text)
        message = str(exc_info.value)
        assert "line 1 column 1 (char 0)" not in message
        assert "Expecting ',' delimiter" in message

    def test_narration_before_valid_fenced_json_parses_successfully(self, adapter):
        """Same narration-then-fence shape as the real captured failure
        above, but with a correctly-escaped JSON body — isolates the fix
        under test (fence-finding) from that fixture's unrelated
        JSON-escaping defect, proving fence-finding works end-to-end once
        the JSON itself is well-formed."""
        text = (
            "I'll fetch all the curriculum resources simultaneously before "
            "writing the assessment.\n"
            "I have sufficient information from the fetched sources to "
            "produce the assessment.\n\n"
            "```json\n"
            '{"assessment_text": "Q1: What is tree-shaking?", '
            '"rubric": "Full marks for a correct definition.", '
            '"duration_minutes": 75}\n'
            "```"
        )
        result = adapter._parse_json(text)
        assert result == {
            "assessment_text": "Q1: What is tree-shaking?",
            "rubric": "Full marks for a correct definition.",
            "duration_minutes": 75,
        }

    def test_narration_after_fenced_json_still_parses(self, adapter):
        """Claude sometimes adds a closing remark after the fenced JSON
        instead of (or in addition to) a preface — the fence must be found
        regardless of what follows it too."""
        text = (
            "```json\n"
            '{"category": "medical", "confidence": "high"}\n'
            "```\n\n"
            "I used the general-knowledge fallback for the ByteByteGo "
            "resource since it requires a paid subscription."
        )
        result = adapter._parse_json(text)
        assert result == {"category": "medical", "confidence": "high"}

    def test_narration_around_malformed_json_still_fails_cleanly(self, adapter):
        """A narration prefix must not cause the fallback brace-search to
        mask genuinely malformed JSON — it should still raise
        LLMValidationError, not silently return garbage or hang."""
        text = (
            "I'll use my general knowledge for this one.\n\n"
            '```json\n{"assessment_text": "unterminated\n```'
        )
        with pytest.raises(LLMValidationError, match="not valid JSON"):
            adapter._parse_json(text)

    def test_pure_json_no_narration_unaffected(self, adapter):
        """The existing no-narration, no-fence happy path must be
        unaffected by the fence-search rewrite."""
        result = adapter._parse_json('{"key": "value", "n": 42}')
        assert result == {"key": "value", "n": 42}


# ── _render ───────────────────────────────────────────────────────────────────


class TestRender:

    def test_renders_all_placeholders(self, adapter):
        result = adapter._render("Topic: {topic}, Level: {level}", topic="Python", level="advanced")
        assert result == "Topic: Python, Level: advanced"

    def test_template_with_no_placeholders(self, adapter):
        result = adapter._render("Static prompt body.")
        assert result == "Static prompt body."

    def test_missing_key_raises_validation_error(self, adapter):
        with pytest.raises(LLMValidationError, match="missing key"):
            adapter._render("Hello {name} from {place}", name="Alice")

    def test_extra_kwargs_are_ignored(self, adapter):
        result = adapter._render("Hello {name}", name="Alice", unused="extra")
        assert result == "Hello Alice"


# ── _call: SDK error mapping ──────────────────────────────────────────────────


class TestCallErrorMapping:

    def test_successful_call_returns_text(self, adapter):
        adapter._client.messages.create = MagicMock(return_value=_fake_response("hello world"))
        assert adapter._call("prompt") == "hello world"

    def test_timeout_error_raises_unavailable(self, adapter):
        adapter._client.messages.create = MagicMock(
            side_effect=anthropic.APITimeoutError(request=_REQUEST)
        )
        with pytest.raises(LLMUnavailableError, match="unreachable"):
            adapter._call("prompt")

    def test_connection_error_raises_unavailable(self, adapter):
        adapter._client.messages.create = MagicMock(
            side_effect=anthropic.APIConnectionError(request=_REQUEST)
        )
        with pytest.raises(LLMUnavailableError, match="unreachable"):
            adapter._call("prompt")

    def test_rate_limit_error_raises_unavailable(self, adapter):
        adapter._client.messages.create = MagicMock(
            side_effect=anthropic.RateLimitError(
                "Rate limit hit", response=_RESPONSE_429, body={}
            )
        )
        with pytest.raises(LLMUnavailableError, match="rate limit"):
            adapter._call("prompt")

    def test_server_error_raises_unavailable_with_status_code(self, adapter):
        adapter._client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                "Internal server error", response=_RESPONSE_500, body={}
            )
        )
        exc = pytest.raises(LLMUnavailableError, match="500")
        with exc:
            adapter._call("prompt")

    def test_503_raises_unavailable(self, adapter):
        adapter._client.messages.create = MagicMock(
            side_effect=anthropic.APIStatusError(
                "Service unavailable", response=_RESPONSE_503, body={}
            )
        )
        with pytest.raises(LLMUnavailableError, match="503"):
            adapter._call("prompt")

    def test_call_passes_correct_model_and_messages(self, adapter):
        mock = MagicMock(return_value=_fake_response("response"))
        adapter._client.messages.create = mock
        adapter._call("my prompt", max_tokens=2048)
        mock.assert_called_once_with(
            model=adapter._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": "my prompt"}],
        )


# ── _retry: logic and call counts ─────────────────────────────────────────────


class TestRetryLogic:

    def _assessment_req(self, template: str = "Generate for {topic} using {curriculum_content}"):
        return AssessmentGenerationRequest(
            topic="Python",
            curriculum_content="notes",
            prompt_template_body=template,
        )

    def _good_assessment_json(self) -> str:
        return json.dumps({
            "assessment_text": "Describe async/await.",
            "rubric": "Award marks for accuracy.",
            "duration_minutes": 60,
        })

    def _bad_json(self) -> str:
        return "not json at all"

    def test_success_on_first_attempt_makes_one_api_call(self, adapter):
        mock = _patch_create(adapter, self._good_assessment_json())
        adapter.generate_assessment(self._assessment_req())
        assert mock.call_count == 1

    def test_recovery_after_one_bad_response_makes_two_api_calls(self, adapter):
        mock = _patch_create(adapter, self._bad_json(), self._good_assessment_json())
        adapter.generate_assessment(self._assessment_req())
        assert mock.call_count == 2

    def test_all_retries_exhausted_raises_validation_error(self, adapter):
        mock = _patch_create(adapter, self._bad_json(), self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(self._assessment_req())
        assert mock.call_count == 3

    def test_exhaustion_call_count_equals_max_retries(self, adapter):
        adapter._max_retries = 2
        mock = _patch_create(adapter, self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(self._assessment_req())
        assert mock.call_count == 2

    def test_unavailable_error_is_not_retried(self, adapter):
        mock = _patch_create(
            adapter,
            anthropic.APITimeoutError(request=_REQUEST),
        )
        with pytest.raises(LLMUnavailableError):
            adapter.generate_assessment(self._assessment_req())
        assert mock.call_count == 1

    def test_retry_nudge_appended_on_second_attempt(self, adapter):
        mock = _patch_create(adapter, self._bad_json(), self._good_assessment_json())
        adapter.generate_assessment(self._assessment_req())

        first_prompt = mock.call_args_list[0][1]["messages"][0]["content"]
        second_prompt = mock.call_args_list[1][1]["messages"][0]["content"]
        assert "Return ONLY valid JSON" not in first_prompt
        assert "Return ONLY valid JSON" in second_prompt

    def test_no_nudge_on_first_attempt(self, adapter):
        mock = _patch_create(adapter, self._good_assessment_json())
        adapter.generate_assessment(self._assessment_req())
        prompt = mock.call_args_list[0][1]["messages"][0]["content"]
        assert "Return ONLY valid JSON" not in prompt

    def test_tool_stop_instruction_appended_on_retry_for_tool_enabled_call(self, adapter):
        """Regression test for the tool-budget-exhaustion fix: on a retry of
        a tool-enabled call, the model must be explicitly told to stop
        calling tools and answer now — not just told to return valid JSON,
        which alone doesn't address a response that ran out of budget
        mid-tool-use (see LLMValidationError: "No text block in response
        content", the real failure this session's dry run hit)."""
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        mock = _patch_create(adapter, self._bad_json(), self._good_assessment_json())
        adapter.generate_assessment(req)

        first_prompt = mock.call_args_list[0][1]["messages"][0]["content"]
        second_prompt = mock.call_args_list[1][1]["messages"][0]["content"]
        assert "STOP calling tools" not in first_prompt
        assert "STOP calling tools" in second_prompt


# ── Regression tests for the 2026-08-30 runaway-spend fix ─────────────────────
# Root cause: a tool-budget-exhaustion failure (LLMValidationError: "No text
# block in response content") was retried up to llm_max_retries (3) times on
# tool-enabled paths, each retry re-running the full (then-16-round) tool-use
# cycle — and in two real incidents, every retry failed identically. The fix
# has three independent layers, each tested below: (1) a lower tool
# round-trip cap, (2) a tighter attempt cap specifically for tool-enabled
# calls, (3) a circuit breaker on cumulative spend across attempts.


class TestToolRoundTripCap:
    """Layer 1: bound the cost of a single attempt directly, not just how
    many times an expensive attempt gets repeated."""

    def test_web_search_max_uses_is_capped_low(self):
        assert WEB_SEARCH_TOOL["max_uses"] <= 3

    def test_web_fetch_max_uses_is_capped_low(self):
        assert WEB_FETCH_TOOL["max_uses"] <= 4

    def test_combined_tool_round_trip_budget_is_in_4_to_6_range(self):
        total = WEB_SEARCH_TOOL["max_uses"] + WEB_FETCH_TOOL["max_uses"]
        assert 4 <= total <= 6


class TestToolPathAttemptCap:
    """Layer 2: tool-enabled calls get fewer total attempts than the
    general schema-mismatch retry budget — both real incidents showed a
    3rd blind repeat never once recovering a tool-budget failure."""

    def _bad_json(self) -> str:
        return "not json at all"

    def _good_assessment_json(self) -> str:
        return json.dumps({
            "assessment_text": "text", "rubric": "rubric", "duration_minutes": 60,
        })

    def test_tool_enabled_call_makes_at_most_2_attempts(self, adapter):
        """3 bad responses queued, but a tool-enabled call must stop after
        2 — proving it doesn't reach for the 3rd even though settings.
        llm_max_retries is 3 and a 3rd response is available."""
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        mock = _patch_create(adapter, self._bad_json(), self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(req)
        assert mock.call_count == 2

    def test_non_tool_call_still_gets_full_3_attempts(self, adapter):
        """The tighter cap is scoped to tool-enabled calls only — a request
        with no resources (no tools attached) must be unaffected."""
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=[],
        )
        mock = _patch_create(adapter, self._bad_json(), self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(req)
        assert mock.call_count == 3

    def test_grade_midterm_submission_tool_enabled_makes_at_most_2_attempts(self, adapter):
        """Confirms the cap applies to grade_midterm_submission too — the
        one grading path that IS tool-enabled, unlike grade_submission."""
        req = MidtermGradingRequest(
            part1_text="p1", part1_rubric="r1", part2_text="p2", part2_rubric="r2",
            part1_max_marks=30.0, part2_max_marks=70.0,
            part1_submission_content="a1", part2_submission_content="a2",
            prompt_template_body="{part1_text}{part1_rubric}{part2_text}{part2_rubric}"
                                  "{part1_max_marks}{part2_max_marks}"
                                  "{part1_submission_content}{part2_submission_content}"
                                  "{resource_guidance}{readme_content}",
            resources=["https://github.com/example/repo"],
        )
        mock = _patch_create(adapter, self._bad_json(), self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError):
            adapter.grade_midterm_submission(req)
        assert mock.call_count == 2


class TestCircuitBreaker:
    """Layer 3: defense-in-depth. Even within the 2-attempt cap, if the
    first attempt alone already spent at/above the ceiling, the second
    attempt must abort WITHOUT making another API call."""

    def _bad_json(self) -> str:
        return "not json at all"

    def _good_assessment_json(self) -> str:
        return json.dumps({
            "assessment_text": "text", "rubric": "rubric", "duration_minutes": 60,
        })

    def test_second_attempt_skipped_when_first_attempt_exceeds_ceiling(self, adapter, monkeypatch):
        monkeypatch.setenv("LLM_TOOL_CALL_BUDGET_TOKENS", "1000")
        from app.config import get_settings
        get_settings.cache_clear()
        adapter2 = AnthropicLLMAdapter()  # picks up the lowered ceiling

        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        # First (and only, if the breaker works) call spends 5000 tokens —
        # already over the 1000-token ceiling — and fails validation.
        big_response = _fake_response(self._bad_json(), input_tokens=4000, output_tokens=1000)
        mock = MagicMock(return_value=big_response)
        adapter2._client.messages.create = mock

        with pytest.raises(LLMValidationError, match="Circuit breaker"):
            adapter2.generate_assessment(req)

        assert mock.call_count == 1  # the would-be 2nd attempt never made a real API call
        get_settings.cache_clear()

    def test_stays_under_ceiling_gets_normal_second_attempt(self, adapter, monkeypatch):
        """Sanity check the breaker isn't over-triggering: a small first
        failure well under the ceiling must still get its normal retry."""
        monkeypatch.setenv("LLM_TOOL_CALL_BUDGET_TOKENS", "1_000_000".replace("_", ""))
        from app.config import get_settings
        get_settings.cache_clear()
        adapter2 = AnthropicLLMAdapter()

        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        mock = _patch_create(adapter2, self._bad_json(), self._good_assessment_json())
        result = adapter2.generate_assessment(req)

        assert result.assessment_text == "text"
        assert mock.call_count == 2
        get_settings.cache_clear()


# ── Regression tests for the 2026-08-30 (second incident) code_execution fix ──
# Root cause, confirmed from the raw captured response, not assumed: a
# *successful* single tool-enabled attempt still cost 316k input tokens,
# even with the max_uses caps above in place and correctly enforced (2
# web_search + 3 web_fetch calls, exactly at the caps). The actual driver was
# the model repeatedly re-printing fetched page content across 19
# code_execution rounds — code_execution is available as a side effect of
# using web_search_20260209/web_fetch_20260209 (Anthropic's own docs: these
# versions "use code execution internally"), not something this file ever
# declared. Fix: allowed_callers=["direct"] on both tools, closing that
# avenue at its source, plus a lower max_content_tokens as a backstop.


class TestCodeExecutionCallerRestriction:
    """Layer 4: prevent web_search/web_fetch from being invoked through a
    code_execution sandbox at all, rather than trying to separately cap a
    tool (code_execution) this file never declares."""

    def test_web_search_restricted_to_direct_callers(self):
        assert WEB_SEARCH_TOOL["allowed_callers"] == ["direct"]

    def test_web_fetch_restricted_to_direct_callers(self):
        assert WEB_FETCH_TOOL["allowed_callers"] == ["direct"]

    def test_web_fetch_max_uses_lowered_to_2(self):
        # Was 3 — the 3rd call in the incident was a same-URL retry after
        # the model's own code hit a TypeError, not a genuine 3rd resource.
        assert WEB_FETCH_TOOL["max_uses"] == 2

    def test_web_fetch_max_content_tokens_lowered(self):
        # Was 20000 — shrinks the raw content available to be re-injected
        # into context, independent of whether allowed_callers alone fully
        # closes the code_execution avenue.
        assert WEB_FETCH_TOOL["max_content_tokens"] <= 8000


# ── Regression tests: nonsensical-search resource filtering ───────────────────
# Root cause, found via a curriculum_seed.json structural audit (2026-08-30):
# two real entries pass labels into own_resources/resources that describe
# internal/personal context, not a real resource — "own cyber-sale workflow
# notes" (System Design Fundamentals) and "cumulative: all Assessments with
# completion_date on/before ..." (every midterm past the first, where
# known_now just redescribes content already supplied via
# cumulative_pool_content). Both would trigger a doomed web_search/web_fetch
# attempt if left in. filter_fetchable_resources() drops them before any
# Request dataclass field or tool-gating decision sees them.


class TestFilterFetchableResources:
    def test_drops_own_notes_label(self):
        # The exact real label from curriculum_seed.json's System Design
        # Fundamentals entry.
        result = filter_fetchable_resources([
            "\"System Design Interview\" by Alex Xu, Vol 1 (scalability, reliability patterns, databases at scale, distributed systems)",
            "ByteByteGo (caching/CDNs, message queues)",
            "Grokking System Design (educative.io — structured practice)",
            "own cyber-sale workflow notes (reliability patterns)",
        ])
        assert "own cyber-sale workflow notes (reliability patterns)" not in result
        assert len(result) == 3

    def test_drops_cumulative_pool_label(self):
        # The exact real label from curriculum_seed.json's Midterm 2/3/4
        # known_now entries.
        result = filter_fetchable_resources([
            "cumulative: all Assessments with completion_date on/before 2026-09-13",
        ])
        assert result == []

    def test_keeps_real_resources_untouched(self):
        real = ["react.dev (rendering, keys, reconciliation, JSX docs)", "Full Stack Open Part 1"]
        assert filter_fetchable_resources(real) == real

    def test_mixed_list_keeps_real_and_drops_internal(self):
        result = filter_fetchable_resources([
            "react.dev (component/rendering resources, for Part 1)",
            "WAI-ARIA APG",
            "cumulative: all Assessments with completion_date on/before 2026-10-04",
            "own design notes",
        ])
        assert result == ["react.dev (component/rendering resources, for Part 1)", "WAI-ARIA APG"]

    def test_empty_list_stays_empty(self):
        assert filter_fetchable_resources([]) == []

    def test_case_insensitive_and_whitespace_tolerant(self):
        result = filter_fetchable_resources(["  OWN team retro notes", "Cumulative: everything so far"])
        assert result == []


class TestFilterAppliedInAdapterMethods:
    """The filter must be applied inside each of the 5 tool-enabled methods
    — not left to callers to remember — so a request built from real seed
    data (a genuine own_resources/resources list containing an internal-
    only label) never enables tools or renders resource_guidance for it."""

    def _good_assessment_json(self) -> str:
        return json.dumps({
            "assessment_text": "text", "rubric": "rubric", "duration_minutes": 60,
        })

    def _good_midterm_json(self) -> str:
        return json.dumps({
            "part1_text": "p1", "part1_rubric": "r1", "part2_text": "p2",
            "part2_rubric": "r2", "duration_minutes": 120,
        })

    def test_generate_assessment_tools_omitted_when_only_internal_label_present(self, adapter):
        # Mirrors System Design Fundamentals with only the internal label
        # left (the other 3 real resources omitted here just to isolate
        # the boundary case: internal-only -> no tools at all).
        req = AssessmentGenerationRequest(
            topic="System Design", curriculum_content="notes",
            prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
            resources=["own cyber-sale workflow notes (reliability patterns)"],
        )
        mock = _patch_create(adapter, self._good_assessment_json())
        adapter.generate_assessment(req)
        assert "tools" not in mock.call_args[1]

    def test_generate_assessment_resource_guidance_omits_internal_label(self, adapter):
        req = AssessmentGenerationRequest(
            topic="System Design", curriculum_content="notes",
            prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
            resources=["ByteByteGo (caching/CDNs, message queues)", "own cyber-sale workflow notes (reliability patterns)"],
        )
        mock = _patch_create(adapter, self._good_assessment_json())
        adapter.generate_assessment(req)
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "own cyber-sale workflow notes" not in prompt
        assert "ByteByteGo" in prompt

    def test_generate_midterm_tools_omitted_for_cumulative_only_label(self, adapter):
        # Mirrors Midterm 2/3/4's known_now exactly: just the self-
        # referential cumulative-pool label, nothing else.
        req = MidtermGenerationRequest(
            topic="Chat App", cumulative_pool_content="[real assembled pool content]",
            own_resources=["cumulative: all Assessments with completion_date on/before 2026-09-13"],
            probe_focus="reconnect logic", part1_max_marks=30.0, part2_max_marks=70.0,
            prompt_template_body="{topic}{cumulative_pool_content}{own_resources_list}{resource_guidance}{probe_focus}{part1_max_marks}{part2_max_marks}{readme_content}",
        )
        mock = _patch_create(adapter, self._good_midterm_json())
        adapter.generate_midterm(req)
        assert "tools" not in mock.call_args[1]

    def test_generate_midterm_falls_back_to_non_tool_attempt_cap(self, adapter):
        """With the internal-only label filtered to empty, this must use
        the plain (non-tool) retry path — settings.llm_max_retries (3),
        not _TOOL_PATH_MAX_ATTEMPTS (2) — since there's no real tool
        budget at stake."""
        bad = "not json"
        req = MidtermGenerationRequest(
            topic="Chat App", cumulative_pool_content="[pool]",
            own_resources=["cumulative: all Assessments with completion_date on/before 2026-09-13"],
            probe_focus="x", part1_max_marks=30.0, part2_max_marks=70.0,
            prompt_template_body="{topic}{cumulative_pool_content}{own_resources_list}{resource_guidance}{probe_focus}{part1_max_marks}{part2_max_marks}{readme_content}",
        )
        mock = _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError) as exc_info:
            adapter.generate_midterm(req)
        assert mock.call_count == 3
        assert not isinstance(exc_info.value, LLMToolBudgetExceededError)

    def test_grade_midterm_submission_tools_omitted_for_internal_only_resources(self, adapter):
        req = MidtermGradingRequest(
            part1_text="p1", part1_rubric="r1", part2_text="p2", part2_rubric="r2",
            part1_max_marks=30.0, part2_max_marks=70.0,
            part1_submission_content="a1", part2_submission_content="a2",
            prompt_template_body="{part1_text}{part1_rubric}{part2_text}{part2_rubric}"
                                  "{part1_max_marks}{part2_max_marks}"
                                  "{part1_submission_content}{part2_submission_content}"
                                  "{resource_guidance}{readme_content}",
            resources=["cumulative: all Assessments with completion_date on/before 2026-10-04"],
        )
        good_grading_json = json.dumps({
            "part1_score": 27.0, "part2_score": 63.0, "weak_areas": [], "overall_feedback": "ok",
        })
        mock = _patch_create(adapter, good_grading_json)
        adapter.grade_midterm_submission(req)
        assert "tools" not in mock.call_args[1]

    def test_real_resources_alongside_internal_label_still_enable_tools(self, adapter):
        """Sanity check the filter isn't over-triggering: a mixed list
        (real resource + internal label) must still enable tools for the
        real resource."""
        req = AssessmentGenerationRequest(
            topic="React", curriculum_content="notes",
            prompt_template_body="{topic}{curriculum_content}{resource_guidance}",
            resources=["react.dev (rendering docs)", "own personal cheat sheet"],
        )
        mock = _patch_create(adapter, self._good_assessment_json())
        adapter.generate_assessment(req)
        assert mock.call_args[1]["tools"]


class TestToolBudgetExceededErrorType:
    """Exhausting a tool-enabled path's attempts must raise the specific
    LLMToolBudgetExceededError (a LLMValidationError subclass) — not the
    plain base class — so send_assessment_job can distinguish "this
    generation is expensive and keeps failing the same way" from an
    ordinary one-off schema mismatch, and route it to needs_manual_
    diagnosis instead of a silent automatic retry."""

    def _bad_json(self) -> str:
        return "not json at all"

    def _good_assessment_json(self) -> str:
        return json.dumps({
            "assessment_text": "text", "rubric": "rubric", "duration_minutes": 60,
        })

    def test_tool_enabled_exhaustion_raises_specific_subclass(self, adapter):
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        _patch_create(adapter, self._bad_json(), self._bad_json())
        with pytest.raises(LLMToolBudgetExceededError) as exc_info:
            adapter.generate_assessment(req)
        assert exc_info.value.attempts_made == 2
        assert exc_info.value.tokens_spent > 0
        assert exc_info.value.ceiling == adapter._tool_call_budget_tokens

    def test_non_tool_exhaustion_raises_plain_validation_error_not_subclass(self, adapter):
        """The tighter exception type is scoped to tool-enabled (budgeted)
        call sites only — a plain schema-mismatch exhaustion on a
        non-tool-enabled call must NOT be the new subclass, since there's
        no tool budget involved and no reason to route it to manual
        diagnosis instead of the normal failure handling."""
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=[],
        )
        _patch_create(adapter, self._bad_json(), self._bad_json(), self._bad_json())
        with pytest.raises(LLMValidationError) as exc_info:
            adapter.generate_assessment(req)
        assert not isinstance(exc_info.value, LLMToolBudgetExceededError)

    def test_diagnostic_message_names_tool_round_counts(self, adapter):
        """The exception's own message must be enough to diagnose the
        failure without re-running anything — tool names and round counts,
        not a raw content dump."""
        response = MagicMock()
        response.content = [
            MagicMock(type="server_tool_use", name="web_search"),
            MagicMock(type="server_tool_use", name="web_search"),
            MagicMock(type="web_fetch_tool_result"),
        ]
        response.usage = MagicMock(
            input_tokens=500, output_tokens=100,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        )
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}{resource_guidance}",
            resources=["https://example.com/docs"],
        )
        adapter._client.messages.create = MagicMock(return_value=response)
        with pytest.raises(LLMToolBudgetExceededError) as exc_info:
            adapter.generate_assessment(req)
        message = str(exc_info.value)
        assert "web_search" in message
        assert "2" in message  # 2 web_search rounds counted


# ── Regression tests: per-call cost estimate + curriculum-entry log tagging ───
# Added 2026-08-30 in response to a direct question: production had zero
# per-call cost/curriculum visibility (raw token counts only, no dollar
# estimate, no way to tell which curriculum entry a given call belonged to).


class TestCostEstimate:
    def _usage(self, input_tokens=0, output_tokens=0, cache_creation=0, cache_read=0):
        return MagicMock(
            input_tokens=input_tokens, output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation, cache_read_input_tokens=cache_read,
        )

    def test_matches_hand_computed_cost_for_the_incident_call(self):
        # The exact usage figures from the 2026-08-30 316k-token incident
        # call — $3/MTok in, $15/MTok out, verified pricing.
        usage = self._usage(input_tokens=316322, output_tokens=16578)
        cost = _estimate_cost_usd(usage)
        expected = 316322 / 1_000_000 * 3.00 + 16578 / 1_000_000 * 15.00
        assert cost == pytest.approx(expected)
        assert 1.10 < cost < 1.30  # sanity: matches the ~$1.20 reported figure

    def test_zero_usage_is_zero_cost(self):
        assert _estimate_cost_usd(self._usage()) == 0.0

    def test_cache_read_is_cheaper_than_fresh_input(self):
        fresh = _estimate_cost_usd(self._usage(input_tokens=1000))
        cached = _estimate_cost_usd(self._usage(cache_read=1000))
        assert cached < fresh

    def test_cache_write_is_more_expensive_than_fresh_input(self):
        fresh = _estimate_cost_usd(self._usage(input_tokens=1000))
        cache_write = _estimate_cost_usd(self._usage(cache_creation=1000))
        assert cache_write > fresh


class TestLogContextTagging:
    def _good_json(self) -> str:
        return json.dumps({
            "assessment_text": "text", "rubric": "rubric", "duration_minutes": 60,
        })

    def test_log_line_includes_the_active_context_label(self, adapter, caplog):
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}",
        )
        _patch_create(adapter, self._good_json())
        with caplog.at_level("INFO"):
            with llm_log_context("curriculum=abc-123 topic='Python' (test)"):
                adapter.generate_assessment(req)
        assert any("curriculum=abc-123" in r.message for r in caplog.records)

    def test_log_line_shows_placeholder_when_no_context_set(self, adapter, caplog):
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}",
        )
        _patch_create(adapter, self._good_json())
        with caplog.at_level("INFO"):
            adapter.generate_assessment(req)
        assert any("(no context set)" in r.message for r in caplog.records)

    def test_context_does_not_leak_across_calls(self, adapter, caplog):
        """The contextvar must be reset on exit — a call made after the
        `with` block must NOT still show the earlier label."""
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}",
        )
        _patch_create(adapter, self._good_json(), self._good_json())
        with caplog.at_level("INFO"):
            with llm_log_context("curriculum=first"):
                adapter.generate_assessment(req)
            adapter.generate_assessment(req)
        assert not any("curriculum=first" in r.message for r in caplog.records[1:])

    def test_log_line_includes_est_cost_usd(self, adapter, caplog):
        req = AssessmentGenerationRequest(
            topic="Python", curriculum_content="notes",
            prompt_template_body="Generate for {topic} using {curriculum_content}",
        )
        _patch_create(adapter, self._good_json())
        with caplog.at_level("INFO"):
            adapter.generate_assessment(req)
        assert any("est_cost_usd=" in r.message for r in caplog.records)


# ── analyze_curriculum ────────────────────────────────────────────────────────


class TestAnalyzeCurriculum:

    _TEMPLATE = "Analyze: {topic}\nContent: {curriculum_content}"

    def _req(self) -> CurriculumAnalysisRequest:
        return CurriculumAnalysisRequest(
            topic="Python",
            curriculum_content="Notes on Python.",
            prompt_template_body=self._TEMPLATE,
        )

    def _good_json(self) -> str:
        return json.dumps({
            "summary": "A concise overview of Python.",
            "key_topics": ["functions", "classes", "async"],
            "complexity_level": "intermediate",
            "estimated_study_hours": 12.5,
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.analyze_curriculum(self._req())
        assert result.summary == "A concise overview of Python."
        assert result.key_topics == ["functions", "classes", "async"]
        assert result.complexity_level == "intermediate"
        assert result.estimated_study_hours == 12.5

    def test_fenced_json_is_parsed_correctly(self, adapter):
        fenced = f"```json\n{self._good_json()}\n```"
        _patch_create(adapter, fenced)
        result = adapter.analyze_curriculum(self._req())
        assert result.summary == "A concise overview of Python."

    def test_missing_summary_key_exhausts_retries(self, adapter):
        bad = json.dumps({"key_topics": [], "complexity_level": "beginner", "estimated_study_hours": 5})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.analyze_curriculum(self._req())

    def test_non_numeric_study_hours_exhausts_retries(self, adapter):
        bad = json.dumps({
            "summary": "ok",
            "key_topics": [],
            "complexity_level": "beginner",
            "estimated_study_hours": "twelve",  # not a float
        })
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.analyze_curriculum(self._req())

    def test_recovery_first_bad_then_good(self, adapter):
        bad = json.dumps({"incomplete": True})
        mock = _patch_create(adapter, bad, self._good_json())
        result = adapter.analyze_curriculum(self._req())
        assert result.estimated_study_hours == 12.5
        assert mock.call_count == 2

    def test_topic_and_content_rendered_into_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.analyze_curriculum(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "Python" in prompt
        assert "Notes on Python." in prompt

    def test_unavailable_error_propagates_immediately(self, adapter):
        _patch_create(adapter, anthropic.APIConnectionError(request=_REQUEST))
        with pytest.raises(LLMUnavailableError):
            adapter.analyze_curriculum(self._req())


# ── generate_assessment ───────────────────────────────────────────────────────


class TestGenerateAssessment:

    _TEMPLATE = "Create assessment on {topic}\nCurriculum: {curriculum_content}"

    def _req(self) -> AssessmentGenerationRequest:
        return AssessmentGenerationRequest(
            topic="FastAPI",
            curriculum_content="REST API design notes.",
            prompt_template_body=self._TEMPLATE,
        )

    def _good_json(self) -> str:
        return json.dumps({
            "assessment_text": "Build a FastAPI CRUD app.",
            "rubric": "Full marks for correct endpoints.",
            "duration_minutes": 90,
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.generate_assessment(self._req())
        assert result.assessment_text == "Build a FastAPI CRUD app."
        assert result.rubric == "Full marks for correct endpoints."
        assert result.duration_minutes == 90

    def test_uses_max_tokens_16000(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_assessment(self._req())
        assert mock.call_args[1]["max_tokens"] == 16000

    def test_missing_assessment_text_key_raises_validation_error(self, adapter):
        bad = json.dumps({"rubric": "some rubric", "duration_minutes": 60})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.generate_assessment(self._req())

    def test_missing_rubric_key_raises_validation_error(self, adapter):
        bad = json.dumps({"assessment_text": "text", "duration_minutes": 60})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(self._req())

    def test_duration_minutes_as_float_string_raises_validation_error(self, adapter):
        # int("90.0") raises ValueError — must be a clean int
        bad = json.dumps({"assessment_text": "text", "rubric": "rubric", "duration_minutes": "90.0"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.generate_assessment(self._req())

    def test_duration_minutes_as_integer_string_is_accepted(self, adapter):
        # int("90") works fine — string digit representations are acceptable
        ok = json.dumps({"assessment_text": "text", "rubric": "rubric", "duration_minutes": "90"})
        _patch_create(adapter, ok)
        result = adapter.generate_assessment(self._req())
        assert result.duration_minutes == 90

    def test_recovery_after_one_bad_response(self, adapter):
        bad = json.dumps({"wrong_key": "value"})
        mock = _patch_create(adapter, bad, self._good_json())
        result = adapter.generate_assessment(self._req())
        assert result.duration_minutes == 90
        assert mock.call_count == 2


# ── generate_retest ───────────────────────────────────────────────────────────


class TestGenerateRetest:

    _TEMPLATE = (
        "Retest for {topic}. Previous score: {previous_mastery_score}. "
        "Weak areas: {weak_areas}. Attempt: {attempt_number}.\n{curriculum_content}"
    )

    def _req(self) -> RetestGenerationRequest:
        return RetestGenerationRequest(
            topic="Python",
            curriculum_content="async notes",
            prompt_template_body=self._TEMPLATE,
            previous_mastery_score=72.0,
            weak_areas=["event loop", "coroutines"],
            attempt_number=2,
        )

    def _good_json(self) -> str:
        return json.dumps({
            "assessment_text": "Retest on event loop and coroutines.",
            "rubric": "Focus on weak areas.",
            "duration_minutes": 45,
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.generate_retest(self._req())
        assert result.assessment_text == "Retest on event loop and coroutines."
        assert result.duration_minutes == 45

    def test_weak_areas_joined_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_retest(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "event loop, coroutines" in prompt

    def test_previous_mastery_score_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_retest(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "72.0" in prompt

    def test_attempt_number_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_retest(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "2" in prompt

    def test_missing_assessment_text_exhausts_retries(self, adapter):
        bad = json.dumps({"rubric": "rubric", "duration_minutes": 45})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.generate_retest(self._req())

    def test_uses_max_tokens_16000(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_retest(self._req())
        assert mock.call_args[1]["max_tokens"] == 16000


# ── generate_midterm ──────────────────────────────────────────────────────────


class TestGenerateMidterm:

    _TEMPLATE = (
        "Midterm on {topic}\nPool: {cumulative_pool_content}\n"
        "Resources: {own_resources_list}\n{resource_guidance}\n"
        "README: {readme_content}\nProbe: {probe_focus}\n"
        "Marks: {part1_max_marks}/{part2_max_marks}"
    )

    def _req(self, own_resources=None, readme_content=None) -> MidtermGenerationRequest:
        return MidtermGenerationRequest(
            topic="Capstone Project",
            cumulative_pool_content="Prior chapters on APIs and databases.",
            own_resources=own_resources if own_resources is not None else ["https://github.com/example/repo"],
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
            prompt_template_body=self._TEMPLATE,
            readme_content=readme_content,
        )

    def _good_json(self) -> str:
        return json.dumps({
            "part1_text": "Part 1 questions.",
            "part1_rubric": "Part 1 rubric.",
            "part2_text": "Part 2 questions.",
            "part2_rubric": "Part 2 rubric.",
            "duration_minutes": 120,
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.generate_midterm(self._req())
        assert result.part1_text == "Part 1 questions."
        assert result.part2_rubric == "Part 2 rubric."
        assert result.duration_minutes == 120

    def test_uses_max_tokens_16000(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm(self._req())
        assert mock.call_args[1]["max_tokens"] == 16000

    def test_readme_content_rendered_in_prompt_when_present(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm(self._req(readme_content="See src/auth/token.py for token verification."))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "See src/auth/token.py for token verification." in prompt

    def test_readme_content_defaults_to_placeholder_when_none(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm(self._req(readme_content=None))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "No project README/design writeup was submitted" in prompt

    def test_tools_enabled_when_own_resources_present(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm(self._req(own_resources=["https://github.com/example/repo"]))
        assert mock.call_args[1]["tools"]

    def test_tools_omitted_when_own_resources_empty(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm(self._req(own_resources=[]))
        assert "tools" not in mock.call_args[1]

    def test_missing_part1_text_key_raises_validation_error(self, adapter):
        bad = json.dumps({
            "part1_rubric": "r1", "part2_text": "t2", "part2_rubric": "r2", "duration_minutes": 120,
        })
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.generate_midterm(self._req())


# ── generate_midterm_retest ───────────────────────────────────────────────────


class TestGenerateMidtermRetest:

    _TEMPLATE = (
        "Midterm retest on {topic}, attempt {attempt_number}\n"
        "Pool: {cumulative_pool_content}\nResources: {own_resources_list}\n"
        "{resource_guidance}\nREADME: {readme_content}\nProbe: {probe_focus}\n"
        "Previous: {previous_part1_score}/{part1_max_marks}, "
        "{previous_part2_score}/{part2_max_marks}\nWeak: {weak_areas}"
    )

    def _req(self, readme_content=None) -> MidtermRetestGenerationRequest:
        return MidtermRetestGenerationRequest(
            topic="Capstone Project",
            cumulative_pool_content="Prior chapters.",
            own_resources=["https://github.com/example/repo"],
            probe_focus="architecture decisions",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
            previous_part1_score=10.0,
            previous_part2_score=20.0,
            weak_areas=["error handling", "test coverage"],
            attempt_number=2,
            prompt_template_body=self._TEMPLATE,
            readme_content=readme_content,
        )

    def _good_json(self) -> str:
        return json.dumps({
            "part1_text": "Retest Part 1.",
            "part1_rubric": "Retest Part 1 rubric.",
            "part2_text": "Retest Part 2.",
            "part2_rubric": "Retest Part 2 rubric.",
            "duration_minutes": 120,
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.generate_midterm_retest(self._req())
        assert result.part1_text == "Retest Part 1."
        assert result.part2_text == "Retest Part 2."

    def test_uses_max_tokens_16000(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm_retest(self._req())
        assert mock.call_args[1]["max_tokens"] == 16000

    def test_readme_content_rendered_in_prompt_when_present(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm_retest(self._req(readme_content="See src/db/repository.py."))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "See src/db/repository.py." in prompt

    def test_weak_areas_joined_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_midterm_retest(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "error handling, test coverage" in prompt


# ── grade_submission ──────────────────────────────────────────────────────────


class TestGradeSubmission:

    _TEMPLATE = (
        "Grade this:\nAssessment: {assessment_text}\nRubric: {rubric}\n"
        "Curriculum: {curriculum_content}\nSubmission: {submission_content}"
    )

    def _req(self) -> GradingRequest:
        return GradingRequest(
            assessment_text="Explain async/await.",
            rubric="Full marks for correctness.",
            curriculum_content="Python async notes.",
            submission_content="async/await allows non-blocking I/O.",
            prompt_template_body=self._TEMPLATE,
        )

    def _good_json(self, score: float = 88.0) -> str:
        return json.dumps({
            "mastery_score": score,
            "weak_areas": ["error handling"],
            "overall_feedback": "Good grasp of the core concept.",
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.grade_submission(self._req())
        assert result.mastery_score == 88.0
        assert result.weak_areas == ["error handling"]
        assert result.overall_feedback == "Good grasp of the core concept."

    def test_mastery_score_zero_is_valid_boundary(self, adapter):
        _patch_create(adapter, self._good_json(score=0.0))
        result = adapter.grade_submission(self._req())
        assert result.mastery_score == 0.0

    def test_mastery_score_100_is_valid_boundary(self, adapter):
        _patch_create(adapter, self._good_json(score=100.0))
        result = adapter.grade_submission(self._req())
        assert result.mastery_score == 100.0

    def test_mastery_score_above_100_raises_validation_error(self, adapter):
        bad = json.dumps({"mastery_score": 101.0, "weak_areas": [], "overall_feedback": "perfect"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.grade_submission(self._req())

    def test_mastery_score_negative_raises_validation_error(self, adapter):
        bad = json.dumps({"mastery_score": -1.0, "weak_areas": [], "overall_feedback": "poor"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.grade_submission(self._req())

    def test_missing_weak_areas_key_raises_validation_error(self, adapter):
        bad = json.dumps({"mastery_score": 80.0, "overall_feedback": "ok"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.grade_submission(self._req())

    def test_missing_mastery_score_raises_validation_error(self, adapter):
        bad = json.dumps({"weak_areas": [], "overall_feedback": "good"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.grade_submission(self._req())

    def test_recovery_after_out_of_range_score(self, adapter):
        bad = json.dumps({"mastery_score": 999.0, "weak_areas": [], "overall_feedback": "wat"})
        mock = _patch_create(adapter, bad, self._good_json())
        result = adapter.grade_submission(self._req())
        assert result.mastery_score == 88.0
        assert mock.call_count == 2

    def test_all_required_fields_rendered_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_submission(self._req())
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "Explain async/await." in prompt
        assert "Full marks for correctness." in prompt
        assert "Python async notes." in prompt
        assert "async/await allows non-blocking I/O." in prompt

    def test_rate_limit_propagates_without_retry(self, adapter):
        _patch_create(adapter, anthropic.RateLimitError("429", response=_RESPONSE_429, body={}))
        with pytest.raises(LLMUnavailableError, match="rate limit"):
            adapter.grade_submission(self._req())


# ── grade_midterm_submission ──────────────────────────────────────────────────


class TestGradeMidtermSubmission:

    _TEMPLATE = (
        "Grade midterm:\nP1: {part1_text}\nP1 rubric: {part1_rubric}\n"
        "P2: {part2_text}\nP2 rubric: {part2_rubric}\n"
        "Marks: {part1_max_marks}/{part2_max_marks}\n"
        "P1 answer: {part1_submission_content}\nP2 answer: {part2_submission_content}\n"
        "{resource_guidance}\nREADME: {readme_content}"
    )

    def _req(self, resources=None, readme_content=None) -> MidtermGradingRequest:
        return MidtermGradingRequest(
            part1_text="Part 1 exam.",
            part1_rubric="Part 1 rubric.",
            part2_text="Part 2 exam.",
            part2_rubric="Part 2 rubric.",
            part1_max_marks=30.0,
            part2_max_marks=70.0,
            part1_submission_content="Part 1 answer.",
            part2_submission_content="Part 2 answer.",
            prompt_template_body=self._TEMPLATE,
            resources=resources if resources is not None else ["https://github.com/example/repo"],
            readme_content=readme_content,
        )

    def _good_json(self, part1: float = 27.0, part2: float = 63.0) -> str:
        return json.dumps({
            "part1_score": part1,
            "part2_score": part2,
            "weak_areas": ["deployment"],
            "overall_feedback": "Solid overall.",
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json())
        result = adapter.grade_midterm_submission(self._req())
        assert result.part1_score == 27.0
        assert result.part2_score == 63.0
        assert result.weak_areas == ["deployment"]

    def test_uses_max_tokens_8000(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_midterm_submission(self._req())
        assert mock.call_args[1]["max_tokens"] == 8000

    def test_readme_content_rendered_in_prompt_when_present(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_midterm_submission(self._req(readme_content="README claims X is in file Y."))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "README claims X is in file Y." in prompt

    def test_readme_content_defaults_to_placeholder_when_none(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_midterm_submission(self._req(readme_content=None))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert "No project README/design writeup was submitted" in prompt

    def test_tools_enabled_when_resources_present(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_midterm_submission(self._req(resources=["https://github.com/example/repo"]))
        assert mock.call_args[1]["tools"]

    def test_tools_omitted_when_resources_empty(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.grade_midterm_submission(self._req(resources=[]))
        assert "tools" not in mock.call_args[1]

    def test_part2_score_above_max_marks_raises_validation_error(self, adapter):
        bad = self._good_json(part2=999.0)
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.grade_midterm_submission(self._req())


# ── classify_reschedule_request ───────────────────────────────────────────────


class TestClassifyRescheduleRequest:

    _TEMPLATE = "Classify this reschedule reason: {reason}"

    def _req(self, reason: str = "I have a medical appointment on that day.") -> RescheduleClassificationRequest:
        return RescheduleClassificationRequest(
            reason=reason,
            prompt_template_body=self._TEMPLATE,
        )

    def _good_json(self, category: str = "medical") -> str:
        return json.dumps({
            "category": category,
            "reasoning": f"User cited a {category} situation.",
        })

    def test_happy_path_returns_correct_result(self, adapter):
        _patch_create(adapter, self._good_json("medical"))
        result = adapter.classify_reschedule_request(self._req())
        assert result.category == "medical"
        assert "medical" in result.reasoning

    @pytest.mark.parametrize("category", [
        "interview",
        "medical",
        "emergency",
        "work_escalation",
        "procrastination",
        "lack_of_preparation",
        "missed_schedule",
    ])
    def test_all_valid_categories_accepted(self, adapter, category):
        _patch_create(adapter, self._good_json(category))
        result = adapter.classify_reschedule_request(self._req())
        assert result.category == category

    def test_unknown_category_raises_validation_error(self, adapter):
        bad = json.dumps({"category": "vacation", "reasoning": "Going on holiday."})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError, match="schema mismatch"):
            adapter.classify_reschedule_request(self._req())

    def test_missing_category_key_raises_validation_error(self, adapter):
        bad = json.dumps({"reasoning": "No category provided."})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.classify_reschedule_request(self._req())

    def test_missing_reasoning_key_raises_validation_error(self, adapter):
        bad = json.dumps({"category": "medical"})
        _patch_create(adapter, bad, bad, bad)
        with pytest.raises(LLMValidationError):
            adapter.classify_reschedule_request(self._req())

    def test_recovery_after_invalid_category(self, adapter):
        bad = json.dumps({"category": "unknown_junk", "reasoning": "..."})
        mock = _patch_create(adapter, bad, self._good_json("interview"))
        result = adapter.classify_reschedule_request(self._req())
        assert result.category == "interview"
        assert mock.call_count == 2

    def test_reason_rendered_in_prompt(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        reason = "Doctor confirmed appointment conflicts with test date."
        adapter.classify_reschedule_request(self._req(reason=reason))
        prompt = mock.call_args[1]["messages"][0]["content"]
        assert reason in prompt

    def test_uses_max_tokens_1024(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.classify_reschedule_request(self._req())
        assert mock.call_args[1]["max_tokens"] == 1024

    def test_timeout_propagates_without_retry(self, adapter):
        _patch_create(adapter, anthropic.APITimeoutError(request=_REQUEST))
        with pytest.raises(LLMUnavailableError):
            adapter.classify_reschedule_request(self._req())
