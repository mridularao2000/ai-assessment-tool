"""Unit tests for AnthropicLLMAdapter.

All Anthropic API calls are monkeypatched — no network traffic.
Tests cover:
  - _parse_json: valid JSON, code-fenced JSON, invalid JSON
  - _render: template substitution, missing keys
  - _call: all SDK error types mapped to LLMUnavailableError
  - _retry: exhaustion, recovery, call counts, retry-nudge injection
  - All 5 public LLMInterface methods: happy paths, schema failures,
    boundary conditions, recovery after one bad response
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, call

import anthropic
import httpx
import pytest

from app.adapters.anthropic_llm import AnthropicLLMAdapter
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    CurriculumAnalysisRequest,
    GradingRequest,
    LLMUnavailableError,
    LLMValidationError,
    RescheduleClassificationRequest,
    RetestGenerationRequest,
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


def _fake_response(text: str) -> MagicMock:
    """Build a fake anthropic.Message with one text content block."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
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

    def test_uses_max_tokens_8192(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_assessment(self._req())
        assert mock.call_args[1]["max_tokens"] == 8192

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

    def test_uses_max_tokens_8192(self, adapter):
        mock = _patch_create(adapter, self._good_json())
        adapter.generate_retest(self._req())
        assert mock.call_args[1]["max_tokens"] == 8192


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
