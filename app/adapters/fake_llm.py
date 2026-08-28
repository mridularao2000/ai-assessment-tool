"""Test-mode LLM adapter — canned, schema-valid responses, no network call.

Selected by app.dependencies._build_llm() only when Settings.test_mode is
true (the same single flag app.adapters.fake_email.FakeEmailAdapter is
gated on — see that module's docstring for why this is one flag, not one
per integration). Production never sets that flag, so this adapter is
never constructed outside an isolated-verification harness — app.
adapters.anthropic_llm (the real Anthropic-backed adapter) is untouched
and remains the only adapter production code paths can reach.

Every method returns a fixed, valid instance of its Protocol-declared
result type immediately — no API call, no cost, no retry/validation logic
to exercise. This is for proving state-transition/scoring/scheduling
mechanics (what app.services.* and app.jobs.* do with an LLM result), not
for exercising the real model's output quality — callers that need to
observe genuine model behavior should keep using AnthropicLLMAdapter.
"""
from __future__ import annotations

from app.interfaces.llm import (
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    GradingRequest,
    GradingResult,
    MidtermGenerationRequest,
    MidtermGenerationResult,
    MidtermGradingRequest,
    MidtermGradingResult,
    MidtermRetestGenerationRequest,
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RetestGenerationRequest,
)


class FakeLLMAdapter:
    """Canned LLMInterface implementation for test mode. See module docstring."""

    def analyze_curriculum(
        self, request: CurriculumAnalysisRequest
    ) -> CurriculumAnalysisResult:
        return CurriculumAnalysisResult(
            summary=f"[FAKE_LLM] Canned analysis of {request.topic!r}.",
            key_topics=["fake-topic-1", "fake-topic-2"],
            complexity_level="intermediate",
            estimated_study_hours=2.0,
        )

    def generate_assessment(
        self, request: AssessmentGenerationRequest
    ) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text=(
                f"[FAKE_LLM] Canned assessment for {request.topic!r}.\n\n"
                "Question 1 (100 marks): Describe the topic in your own words."
            ),
            rubric="[FAKE_LLM] Award full marks for any substantive, on-topic answer.",
            duration_minutes=30,
        )

    def generate_retest(
        self, request: RetestGenerationRequest
    ) -> AssessmentGenerationResult:
        return AssessmentGenerationResult(
            assessment_text=(
                f"[FAKE_LLM] Canned retest for {request.topic!r} "
                f"(attempt {request.attempt_number}), focused on: "
                f"{', '.join(request.weak_areas) or 'general review'}.\n\n"
                "Question 1 (100 marks): Describe the topic in your own words."
            ),
            rubric="[FAKE_LLM] Award full marks for any substantive, on-topic answer.",
            duration_minutes=30,
        )

    def generate_midterm(
        self, request: MidtermGenerationRequest
    ) -> MidtermGenerationResult:
        return MidtermGenerationResult(
            part1_text=f"[FAKE_LLM] Canned Part 1 for {request.topic!r}.",
            part1_rubric="[FAKE_LLM] Award full marks for any substantive answer.",
            part2_text=f"[FAKE_LLM] Canned Part 2 for {request.topic!r}.",
            part2_rubric="[FAKE_LLM] Award full marks for any substantive answer.",
            duration_minutes=45,
        )

    def generate_midterm_retest(
        self, request: MidtermRetestGenerationRequest
    ) -> MidtermGenerationResult:
        return MidtermGenerationResult(
            part1_text=(
                f"[FAKE_LLM] Canned Part 1 retest for {request.topic!r} "
                f"(attempt {request.attempt_number})."
            ),
            part1_rubric="[FAKE_LLM] Award full marks for any substantive answer.",
            part2_text=f"[FAKE_LLM] Canned Part 2 retest for {request.topic!r}.",
            part2_rubric="[FAKE_LLM] Award full marks for any substantive answer.",
            duration_minutes=45,
        )

    def grade_submission(self, request: GradingRequest) -> GradingResult:
        return GradingResult(
            mastery_score=92.0,
            weak_areas=[],
            overall_feedback="[FAKE_LLM] Canned passing grade — test mode, not real grading.",
        )

    def grade_midterm_submission(
        self, request: MidtermGradingRequest
    ) -> MidtermGradingResult:
        return MidtermGradingResult(
            part1_score=request.part1_max_marks,
            part2_score=request.part2_max_marks,
            weak_areas=[],
            overall_feedback="[FAKE_LLM] Canned passing grade — test mode, not real grading.",
        )

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        return RescheduleClassificationResult(
            category="medical",
            reasoning="[FAKE_LLM] Canned classification — test mode, not real classification.",
        )
