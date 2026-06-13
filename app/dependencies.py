"""FastAPI dependency providers for all services.

Singletons (created once at module load):
  _scheduler_adapter — APSchedulerAdapter instance shared across requests
  _llm               — AnthropicLLMAdapter when ANTHROPIC_API_KEY is set,
                       StubLLMAdapter otherwise (fails loudly on any call)
  _email             — ResendEmailAdapter when RESEND_API_KEY is set,
                       StubEmailAdapter otherwise (fails loudly on any call)

Per-request dependencies (instantiated per request via Depends):
  get_db                — SQLAlchemy Session
  get_scheduler_service — SchedulerService
  get_curriculum_service
  get_assessment_service
  get_submission_service
  get_grading_service
  get_reschedule_service
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.adapters.apscheduler_adapter import APSchedulerAdapter
from app.config import get_settings
from app.database import get_db
from app.interfaces.email import (
    AssessmentEmailData,
    EmailInterface,
    ReminderEmailData,
    ResultsEmailData,
)
from app.interfaces.llm import (
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    GradingRequest,
    GradingResult,
    LLMInterface,
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RetestGenerationRequest,
)
from app.services.assessment_service import AssessmentService
from app.services.curriculum_service import CurriculumService
from app.services.grading_service import GradingService
from app.services.reschedule_service import RescheduleService
from app.services.scheduler_service import SchedulerService
from app.services.submission_service import SubmissionService


# ── Stub email adapter ────────────────────────────────────────────────────────

class StubEmailAdapter:
    """Development stub — raises NotImplementedError for all email sends."""

    def send_assessment_email(self, data: AssessmentEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: set RESEND_API_KEY to enable email.")

    def send_reminder_email(self, data: ReminderEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: set RESEND_API_KEY to enable email.")

    def send_results_email(self, data: ResultsEmailData) -> None:
        raise NotImplementedError("StubEmailAdapter: set RESEND_API_KEY to enable email.")


# ── Stub LLM adapter ──────────────────────────────────────────────────────────

class StubLLMAdapter:
    """Development stub — raises NotImplementedError for all LLM calls."""

    def analyze_curriculum(
        self, request: CurriculumAnalysisRequest
    ) -> CurriculumAnalysisResult:
        raise NotImplementedError("StubLLMAdapter: set ANTHROPIC_API_KEY to enable LLM.")

    def generate_assessment(
        self, request: AssessmentGenerationRequest
    ) -> AssessmentGenerationResult:
        raise NotImplementedError("StubLLMAdapter: set ANTHROPIC_API_KEY to enable LLM.")

    def generate_retest(
        self, request: RetestGenerationRequest
    ) -> AssessmentGenerationResult:
        raise NotImplementedError("StubLLMAdapter: set ANTHROPIC_API_KEY to enable LLM.")

    def grade_submission(self, request: GradingRequest) -> GradingResult:
        raise NotImplementedError("StubLLMAdapter: set ANTHROPIC_API_KEY to enable LLM.")

    def classify_reschedule_request(
        self, request: RescheduleClassificationRequest
    ) -> RescheduleClassificationResult:
        raise NotImplementedError("StubLLMAdapter: set ANTHROPIC_API_KEY to enable LLM.")


# ── Module-level singletons ───────────────────────────────────────────────────

_scheduler_adapter = APSchedulerAdapter()


def _build_llm() -> LLMInterface:
    if get_settings().anthropic_api_key:
        from app.adapters.anthropic_llm import AnthropicLLMAdapter
        return AnthropicLLMAdapter()  # type: ignore[return-value]
    return StubLLMAdapter()  # type: ignore[return-value]


_llm: LLMInterface = _build_llm()


def _build_email() -> EmailInterface:
    if get_settings().resend_api_key:
        from app.adapters.resend_email import ResendEmailAdapter
        return ResendEmailAdapter()  # type: ignore[return-value]
    return StubEmailAdapter()  # type: ignore[return-value]


_email: EmailInterface = _build_email()


# ── Scheduler ─────────────────────────────────────────────────────────────────

def get_scheduler_adapter() -> APSchedulerAdapter:
    """Return the shared APSchedulerAdapter singleton."""
    return _scheduler_adapter


def get_scheduler_service(
    db: Annotated[Session, Depends(get_db)],
) -> SchedulerService:
    return SchedulerService(db, _scheduler_adapter)


# ── Services ──────────────────────────────────────────────────────────────────

def get_curriculum_service(
    db: Annotated[Session, Depends(get_db)],
    scheduler_service: Annotated[SchedulerService, Depends(get_scheduler_service)],
) -> CurriculumService:
    return CurriculumService(db, _llm, scheduler_service)


def get_assessment_service(
    db: Annotated[Session, Depends(get_db)],
) -> AssessmentService:
    return AssessmentService(db, _llm)


def get_submission_service(
    db: Annotated[Session, Depends(get_db)],
    scheduler_service: Annotated[SchedulerService, Depends(get_scheduler_service)],
) -> SubmissionService:
    return SubmissionService(db, scheduler_service)


def get_grading_service(
    db: Annotated[Session, Depends(get_db)],
) -> GradingService:
    return GradingService(db, _llm)


def get_reschedule_service(
    db: Annotated[Session, Depends(get_db)],
    scheduler_service: Annotated[SchedulerService, Depends(get_scheduler_service)],
) -> RescheduleService:
    return RescheduleService(db, _llm, scheduler_service)
