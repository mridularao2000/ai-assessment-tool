from app.interfaces.llm import (
    LLMInterface,
    CurriculumAnalysisRequest,
    CurriculumAnalysisResult,
    AssessmentGenerationRequest,
    AssessmentGenerationResult,
    RetestGenerationRequest,
    GradingRequest,
    GradingResult,
    RescheduleClassificationRequest,
    RescheduleClassificationResult,
    RescheduleCategory,
    LLMError,
    LLMValidationError,
    LLMUnavailableError,
)
from app.interfaces.email import (
    EmailInterface,
    AssessmentEmailData,
    ReminderEmailData,
    ResultsEmailData,
    EmailError,
    EmailDeliveryError,
)
from app.interfaces.scheduler import (
    SchedulerInterface,
    AssessmentJobIds,
    SchedulerError,
    SchedulerNotRunningError,
    JobNotFoundError,
)

__all__ = [
    # LLM
    "LLMInterface",
    "CurriculumAnalysisRequest",
    "CurriculumAnalysisResult",
    "AssessmentGenerationRequest",
    "AssessmentGenerationResult",
    "RetestGenerationRequest",
    "GradingRequest",
    "GradingResult",
    "RescheduleClassificationRequest",
    "RescheduleClassificationResult",
    "RescheduleCategory",
    "LLMError",
    "LLMValidationError",
    "LLMUnavailableError",
    # Email
    "EmailInterface",
    "AssessmentEmailData",
    "ReminderEmailData",
    "ResultsEmailData",
    "EmailError",
    "EmailDeliveryError",
    # Scheduler
    "SchedulerInterface",
    "AssessmentJobIds",
    "SchedulerError",
    "SchedulerNotRunningError",
    "JobNotFoundError",
]
