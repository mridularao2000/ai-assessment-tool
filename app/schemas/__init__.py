from app.schemas.resource import ResourceResponse
from app.schemas.assessment import AssessmentSummary, AssessmentDetailResponse
from app.schemas.grade import GradeResponse
from app.schemas.submission import SubmissionCreate, SubmissionResponse
from app.schemas.reschedule import RescheduleRequestCreate, RescheduleResponse
from app.schemas.prompt_template import (
    PromptTemplateCreate,
    PromptTemplateUpdate,
    PromptTemplateResponse,
)
from app.schemas.curriculum import (
    CurriculumCreate,
    CurriculumCreateResponse,
    CurriculumResponse,
)

__all__ = [
    "ResourceResponse",
    "AssessmentSummary",
    "AssessmentDetailResponse",
    "GradeResponse",
    "SubmissionCreate",
    "SubmissionResponse",
    "RescheduleRequestCreate",
    "RescheduleResponse",
    "PromptTemplateCreate",
    "PromptTemplateUpdate",
    "PromptTemplateResponse",
    "CurriculumCreate",
    "CurriculumCreateResponse",
    "CurriculumResponse",
]
