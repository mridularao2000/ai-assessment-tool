# Import all models here so that Base.metadata is fully populated
# when Alembic or Base.metadata.create_all() is called.

from app.models.curriculum import Curriculum, CurriculumStatus
from app.models.resource import Resource, ResourceType
from app.models.prompt_template import PromptTemplate
from app.models.assessment import Assessment, AssessmentStatus
from app.models.submission import Submission, SubmissionType
from app.models.grade import Grade
from app.models.reschedule_request import RescheduleRequest
from app.models.late_submission_token import LateSubmissionTokenBalance

__all__ = [
    "Curriculum",
    "CurriculumStatus",
    "Resource",
    "ResourceType",
    "PromptTemplate",
    "Assessment",
    "AssessmentStatus",
    "Submission",
    "SubmissionType",
    "Grade",
    "RescheduleRequest",
    "LateSubmissionTokenBalance",
]
