# Import all models here so that Base.metadata is fully populated
# when Alembic or Base.metadata.create_all() is called.

from app.models.curriculum import Curriculum, CurriculumEntryType, CurriculumStatus
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.models.resource import Resource, ResourceType
from app.models.prompt_template import PromptTemplate
from app.models.assessment import Assessment, AssessmentStatus
from app.models.submission import Submission, SubmissionType
from app.models.grade import Grade
from app.models.reschedule_request import RescheduleRequest
from app.models.late_submission_token import LateSubmissionToken

__all__ = [
    "Curriculum",
    "CurriculumStatus",
    "CurriculumEntryType",
    "CurriculumUpload",
    "MidtermDetail",
    "Resource",
    "ResourceType",
    "PromptTemplate",
    "Assessment",
    "AssessmentStatus",
    "Submission",
    "SubmissionType",
    "Grade",
    "RescheduleRequest",
    "LateSubmissionToken",
]
