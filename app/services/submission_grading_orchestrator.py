from sqlalchemy.orm import Session

from app.interfaces.llm import LLMInterface
from app.models.grade import Grade
from app.services.grading_service import GradingService


def process_submission_for_grading(
    submission_id: str,
    db: Session,
    llm: LLMInterface,
) -> Grade:
    return GradingService(db, llm).grade(submission_id)
