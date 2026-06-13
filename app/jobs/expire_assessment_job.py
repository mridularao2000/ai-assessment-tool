import logging

from app.database import SessionLocal
from app.models.assessment import Assessment, AssessmentStatus

logger = logging.getLogger(__name__)


def expire_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: mark an assessment expired if still active at due_date."""
    logger.info("Starting job: expire_%s", assessment_id)
    db = SessionLocal()
    try:
        assessment = db.get(Assessment, assessment_id)
        if assessment is not None and assessment.status == AssessmentStatus.active:
            assessment.status = AssessmentStatus.expired
            db.commit()
    finally:
        db.close()
