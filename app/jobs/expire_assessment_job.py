from app.database import SessionLocal
from app.models.assessment import Assessment, AssessmentStatus


def expire_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: mark an assessment expired if still active at due_date."""
    db = SessionLocal()
    try:
        assessment = db.get(Assessment, assessment_id)
        if assessment is not None and assessment.status == AssessmentStatus.active:
            assessment.status = AssessmentStatus.expired
            db.commit()
    finally:
        db.close()
