from app.database import SessionLocal
from app.services.email_service import EmailService, StubEmailAdapter
from app.models.assessment import Assessment, AssessmentStatus


def send_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send assessment email and activate assessment."""
    db = SessionLocal()
    try:
        email_service = EmailService(db, StubEmailAdapter())
        email_service.send_assessment_email(assessment_id)

        assessment = db.get(Assessment, assessment_id)
        if assessment and assessment.status == AssessmentStatus.scheduled:
            assessment.status = AssessmentStatus.active
            db.commit()

    finally:
        db.close()