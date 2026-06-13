from app.database import SessionLocal
from app.services.email_service import EmailService, StubEmailAdapter


def send_reminder_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send the pre-assessment reminder email."""
    db = SessionLocal()
    try:
        EmailService(db, StubEmailAdapter()).send_reminder_email(assessment_id)
    finally:
        db.close()