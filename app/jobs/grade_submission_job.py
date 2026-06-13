from app.database import SessionLocal
from app.dependencies import _stub_llm
from app.services.grading_service import GradingService


def grade_submission_job(submission_id: str) -> None:
    """Scheduler entrypoint: grade a submitted assessment."""
    db = SessionLocal()
    try:
        GradingService(db, _stub_llm).grade(submission_id)
    finally:
        db.close()