import logging

from app.database import SessionLocal
from app.dependencies import _email, _llm
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import CurriculumEntryType
from app.services.assessment_service import AssessmentService
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)


def send_assessment_job(assessment_id: str) -> None:
    """Scheduler entrypoint: send assessment email and activate assessment.

    For curriculum-upload entries, content isn't generated until this exact
    moment (assessment_text and part1_text are both still None) — deferring
    generation to send-time keeps resource-grounding maximally current and
    avoids spending LLM calls on exams that were scheduled weeks/months
    earlier. Standalone assessments already have their content populated
    eagerly at creation (today's unmodified behavior), so this branch is a
    no-op for them.
    """
    logger.info("Starting job: assessment_%s", assessment_id)
    db = SessionLocal()
    try:
        assessment = db.get(Assessment, assessment_id)
        if assessment and assessment.assessment_text is None and assessment.part1_text is None:
            service = AssessmentService(db, _llm)
            if assessment.curriculum.entry_type == CurriculumEntryType.midterm:
                service.generate_midterm_content(assessment)
            else:
                service.generate_assessment_content(assessment)
            db.commit()

        EmailService(db, _email).send_assessment_email(assessment_id)

        assessment = db.get(Assessment, assessment_id)
        if assessment and assessment.status == AssessmentStatus.scheduled:
            assessment.status = AssessmentStatus.active
            db.commit()
    finally:
        db.close()
