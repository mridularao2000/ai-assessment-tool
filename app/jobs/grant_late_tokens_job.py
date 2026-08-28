import logging

from app.database import SessionLocal
from app.models.curriculum_upload import CurriculumUpload
from app.services.late_token_service import LateTokenService

logger = logging.getLogger(__name__)


def grant_late_tokens_job() -> None:
    """Scheduler entrypoint: top up every late-submission token pool.

    Each curriculum_upload has its own independent pool; standalone
    assessments (curriculum_upload_id is None) have their own separate pool
    too. Tops up every pool that currently exists — new uploads created
    after this job last ran get their first top-up on the next run.
    """
    logger.info("Starting job: grant_late_tokens")
    db = SessionLocal()
    try:
        service = LateTokenService(db)

        standalone_balance = service.grant_monthly(None)
        logger.info("Standalone late-submission token balance topped up to %d", standalone_balance)

        upload_ids = [row.id for row in db.query(CurriculumUpload.id).all()]
        for upload_id in upload_ids:
            balance = service.grant_monthly(upload_id)
            logger.info(
                "curriculum_upload %s late-submission token balance topped up to %d",
                upload_id, balance,
            )
    finally:
        db.close()
