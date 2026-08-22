import logging

from app.database import SessionLocal
from app.services.late_token_service import LateTokenService

logger = logging.getLogger(__name__)


def grant_late_tokens_job() -> None:
    """Scheduler entrypoint: top up the monthly late-submission token balance."""
    logger.info("Starting job: grant_late_tokens")
    db = SessionLocal()
    try:
        new_balance = LateTokenService(db).grant_monthly()
        logger.info("Late-submission token balance topped up to %d", new_balance)
    finally:
        db.close()
