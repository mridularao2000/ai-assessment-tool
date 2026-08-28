from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.exceptions import InvalidStateError
from app.models._utils import utcnow
from app.models.late_submission_token import LateSubmissionToken

MAX_BALANCE = 2


class LateTokenService:
    """Issues and spends discrete UUID late-submission tokens.

    Each token is its own row (see LateSubmissionToken). Every method takes
    an optional curriculum_upload_id: each curriculum_upload has its own
    independent 2-per-month pool, and None means the pool for standalone
    assessments (never belongs to an upload) — two curricula, or a
    curriculum and standalone, never contend for the same tokens. One
    shared service/query surface, scoped per call, rather than a separate
    service instance per curriculum.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self, curriculum_upload_id: Optional[str] = None) -> int:
        return self._unused_query(curriculum_upload_id).count()

    def list_unused_tokens(self, curriculum_upload_id: Optional[str] = None) -> list[str]:
        """Return the UUIDs of currently unused tokens for this pool, oldest first."""
        rows = (
            self._unused_query(curriculum_upload_id)
            .order_by(LateSubmissionToken.issued_at)
            .all()
        )
        return [row.id for row in rows]

    def grant_monthly(self, curriculum_upload_id: Optional[str] = None) -> int:
        """Issue new UUID tokens so this pool's unused count reaches MAX_BALANCE.

        Called by the monthly cron job, once per pool (standalone + every
        curriculum_upload). Returns the resulting balance for this pool.
        """
        unused = self.get_balance(curriculum_upload_id)
        to_issue = max(MAX_BALANCE - unused, 0)
        for _ in range(to_issue):
            self.db.add(LateSubmissionToken(curriculum_upload_id=curriculum_upload_id))
        self.db.commit()
        return unused + to_issue

    def spend(self, assessment_id: str, curriculum_upload_id: Optional[str] = None) -> str:
        """Mark the oldest unused token in this pool as used by assessment_id.

        Does not commit — caller owns the transaction.
        Returns the spent token's id.

        Raises:
            InvalidStateError: if no unused token is available in this pool.
        """
        token = (
            self._unused_query(curriculum_upload_id)
            .order_by(LateSubmissionToken.issued_at)
            .first()
        )
        if token is None:
            raise InvalidStateError("No late-submission tokens available.")
        token.used_at = utcnow()
        token.used_by_assessment_id = assessment_id
        return token.id

    def _unused_query(self, curriculum_upload_id: Optional[str] = None):
        return self.db.query(LateSubmissionToken).filter(
            LateSubmissionToken.used_at.is_(None),
            LateSubmissionToken.curriculum_upload_id == curriculum_upload_id,
        )
