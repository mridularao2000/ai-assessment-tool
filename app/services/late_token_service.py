from __future__ import annotations

from sqlalchemy.orm import Session

from app.exceptions import InvalidStateError
from app.models._utils import utcnow
from app.models.late_submission_token import LateSubmissionToken

MAX_BALANCE = 2


class LateTokenService:
    """Issues and spends discrete UUID late-submission tokens.

    Single-tenant system. Each token is its own row (see
    LateSubmissionToken). The monthly cron job tops the pool of unused
    tokens up to MAX_BALANCE by issuing new UUID tokens; spending marks
    the oldest unused token as used by SubmissionService.create().
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self) -> int:
        return self._unused_query().count()

    def list_unused_tokens(self) -> list[str]:
        """Return the UUIDs of currently unused tokens, oldest first."""
        rows = self._unused_query().order_by(LateSubmissionToken.issued_at).all()
        return [row.id for row in rows]

    def grant_monthly(self) -> int:
        """Issue new UUID tokens so the unused pool reaches MAX_BALANCE.

        Called by the monthly cron job. Returns the resulting balance.
        """
        unused = self.get_balance()
        to_issue = max(MAX_BALANCE - unused, 0)
        for _ in range(to_issue):
            self.db.add(LateSubmissionToken())
        self.db.commit()
        return unused + to_issue

    def spend(self, assessment_id: str) -> str:
        """Mark the oldest unused token as used by assessment_id.

        Does not commit — caller owns the transaction.
        Returns the spent token's id.

        Raises:
            InvalidStateError: if no unused token is available.
        """
        token = (
            self._unused_query().order_by(LateSubmissionToken.issued_at).first()
        )
        if token is None:
            raise InvalidStateError("No late-submission tokens available.")
        token.used_at = utcnow()
        token.used_by_assessment_id = assessment_id
        return token.id

    def _unused_query(self):
        return self.db.query(LateSubmissionToken).filter(
            LateSubmissionToken.used_at.is_(None)
        )
