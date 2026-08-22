from __future__ import annotations

from sqlalchemy.orm import Session

from app.exceptions import InvalidStateError
from app.models.late_submission_token import LateSubmissionTokenBalance

MONTHLY_GRANT = 2
MAX_BALANCE = 2


class LateTokenService:
    """Tracks the monthly late-submission token allowance.

    Single-tenant system: exactly one balance row (id=1) exists, created
    lazily on first access. A token lets SubmissionService.create() accept
    a submission for an assessment that has already expired.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_balance(self) -> int:
        return self._get_or_create_row().balance

    def grant_monthly(self) -> int:
        """Top the balance up to MAX_BALANCE. Called by the monthly cron job.

        Returns the resulting balance.
        """
        row = self._get_or_create_row()
        row.balance = min(row.balance + MONTHLY_GRANT, MAX_BALANCE)
        self.db.commit()
        return row.balance

    def spend(self) -> None:
        """Deduct one token. Does not commit — caller owns the transaction.

        Raises:
            InvalidStateError: if the balance is already 0.
        """
        row = self._get_or_create_row()
        if row.balance <= 0:
            raise InvalidStateError("No late-submission tokens available.")
        row.balance -= 1

    def _get_or_create_row(self) -> LateSubmissionTokenBalance:
        row = self.db.get(LateSubmissionTokenBalance, 1)
        if row is None:
            row = LateSubmissionTokenBalance(id=1, balance=0)
            self.db.add(row)
            self.db.flush()
        return row
