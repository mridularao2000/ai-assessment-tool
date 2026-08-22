from datetime import datetime

from sqlalchemy import DateTime, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models._utils import utcnow


class LateSubmissionTokenBalance(Base):
    """Singleton row (id=1) tracking the late-submission token balance.

    Single-tenant system — exactly one row ever exists, created lazily by
    LateTokenService on first access. A monthly cron job tops the balance
    up to a fixed cap; spending a token lets a submission be accepted after
    an assessment's due_date has passed.
    """

    __tablename__ = "late_submission_token_balance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    balance: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )
