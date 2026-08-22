from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_late_token_service
from app.services.late_token_service import LateTokenService

router = APIRouter()


class LateTokenBalanceResponse(BaseModel):
    balance: int
    tokens: list[str]


@router.get("/", response_model=LateTokenBalanceResponse)
def get_late_token_balance(
    late_token_svc: Annotated[LateTokenService, Depends(get_late_token_service)],
) -> LateTokenBalanceResponse:
    tokens = late_token_svc.list_unused_tokens()
    return LateTokenBalanceResponse(balance=len(tokens), tokens=tokens)


@router.post("/grant", response_model=LateTokenBalanceResponse)
def grant_late_tokens(
    late_token_svc: Annotated[LateTokenService, Depends(get_late_token_service)],
) -> LateTokenBalanceResponse:
    """Manually run the same top-up the monthly cron job runs.

    Idempotent within a top-up cycle — safe to call more than once since it
    only ever tops the unused pool up to the cap, never issues extra tokens.
    Useful when the scheduled grant was missed (e.g. right after deploying
    this feature, before the 1st of the month has passed).
    """
    late_token_svc.grant_monthly()
    tokens = late_token_svc.list_unused_tokens()
    return LateTokenBalanceResponse(balance=len(tokens), tokens=tokens)
