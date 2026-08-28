from __future__ import annotations

from typing import Annotated, Optional

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
    curriculum_upload_id: Optional[str] = None,
) -> LateTokenBalanceResponse:
    """Defaults to the standalone pool (curriculum_upload_id unset) — pass
    it to check a specific curriculum_upload's own independent pool."""
    tokens = late_token_svc.list_unused_tokens(curriculum_upload_id)
    return LateTokenBalanceResponse(balance=len(tokens), tokens=tokens)


@router.post("/grant", response_model=LateTokenBalanceResponse)
def grant_late_tokens(
    late_token_svc: Annotated[LateTokenService, Depends(get_late_token_service)],
    curriculum_upload_id: Optional[str] = None,
) -> LateTokenBalanceResponse:
    """Manually run the same top-up the monthly cron job runs, for one pool
    (defaults to the standalone pool; pass curriculum_upload_id for a
    specific curriculum's pool).

    Idempotent within a top-up cycle — safe to call more than once since it
    only ever tops the unused pool up to the cap, never issues extra tokens.
    Useful when the scheduled grant was missed (e.g. right after deploying
    this feature, before the 1st of the month has passed, or right after a
    new curriculum_upload is created).
    """
    late_token_svc.grant_monthly(curriculum_upload_id)
    tokens = late_token_svc.list_unused_tokens(curriculum_upload_id)
    return LateTokenBalanceResponse(balance=len(tokens), tokens=tokens)
