from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import get_late_token_service
from app.services.late_token_service import LateTokenService

router = APIRouter()


class LateTokenBalanceResponse(BaseModel):
    balance: int


@router.get("/", response_model=LateTokenBalanceResponse)
def get_late_token_balance(
    late_token_svc: Annotated[LateTokenService, Depends(get_late_token_service)],
) -> LateTokenBalanceResponse:
    return LateTokenBalanceResponse(balance=late_token_svc.get_balance())
