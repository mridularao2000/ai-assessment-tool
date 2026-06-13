from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.seed import REQUIRED_SLUGS, check_missing_templates
from app.dependencies import get_db

router = APIRouter()


class PromptsHealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    present: list[str]
    missing: list[str]


@router.get("/prompts", response_model=PromptsHealthResponse)
def prompts_health(db: Annotated[Session, Depends(get_db)]) -> PromptsHealthResponse:
    """Check whether all required prompt templates are seeded and active."""
    missing = check_missing_templates(db)
    present = sorted(REQUIRED_SLUGS - set(missing))
    return PromptsHealthResponse(
        status="ok" if not missing else "degraded",
        present=present,
        missing=missing,
    )
