from __future__ import annotations

# ── Route responsibility ───────────────────────────────────────────────────────
#
# This module is a thin HTTP boundary layer ONLY.
# All orchestration (curriculum creation, assessment generation, scheduling)
# lives in CurriculumService.create(). This route:
#   1. Parses multipart form data
#   2. Calls curriculum_svc.create() — the single orchestration entry point
#   3. Maps service exceptions to HTTP status codes
#   4. Returns the response schema
#
# Do NOT add assessment creation or scheduler calls here.
#
# ─────────────────────────────────────────────────────────────────────────────

from datetime import date
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_curriculum_service
from app.exceptions import IngestionError, InvalidStateError, NotFoundError
from app.interfaces.llm import LLMValidationError
from app.schemas.curriculum import CurriculumCreateResponse, CurriculumResponse
from app.services.curriculum_service import CurriculumService

router = APIRouter()


@router.post("/", response_model=CurriculumCreateResponse, status_code=201)
def create_curriculum(
    topic: Annotated[str, Form()],
    target_completion_date: Annotated[date, Form()],
    links: Annotated[List[str], Form()] = [],
    github_repos: Annotated[List[str], Form()] = [],
    notes: Annotated[Optional[str], Form()] = None,
    files: Annotated[List[UploadFile], File()] = [],
    curriculum_svc: CurriculumService = Depends(get_curriculum_service),
) -> CurriculumCreateResponse:
    uploaded_files = [(f.filename or "upload", f.file.read()) for f in files]
    try:
        curriculum = curriculum_svc.create(
            topic=topic,
            target_completion_date=target_completion_date,
            links=links,
            github_repos=github_repos,
            notes=notes,
            uploaded_files=uploaded_files,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LLMValidationError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Assessment generation failed after all retries: {exc}",
        )

    return CurriculumCreateResponse(curriculum_id=curriculum.id)


@router.get("/{curriculum_id}", response_model=CurriculumResponse)
def get_curriculum(
    curriculum_id: str,
    curriculum_svc: CurriculumService = Depends(get_curriculum_service),
) -> CurriculumResponse:
    try:
        curriculum = curriculum_svc.get(curriculum_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return CurriculumResponse.model_validate(curriculum)
