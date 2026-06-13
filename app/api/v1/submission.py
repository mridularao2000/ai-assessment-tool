from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.dependencies import get_submission_service
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.models.submission import SubmissionType
from app.schemas.submission import SubmissionResponse
from app.services.submission_service import SubmissionService

router = APIRouter()


@router.post("/", response_model=SubmissionResponse, status_code=201)
def create_submission(
    assessment_id: Annotated[str, Form()],
    token: Annotated[str, Form()],
    submission_type: Annotated[SubmissionType, Form()],
    github_url: Annotated[Optional[str], Form()] = None,
    text_content: Annotated[Optional[str], Form()] = None,
    file: Annotated[Optional[UploadFile], File()] = None,
    submission_svc: SubmissionService = Depends(get_submission_service),
) -> SubmissionResponse:
    uploaded_file = None
    if file is not None:
        uploaded_file = (file.filename or "upload", file.file.read())

    try:
        submission = submission_svc.create(
            assessment_id=assessment_id,
            token=token,
            submission_type=submission_type,
            github_url=github_url,
            text_content=text_content,
            uploaded_file=uploaded_file,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTokenError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return SubmissionResponse(submission_id=submission.id)


@router.get("/{submission_id}", response_model=SubmissionResponse)
def get_submission(
    submission_id: str,
    submission_svc: SubmissionService = Depends(get_submission_service),
) -> SubmissionResponse:
    try:
        submission = submission_svc.get(submission_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SubmissionResponse(submission_id=submission.id)
