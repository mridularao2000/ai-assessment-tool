from __future__ import annotations

import json
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
    part1_text_content: Annotated[Optional[str], Form()] = None,
    # JSON-encoded array of 5 option letters, e.g. '["A","C","B","D","A"]' —
    # see SubmissionService.create's mcq_answers docstring. A plain string
    # Form field (not List[str] = Form(...)) since this is the MCQ +
    # Coding Project format's Section 1 answer set, not a repeated field.
    mcq_answers: Annotated[Optional[str], Form()] = None,
    submission_svc: SubmissionService = Depends(get_submission_service),
) -> SubmissionResponse:
    uploaded_file = None
    if file is not None:
        uploaded_file = (file.filename or "upload", file.file.read())

    parsed_mcq_answers: Optional[list[str]] = None
    if mcq_answers is not None:
        try:
            parsed_mcq_answers = json.loads(mcq_answers)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="mcq_answers must be a JSON array of strings.")
        if not isinstance(parsed_mcq_answers, list) or not all(
            isinstance(a, str) for a in parsed_mcq_answers
        ):
            raise HTTPException(status_code=422, detail="mcq_answers must be a JSON array of strings.")

    try:
        submission = submission_svc.create(
            assessment_id=assessment_id,
            token=token,
            submission_type=submission_type,
            github_url=github_url,
            text_content=text_content,
            uploaded_file=uploaded_file,
            part1_text_content=part1_text_content,
            mcq_answers=parsed_mcq_answers,
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
