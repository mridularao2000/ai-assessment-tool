from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query

from app.dependencies import get_assessment_service, get_scheduler_service
from app.exceptions import InvalidStateError, InvalidTokenError, NotFoundError
from app.interfaces.email import EmailDeliveryError
from app.interfaces.llm import LLMUnavailableError, LLMValidationError
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.schemas.assessment import AssessmentDetailResponse, AssessmentSummary
from app.services.assessment_service import AssessmentService
from app.services.scheduler_service import SchedulerService

router = APIRouter()


@router.post("/{curriculum_id}", response_model=AssessmentSummary, status_code=201)
def create_assessment(
    curriculum_id: str,
    assessment_svc: AssessmentService = Depends(get_assessment_service),
    scheduler_svc: SchedulerService = Depends(get_scheduler_service),
) -> AssessmentSummary:
    try:
        curriculum = assessment_svc.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum {curriculum_id!r} not found.")

        # create_for_curriculum() is a pure factory (no DB writes) — this
        # route, not CurriculumService.create(), is the transaction boundary
        # here, so it owns add/commit/refresh before scheduling jobs.
        assessment = assessment_svc.create_for_curriculum(curriculum)
        assessment_svc.db.add(assessment)
        assessment_svc.db.commit()
        assessment_svc.db.refresh(assessment)

        scheduler_svc.schedule_assessment_jobs(
            assessment_id=assessment.id,
            scheduled_at=assessment.scheduled_at,
            reminder_at=assessment.reminder_at,
            due_date=assessment.due_date,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return AssessmentSummary.model_validate(assessment)


@router.post("/{assessment_id}/resend", response_model=AssessmentSummary)
def resend_assessment(
    assessment_id: str,
    assessment_svc: AssessmentService = Depends(get_assessment_service),
) -> AssessmentSummary:
    """Manually re-run the scheduled send for an assessment stuck in
    `scheduled` status past its scheduled_at, or one moved to
    `needs_manual_diagnosis` after exhausting its tool-call budget.

    send_assessment_job is registered as a one-shot APScheduler `date`
    trigger — if that single execution raised (an LLM call failure, an
    email send failure, ...), the row's send_job_claimed_at is released
    so a retry CAN succeed, but nothing retries it automatically; the job
    itself is already consumed. This endpoint is that manual retry.

    needs_manual_diagnosis is accepted here deliberately: it's the one
    failure mode recheck_stuck_assessments_job's 15-minute sweep will never
    pick back up on its own (its query only matches status == scheduled),
    so this endpoint is the only path back to a working state once the
    underlying cause has been fixed — both the sweep's give-up log message
    and the needs_manual_diagnosis alert email point here.

    Safe to call even if the original execution actually did succeed
    concurrently — send_assessment_job's atomic claim on
    send_job_claimed_at makes re-running it idempotent, not a duplicate
    send.

    Raises:
        NotFoundError: assessment_id doesn't exist.
        InvalidStateError: not currently `scheduled` or
                            `needs_manual_diagnosis`, or scheduled_at
                            hasn't arrived yet (nothing to force).
    """
    assessment = assessment_svc.db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail=f"Assessment {assessment_id!r} not found.")
    if assessment.status not in (AssessmentStatus.scheduled, AssessmentStatus.needs_manual_diagnosis):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Assessment {assessment_id!r} is not in 'scheduled' or "
                f"'needs_manual_diagnosis' status (currently "
                f"{assessment.status.value!r}) — nothing to resend."
            ),
        )
    if datetime.utcnow() < assessment.scheduled_at:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Assessment {assessment_id!r} isn't due to send until "
                f"{assessment.scheduled_at.isoformat()} — refusing to send it early."
            ),
        )

    from app.jobs.send_assessment_job import send_assessment_job

    try:
        send_assessment_job(assessment_id)
    except LLMValidationError as exc:
        raise HTTPException(
            status_code=503, detail=f"Assessment generation failed after all retries: {exc}"
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Claude API unavailable: {exc}. If this is a billing/rate-limit "
                "error, check your Anthropic console balance before retrying — "
                "each retry re-attempts a full (expensive, tool-using) generation."
            ),
        )
    except EmailDeliveryError as exc:
        raise HTTPException(status_code=502, detail=f"Email send failed: {exc}")
    except NotImplementedError as exc:
        raise HTTPException(status_code=503, detail=f"Email provider is not configured: {exc}")

    assessment_svc.db.refresh(assessment)
    return AssessmentSummary.model_validate(assessment)


@router.get("/{assessment_id}", response_model=AssessmentDetailResponse)
def get_assessment(
    assessment_id: str,
    token: str = Query(...),
    assessment_svc: AssessmentService = Depends(get_assessment_service),
) -> AssessmentDetailResponse:
    try:
        assessment = assessment_svc.get_by_id_and_token(assessment_id, token)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except InvalidTokenError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

    is_midterm = assessment.curriculum.entry_type == CurriculumEntryType.midterm

    # Retroactively-created entries (window already closed before upload —
    # see CurriculumUploadService._create_retroactive_expired_assessment)
    # have no content yet. Generate it lazily, the first time it's viewed,
    # rather than paying for LLM calls on entries nobody may ever revisit.
    # Scoped strictly to status == expired: a normally-scheduled entry
    # (status == scheduled) must wait for send_assessment_job, not be
    # peekable early through this endpoint.
    if (
        assessment.status == AssessmentStatus.expired
        and assessment.assessment_text is None
        and assessment.part1_text is None
    ):
        if is_midterm:
            assessment_svc.generate_midterm_content(assessment)
        else:
            assessment_svc.generate_assessment_content(assessment)
        assessment_svc.db.commit()

    return AssessmentDetailResponse(
        assessment_id=assessment.id,
        topic=assessment.curriculum.topic,
        assessment_text=(assessment.part1_text if is_midterm else assessment.assessment_text),
        duration_minutes=assessment.duration_minutes,
        scheduled_at=assessment.scheduled_at,
        due_date=assessment.due_date,
        status=assessment.status,
        is_midterm=is_midterm,
    )
