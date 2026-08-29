from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.exceptions import CurriculumUploadValidationError, InvalidStateError, NotFoundError
from app.interfaces.email import EmailInterface
from app.interfaces.scheduler import AssessmentJobIds
from app.models._utils import utcnow
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType, CurriculumStatus
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.models.resource import Resource, ResourceType
from app.services.assessment_service import calculate_scheduled_at, build_assessment_dates
from app.services.email_service import EmailService
from app.services.late_token_service import LateTokenService
from app.services.scheduler_service import SchedulerService
from app.utils.token_auth import generate_submission_token

logger = logging.getLogger(__name__)

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")


def assemble_part1_pool(db: Session, curriculum: Curriculum) -> tuple[list[Curriculum], bool]:
    """Return (qualifying assessment-type entries, used_known_now_fallback).

    Free function — pure DB query, no service-instance state needed — so
    AssessmentService.generate_midterm_content() can call it directly
    without depending on a whole CurriculumUploadService instance.

    Purely date-driven and generic — no entry is ever special-cased by
    name. The known_now fallback triggers on zero qualifying Assessments
    existing at/before this Midterm's completion_date (e.g. PM System,
    the chronologically-first entry in the current seed).
    MidtermDetail.special_case is never read here — audit/display only.
    """
    qualifying = (
        db.query(Curriculum)
        .filter(
            Curriculum.upload_id == curriculum.upload_id,
            Curriculum.entry_type == CurriculumEntryType.assessment,
            Curriculum.target_completion_date <= curriculum.target_completion_date,
        )
        .order_by(Curriculum.target_completion_date)
        .all()
    )
    return qualifying, len(qualifying) == 0


def _slugify_label(label: str) -> str:
    slug = _SLUG_STRIP_RE.sub("_", label.lower().strip()).strip("_")
    return slug or "resource"


def _build_pending_slots(labels: list[str]) -> tuple[dict[str, str], dict[str, Optional[str]]]:
    """Restructure placeholder labels (e.g. "project README") into named
    slots, defaulting to unfilled. Every label becomes a slot regardless of
    whether it reads like a real value or a placeholder — the upload never
    has real pending_completion content at ingestion time, per spec.
    """
    labels_by_slug: dict[str, str] = {}
    slots: dict[str, Optional[str]] = {}
    used: set[str] = set()
    for label in labels:
        base_slug = _slugify_label(label)
        slug = base_slug
        n = 2
        while slug in used:
            slug = f"{base_slug}_{n}"
            n += 1
        used.add(slug)
        labels_by_slug[slug] = label
        slots[slug] = None
    return labels_by_slug, slots


@dataclass
class _ParsedEntry:
    topic: str
    entry_type: CurriculumEntryType
    chapter_label: str
    completion_date: date
    max_marks: float
    term: Optional[str]
    prerequisites: Optional[list]
    resources: Optional[list[str]]                    # assessment-only
    known_now: Optional[list[str]]                     # midterm-only
    pending_completion_labels_raw: Optional[list[str]] # midterm-only
    probe_focus: Optional[str]
    special_case: Optional[str]
    part1_max_marks: Optional[float]
    part2_max_marks: Optional[float]


def _require(item: dict, field_name: str, entry_label: str) -> Any:
    value = item.get(field_name)
    if value in (None, ""):
        raise CurriculumUploadValidationError(
            f"Entry {entry_label!r}: missing required field {field_name!r}."
        )
    return value


def _parse_entries(topics_raw: list) -> list[_ParsedEntry]:
    parsed: list[_ParsedEntry] = []
    for i, item in enumerate(topics_raw):
        if not isinstance(item, dict):
            raise CurriculumUploadValidationError(f"topics[{i}] must be an object.")
        label = item.get("topic") or f"topics[{i}]"

        entry_type_raw = _require(item, "type", label)
        if entry_type_raw not in ("assessment", "midterm"):
            raise CurriculumUploadValidationError(
                f"Entry {label!r}: type must be 'assessment' or 'midterm', got {entry_type_raw!r}."
            )
        entry_type = CurriculumEntryType(entry_type_raw)

        topic = _require(item, "topic", label)
        chapter_label = _require(item, "chapter", label)
        completion_date_raw = _require(item, "completion_date", label)
        try:
            completion_date = date.fromisoformat(str(completion_date_raw))
        except ValueError as exc:
            raise CurriculumUploadValidationError(
                f"Entry {label!r}: completion_date {completion_date_raw!r} is not a valid ISO date (YYYY-MM-DD)."
            ) from exc
        try:
            max_marks = float(_require(item, "max_marks", label))
        except (TypeError, ValueError) as exc:
            raise CurriculumUploadValidationError(
                f"Entry {label!r}: max_marks must be a number."
            ) from exc

        resources_raw = _require(item, "resources", label)
        resources = known_now = pending_labels = None
        if entry_type == CurriculumEntryType.assessment:
            if not isinstance(resources_raw, list) or not all(isinstance(r, str) for r in resources_raw):
                raise CurriculumUploadValidationError(
                    f"Entry {label!r}: assessment resources must be a list of strings."
                )
            resources = resources_raw
        else:
            if (
                not isinstance(resources_raw, dict)
                or "known_now" not in resources_raw
                or "pending_completion" not in resources_raw
            ):
                raise CurriculumUploadValidationError(
                    f"Entry {label!r}: midterm resources must be an object with "
                    "'known_now' and 'pending_completion' lists."
                )
            known_now = list(resources_raw["known_now"])
            pending_labels = list(resources_raw["pending_completion"])

        parsed.append(
            _ParsedEntry(
                topic=topic,
                entry_type=entry_type,
                chapter_label=chapter_label,
                completion_date=completion_date,
                max_marks=max_marks,
                term=item.get("term"),
                prerequisites=item.get("prerequisites"),
                resources=resources,
                known_now=known_now,
                pending_completion_labels_raw=pending_labels,
                probe_focus=item.get("probe_focus"),
                special_case=item.get("special_case"),
                part1_max_marks=item.get("part1_max_marks"),
                part2_max_marks=item.get("part2_max_marks"),
            )
        )
    return parsed


def _build_entry_dates(scheduled_at: datetime) -> tuple[datetime, datetime]:
    """Entry-specific date math (flow b in the spec): due_date uses the
    exact same formula as standalone (reused from build_assessment_dates),
    but reminder_at counts back from due_date — the deadline — using a
    configurable offset, rather than from scheduled_at like standalone's
    hardcoded 1-day-before-send reminder. Standalone's own date math
    (AssessmentService._build_dates) is untouched.
    """
    _, due_date = build_assessment_dates(scheduled_at)
    hours = get_settings().entry_reminder_hours_before_deadline
    reminder_at = due_date - timedelta(hours=hours)
    return reminder_at, due_date


class CurriculumUploadService:
    """Ingests a curriculum-template upload file, schedules its entries, and
    sends the syllabus email.

    Scheduling note: content generation is deliberately deferred to
    send-time (see send_assessment_job / AssessmentService.
    generate_assessment_content / generate_midterm_content) rather than
    happening eagerly here — this service only computes scheduled_at/
    reminder_at/due_date and registers the same three APScheduler jobs the
    standalone flow uses. This keeps resource-grounding maximally current
    and avoids spending real LLM calls today on exams that are weeks or
    months away.

    Depends on:
      db                — SQLAlchemy session for all persistence
      email             — EmailInterface, wrapped in EmailService for sends
      scheduler_service — SchedulerService, reused as-is from the
                          standalone flow (same schedule_assessment_jobs())
    """

    def __init__(
        self, db: Session, email: EmailInterface, scheduler_service: SchedulerService
    ) -> None:
        self.db = db
        self.email = email
        self.scheduler_service = scheduler_service

    # ── Public methods ─────────────────────────────────────────────────────────

    def ingest(self, raw_json: dict, source_filename: str) -> CurriculumUpload:
        """Parse, validate, and persist an uploaded curriculum file.

        Raises:
            CurriculumUploadValidationError: on any schema mismatch. The
            whole upload is rejected atomically — nothing is persisted.
        """
        topics_raw = raw_json.get("topics")
        if not isinstance(topics_raw, list) or not topics_raw:
            raise CurriculumUploadValidationError(
                "Upload file must contain a non-empty 'topics' list."
            )
        parsed_entries = _parse_entries(topics_raw)

        upload = CurriculumUpload(
            id=str(uuid.uuid4()),
            source_filename=source_filename,
            note_on_pending_resources=raw_json.get("note_on_pending_resources"),
            chapters_with_no_standalone_assessment=raw_json.get(
                "chapters_with_no_standalone_assessment"
            ),
        )
        self.db.add(upload)

        today = date.today()
        created: list[Curriculum] = [
            self._create_entry(upload.id, entry, today) for entry in parsed_entries
        ]

        self.db.commit()
        for c in created:
            self.db.refresh(c)
        self.db.refresh(upload)

        # Fund this upload's own late-submission-token pool immediately —
        # otherwise it sits at 0 until the next monthly cron tick (which
        # only tops up pools that already existed when it last ran; see
        # grant_late_tokens_job.py), silently blocking every Missed —
        # Late-Eligible entry from actually being recoverable.
        LateTokenService(self.db).grant_monthly(upload.id)

        # Non-fatal side effects below — the upload itself is already
        # committed and durable regardless of what happens here.
        email_service = EmailService(self.db, self.email)
        try:
            email_service.send_syllabus_email(upload.id)
            upload.syllabus_email_sent_at = utcnow()
            self.db.commit()
        except Exception:
            logger.exception("Syllabus email failed for upload %s", upload.id)

        for c in created:
            if c.resources_hold:
                self.send_hold_reminder(c, email_service)

        self.schedule_ready_entries()

        return upload

    def _create_entry(self, upload_id: str, entry: _ParsedEntry, today: date) -> Curriculum:
        """Build one Curriculum row (+ its Resource or MidtermDetail
        children) from a parsed entry. Shared by ingest() (many entries at
        once) and add_entry() (one entry into an existing upload) — the
        per-entry construction logic must stay identical between the two,
        or an added entry could end up subtly different from one that
        arrived with the original upload.
        """
        curriculum = Curriculum(
            id=str(uuid.uuid4()),
            topic=entry.topic,
            target_completion_date=entry.completion_date,
            status=CurriculumStatus.ready,
            entry_type=entry.entry_type,
            upload_id=upload_id,
            chapter_label=entry.chapter_label,
            max_marks=entry.max_marks,
        )
        self.db.add(curriculum)

        if entry.entry_type == CurriculumEntryType.assessment:
            for resource_label in entry.resources or []:
                self.db.add(
                    Resource(
                        curriculum_id=curriculum.id,
                        type=ResourceType.note,
                        source_ref=resource_label,
                        raw_content=None,
                    )
                )
        else:
            labels_by_slug, slots = _build_pending_slots(
                entry.pending_completion_labels_raw or []
            )
            # Default split: Part 2 (finish the project and submit) = 70%,
            # Part 1 (pass the assignment — cumulative questions) = 30%.
            # Either can be overridden per-entry via part1_max_marks/
            # part2_max_marks in the upload file.
            part1 = (
                entry.part1_max_marks
                if entry.part1_max_marks is not None
                else entry.max_marks * 0.30
            )
            part2 = (
                entry.part2_max_marks
                if entry.part2_max_marks is not None
                else entry.max_marks * 0.70
            )
            self.db.add(
                MidtermDetail(
                    curriculum_id=curriculum.id,
                    known_now=entry.known_now or [],
                    pending_completion_labels=labels_by_slug,
                    pending_completion_slots=slots,
                    probe_focus=entry.probe_focus,
                    special_case=entry.special_case,
                    part1_max_marks=part1,
                    part2_max_marks=part2,
                )
            )
            all_filled = all(v is not None for v in slots.values())
            if entry.completion_date <= today and not all_filled:
                curriculum.resources_hold = True

        return curriculum

    def assemble_part1_pool(self, curriculum: Curriculum) -> tuple[list[Curriculum], bool]:
        return assemble_part1_pool(self.db, curriculum)

    def add_entry(self, upload_id: str, entry_raw: dict) -> Curriculum:
        """Add one new entry to an existing, still-open curriculum_upload.

        Uses the exact same per-entry parse/create/schedule pipeline as
        ingest() (one entry instead of a whole file), so an entry added
        after the fact behaves identically to one that arrived with the
        original upload — same validation, same hold/scheduling logic.

        Raises:
            NotFoundError: upload_id doesn't exist.
            InvalidStateError: the upload is closed.
            CurriculumUploadValidationError: the entry fails validation.
        """
        upload = self.get_upload(upload_id)
        if upload.closed_at is not None:
            raise InvalidStateError(
                f"CurriculumUpload {upload_id!r} is closed — no new entries can be added."
            )

        parsed = _parse_entries([entry_raw])[0]
        curriculum = self._create_entry(upload.id, parsed, date.today())
        self.db.commit()
        self.db.refresh(curriculum)

        email_service = EmailService(self.db, self.email)
        if curriculum.resources_hold:
            self.send_hold_reminder(curriculum, email_service)

        self.schedule_ready_entries()
        self.db.refresh(curriculum)
        return curriculum

    def _is_editable(self, curriculum: Curriculum) -> bool:
        """True if this entry hasn't reached its exam window yet — no
        Assessment exists for it, or the sole Assessment is still
        `scheduled` (job registered, content not yet generated, exam not
        yet sent). Any attempt or grade on record means the exam was
        already generated from this entry's current resources/dates, or a
        submission already references it — editing now would silently
        corrupt what's already on record, so update_entry() refuses it
        outright rather than allowing it.
        """
        return all(a.status == AssessmentStatus.scheduled for a in curriculum.assessments)

    def update_entry(self, curriculum_id: str, updates: dict) -> Curriculum:
        """Edit an entry that hasn't reached its exam window yet.

        Allowed fields: topic, chapter_label, target_completion_date,
        max_marks; resources (assessment-type, full replace); known_now/
        probe_focus (midterm-type). pending_completion slots are NOT
        editable here — that's fill_pending_resources()'s dedicated flow.

        If the entry already has a not-yet-fired Assessment row (status
        scheduled), it's cancelled and deleted so schedule_ready_entries()
        recreates it fresh against the new data — nothing user-facing has
        happened yet for that row (deferred generation means its
        assessment_text is still None), so there's nothing to corrupt.

        Raises:
            NotFoundError: curriculum_id doesn't exist.
            InvalidStateError: the entry already has an attempt or grade on
                record (see _is_editable), its upload is closed, or an
                unknown/non-editable field was supplied.
        """
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None:
            raise NotFoundError(f"Curriculum entry {curriculum_id!r} not found.")
        if curriculum.upload is not None and curriculum.upload.closed_at is not None:
            raise InvalidStateError(
                f"Curriculum entry {curriculum_id!r} belongs to a closed curriculum "
                "and cannot be modified."
            )
        if not self._is_editable(curriculum):
            raise InvalidStateError(
                f"Curriculum entry {curriculum_id!r} already has an attempt or grade on "
                "record — it cannot be silently modified. Changing its resources or "
                "dates after an exam was already generated from them would corrupt "
                "what's already on record."
            )

        for assessment in list(curriculum.assessments):
            if assessment.scheduled_job_ids:
                job_ids = AssessmentJobIds(**assessment.scheduled_job_ids)
                self.scheduler_service.cancel_jobs_for_assessment(job_ids)
            self.db.delete(assessment)

        simple_fields = {"topic", "chapter_label", "target_completion_date", "max_marks"}
        for field, value in updates.items():
            if field == "resources":
                if curriculum.entry_type != CurriculumEntryType.assessment:
                    raise InvalidStateError(
                        "'resources' is only editable on assessment-type entries "
                        "(midterm entries use known_now/probe_focus)."
                    )
                for resource in list(curriculum.resources):
                    self.db.delete(resource)
                for label in value:
                    self.db.add(Resource(
                        curriculum_id=curriculum.id, type=ResourceType.note,
                        source_ref=label, raw_content=None,
                    ))
            elif field in ("known_now", "probe_focus"):
                if curriculum.midterm_detail is None:
                    raise InvalidStateError(f"{field!r} is only editable on midterm-type entries.")
                setattr(curriculum.midterm_detail, field, value)
            elif field in simple_fields:
                if field == "target_completion_date" and isinstance(value, str):
                    value = date.fromisoformat(value)
                setattr(curriculum, field, value)
            else:
                raise InvalidStateError(f"Unknown or non-editable field {field!r}.")

        self.db.commit()
        self.db.refresh(curriculum)

        self.schedule_ready_entries()
        self.db.refresh(curriculum)
        return curriculum

    def close_upload(self, upload_id: str) -> CurriculumUpload:
        """Archive (soft-delete) a curriculum_upload — never hard-deleted,
        since the whole point of a transcript is a historical record.

        Order matters: sends the ONE final transcript snapshot FIRST
        (capturing this upload's exact final state, to the currently
        configured primary recipient — recipients are global/shared by
        design, same as every other transcript send); only if that
        succeeds does it cancel every still-pending scheduled action
        (reminders, exam generation, expiry checks) across every entry in
        this upload, then mark it closed. A failed final-send leaves the
        upload open and its jobs untouched, rather than silently closing
        without the record being sent — the caller can retry.

        Once closed_at is set, recheck_pending_midterms_job's periodic
        transcript check and its hold-recheck loop both skip this upload's
        entries permanently — see their queries.

        Raises:
            NotFoundError: upload_id doesn't exist.
            InvalidStateError: already closed.
        """
        upload = self.get_upload(upload_id)
        if upload.closed_at is not None:
            raise InvalidStateError(f"CurriculumUpload {upload_id!r} is already closed.")

        EmailService(self.db, self.email).send_transcript_email(upload_id)

        for curriculum in upload.entries:
            for assessment in curriculum.assessments:
                if assessment.scheduled_job_ids:
                    job_ids = AssessmentJobIds(**assessment.scheduled_job_ids)
                    self.scheduler_service.cancel_jobs_for_assessment(job_ids)

        upload.closed_at = utcnow()
        self.db.commit()
        self.db.refresh(upload)
        return upload

    def schedule_ready_entries(self) -> list[Assessment]:
        """Schedule every upload entry that isn't held, in one of two ways.

        Future/current window: content generation is deferred to send-time
        (see send_assessment_job) — this only computes scheduled_at/
        reminder_at/due_date and registers the same three APScheduler jobs
        the standalone flow uses.

        Already-past window (target_completion_date+3 < today, e.g. an
        entry whose due date had already passed before this curriculum was
        even uploaded): creates the Assessment directly in `expired` status
        with no jobs registered — no email is sent for a window that's
        already closed, but this makes the entry late-submittable with a
        token like any other expired assessment. Without this row, "Missed
        — Late-Eligible" would be a label with nothing to actually submit
        against. Content for these generates lazily the first time the
        entry is accessed (see AssessmentService.get_by_id_and_token's
        caller in the assessment-detail route), not here — generating it
        now would mean paying for LLM calls on entries nobody may ever
        revisit. The existing calendar-month check in
        SubmissionService.create() still governs whether a token can
        actually be spent — an entry that's been past-due for more than a
        month is correctly rejected there, same as any other case.

        Idempotent — already-scheduled entries (an Assessment child already
        exists) are skipped, so this is safe to call repeatedly (e.g. once
        per ingest(), and again whenever a hold clears).
        """
        today = date.today()
        ready = (
            self.db.query(Curriculum)
            .filter(
                Curriculum.entry_type.isnot(None),
                Curriculum.resources_hold.is_(False),
            )
            .all()
        )
        created: list[Assessment] = []
        for curriculum in ready:
            if curriculum.assessments:
                continue
            if curriculum.target_completion_date + timedelta(days=3) < today:
                created.append(self._create_retroactive_expired_assessment(curriculum))
            else:
                created.append(
                    self._schedule_entry_assessment(curriculum, curriculum.target_completion_date)
                )
        return created

    def _create_retroactive_expired_assessment(self, curriculum: Curriculum) -> Assessment:
        """Create an Assessment directly in `expired` status for an entry
        whose window already closed before upload — no jobs registered, no
        email sent. See schedule_ready_entries() for why this exists.
        """
        scheduled_at = calculate_scheduled_at(curriculum.target_completion_date)
        reminder_at, due_date = _build_entry_dates(scheduled_at)

        assessment_id = str(uuid.uuid4())
        assessment = Assessment(
            id=assessment_id,
            curriculum_id=curriculum.id,
            attempt_number=1,
            assessment_text=None,
            rubric=None,
            duration_minutes=None,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
            status=AssessmentStatus.expired,
            submission_token=generate_submission_token(assessment_id),
        )
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)
        return assessment

    def _schedule_entry_assessment(
        self, curriculum: Curriculum, effective_date: date
    ) -> Assessment:
        """Create the (content-empty) Assessment row for one entry and
        register its jobs, using effective_date for the window-date math.

        effective_date is curriculum.target_completion_date for the normal
        bulk-schedule path, or date.today() when a Midterm's hold has just
        cleared — "open the normal window from whenever they arrive, not
        the original completion_date," per spec.
        """
        scheduled_at = calculate_scheduled_at(effective_date)
        reminder_at, due_date = _build_entry_dates(scheduled_at)

        assessment_id = str(uuid.uuid4())
        assessment = Assessment(
            id=assessment_id,
            curriculum_id=curriculum.id,
            attempt_number=1,
            assessment_text=None,
            rubric=None,
            duration_minutes=None,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
            status=AssessmentStatus.scheduled,
            submission_token=generate_submission_token(assessment_id),
        )
        self.db.add(assessment)
        self.db.commit()
        self.db.refresh(assessment)

        self.scheduler_service.schedule_assessment_jobs(
            assessment_id=assessment.id,
            scheduled_at=scheduled_at,
            reminder_at=reminder_at,
            due_date=due_date,
        )
        return assessment

    def fill_pending_resources(self, curriculum_id: str, values: dict[str, str]) -> Curriculum:
        """Write real values into a Midterm's pending_completion slots.

        If every slot is now filled, clears resources_hold immediately
        (rather than waiting for the next daily recheck).

        Raises:
            NotFoundError: curriculum_id doesn't exist or isn't a midterm.
            InvalidStateError: an unknown slot slug was supplied.
        """
        curriculum = self.db.get(Curriculum, curriculum_id)
        if curriculum is None or curriculum.midterm_detail is None:
            raise NotFoundError(f"Midterm curriculum entry {curriculum_id!r} not found.")
        if curriculum.upload is not None and curriculum.upload.closed_at is not None:
            raise InvalidStateError(
                f"Curriculum entry {curriculum_id!r} belongs to a closed curriculum "
                "and cannot be modified."
            )

        detail = curriculum.midterm_detail
        slots = dict(detail.pending_completion_slots)
        for slug, value in values.items():
            if slug not in slots:
                raise InvalidStateError(
                    f"Unknown pending-resource slot {slug!r} for {curriculum_id!r}. "
                    f"Valid slots: {sorted(slots)}"
                )
            if value:
                slots[slug] = value
        detail.pending_completion_slots = slots
        self.db.commit()

        self.check_and_clear_hold(curriculum)
        self.db.refresh(curriculum)
        return curriculum

    def check_and_clear_hold(self, curriculum: Curriculum) -> bool:
        """Clear resources_hold if every pending slot is now filled AND
        the late-submission grace window still allows it, then
        immediately schedule its exam window from today (not the original
        completion_date), per spec.

        A held midterm's completion_date has, by construction, already
        passed the moment resources_hold is set (see _create_entry /
        recheck_pending_midterms_job) — so clearing it is always a late
        recovery, gated the same way SubmissionService.create() gates a
        late submission for an expired assessment:
          - target_completion_date's calendar month == today's month:
            clearing spends one late-submission token from this entry's
            pool (upload-scoped, same pool late assessment submissions
            use). No token available -> stays held (caller/UI sees
            resources_hold still True; a manual re-check or a future
            grant can succeed later, same month).
          - a LATER calendar month: permanently unscoreable, no token can
            recover it — see transcript_service.display_status, which
            classifies this exact state as MISSED_NO_SCORE, "same
            terminal state as a permanently-missed assessment" by
            explicit design, not a distinct label.

        Returns True if this call flipped resources_hold off. Shared by
        fill_pending_resources() and the daily recheck job so a manual
        PATCH resolves immediately without waiting for the next cron tick.
        """
        detail = curriculum.midterm_detail
        if detail is None or not curriculum.resources_hold:
            return False
        if not all(v is not None for v in detail.pending_completion_slots.values()):
            return False

        today = date.today()
        due = curriculum.target_completion_date
        if (due.year, due.month) != (today.year, today.month):
            # Permanently past its grace month — see display_status().
            return False

        late_tokens = LateTokenService(self.db)
        if late_tokens.get_balance(curriculum.upload_id) <= 0:
            return False

        curriculum.resources_hold = False
        self.db.commit()
        assessment = self._schedule_entry_assessment(curriculum, today)
        late_tokens.spend(assessment.id, curriculum.upload_id)
        self.db.commit()
        return True

    def send_hold_reminder(self, curriculum: Curriculum, email_service: EmailService) -> None:
        """Send (or re-send) the "resources still missing" reminder. Non-fatal."""
        try:
            email_service.send_midterm_hold_reminder_email(curriculum.id)
            curriculum.last_hold_reminder_at = utcnow()
            self.db.commit()
        except Exception:
            logger.exception("Hold-reminder email failed for curriculum %s", curriculum.id)

    def send_periodic_transcript_if_due(
        self, upload: CurriculumUpload, email_service: EmailService
    ) -> bool:
        """Send the secondary-recipient transcript copy if real elapsed
        time since the last one (or since upload, if never sent) has
        reached transcript_secondary_interval_days. Returns True if a send
        happened.

        Deliberately gated on a persisted timestamp compared against wall
        clock, not on a scheduler trigger's own cadence: this is called
        once per tick of recheck_pending_midterms_daily (an absolute-time
        CronTrigger, hour=6), the same job that already re-checks
        Midterm resource holds daily. An IntervalTrigger's next_run_time is
        recomputed relative to *registration* time on every add_job() call,
        so re-registering it on every process restart (as start() does,
        with replace_existing=True) silently resets its countdown to zero —
        this is why the old separate "send_biweekly_transcript" interval
        job could drift arbitrarily far past its configured interval on a
        server that restarts often. Deriving "is it due" from a stored
        timestamp instead of scheduler state is immune to that: any number
        of restarts between two daily 6am ticks changes nothing here.
        """
        settings = get_settings()
        if not settings.transcript_secondary_recipient_email:
            return False
        anchor = upload.last_secondary_transcript_sent_at or upload.uploaded_at
        interval = timedelta(days=settings.transcript_secondary_interval_days)
        if utcnow() - anchor < interval:
            return False
        try:
            email_service.send_transcript_email(
                upload.id, recipient_emails=[settings.transcript_secondary_recipient_email]
            )
            upload.last_secondary_transcript_sent_at = utcnow()
            self.db.commit()
            return True
        except Exception:
            logger.exception("Periodic transcript send failed for upload %s", upload.id)
            return False

    def get_upload(self, upload_id: str) -> CurriculumUpload:
        upload = (
            self.db.query(CurriculumUpload)
            .options(
                joinedload(CurriculumUpload.entries).joinedload(Curriculum.resources),
                joinedload(CurriculumUpload.entries).joinedload(Curriculum.midterm_detail),
            )
            .filter(CurriculumUpload.id == upload_id)
            .first()
        )
        if upload is None:
            raise NotFoundError(f"CurriculumUpload {upload_id!r} not found.")
        return upload
