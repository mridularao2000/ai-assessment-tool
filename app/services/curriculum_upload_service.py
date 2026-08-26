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
from app.models._utils import utcnow
from app.models.assessment import Assessment, AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType, CurriculumStatus
from app.models.curriculum_upload import CurriculumUpload
from app.models.midterm_detail import MidtermDetail
from app.models.resource import Resource, ResourceType
from app.services.assessment_service import calculate_scheduled_at, build_assessment_dates
from app.services.email_service import EmailService
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
        created: list[Curriculum] = []
        for entry in parsed_entries:
            curriculum = Curriculum(
                id=str(uuid.uuid4()),
                topic=entry.topic,
                target_completion_date=entry.completion_date,
                status=CurriculumStatus.ready,
                entry_type=entry.entry_type,
                upload_id=upload.id,
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

            created.append(curriculum)

        self.db.commit()
        for c in created:
            self.db.refresh(c)
        self.db.refresh(upload)

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

    def assemble_part1_pool(self, curriculum: Curriculum) -> tuple[list[Curriculum], bool]:
        return assemble_part1_pool(self.db, curriculum)

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
        """Clear resources_hold if every pending slot is now filled, and
        immediately schedule its exam window from today (not the original
        completion_date), per spec.

        Returns True if this call flipped it off. Shared by
        fill_pending_resources() and the daily recheck job so a manual
        PATCH resolves immediately without waiting for the next cron tick.
        """
        detail = curriculum.midterm_detail
        if detail is None or not curriculum.resources_hold:
            return False
        if all(v is not None for v in detail.pending_completion_slots.values()):
            curriculum.resources_hold = False
            self.db.commit()
            self._schedule_entry_assessment(curriculum, date.today())
            return True
        return False

    def send_hold_reminder(self, curriculum: Curriculum, email_service: EmailService) -> None:
        """Send (or re-send) the "resources still missing" reminder. Non-fatal."""
        try:
            email_service.send_midterm_hold_reminder_email(curriculum.id)
            curriculum.last_hold_reminder_at = utcnow()
            self.db.commit()
        except Exception:
            logger.exception("Hold-reminder email failed for curriculum %s", curriculum.id)

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
