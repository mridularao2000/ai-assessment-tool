from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session, joinedload

from app.models.assessment import AssessmentStatus
from app.models.curriculum import Curriculum, CurriculumEntryType
from app.exceptions import NotFoundError
from app.models.curriculum_upload import CurriculumUpload
from app.models.late_submission_token import LateSubmissionToken
from app.services.late_token_service import LateTokenService
from app.services.syllabus_builder import _chapter_number

# Status labels shown on the transcript (Section 7) and used by GPA (Section 6)
# to decide whether/how an entry counts toward the denominator. Section 7
# builds the full transcript email/HTML on top of this same classifier
# rather than duplicating it.
NOT_YET_DUE = "Not Yet Due"
EXAM_SENT = "Exam Sent"
GRADED = "Graded"
MISSED_NO_SCORE = "Missed — No Score"
HELD = "Awaiting Resources (Held)"
NEEDS_MANUAL_DIAGNOSIS = "Needs Manual Diagnosis"


def display_status(db: Session, curriculum: Curriculum) -> str:
    """Classify a curriculum-upload entry's current state for GPA/transcript.

    Every non-held entry gets an Assessment row at ingestion time (deferred
    *content* generation, not deferred row creation — see
    CurriculumUploadService.schedule_ready_entries), so "no Assessment row"
    is not a real state to branch on here; classification is driven by the
    latest attempt's Assessment.status instead.
    """
    if curriculum.resources_hold:
        # A held midterm's completion_date has already passed by
        # construction (that's what set resources_hold — see
        # CurriculumUploadService._create_entry). Same token-gated
        # monthly window as an expired assessment: recoverable with a
        # token within completion_date's calendar month (see
        # CurriculumUploadService.check_and_clear_hold), permanently
        # unscoreable once that month has passed — reusing MISSED_NO_SCORE
        # verbatim rather than a distinct label, since it's the exact same
        # terminal state, not just a similar one.
        now = datetime.utcnow()
        due = curriculum.target_completion_date
        if (due.year, due.month) != (now.year, now.month):
            return MISSED_NO_SCORE
        return HELD

    assessments = sorted(curriculum.assessments, key=lambda a: a.attempt_number)
    if not assessments:
        return NOT_YET_DUE

    latest = assessments[-1]

    if latest.status == AssessmentStatus.completed:
        return GRADED

    if latest.status == AssessmentStatus.expired:
        now = datetime.utcnow()
        due = latest.due_date
        if (due.year, due.month) == (now.year, now.month):
            balance = LateTokenService(db).get_balance(curriculum.upload_id)
            return f"Missed — Late-Eligible ({balance} left)"
        return MISSED_NO_SCORE

    if latest.status == AssessmentStatus.scheduled:
        return NOT_YET_DUE

    if latest.status == AssessmentStatus.needs_manual_diagnosis:
        # Distinct from EXAM_SENT below on purpose: nothing was ever
        # generated or sent for this row (generation exhausted its
        # tool-call budget before producing valid output — see
        # send_assessment_job's LLMToolBudgetExceededError handler), so
        # grouping it with "exam is out, grading not yet final" would be
        # actively misleading — a real production incident (Performance
        # Optimization I, 2026-08-31) where the transcript/entries UI
        # showed "Exam Sent" for a row that was never sent at all.
        return NEEDS_MANUAL_DIAGNOSIS

    # active, submitted, late_submitted — exam is out, grading not yet final.
    return EXAM_SENT


_RESOLVED_PREFIXES = (GRADED, "Missed")


def _is_resolved(status: str) -> bool:
    """An entry counts as "resolved" for the transcript once its outcome is
    final: Graded, or Missed in either flavor. Not-yet-due, held, and
    exam-sent-but-ungraded entries are excluded — the reversal from "never
    omit any entry" to "only show resolved entries" was an explicit,
    deliberate spec change (see checkpoint history)."""
    return status.startswith(_RESOLVED_PREFIXES)


def _compact_status_label(status: str) -> str:
    """Compact, uppercase form for the transcript table column — e.g.
    "Missed — Late-Eligible (2 left)" -> "MISSED–LATE (2)"."""
    if status == GRADED:
        return "GRADED"
    if status.startswith("Missed — Late-Eligible"):
        count = status.split("(")[1].split(" ")[0]
        return f"MISSED–LATE ({count})"
    if status == MISSED_NO_SCORE:
        return "MISSED"
    return status.upper()


@dataclass
class TranscriptEntryRow:
    row_id: str            # e.g. "CH1-A" or "MT-A"
    topic: str
    chapter_number: Optional[int]
    max_marks: float
    status_label: str      # compact form, e.g. "GRADED", "GRADED (LATE)", "MISSED–LATE (2)"
    points: Optional[float]  # None -> rendered as "—"
    retake_note: Optional[str] = None  # e.g. "retake, was 38.00"
    was_late: bool = False  # graded from a late-token-covered submission —
    # the transcript must show this distinctly, not render identically to
    # an on-time grade (a missed-then-late-graded entry is real history,
    # not something the final "completed" status alone can show — see
    # AssessmentStatus.late_submitted getting overwritten to `completed`
    # by GradingService.grade(), which loses the distinction unless it's
    # re-derived here from LateSubmissionToken.used_by_assessment_id).


@dataclass
class TranscriptChapterGroup:
    chapter_label: str
    rows: list[TranscriptEntryRow]


@dataclass
class TranscriptContent:
    upload_id: str
    source_filename: str
    chapter_groups: list[TranscriptChapterGroup]
    resolved_count: int
    total_entry_count: int
    graded_count: int
    total_credits: float
    total_points: float
    gpa: float
    course_material: Optional[dict]     # frozen snapshot, see CurriculumUpload
    course_material_captured_at: Optional[datetime]


def _row_for(db: Session, curriculum: Curriculum) -> Optional[TranscriptEntryRow]:
    status = display_status(db, curriculum)
    if not _is_resolved(status):
        return None

    attempts = sorted(curriculum.assessments, key=lambda a: a.attempt_number)
    # A permanently-missed HELD midterm (see display_status) never got an
    # Assessment row at all — the window closed before it ever opened, so
    # there's nothing to point at. final stays None for that case; every
    # branch below that needs it is already guarded on status/attempts.
    final = attempts[-1] if attempts else None

    points: Optional[float] = None
    retake_note: Optional[str] = None
    was_late = False
    if status == GRADED and final.submission is not None and final.submission.grade is not None:
        points = final.submission.grade.score_earned
        # A token is spent to get access past a closed window, not for a
        # normal fail-then-retry — so it may be recorded against an
        # EARLIER attempt than the one that produced the final grade (a
        # token-covered attempt that failed, followed by a free retake
        # that passed). Checking only `final.id` misses that case; the
        # entry is late if a token was used at ANY point in its history.
        attempt_ids = [a.id for a in attempts]
        was_late = (
            db.query(LateSubmissionToken)
            .filter(LateSubmissionToken.used_by_assessment_id.in_(attempt_ids))
            .first()
            is not None
        )

    if final is not None and final.attempt_number > 1:
        first = attempts[0]
        if first.submission is not None and first.submission.grade is not None:
            prior_points = first.submission.grade.score_earned
            if prior_points is not None:
                retake_note = f"retake, was {prior_points:.2f}"

    status_label = _compact_status_label(status)
    if was_late:
        status_label += " (LATE)"

    return TranscriptEntryRow(
        row_id="",  # assigned by compute_transcript once chapter grouping/order is known
        topic=curriculum.topic,
        chapter_number=_chapter_number(curriculum.chapter_label or ""),
        max_marks=curriculum.max_marks or 0.0,
        status_label=status_label,
        points=points,
        retake_note=retake_note,
        was_late=was_late,
    )


def compute_transcript(db: Session, upload_id: str) -> TranscriptContent:
    """Build the transcript's entry table + footer totals, fresh from
    current state (never cached) — only the "Course Material" section is
    frozen (see CurriculumUpload.course_material_snapshot).

    Chapter grouping/ordering mirrors syllabus_builder.build_syllabus()
    exactly (numeric chapter order, then unparseable labels — which covers
    every Midterm, since its chapter_label is a cumulative-span
    description, not a "Chapter N" pattern — sorted last, alphabetically)
    so the transcript and syllabus present entries in a consistent order.
    """
    upload = (
        db.query(CurriculumUpload)
        .options(joinedload(CurriculumUpload.entries).joinedload(Curriculum.assessments))
        .filter(CurriculumUpload.id == upload_id)
        .first()
    )
    if upload is None:
        raise NotFoundError(f"CurriculumUpload {upload_id!r} not found.")

    # Local import — gpa_service imports display_status from this module,
    # so a top-level import here would be circular.
    from app.services.gpa_service import compute_gpa

    entries = list(upload.entries)
    by_chapter: dict[str, list[Curriculum]] = {}
    for e in entries:
        by_chapter.setdefault(e.chapter_label or "", []).append(e)

    ordered_labels = sorted(
        by_chapter,
        key=lambda label: (_chapter_number(label) is None, _chapter_number(label) or 0, label),
    )

    chapter_groups: list[TranscriptChapterGroup] = []
    graded_count = 0
    resolved_count = 0
    for label in ordered_labels:
        rows: list[TranscriptEntryRow] = []
        letter_index = 0
        prefix = f"CH{_chapter_number(label)}" if _chapter_number(label) is not None else "MT"
        for curriculum in sorted(by_chapter[label], key=lambda e: e.target_completion_date):
            row = _row_for(db, curriculum)
            if row is None:
                continue
            letter_index += 1
            row.row_id = f"{prefix}-{chr(ord('A') + letter_index - 1)}"
            rows.append(row)
            resolved_count += 1
            if row.status_label == "GRADED":
                graded_count += 1
        if rows:
            chapter_groups.append(TranscriptChapterGroup(chapter_label=label, rows=rows))

    gpa_summary = compute_gpa(db, upload_id)

    return TranscriptContent(
        upload_id=upload.id,
        source_filename=upload.source_filename,
        chapter_groups=chapter_groups,
        resolved_count=resolved_count,
        total_entry_count=len(entries),
        graded_count=graded_count,
        total_credits=gpa_summary.total_max,
        total_points=gpa_summary.total_earned,
        gpa=gpa_summary.gpa,
        course_material=upload.course_material_snapshot,
        course_material_captured_at=upload.course_material_captured_at,
    )
