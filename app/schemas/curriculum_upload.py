from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel


class CurriculumUploadResponse(BaseModel):
    upload_id: str
    entry_count: int
    syllabus_email_sent: bool


class PendingCompletionSlot(BaseModel):
    slug: str
    label: str
    value: Optional[str] = None


class CurriculumEntrySummary(BaseModel):
    id: str
    topic: str
    entry_type: str
    chapter_label: str
    completion_date: date
    max_marks: float
    resources_hold: bool
    # Live classification from transcript_service.display_status — e.g.
    # "Missed — Late-Eligible (2 left)". The UI uses the "Missed —
    # Late-Eligible" prefix to decide when to show the late-send trigger.
    status: str
    # Populated only for a midterm-type entry currently on hold — the
    # pending_completion slots the "Submit completed project" UI section
    # needs to render. None for every other entry (not just False/empty),
    # so the frontend can distinguish "nothing to show" from "not a held
    # midterm at all".
    pending_completion: Optional[list[PendingCompletionSlot]] = None


class CurriculumUploadDetailResponse(BaseModel):
    upload_id: str
    source_filename: str
    uploaded_at: datetime
    entries: list[CurriculumEntrySummary]


class PendingResourcesUpdate(BaseModel):
    values: dict[str, str]


class PendingResourcesResponse(BaseModel):
    curriculum_id: str
    resources_hold: bool
    pending_completion_slots: dict[str, Optional[str]]


class GPAResponse(BaseModel):
    upload_id: str
    total_earned: float
    total_max: float
    gpa: float
    graded_count: int
    missed_count: int


class TranscriptRowResponse(BaseModel):
    row_id: str
    topic: str
    chapter_number: Optional[int]
    max_marks: float
    status_label: str
    points: Optional[float]
    retake_note: Optional[str]
    was_late: bool


class TranscriptChapterGroupResponse(BaseModel):
    chapter_label: str
    rows: list[TranscriptRowResponse]


class TranscriptResponse(BaseModel):
    upload_id: str
    source_filename: str
    chapter_groups: list[TranscriptChapterGroupResponse]
    resolved_count: int
    total_entry_count: int
    graded_count: int
    total_credits: float
    total_points: float
    gpa: float
    course_material_captured_at: Optional[datetime]


class AddEntryRequest(BaseModel):
    """Same shape as one item in the upload file's `topics` list."""
    entry: dict[str, Any]


class UpdateEntryRequest(BaseModel):
    """Field -> new value. Allowed: topic, chapter_label,
    target_completion_date, max_marks, resources (assessment-type),
    known_now/probe_focus (midterm-type). Refused entirely if the entry
    already has an attempt or grade on record."""
    updates: dict[str, Any]


class CloseUploadResponse(BaseModel):
    upload_id: str
    closed_at: datetime


class LateSendResponse(BaseModel):
    curriculum_id: str
    assessment_id: str
    sent: bool = True


class ResendSyllabusResponse(BaseModel):
    upload_id: str
    sent: bool = True


class BulkRescheduleRequest(BaseModel):
    """Shift due_date/scheduled_date forward by shift_days for every listed
    entry, without regenerating already-generated content. Only entries
    whose latest assessment is currently 'expired' (missed) qualify — see
    CurriculumUploadService.bulk_reschedule_entries."""
    curriculum_ids: list[str]
    shift_days: int


class BulkRescheduleResponse(BaseModel):
    updated: list[CurriculumEntrySummary]
