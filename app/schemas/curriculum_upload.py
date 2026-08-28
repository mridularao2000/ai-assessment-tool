from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel


class CurriculumUploadResponse(BaseModel):
    upload_id: str
    entry_count: int
    syllabus_email_sent: bool


class CurriculumEntrySummary(BaseModel):
    id: str
    topic: str
    entry_type: str
    chapter_label: str
    completion_date: date
    max_marks: float
    resources_hold: bool


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
