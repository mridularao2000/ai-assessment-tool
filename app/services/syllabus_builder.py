from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from app.models.curriculum import Curriculum, CurriculumEntryType
from app.models.curriculum_upload import CurriculumUpload

_CHAPTER_NUMBER_RE = re.compile(r"\s*Chapter\s+(\d+)", re.IGNORECASE)


@dataclass
class SyllabusAssessmentRow:
    topic: str
    resources: list[str]           # verbatim, never paraphrased
    completion_date: date
    window_start: date
    window_end: date


@dataclass
class SyllabusChapterSection:
    chapter_label: str              # verbatim `chapter` field
    assessments: list[SyllabusAssessmentRow] = field(default_factory=list)
    no_standalone_note: Optional[str] = None


@dataclass
class SyllabusMidtermRow:
    topic: str
    chapter_label: str               # verbatim cumulative-span description
    completion_date: date
    known_now: list[str]             # verbatim
    pending_status: list[tuple[str, bool]]   # (verbatim label, is_filled)
    probe_focus: Optional[str]
    special_case: Optional[str]
    resources_hold: bool


@dataclass
class SyllabusContent:
    upload_id: str
    source_filename: str
    chapters: list[SyllabusChapterSection]
    midterms: list[SyllabusMidtermRow]


def _chapter_number(chapter_label: str) -> Optional[int]:
    m = _CHAPTER_NUMBER_RE.match(chapter_label)
    return int(m.group(1)) if m else None


def serialize_syllabus_content(content: SyllabusContent) -> dict:
    """JSON-safe snapshot of a SyllabusContent, for freezing the transcript
    email's "Course Material" section at capture time (see
    CurriculumUpload.course_material_snapshot) — dates become ISO strings,
    everything else is already JSON-safe.
    """
    return {
        "upload_id": content.upload_id,
        "source_filename": content.source_filename,
        "chapters": [
            {
                "chapter_label": ch.chapter_label,
                "no_standalone_note": ch.no_standalone_note,
                "assessments": [
                    {
                        "topic": a.topic,
                        "resources": a.resources,
                        "completion_date": a.completion_date.isoformat(),
                        "window_start": a.window_start.isoformat(),
                        "window_end": a.window_end.isoformat(),
                    }
                    for a in ch.assessments
                ],
            }
            for ch in content.chapters
        ],
        "midterms": [
            {
                "topic": m.topic,
                "chapter_label": m.chapter_label,
                "completion_date": m.completion_date.isoformat(),
                "known_now": m.known_now,
                "pending_status": [[label, filled] for label, filled in m.pending_status],
                "probe_focus": m.probe_focus,
                "special_case": m.special_case,
                "resources_hold": m.resources_hold,
            }
            for m in content.midterms
        ],
    }


def build_syllabus(upload: CurriculumUpload, entries: list[Curriculum]) -> SyllabusContent:
    """Build the syllabus email's content from already-loaded ORM rows.

    Pure function — no DB/LLM/email calls. `entries` must have `.resources`
    and `.midterm_detail` eager-loaded by the caller.

    Layout choice (flagged — the spec's "Midterms listed at their point in
    the sequence by due date" is ambiguous between a fully merged
    chapter+date timeline and Midterms as their own date-ordered list):
    this builds two separate ordered sections — chapters 1..N in numeric
    order, then Midterms in their own due-date order — rather than
    interleaving them into one timeline, since a literal merge produces
    out-of-numeric-order chapter headings whenever a later chapter's
    Assessment completes before an earlier chapter's (as happens in the
    current seed: Chapter 11 completes before Chapter 2).
    """
    assessments = [e for e in entries if e.entry_type == CurriculumEntryType.assessment]
    midterms = [e for e in entries if e.entry_type == CurriculumEntryType.midterm]

    by_chapter: dict[str, list[Curriculum]] = defaultdict(list)
    for e in assessments:
        by_chapter[e.chapter_label].append(e)

    no_standalone_notes: dict[str, str] = {
        row["chapter"]: row["note"]
        for row in (upload.chapters_with_no_standalone_assessment or [])
    }

    all_labels = set(by_chapter) | set(no_standalone_notes)
    ordered_labels = sorted(
        all_labels,
        key=lambda label: (
            _chapter_number(label) is None,
            _chapter_number(label) or 0,
            label,
        ),
    )

    chapters: list[SyllabusChapterSection] = []
    for label in ordered_labels:
        rows = [
            SyllabusAssessmentRow(
                topic=e.topic,
                resources=[r.source_ref for r in e.resources],
                completion_date=e.target_completion_date,
                window_start=e.target_completion_date + timedelta(days=1),
                window_end=e.target_completion_date + timedelta(days=3),
            )
            for e in sorted(by_chapter.get(label, []), key=lambda e: e.target_completion_date)
        ]
        chapters.append(
            SyllabusChapterSection(
                chapter_label=label,
                assessments=rows,
                no_standalone_note=no_standalone_notes.get(label),
            )
        )

    midterm_rows: list[SyllabusMidtermRow] = []
    for e in sorted(midterms, key=lambda e: e.target_completion_date):
        detail = e.midterm_detail
        pending_status = [
            (detail.pending_completion_labels[slug], detail.pending_completion_slots.get(slug) is not None)
            for slug in detail.pending_completion_labels
        ]
        midterm_rows.append(
            SyllabusMidtermRow(
                topic=e.topic,
                chapter_label=e.chapter_label,
                completion_date=e.target_completion_date,
                known_now=list(detail.known_now),
                pending_status=pending_status,
                probe_focus=detail.probe_focus,
                special_case=detail.special_case,
                resources_hold=e.resources_hold,
            )
        )

    return SyllabusContent(
        upload_id=upload.id,
        source_filename=upload.source_filename,
        chapters=chapters,
        midterms=midterm_rows,
    )
