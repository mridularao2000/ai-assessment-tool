from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.assessment import AssessmentStatus
from app.models.curriculum import Curriculum
from app.services.transcript_service import MISSED_NO_SCORE, display_status


@dataclass
class GPASummary:
    """Weighted GPA over one curriculum-upload's entries.

    gpa is total_earned/total_max as a 0-100 percentage, weighted by each
    entry's max_marks (not a simple average of per-entry percentages) —
    entries with more marks at stake count for more.
    """

    total_earned: float
    total_max: float
    gpa: float
    graded_count: int
    missed_count: int


def compute_gpa(db: Session, upload_id: str) -> GPASummary:
    """Compute GPA fresh from current state — never cached, so it's always
    correct immediately after any grading event.

    Only the FINAL attempt counts for an entry that was retaken (its Grade
    already reflects that attempt only — retakes overwrite nothing, each
    attempt is its own Assessment/Grade row, so "final attempt" means the
    highest attempt_number that EXISTS, not the highest one that happens
    to be graded — an entry with a fresh, not-yet-resolved retake must not
    silently fall back to its earlier failing grade; it isn't counted at
    all until that retake resolves, mirroring transcript_service._row_for's
    same "not yet resolved -> excluded" rule).

    "Missed — No Score" entries (due date passed, later calendar month, no
    late-eligible window left) count as 0 earned / max_marks in the
    denominator, per the confirmed GPA rule. "Missed — Late-Eligible"
    entries (still within this calendar month, a token could still be
    spent) are NOT counted yet — they're still a live outcome, not final.
    Not-yet-due / held / in-progress entries aren't counted at all.
    """
    entries = (
        db.query(Curriculum).filter(Curriculum.upload_id == upload_id).all()
    )

    total_earned = 0.0
    total_max = 0.0
    graded_count = 0
    missed_count = 0

    for curriculum in entries:
        if curriculum.assessments:
            final = max(curriculum.assessments, key=lambda a: a.attempt_number)
            if (
                final.status == AssessmentStatus.completed
                and final.submission is not None
                and final.submission.grade is not None
            ):
                grade = final.submission.grade
                if grade.score_earned is not None and grade.max_marks:
                    total_earned += grade.score_earned
                    total_max += grade.max_marks
                    graded_count += 1
                continue
            if final.status != AssessmentStatus.expired:
                # The latest attempt exists but isn't resolved yet (e.g. a
                # retake was just generated and hasn't been graded, or the
                # first attempt is still scheduled/active/submitted) — not
                # counted yet, and NOT a fallback to an earlier attempt's
                # stale grade.
                continue
            # else: latest attempt is `expired` — due_date passed with
            # nothing ever submitted for it — fall through to the
            # missed-check below exactly as if there were no assessments.

        if display_status(db, curriculum) == MISSED_NO_SCORE:
            total_max += curriculum.max_marks or 0.0
            missed_count += 1

    gpa = (total_earned / total_max * 100.0) if total_max else 0.0
    return GPASummary(
        total_earned=total_earned,
        total_max=total_max,
        gpa=gpa,
        graded_count=graded_count,
        missed_count=missed_count,
    )
