"""One-time admin fix: re-send the exam delivery email for an assessment
that already received it broken (b1b985e fixed EmailService to actually
render MCQ questions and the project brief; before that fix, the email
showed the literal string "None" for Section 1 and dropped the project
brief entirely).

The /resend API route only accepts status in (scheduled,
needs_manual_diagnosis) — this assessment is already `active` (content
generated, first email already sent), so that route correctly refuses it
as "nothing to resend" from its point of view. This script bypasses that
status gate and calls EmailService.send_assessment_email() directly: the
content (part1_text/part2_text) is already correct and unchanged, only
the email RENDERING was broken, so re-sending needs no regeneration and
no status change.

Run once, from the repo root (or in Render Shell after the fix is
deployed):
    python3 scripts/resend_corrected_exam_email.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal
from app.dependencies import _email
from app.services.email_service import EmailService

ASSESSMENT_ID = "68ca58e6-8a6e-4a1e-81c1-378d451756f2"


def main() -> None:
    db = SessionLocal()
    try:
        EmailService(db, _email).send_assessment_email(ASSESSMENT_ID)
        print(f"Resent corrected exam email for assessment {ASSESSMENT_ID}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
