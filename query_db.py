#!/usr/bin/env python3
"""
query_db.py — quick read-only view of curricula and their scheduled assessments.

Usage:
    python query_db.py               # uses ./assessment.db
    python query_db.py /data/assessment.db
"""
import sqlite3
import sys
from pathlib import Path


# ── DB path resolution ────────────────────────────────────────────────────────

def resolve_db_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    # Try DATABASE_URL from .env if present
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                url = line.split("=", 1)[1].strip()
                # sqlite:////absolute/path  or  sqlite:///./relative
                path_part = url.replace("sqlite:///", "").lstrip("/")
                if path_part.startswith("."):
                    return Path(__file__).parent / path_part
                return Path("/" + path_part)
    return Path(__file__).parent / "assessment.db"


# ── Formatting helpers ────────────────────────────────────────────────────────

def short(uuid_str: str | None, n: int = 8) -> str:
    if not uuid_str:
        return "—"
    return uuid_str[:n] + "…"


def fmt_dt(val: str | None) -> str:
    if not val:
        return "—"
    # SQLite stores datetimes as strings; trim microseconds for readability
    return str(val)[:16]


def divider(char: str = "─", width: int = 72) -> str:
    return char * width


# ── Queries ───────────────────────────────────────────────────────────────────

CURRICULUM_SQL = """
    SELECT
        id,
        topic,
        target_completion_date,
        status,
        mastery_achieved,
        created_at
    FROM curricula
    ORDER BY target_completion_date ASC, created_at ASC
"""

ASSESSMENT_SQL = """
    SELECT
        id,
        attempt_number,
        status,
        scheduled_at,
        reminder_at,
        due_date,
        duration_minutes,
        submission_token
    FROM assessments
    WHERE curriculum_id = ?
    ORDER BY attempt_number ASC
"""


def run(db_path: Path) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    curricula = cur.execute(CURRICULUM_SQL).fetchall()

    if not curricula:
        print("No curricula found.")
        return

    print(divider("═"))
    print(f"  Curricula ({len(curricula)} total)  —  database: {db_path}")
    print(divider("═"))

    for c in curricula:
        mastery = (
            "✓ mastered" if c["mastery_achieved"]
            else ("✗ not mastered" if c["mastery_achieved"] is not None else "pending")
        )
        print(f"\n  ID       {short(c['id'])}  ({c['id']})")
        print(f"  Topic    {c['topic']}")
        print(f"  Target   {c['target_completion_date']}")
        print(f"  Status   {c['status']}   Mastery: {mastery}")
        print(f"  Created  {fmt_dt(c['created_at'])}")

        assessments = cur.execute(ASSESSMENT_SQL, (c["id"],)).fetchall()

        if not assessments:
            print("  Assessments  (none)")
        else:
            print(f"  Assessments  ({len(assessments)} attempt(s))")
            print("  " + divider("·", 68))
            for a in assessments:
                token_hint = (a["submission_token"] or "")[:16] + "…"
                print(f"    Attempt #{a['attempt_number']}  [{a['status']}]  "
                      f"id: {short(a['id'])}")
                print(f"      Scheduled   {fmt_dt(a['scheduled_at'])}")
                print(f"      Reminder    {fmt_dt(a['reminder_at'])}")
                print(f"      Due         {fmt_dt(a['due_date'])}")
                if a["duration_minutes"]:
                    print(f"      Duration    {a['duration_minutes']} min")
                print(f"      Token       {token_hint}")
            print("  " + divider("·", 68))

        print("  " + divider())

    con.close()


if __name__ == "__main__":
    run(resolve_db_path())
