"""One-off schema-alignment CLI for a specific SQLite file (e.g. a local
copy of assessment.db, or a Render disk snapshot pulled down for
inspection).

Why this exists: the app only ever calls Base.metadata.create_all() (see
app/main.py's lifespan) — that creates missing TABLES but never ALTERs an
existing table to add new columns. As of this script's latest update, app
startup now also calls app.db.schema_sync.sync_schema() automatically
(same column list, same idempotent logic) right after create_all(), so a
running deployment — Render included — self-heals on its next restart
without anyone needing to run this by hand. This script remains useful for
inspecting/fixing a DB file directly without spinning up the app (e.g. a
local dev DB, or a downloaded copy of the production disk).

This project has no Alembic (migrations/versions/ is an empty, untracked
scaffold — never wired up). This script is intentionally a plain, explicit
CLI over the shared column list in app.db.schema_sync rather than a full
migration framework for a single-file SQLite DB.

Usage:
    .venv/bin/python migrations/align_assessment_db_schema.py [path-to-db]

Defaults to ./assessment.db (the app's configured DATABASE_URL) if no path
is given. Idempotent — safe to re-run; already-present columns/tables are
skipped.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from app.db.schema_sync import PENDING_COLUMNS as ADD_COLUMNS

# ── New tables (created via SQLAlchemy so they exactly match the models —
# indexes, constraints, column order, everything) ──────────────────────────
NEW_TABLES = ["curriculum_uploads", "midterm_details"]


def _existing_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _existing_tables(cur: sqlite3.Cursor) -> set[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def migrate(db_path: str) -> None:
    print(f"Migrating {db_path}")

    # New tables first, via SQLAlchemy so they're byte-for-byte what the
    # models declare (this needs the real engine/metadata, not raw sqlite3).
    from sqlalchemy import create_engine

    from app.database import Base
    import app.models  # noqa: F401 - populates Base.metadata

    engine = create_engine(f"sqlite:///{db_path}")
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        live_tables = _existing_tables(cur)

        to_create = [Base.metadata.tables[t] for t in NEW_TABLES if t not in live_tables]
        if to_create:
            Base.metadata.create_all(bind=engine, tables=to_create)
            for t in to_create:
                print(f"  created table: {t.name}")
        else:
            print("  no missing tables")

        for table, statements in ADD_COLUMNS.items():
            existing_cols = _existing_columns(cur, table)
            for stmt in statements:
                # e.g. "ALTER TABLE assessments ADD COLUMN part1_text TEXT" -> "part1_text"
                col_name = stmt.split("ADD COLUMN")[1].strip().split(" ")[0]
                if col_name in existing_cols:
                    continue
                cur.execute(stmt)
                print(f"  added column: {table}.{col_name}")
        conn.commit()
    finally:
        conn.close()
        engine.dispose()

    print("Done.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent / "assessment.db")
    migrate(db_path)
