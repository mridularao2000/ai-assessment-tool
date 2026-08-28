"""Regression coverage for a real bug found via a live dry run: app.main's
lifespan called Base.metadata.create_all()/seed_prompt_templates() against
app.database.engine — a process-global singleton bound to the real
DATABASE_URL default (sqlite:///./assessment.db) at import time. FastAPI's
TestClient triggers lifespan on startup, and app.dependency_overrides only
intercepts routes' Depends(get_db) — it never touched what engine lifespan
itself used. Result: running the test suite silently created new tables in
the tracked dev DB.

Fixed by gating that whole startup block behind settings.run_schema_bootstrap
(see app/config.py, app/main.py), forced False for the entire test session
at the top of conftest.py, before any app.* import. These tests prove the
fix holds: the real file's schema is provably untouched by a full lifespan
startup/shutdown cycle and by ordinary route access through the `client`
fixture every other test in the suite uses.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

ASSESSMENT_DB_PATH = Path(__file__).resolve().parent.parent / "assessment.db"


def _schema_fingerprint(db_path: Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = [row[0] for row in cur.fetchall()]
        fingerprint = {}
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            fingerprint[table] = sorted(row[1] for row in cur.fetchall())
        return fingerprint
    finally:
        conn.close()


class TestRealDbSchemaNeverMutatedByTests:

    def test_run_schema_bootstrap_is_disabled_for_the_whole_test_session(self):
        from app.config import get_settings

        assert get_settings().run_schema_bootstrap is False
        assert os.environ.get("RUN_SCHEMA_BOOTSTRAP") == "false"

    def test_app_lifespan_startup_via_testclient_does_not_touch_real_db_schema(self):
        """Explicit `with` forces a full lifespan startup+shutdown cycle —
        the exact path that silently ran create_all()/seed against the real
        file before this was gated. Hits /docs (no DB dependency at all)
        purely to force the ASGI app through a real request cycle."""
        assert ASSESSMENT_DB_PATH.exists(), "expected the tracked dev DB to exist for this check"
        tables_before = set(_schema_fingerprint(ASSESSMENT_DB_PATH))
        schema_before = _schema_fingerprint(ASSESSMENT_DB_PATH)
        mtime_before = ASSESSMENT_DB_PATH.stat().st_mtime_ns

        from app.main import app as real_app

        with TestClient(real_app) as fresh_client:
            response = fresh_client.get("/docs")
            assert response.status_code == 200

        schema_after = _schema_fingerprint(ASSESSMENT_DB_PATH)
        mtime_after = ASSESSMENT_DB_PATH.stat().st_mtime_ns

        # The two tables this bug actually created, named explicitly so a
        # regression here fails loudly rather than as a generic diff.
        assert "curriculum_uploads" not in (set(schema_after) - tables_before)
        assert "midterm_details" not in (set(schema_after) - tables_before)
        assert schema_after == schema_before, "real assessment.db schema changed from a lifespan startup/shutdown cycle"
        assert mtime_after == mtime_before, "real assessment.db file was written to by a lifespan startup/shutdown cycle"

    def test_client_fixture_requests_do_not_touch_real_db_schema(self, client):
        """Same guarantee through the actual `client` fixture every other
        test in the suite uses — not a special one-off TestClient."""
        schema_before = _schema_fingerprint(ASSESSMENT_DB_PATH)
        mtime_before = ASSESSMENT_DB_PATH.stat().st_mtime_ns

        response = client.get("/health/prompts")
        assert response.status_code == 200

        schema_after = _schema_fingerprint(ASSESSMENT_DB_PATH)
        mtime_after = ASSESSMENT_DB_PATH.stat().st_mtime_ns

        assert schema_after == schema_before
        assert mtime_after == mtime_before
