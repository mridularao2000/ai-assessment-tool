"""Deterministic HMAC-SHA256 submission tokens.

Tokens are bound to a specific assessment_id and cannot be reused across
assessments. There are no expiry timestamps — expiry is enforced by
Assessment.due_date / Assessment.status in the database.

Usage:
    from app.utils.token_auth import generate_submission_token, verify_submission_token

    token = generate_submission_token("assessment-uuid-123")
    # → "dGhpcyBpcyBhIHRlc3Q..."  (URL-safe base64, no padding)

    ok = verify_submission_token("assessment-uuid-123", token)
    # → True

    bad = verify_submission_token("assessment-uuid-123", "tampered")
    # → False

Secret key resolution (in priority order):
    1. settings.submission_token_secret  — preferred; set in .env
    2. settings.database_url             — fallback; unique per deployment
"""
from __future__ import annotations

import base64
import hashlib
import hmac

from app.config import get_settings


def _secret_key() -> bytes:
    settings = get_settings()
    secret = getattr(settings, "submission_token_secret", None) or settings.database_url
    return secret.encode()


def generate_submission_token(assessment_id: str) -> str:
    """Return a URL-safe base64-encoded HMAC-SHA256 token for assessment_id."""
    digest = hmac.new(_secret_key(), assessment_id.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def verify_submission_token(assessment_id: str, token: str) -> bool:
    """Return True iff token is the valid HMAC token for assessment_id."""
    expected = generate_submission_token(assessment_id)
    return hmac.compare_digest(expected, token)
