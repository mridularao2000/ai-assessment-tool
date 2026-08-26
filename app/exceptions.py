"""Application-level exceptions shared across all services."""


class NotFoundError(Exception):
    """Raised when a requested DB record does not exist."""


class InvalidStateError(Exception):
    """Raised when an operation is attempted on a record in the wrong state."""


class InvalidTokenError(Exception):
    """Raised when a submission token fails HMAC verification."""


class IngestionError(Exception):
    """Raised when a URL, repo, or file cannot be fetched or parsed."""


class CurriculumUploadValidationError(Exception):
    """Raised when an uploaded curriculum file doesn't match the template
    contract. Carries a per-entry, per-field message; the whole upload is
    rejected atomically — no partial ingestion."""
