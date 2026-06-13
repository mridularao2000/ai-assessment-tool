from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (no tzinfo).

    Used as the Python-side default for all DateTime columns so that
    ORM instances have a populated timestamp immediately after construction,
    without waiting for a DB flush.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
