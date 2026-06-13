from collections.abc import Generator

from sqlalchemy import event, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    # Required for SQLite to allow the same connection across threads.
    # Safe here because SessionLocal manages thread-local sessions.
    connect_args={"check_same_thread": False},
    # Surface SQLite constraint errors immediately rather than at commit time.
    echo=False,
)


@event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_conn, _connection_record) -> None:
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models.

    Import this in every model file:
        from app.database import Base
    """


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a database session per request.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()