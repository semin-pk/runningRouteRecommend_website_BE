"""Common dependency functions for database session management."""

from typing import Generator

from sqlalchemy.orm import Session

from app.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Dependency function to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



