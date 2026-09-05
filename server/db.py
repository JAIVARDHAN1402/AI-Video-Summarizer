"""Database layer: SQLAlchemy engine, session, and the Job table.

SQLite by default so the service runs with no external database; the models
are plain SQLAlchemy 2.0 and move to Postgres by changing ``VS_DATABASE_URL``
alone. A job row is the single source of truth for one processing request —
its status, live progress, timings, and where the result lives.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any, Optional

from sqlalchemy import JSON, DateTime, Enum, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from server import config

_engine = create_engine(
    config.DATABASE_URL,
    echo=False,
    future=True,
    # SQLite + a background worker thread need this; harmless on Postgres.
    connect_args={"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Job(Base):
    """One video-processing request and everything known about its progress."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)

    # --- request ---------------------------------------------------------
    source_ref: Mapped[str] = mapped_column(Text)          # URL, or stored upload path
    is_upload: Mapped[bool] = mapped_column(default=False)
    video_type: Mapped[str] = mapped_column(String(24), default="lecture")
    topic: Mapped[str] = mapped_column(Text, default="")
    profile: Mapped[str] = mapped_column(String(16), default="balanced")
    force: Mapped[bool] = mapped_column(default=False)

    # --- identity of the resulting video ------------------------------------
    video_id: Mapped[Optional[str]] = mapped_column(String(32), index=True, default=None)
    title: Mapped[Optional[str]] = mapped_column(Text, default=None)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)

    # --- lifecycle -----------------------------------------------------------
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False, length=16), default=JobStatus.QUEUED, index=True
    )
    stage: Mapped[str] = mapped_column(String(24), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    timings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    worker_class: Mapped[Optional[str]] = mapped_column(String(48), default=None)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), default=None)
    finished_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), default=None)

    def elapsed_sec(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at or _now()
        return round((end - self.started_at).total_seconds(), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "video_id": self.video_id,
            "title": self.title,
            "duration_sec": self.duration_sec,
            "video_type": self.video_type,
            "topic": self.topic,
            "profile": self.profile,
            "timings": self.timings or {},
            "worker_class": self.worker_class,
            "elapsed_sec": self.elapsed_sec(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    config.ensure_dirs()
    Base.metadata.create_all(_engine)


def session() -> Session:
    return SessionLocal()
