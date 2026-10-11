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
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED)


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _aware(value: dt.datetime) -> dt.datetime:
    """SQLite round-trips ``DateTime(timezone=True)`` as naive UTC — put the
    tzinfo back so arithmetic against :func:`_now` doesn't raise."""
    return value if value.tzinfo is not None else value.replace(tzinfo=dt.timezone.utc)


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
    routed_worker_class: Mapped[Optional[str]] = mapped_column(String(48), default=None)
    predicted_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    sla_target_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    sla_met: Mapped[Optional[bool]] = mapped_column(default=None)

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
        return round((_aware(end) - _aware(self.started_at)).total_seconds(), 1)

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
            "routed_worker_class": self.routed_worker_class,
            "predicted_sec": self.predicted_sec,
            "sla_target_sec": self.sla_target_sec,
            "sla_met": self.sla_met,
            "timings": self.timings or {},
            "worker_class": self.worker_class,
            "elapsed_sec": self.elapsed_sec(),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


class Notebook(Base):
    """A named set of processed videos, asked questions as one body of work.

    Membership is a list of video ids rather than a join table: a notebook is
    small, always read whole, and the videos it names live in the cache rather
    than in this database, so there is no foreign key to enforce. A member
    whose cache has since been cleared is skipped at read time instead of
    breaking the notebook.
    """

    __tablename__ = "notebooks"

    id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:12]
    )
    title: Mapped[str] = mapped_column(String(200), default="Untitled notebook")
    video_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "video_ids": list(self.video_ids or []),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SchedulerCost(Base):
    """One learned cost figure for a (worker class, metric) pair.

    The distributed scheduler's online cost model. ``key`` is a coarse metric
    like "total:per_min"; ``seconds`` is the blended estimate; ``samples`` is
    how many completed jobs have fed it. Kept per worker class so a GPU node
    and a CPU node each carry their own numbers.
    """

    __tablename__ = "scheduler_costs"

    worker_class: Mapped[str] = mapped_column(String(48), primary_key=True)
    key: Mapped[str] = mapped_column(String(48), primary_key=True)
    seconds: Mapped[float] = mapped_column(Float)
    samples: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SlaObservation(Base):
    """Predicted vs actual wall-clock for one completed job.

    This is the evidence table: aggregate it to show how often the scheduler's
    routing kept the SLA, and by how much a fixed-configuration baseline would
    have overrun.
    """

    __tablename__ = "sla_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    worker_class: Mapped[str] = mapped_column(String(48), index=True)
    duration_sec: Mapped[float] = mapped_column(Float)
    predicted_sec: Mapped[float] = mapped_column(Float)
    actual_sec: Mapped[float] = mapped_column(Float)
    sla_target_sec: Mapped[float] = mapped_column(Float)
    met: Mapped[bool] = mapped_column()
    baseline_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    baseline_met: Mapped[Optional[bool]] = mapped_column(default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


def init_db() -> None:
    """Create tables if they don't exist. Called once at startup."""
    config.ensure_dirs()
    Base.metadata.create_all(_engine)


def session() -> Session:
    return SessionLocal()
