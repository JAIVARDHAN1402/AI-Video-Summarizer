"""Request and response shapes for the API.

Kept separate from the ORM so the wire contract can evolve independently of
the table, and so FastAPI can generate accurate OpenAPI docs.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

VideoType = Literal[
    "lecture", "coding", "meeting", "interview", "talk", "news", "documentary", "other"
]
Profile = Literal["fast", "balanced", "accurate"]


class CreateJobRequest(BaseModel):
    url: str = Field(..., description="A YouTube or other yt-dlp-supported URL.")
    video_type: VideoType = "lecture"
    topic: str = ""
    profile: Profile = "balanced"
    force: bool = False


class JobRef(BaseModel):
    id: str
    status: str


class JobView(BaseModel):
    id: str
    status: str
    stage: str
    progress: float
    message: str
    error: Optional[str] = None
    video_id: Optional[str] = None
    title: Optional[str] = None
    duration_sec: Optional[float] = None
    video_type: str
    topic: str
    profile: str
    timings: dict[str, Any] = {}
    worker_class: Optional[str] = None
    elapsed_sec: Optional[float] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[dict[str, Any]]


class VideoResult(BaseModel):
    """Everything the frontend needs to render one processed video."""

    video_id: str
    meta: dict[str, Any]
    understanding: Optional[dict[str, Any]] = None
    summary: Optional[dict[str, Any]] = None
    scenes: Optional[dict[str, Any]] = None
    transcript: Optional[dict[str, Any]] = None
    state: dict[str, Any] = {}
