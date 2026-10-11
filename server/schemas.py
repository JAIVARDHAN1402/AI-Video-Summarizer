"""Request and response shapes for the API.

Kept separate from the ORM so the wire contract can evolve independently of
the table, and so FastAPI can generate accurate OpenAPI docs.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

VideoType = Literal[
    "lecture", "coding", "meeting", "interview", "talk", "news", "documentary", "other"
]
Profile = Literal["fast", "balanced", "accurate"]


class CreateJobRequest(BaseModel):
    url: Optional[str] = Field(
        None, description="A YouTube or other yt-dlp-supported URL. "
                          "Give this or upload_id, not both.",
    )
    upload_id: Optional[str] = Field(
        None, description="id returned by POST /api/uploads, to process a file "
                          "already uploaded instead of a URL.",
    )
    video_type: VideoType = "lecture"
    topic: str = ""
    profile: Profile = "balanced"
    force: bool = False
    sla_sec: Optional[float] = Field(
        None, gt=0,
        description="Wall-clock target in seconds. Overrides the profile's "
                    "default budget for routing and SLA scoring.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "CreateJobRequest":
        if bool(self.url and self.url.strip()) == bool(self.upload_id and self.upload_id.strip()):
            raise ValueError("provide exactly one of 'url' or 'upload_id'")
        return self


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    bytes: int


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
    routed_worker_class: Optional[str] = None
    predicted_sec: Optional[float] = None
    sla_target_sec: Optional[float] = None
    sla_met: Optional[bool] = None
    timings: dict[str, Any] = {}
    worker_class: Optional[str] = None
    elapsed_sec: Optional[float] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class AskTurn(BaseModel):
    q: str
    a: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    # Prior turns, so a follow-up ("explain that further") resolves against the
    # conversation instead of being read as a standalone question.
    history: list[AskTurn] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    confidence: str
    sources: list[dict[str, Any]]
    beyond_video: bool = False
    off_topic: bool = False


class VideoResult(BaseModel):
    """Everything the frontend needs to render one processed video."""

    video_id: str
    meta: dict[str, Any]
    understanding: Optional[dict[str, Any]] = None
    summary: Optional[dict[str, Any]] = None
    scenes: Optional[dict[str, Any]] = None
    transcript: Optional[dict[str, Any]] = None
    state: dict[str, Any] = {}
    highlight: Optional[dict[str, Any]] = None  # {url, seconds, scene_ids} when a reel exists


class NoteRequest(BaseModel):
    """A reader's own note. Unlike everything else in the cache this is not
    derived from the video, so it survives a re-run untouched."""

    text: str = Field(..., min_length=1, max_length=8000)
    # Where in the video the thought belongs, when it belongs anywhere.
    start_sec: Optional[float] = None
    # "manual" when typed, "chat" when saved from an answer.
    source: str = "manual"


class NotebookRequest(BaseModel):
    """Create or update a notebook. Both fields are optional on update, so a
    rename does not have to resend the membership and vice versa."""

    title: Optional[str] = Field(None, max_length=200)
    video_ids: Optional[list[str]] = None
