"""Typed data contracts shared across every pipeline module.

Each dataclass mirrors one of the JSON artifacts written to
``cache/<video_id>/``. Changing a field here means updating every module
that reads or writes the corresponding JSON file (see BUILD_PLAN.md).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# --- Exceptions --------------------------------------------------------------


class VideoSummarizerError(Exception):
    """Base class for all pipeline errors."""


class IngestError(VideoSummarizerError):
    """Raised when ingestion (download, probe, or extraction) fails."""


class VideoTooLongError(IngestError):
    """Raised when a video exceeds ``config.MAX_VIDEO_DURATION_SEC``."""


class DependencyError(VideoSummarizerError):
    """Raised when a required system dependency (e.g. FFmpeg) is missing."""


class TranscriptionError(VideoSummarizerError):
    """Raised when speech-to-text transcription fails."""


# --- meta.json -----------------------------------------------------------------


@dataclass
class VideoMeta:
    video_id: str
    source: str  # "youtube" | "upload"
    source_ref: str
    title: str
    duration_sec: float
    width: int
    height: int
    fps: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMeta":
        return cls(**data)


# --- transcript.json -------------------------------------------------------------


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        return cls(**data)


@dataclass
class Transcript:
    language: str
    source: str  # "whisper" | "youtube_captions"
    full_text: str
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "source": self.source,
            "full_text": self.full_text,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            language=data["language"],
            source=data["source"],
            full_text=data["full_text"],
            segments=[TranscriptSegment.from_dict(s) for s in data.get("segments", [])],
        )


# --- scenes.json -----------------------------------------------------------------


@dataclass
class ObjectTag:
    label: str
    conf: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectTag":
        return cls(**data)


@dataclass
class Scene:
    id: int
    start: float
    end: float
    keyframe_path: str
    objects: list[ObjectTag] = field(default_factory=list)
    clip_embedding_idx: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "keyframe_path": self.keyframe_path,
            "objects": [o.to_dict() for o in self.objects],
            "clip_embedding_idx": self.clip_embedding_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scene":
        return cls(
            id=data["id"],
            start=data["start"],
            end=data["end"],
            keyframe_path=data["keyframe_path"],
            objects=[ObjectTag.from_dict(o) for o in data.get("objects", [])],
            clip_embedding_idx=data.get("clip_embedding_idx", -1),
        )


@dataclass
class ScenesResult:
    scenes: list[Scene] = field(default_factory=list)
    embeddings_file: str = ""
    segmentation_method: str = "content_detector"  # or "fixed_interval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "embeddings_file": self.embeddings_file,
            "segmentation_method": self.segmentation_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScenesResult":
        return cls(
            scenes=[Scene.from_dict(s) for s in data.get("scenes", [])],
            embeddings_file=data.get("embeddings_file", ""),
            segmentation_method=data.get("segmentation_method", "content_detector"),
        )


# --- fused.json ------------------------------------------------------------------


@dataclass
class FusedBlock:
    id: int
    start: float
    end: float
    scene_ids: list[int]
    visual_tags: list[str]
    text: str
    fused_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusedBlock":
        return cls(**data)


@dataclass
class FusedResult:
    blocks: list[FusedBlock] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FusedResult":
        return cls(blocks=[FusedBlock.from_dict(b) for b in data.get("blocks", [])])


# --- summary.json ----------------------------------------------------------------


@dataclass
class Chapter:
    start: float
    end: float
    title: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chapter":
        return cls(**data)


@dataclass
class ChunkSummary:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkSummary":
        return cls(**data)


@dataclass
class Summary:
    overall_summary: str
    chapters: list[Chapter] = field(default_factory=list)
    chunk_summaries: list[ChunkSummary] = field(default_factory=list)
    warning: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "overall_summary": self.overall_summary,
            "chapters": [c.to_dict() for c in self.chapters],
            "chunk_summaries": [c.to_dict() for c in self.chunk_summaries],
        }
        if self.warning:
            data["warning"] = self.warning
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Summary":
        return cls(
            overall_summary=data["overall_summary"],
            chapters=[Chapter.from_dict(c) for c in data.get("chapters", [])],
            chunk_summaries=[
                ChunkSummary.from_dict(c) for c in data.get("chunk_summaries", [])
            ],
            warning=data.get("warning"),
        )


# --- qa_index/chunks.json ---------------------------------------------------------


@dataclass
class RagChunk:
    id: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagChunk":
        return cls(**data)


@dataclass
class QASource:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QASource":
        return cls(**data)


@dataclass
class QAAnswer:
    answer: str
    sources: list[QASource]
    confidence: str  # "high" | "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAAnswer":
        return cls(
            answer=data["answer"],
            sources=[QASource.from_dict(s) for s in data.get("sources", [])],
            confidence=data["confidence"],
        )


# --- state.json ------------------------------------------------------------------

STAGES = ("ingest", "transcribe", "vision", "fuse", "summarize", "index", "done", "error")


@dataclass
class JobState:
    stage: str
    progress: float
    message: str
    error: Optional[str]
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        return cls(**data)
