from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class LyricsLine:
    index: int
    start: float
    end: float
    duration: float
    text: str
    augmented_query: str
    text_path: Path
    audio_path: Path
    embedding_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "text": self.text,
            "augmented_query": self.augmented_query,
            "text_path": str(self.text_path),
            "audio_path": str(self.audio_path),
            "embedding_id": self.embedding_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LyricsLine":
        duration = data.get("duration")
        if duration is None:
            duration = float(data["end"]) - float(data["start"])
        return cls(
            index=int(data["index"]),
            start=float(data["start"]),
            end=float(data["end"]),
            duration=float(duration),
            text=str(data.get("text", "")),
            augmented_query=str(data.get("augmented_query", "")),
            text_path=Path(data["text_path"]),
            audio_path=Path(data["audio_path"]),
            embedding_id=data.get("embedding_id"),
        )


@dataclass
class Song:
    name: str
    lyrics_lines: list[LyricsLine]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lyrics_lines": [line.to_dict() for line in self.lyrics_lines],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], name_fallback: Optional[str] = None) -> "Song":
        name = data.get("name") or name_fallback or "unknown"
        lines = [LyricsLine.from_dict(item) for item in data.get("lyrics_lines", [])]
        return cls(name=name, lyrics_lines=lines)


@dataclass
class Candidate:
    segment_id: Optional[str]
    segment_path: Optional[Path]
    duration: Optional[float]
    vibe_card: Optional[str]
    video_score: Optional[float] = None
    vibe_card_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "segment_path": str(self.segment_path) if self.segment_path else None,
            "duration": self.duration,
            "vibe_card": self.vibe_card,
            "video_score": self.video_score,
            "vibe_card_score": self.vibe_card_score,
        }
