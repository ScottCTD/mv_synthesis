from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current.parents[1]


PROJECT_ROOT = _resolve_project_root()
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets" / "ds1"

EMBEDDING_DIM = 3072

VIDEO_SEGMENTS_COLLECTION = "video-segments"
VIDEO_VIBE_CARDS_COLLECTION = "video-vibe_cards"


def lyrics_collection_name(embedding_purpose: str, query_source: str) -> str:
    return f"lyrics-{embedding_purpose}-{query_source}"


LYRICS_COLLECTIONS = {
    "text": {
        "text_video": lyrics_collection_name("text_video", "text"),
        "text_text": lyrics_collection_name("text_text", "text"),
    },
    "augment": {
        "text_video": lyrics_collection_name("text_video", "augment"),
        "text_text": lyrics_collection_name("text_text", "augment"),
    },
    "combined": {
        "text_video": lyrics_collection_name("text_video", "combined"),
        "text_text": lyrics_collection_name("text_text", "combined"),
    },
    "audio": {
        "audio_video": lyrics_collection_name("audio_video", "audio"),
        "audio_text": lyrics_collection_name("audio_text", "audio"),
    },
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path

    @property
    def songs_dir(self) -> Path:
        return self.root / "songs"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def db_dir(self) -> Path:
        return self.root / "db"

    def song_dir(self, song_name: str) -> Path:
        return self.songs_dir / song_name


@dataclass(frozen=True)
class SongPaths:
    dataset: DatasetPaths
    song_name: str

    @property
    def song_dir(self) -> Path:
        return self.dataset.song_dir(self.song_name)

    @property
    def clips_and_lyrics_dir(self) -> Path:
        return self.song_dir / "clips_and_lyrics"

    @property
    def lyrics_lines_path(self) -> Path:
        return self.song_dir / "lyrics_lines.json"

    def output_dir(self, outputs_root: Path) -> Path:
        return outputs_root / "synthesis" / self.song_name
