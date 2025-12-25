from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

try:
    from synthesis.models import LyricsLine, Song
except ImportError:
    from models import LyricsLine, Song

LYRIC_FILE_RE = re.compile(
    r"^(?P<idx>\d+?)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$"
)
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".aac")


def path_to_relative(path: Path, base_dir: Path) -> Path:
    try:
        return path.relative_to(base_dir)
    except ValueError:
        return path


def resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return base_dir / path


def find_audio_clip(base_path: Path) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def find_song_audio(song_dir: Path) -> Optional[Path]:
    for ext in AUDIO_EXTS:
        for audio_file in sorted(song_dir.glob(f"*{ext}")):
            return audio_file
    return None


def load_lyrics_lines(
    lyrics_dir: Path,
    dataset_root: Path,
    include_empty: bool = True,
    strict_audio: bool = True,
) -> list[LyricsLine]:
    lines: list[LyricsLine] = []
    for txt_path in sorted(lyrics_dir.glob("*.txt")):
        match = LYRIC_FILE_RE.match(txt_path.stem)
        if not match:
            continue
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end <= start:
            continue
        index = int(match.group("idx"))
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text and not include_empty:
            continue
        audio_path = find_audio_clip(txt_path.with_suffix(""))
        if audio_path is None:
            if strict_audio:
                raise FileNotFoundError(f"Audio clip not found for {txt_path}")
            continue
        relative_txt = path_to_relative(txt_path, dataset_root)
        relative_audio = path_to_relative(audio_path, dataset_root)
        lines.append(
            LyricsLine(
                index=index,
                start=start,
                end=end,
                duration=end - start,
                text=text,
                augmented_query="",
                text_path=relative_txt,
                audio_path=relative_audio,
            )
        )
    return sorted(lines, key=lambda line: (line.index, line.start))


def build_song_from_clips(
    song_name: str,
    lyrics_dir: Path,
    dataset_root: Path,
    include_empty: bool = True,
    strict_audio: bool = True,
) -> Song:
    lines = load_lyrics_lines(
        lyrics_dir,
        dataset_root=dataset_root,
        include_empty=include_empty,
        strict_audio=strict_audio,
    )
    return Song(name=song_name, lyrics_lines=lines)


def write_song(song: Song, output_path: Path) -> None:
    output_path.write_text(
        json.dumps(song.to_dict(), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def load_song(song_path: Path, name_fallback: Optional[str] = None) -> Song:
    data = json.loads(song_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        lines = [LyricsLine.from_dict(item) for item in data]
        name = name_fallback or song_path.parent.name
        return Song(name=name, lyrics_lines=lines)
    return Song.from_dict(data, name_fallback=name_fallback)
