#!/usr/bin/env python3
"""
Retrieve top-k video clips per lyric line and synthesize a single MV.

Pipeline:
1) Load lyric lines from a clips_and_lyrics directory.
2) Embed each line and query Qdrant for top-k video segments.
3) Pick a clip per line and render trimmed segments.
4) Concatenate segments and add the song as background music.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from qdrant_client import QdrantClient

try:
    from synthesis.nova_embedding_model import Nova2OmniEmbeddings
except ImportError:
    from nova_embedding_model import Nova2OmniEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LYRIC_FILE_RE = re.compile(r"^(?P<idx>\d+?)_(?P<start>\d+\.\d+)-(?P<end>\d+\.\d+)$")


@dataclass(frozen=True)
class LyricLine:
    index: int
    start: float
    end: float
    duration: float
    text: str
    txt_path: Path
    audio_path: Optional[Path]


@dataclass(frozen=True)
class RetrievalCandidate:
    segment_id: Optional[str]
    segment_path: Optional[str]
    score: Optional[float]


def resolve_path(path: Optional[str], fallback: Path) -> Path:
    if path is None:
        return fallback
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return PROJECT_ROOT / path_obj


def find_audio_path(base_path: Path) -> Optional[Path]:
    for ext in (".wav", ".mp3", ".m4a", ".aac"):
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def find_song_audio(song_dir: Path) -> Optional[Path]:
    """Find the song audio file in a song directory."""
    for ext in (".mp3", ".wav", ".m4a", ".aac"):
        for audio_file in song_dir.glob(f"*{ext}"):
            return audio_file
    return None


def load_lyric_lines(lyrics_dir: Path, include_empty: bool) -> list[LyricLine]:
    lines: list[LyricLine] = []
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
        audio_path = find_audio_path(txt_path.with_suffix(""))
        lines.append(
            LyricLine(
                index=index,
                start=start,
                end=end,
                duration=end - start,
                text=text,
                txt_path=txt_path,
                audio_path=audio_path,
            )
        )
    return sorted(lines, key=lambda line: (line.index, line.start))


def run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command list and raise if the subprocess fails."""
    subprocess.run(cmd, check=True)


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe. Returns None on error."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def normalize_hits(hits: Iterable) -> list:
    if hasattr(hits, "points"):
        return list(hits.points)
    return list(hits)


def parse_candidates(hits: Iterable) -> list[RetrievalCandidate]:
    candidates: list[RetrievalCandidate] = []
    for hit in normalize_hits(hits):
        if isinstance(hit, dict):
            payload = hit.get("payload", {}) or {}
            score = hit.get("score")
        else:
            payload = getattr(hit, "payload", {}) or {}
            score = getattr(hit, "score", None)
        candidates.append(
            RetrievalCandidate(
                segment_id=payload.get("segment_id"),
                segment_path=payload.get("segment_path"),
                score=score,
            )
        )
    return candidates


def select_candidate(
    candidates: list[RetrievalCandidate],
    recent_paths: list[str],
    avoid_recent: int,
    min_duration: float = 0.0,
) -> Optional[RetrievalCandidate]:
    if not candidates:
        return None
    
    # Filter candidates by minimum duration if specified
    filtered_candidates = candidates
    if min_duration > 0:
        filtered_candidates = []
        for candidate in candidates:
            if not candidate.segment_path:
                continue
            candidate_path = Path(candidate.segment_path)
            if not candidate_path.is_absolute():
                candidate_path = PROJECT_ROOT / candidate_path
            if not candidate_path.exists():
                continue
            duration = get_video_duration(candidate_path)
            if duration is not None and duration >= min_duration:
                filtered_candidates.append(candidate)
        
        if not filtered_candidates:
            return None
    
    if avoid_recent <= 0:
        return filtered_candidates[0]
    recent_set = set(recent_paths[-avoid_recent:])
    for candidate in filtered_candidates:
        if candidate.segment_path and candidate.segment_path not in recent_set:
            return candidate
    return filtered_candidates[0]


def render_clip(
    input_path: Path,
    output_path: Path,
    duration: Optional[float],
    loop_short: bool,
    no_trim: bool,
) -> None:
    """Render a single video clip, optionally looping/trimming to match duration."""
    cmd = ["ffmpeg", "-y"]
    pad_duration = 0.0
    if duration and duration > 0 and not loop_short and not no_trim:
        input_duration = get_video_duration(input_path)
        if input_duration is not None and input_duration < duration:
            pad_duration = duration - input_duration
    if loop_short and duration and duration > 0:
        cmd += ["-stream_loop", "-1"]
    cmd += ["-i", str(input_path)]
    if pad_duration > 0:
        cmd += [
            "-vf",
            f"tpad=stop_mode=add:stop_duration={pad_duration:.3f}:color=black",
        ]
    if duration and duration > 0 and not no_trim:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def render_clips_parallel(
    tasks: list[tuple[Path, Path, Optional[float], bool, bool]],
    workers: int,
) -> None:
    """Render multiple clips concurrently using a thread pool."""
    if not tasks:
        return
    max_workers = max(1, workers)
    if max_workers == 1:
        for args in tasks:
            render_clip(*args)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(render_clip, *args) for args in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def write_concat_list(paths: Iterable[Path], list_path: Path) -> None:
    """Write an ffmpeg concat list file from ordered clip paths."""
    lines = [f"file '{path.as_posix()}'" for path in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_clips(list_path: Path, output_path: Path) -> None:
    """Concatenate clips using ffmpeg concat demuxer."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def add_background_music(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Mux background audio with the stitched video, trimming to the shorter stream."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def write_srt(lines: Iterable[LyricLine], srt_path: Path) -> None:
    """Write lyric lines to an SRT subtitle file."""
    blocks: list[str] = []
    index = 1
    for line in lines:
        if not line.text:
            continue
        start = format_srt_timestamp(line.start)
        end = format_srt_timestamp(line.end)
        blocks.append(f"{index}\n{start} --> {end}\n{line.text}\n")
        index += 1
    srt_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def escape_subtitles_path(path: Path) -> str:
    """Escape a path for ffmpeg subtitles filter usage."""
    escaped = str(path).replace("\\", "\\\\").replace(":", "\\:")
    return escaped


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    """Burn subtitles into the video using ffmpeg subtitles filter."""
    subtitle_filter = f"subtitles={escape_subtitles_path(srt_path)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_ffmpeg(cmd)


async def embed_lyrics(
    embed_model: Nova2OmniEmbeddings,
    lines: list[LyricLine],
    concurrency: int,
) -> list[list[float]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _embed_line(line: LyricLine) -> list[float]:
        async with semaphore:
            response = await embed_model.embed_text(
                line.text, embedding_purpose="VIDEO_RETRIEVAL"
            )
            return response["embeddings"][0]["embedding"]

    tasks = [_embed_line(line) for line in lines]
    return await asyncio.gather(*tasks)


async def run_pipeline(
    lyrics_dir: Path,
    song_audio: Path,
    db_path: Path,
    output_dir: Path,
    top_k: int,
    collection: str,
    embed_concurrency: int,
    render_workers: int,
    include_empty: bool,
    avoid_recent: int,
    loop_short: bool,
    no_trim: bool,
    dry_run: bool,
    min_clip_duration: float = 0.0,
) -> Path:
    lines = load_lyric_lines(lyrics_dir, include_empty=include_empty)
    if not lines:
        raise ValueError(f"No lyric lines found in {lyrics_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    embed_model = Nova2OmniEmbeddings()
    embeddings = await embed_lyrics(embed_model, lines, embed_concurrency)

    client = QdrantClient(path=str(db_path))

    manifest: list[dict] = []
    selected_paths: list[Path] = []
    render_tasks: list[tuple[Path, Path, Optional[float], bool, bool]] = []
    recent_paths: list[str] = []

    try:
        for line, embedding in zip(lines, embeddings):
            hits = client.query_points(
                collection_name=collection,
                query=embedding,
                limit=top_k,
                with_payload=True,
            )
            candidates = parse_candidates(hits)
            selected = select_candidate(
                candidates, recent_paths, avoid_recent, min_duration=min_clip_duration
            )

            selected_path = None
            if selected and selected.segment_path:
                selected_path = Path(selected.segment_path)
                if not selected_path.is_absolute():
                    selected_path = PROJECT_ROOT / selected_path
                recent_paths.append(selected.segment_path)

            manifest.append(
                {
                    "index": line.index,
                    "start": line.start,
                    "end": line.end,
                    "duration": line.duration,
                    "lyric_txt": str(line.txt_path),
                    "lyric_text": line.text,
                    "candidates": [
                        {
                            "segment_id": candidate.segment_id,
                            "segment_path": candidate.segment_path,
                            "score": candidate.score,
                        }
                        for candidate in candidates
                    ],
                    "selected": {
                        "segment_id": selected.segment_id if selected else None,
                        "segment_path": selected.segment_path if selected else None,
                        "score": selected.score if selected else None,
                    },
                }
            )

            if selected_path is None:
                continue
            if not selected_path.exists():
                continue

            output_clip = clips_dir / f"{line.index:04d}.mp4"
            duration = None if no_trim else line.duration
            render_tasks.append(
                (selected_path, output_clip, duration, loop_short, no_trim)
            )
            selected_paths.append(output_clip)
    finally:
        client.close()

    manifest_path = output_dir / "retrieval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if dry_run:
        return manifest_path

    if not selected_paths:
        raise ValueError("No clips were selected; cannot synthesize video.")

    render_clips_parallel(render_tasks, render_workers)

    concat_list = output_dir / "concat_list.txt"
    write_concat_list(selected_paths, concat_list)

    stitched_video = output_dir / "mv_video.mp4"
    concat_clips(concat_list, stitched_video)

    final_video = output_dir / "mv_with_music.mp4"
    add_background_music(stitched_video, song_audio, final_video)

    subtitle_path = output_dir / "lyrics.srt"
    write_srt(lines, subtitle_path)

    subtitled_video = output_dir / "mv_with_music_subtitled.mp4"
    burn_subtitles(final_video, subtitle_path, subtitled_video)

    return subtitled_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k clips per lyric line and synthesize a MV."
    )
    parser.add_argument(
        "--song-dir",
        default=str(PROJECT_ROOT / "datasets/explore/songs/renwoxing"),
        help="Song directory containing clips_and_lyrics/ subdirectory and song audio file.",
    )
    parser.add_argument(
        "--db-path",
        default=str(PROJECT_ROOT / "video_embeddings.db"),
        help="Path to the local Qdrant database.",
    )
    parser.add_argument(
        "--collection",
        default="segment_video_embeddings",
        help="Qdrant collection name for video embeddings.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k retrieval size.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs; defaults to outputs/synthesis/<song>.",
    )
    parser.add_argument(
        "--embed-concurrency",
        type=int,
        default=16,
        help="Concurrent embedding requests to Bedrock.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=4,
        help="Concurrent ffmpeg workers for rendering clips.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include empty lyric lines in retrieval.",
    )
    parser.add_argument(
        "--avoid-recent",
        type=int,
        default=0,
        help="Avoid reusing any of the last N selected clips.",
    )
    parser.add_argument(
        "--loop-short",
        action="store_true",
        help="Loop clips that are shorter than the lyric duration.",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Do not trim clips to lyric durations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write retrieval manifest; skip ffmpeg synthesis.",
    )
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=0.0,
        help="Minimum duration in seconds for retrieved clips. Clips shorter than this will be filtered out. Default: 0.0 (no filtering).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    song_dir = resolve_path(args.song_dir, PROJECT_ROOT)
    lyrics_dir = song_dir / "clips_and_lyrics"
    song_audio = find_song_audio(song_dir)
    db_path = resolve_path(args.db_path, PROJECT_ROOT)

    if args.output_dir:
        output_dir = resolve_path(args.output_dir, PROJECT_ROOT)
    else:
        song_name = song_dir.name
        output_dir = PROJECT_ROOT / "outputs" / "synthesis" / song_name

    if not song_dir.exists():
        raise FileNotFoundError(f"song dir not found: {song_dir}")
    if not lyrics_dir.exists():
        raise FileNotFoundError(f"lyrics dir not found: {lyrics_dir}")
    if song_audio is None or not song_audio.exists():
        raise FileNotFoundError(f"song audio not found in {song_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"qdrant db not found: {db_path}")

    result = asyncio.run(
        run_pipeline(
            lyrics_dir=lyrics_dir,
            song_audio=song_audio,
            db_path=db_path,
            output_dir=output_dir,
            top_k=args.top_k,
            collection=args.collection,
            embed_concurrency=args.embed_concurrency,
            render_workers=args.render_workers,
            include_empty=args.include_empty,
            avoid_recent=args.avoid_recent,
            loop_short=args.loop_short,
            no_trim=args.no_trim,
            dry_run=args.dry_run,
            min_clip_duration=args.min_clip_duration,
        )
    )
    print(f"Done. Output: {result}")


if __name__ == "__main__":
    main()
