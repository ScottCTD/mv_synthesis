#!/usr/bin/env python3
"""
Retrieve top-k video clips per lyric line and synthesize a single MV.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from synthesis.config import (
    DEFAULT_DATASET_ROOT,
    LYRICS_AUGMENTED_QUERY_COLLECTION,
    LYRICS_TEXT_COLLECTION,
    LYRICS_TEXT_AUGMENTED_QUERY_COLLECTION,
    PROJECT_ROOT,
)
from synthesis.db import QdrantStore
from synthesis.lyrics_io import find_song_audio, load_song
from synthesis.models import Candidate, LyricsLine
from synthesis.postprocess import PostprocessPlan, build_postprocess_plan
from synthesis.render import render_clips_parallel, stitch_video
from synthesis.retrieval import retrieve_candidates
from synthesis.selection import SelectionResult, select_candidate


@dataclass(frozen=True)
class PipelineConfig:
    dataset_root: Path
    song_dir: Path
    song_name: str
    db_url: str
    output_dir: Path
    query_source: str
    top_k: int
    selection_strategy: str
    duration_threshold: float
    render_workers: int
    video_encoder: str
    dry_run: bool


def resolve_song_dir(
    song_dir_arg: Optional[str], dataset_root: Path, song_name: Optional[str]
) -> Path:
    if song_dir_arg:
        path = Path(song_dir_arg)
        if not path.is_absolute():
            return PROJECT_ROOT / path
        return path
    if not song_name:
        raise ValueError("song name is required when song-dir is not provided.")
    return dataset_root / "songs" / song_name


def infer_dataset_root(song_dir: Path, dataset_root_arg: Optional[Path]) -> Path:
    if dataset_root_arg is not None:
        return dataset_root_arg
    if song_dir.parent.name == "songs":
        return song_dir.parent.parent
    return DEFAULT_DATASET_ROOT


def resolve_output_dir(output_dir_arg: Optional[str], song_name: str) -> Path:
    if output_dir_arg:
        path = Path(output_dir_arg)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return PROJECT_ROOT / "outputs" / "synthesis" / song_name


def choose_query_collection(query_source: str) -> str:
    if query_source == "augmented":
        return LYRICS_AUGMENTED_QUERY_COLLECTION
    if query_source == "text":
        return LYRICS_TEXT_COLLECTION
    if query_source == "text_augmented":
        return LYRICS_TEXT_AUGMENTED_QUERY_COLLECTION
    raise ValueError(f"Unknown query source: {query_source}")


def get_query_embedding_id(line: LyricsLine, query_source: str) -> str:
    embedding_id = line.embedding_id
    if not embedding_id:
        raise ValueError(
            f"Missing embedding id for line {line.index} ({query_source} query)."
        )
    return embedding_id


def candidates_to_dicts(candidates: list[Candidate]) -> list[dict]:
    return [candidate.to_dict() for candidate in candidates]


def selection_to_dict(selection: SelectionResult) -> Optional[dict]:
    if selection.candidate is None:
        return None
    data = selection.candidate.to_dict()
    data["strategy"] = selection.strategy
    return data


def plan_to_dict(plan: Optional[PostprocessPlan]) -> Optional[dict]:
    if plan is None:
        return None
    return {
        "input_path": str(plan.input_path),
        "output_path": str(plan.output_path),
        "start_offset": plan.start_offset,
        "duration": plan.duration,
        "pad_black": plan.pad_black,
        "clip_duration": plan.clip_duration,
        "speed_factor": plan.speed_factor,
        "note": plan.note,
        "input_exists": plan.input_path.exists(),
    }


def run_pipeline(config: PipelineConfig) -> Path:
    lyrics_lines_path = config.song_dir / "lyrics_lines.json"
    if not lyrics_lines_path.exists():
        raise FileNotFoundError(
            f"lyrics_lines.json not found: {lyrics_lines_path}. "
            "Run embed_lyrics_lines.py first."
        )

    song = load_song(lyrics_lines_path, name_fallback=config.song_name)
    if not song.lyrics_lines:
        raise ValueError(f"No lyric lines found in {lyrics_lines_path}")
    song.lyrics_lines = sorted(
        song.lyrics_lines, key=lambda line: (line.index, line.start)
    )

    song_audio = find_song_audio(config.song_dir)
    if song_audio is None or not song_audio.exists():
        raise FileNotFoundError(f"song audio not found in {config.song_dir}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = config.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    store = QdrantStore(db_url=config.db_url)
    query_collection = choose_query_collection(config.query_source)

    manifest: list[dict] = []
    render_plans: list[PostprocessPlan] = []
    selected_paths: list[Path] = []

    try:
        for line in song.lyrics_lines:
            embedding_id = get_query_embedding_id(line, config.query_source)
            query_vector = store.retrieve_vector(query_collection, embedding_id)
            retrieval = retrieve_candidates(store, query_vector, config.top_k)
            selection = select_candidate(
                retrieval,
                config.selection_strategy,
                line.duration,
                config.duration_threshold,
            )

            output_clip = clips_dir / f"{line.index:04d}.mp4"
            plan = None
            if selection.candidate is not None:
                plan = build_postprocess_plan(
                    selection.candidate,
                    line.duration,
                    output_clip,
                    config.dataset_root,
                )
                if plan and plan.input_path.exists():
                    render_plans.append(plan)
                    selected_paths.append(output_clip)

            manifest.append(
                {
                    "index": line.index,
                    "start": line.start,
                    "end": line.end,
                    "duration": line.duration,
                    "lyric_text": line.text,
                    "augmented_query": line.augmented_query,
                    "text_path": str(line.text_path),
                    "audio_path": str(line.audio_path),
                    "query_source": config.query_source,
                    "query_collection": query_collection,
                    "query_embedding_id": embedding_id,
                    "video_candidates": candidates_to_dicts(retrieval.video_candidates),
                    "vibe_candidates": candidates_to_dicts(retrieval.vibe_candidates),
                    "selected": selection_to_dict(selection),
                    "postprocess": plan_to_dict(plan),
                }
            )
    finally:
        store.close()

    manifest_path = config.output_dir / "retrieval_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    if config.dry_run:
        return manifest_path

    if not selected_paths:
        raise ValueError("No clips were selected; cannot synthesize video.")

    render_clips_parallel(render_plans, config.render_workers, config.video_encoder)
    return stitch_video(
        selected_paths,
        song_audio,
        song.lyrics_lines,
        config.output_dir,
        config.video_encoder,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k clips per lyric line and synthesize a MV."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root containing songs/ videos/ and db/.",
    )
    parser.add_argument(
        "--song-name",
        default=None,
        help="Song name under the dataset songs directory.",
    )
    parser.add_argument(
        "--song-dir",
        default=None,
        help="Absolute or project-relative path to a song directory.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="http://localhost:6333",
        help="URL to remote Qdrant server (e.g., http://localhost:6333) or path to local Qdrant database directory (e.g., /path/to/db). Defaults to http://localhost:6333.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs; defaults to outputs/synthesis/<song>.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-k retrieval size.")
    parser.add_argument(
        "--query-source",
        choices=["augmented", "text", "text_augmented"],
        default="text_augmented",
        help="Which lyric embedding to use for retrieval.",
    )
    parser.add_argument(
        "--selection-strategy",
        choices=[
            "top_vibe",
            "top_vibe_duration",
            "top_video",
            "top_video_duration",
            "intersection",
        ],
        default="top_vibe_duration",
        help="Candidate selection strategy.",
    )
    parser.add_argument(
        "--duration-threshold",
        type=float,
        default=0.5,
        help="Tolerance window in seconds for duration-based selection.",
    )
    parser.add_argument(
        "--render-workers",
        type=int,
        default=16,
        help="Concurrent ffmpeg workers for rendering clips.",
    )
    parser.add_argument(
        "--video-encoder",
        choices=["libx264", "h264_videotoolbox"],
        default="h264_videotoolbox",
        help="FFmpeg video encoder; use h264_videotoolbox for macOS GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write retrieval manifest; skip ffmpeg synthesis.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root_for_song = args.dataset_root or DEFAULT_DATASET_ROOT
    song_dir = resolve_song_dir(args.song_dir, dataset_root_for_song, args.song_name)
    dataset_root = infer_dataset_root(song_dir, args.dataset_root)
    song_name = args.song_name or song_dir.name
    output_dir = resolve_output_dir(args.output_dir, song_name)
    
    db_url = args.db_url
    # If it's a local path (not a URL), check if it exists
    if not db_url.startswith("http://") and not db_url.startswith("https://"):
        db_path = Path(db_url)
        if not db_path.exists():
            raise FileNotFoundError(f"qdrant db not found: {db_path}")

    if not song_dir.exists():
        raise FileNotFoundError(f"song dir not found: {song_dir}")

    config = PipelineConfig(
        dataset_root=dataset_root,
        song_dir=song_dir,
        song_name=song_name,
        db_url=db_url,
        output_dir=output_dir,
        query_source=args.query_source,
        top_k=args.top_k,
        selection_strategy=args.selection_strategy,
        duration_threshold=args.duration_threshold,
        render_workers=args.render_workers,
        video_encoder=args.video_encoder,
        dry_run=args.dry_run,
    )

    result = run_pipeline(config)
    print(f"Done. Output: {result}")


if __name__ == "__main__":
    main()
