"""Create a segmented dataset from raw video files using DP-based segmentation.

This script processes all movies (E01, E02, etc.) in datasets/raw/ and creates
segmented datasets using the embedding_dp.py segmentation pipeline.
"""

import argparse
import asyncio
import re
import sys
from pathlib import Path
from typing import Optional

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from segmentation.embedding_dp import (
    DEFAULT_VIDEO_ENCODER,
    segment_video,
    split_segments_ffmpeg,
)

PROJECT_ROOT = SRC_ROOT.parent
RAW_VIDEOS_DIR = PROJECT_ROOT / "datasets" / "raw"

# Dictionary mapping movie name to (start_time, end_time) for skipping intro/outro
# None means use full video duration.
MOVIE_TIME_RANGES: dict[str, Optional[tuple[float, Optional[float]]]] = {
    # "E01-Little_Quacker": (25.0, 419.0),
    # "E02-Saturday_Evening_Puss": (27.0, 366.0),
    "E03-Texas_Tom": (25.0, 394.0),
    "E04-Jerry_And_The_Lion": (26.0, 415.0),
    "E05-Safety_Second": (27, 403),
    "E06-Tom_and_Jerry_in_the_Hollywood_Bowl": (33, 434),
    "E07-The_Framed_Cat": (27, 420),
    "E08-Cue_Ball_Cat": (30, 414),
    "E09-Casanova_Cat": (26, 416),
    "E10-Jerry_And_The_Goldfish": (28, 433),
    "S03-The_Night_Before_Christmas": (32, 510),
}


def find_movie_files(raw_dir: Path) -> list[Path]:
    """Find all movie files matching E01, E02, etc. pattern."""
    if not raw_dir.exists():
        raise ValueError(f"Raw videos directory does not exist: {raw_dir}")
    
    movie_files: list[Path] = []
    pattern = re.compile(r"^E\d{2}-.*\.mp4$")
    
    for file_path in raw_dir.iterdir():
        if file_path.is_file() and pattern.match(file_path.name):
            movie_files.append(file_path)
    
    return sorted(movie_files)


def get_movie_name(video_path: Path) -> str:
    """Extract movie name from video path (without extension)."""
    return video_path.stem


async def process_movie(
    video_path: Path,
    dataset_dir: Path,
    cache_dir: Path,
    movie_name: str,
    hop: float,
    win: float,
    min_len: float,
    max_len: float,
    embedding_purpose: str,
    segment_penalty: float,
    max_concurrent: int,
    video_encoder: str,
) -> None:
    """Process a single movie: segment it and save segments to dataset directory."""
    print(f"\nProcessing {movie_name}...")
    
    # Get start/end times from dictionary (must exist since we filtered)
    time_range = MOVIE_TIME_RANGES[movie_name]
    if time_range is None or time_range == (None, None):
        start_time = 0.0
        end_time = None
    else:
        start_time, end_time = time_range
        if start_time is None:
            start_time = 0.0
    
    # Set up directories
    movie_output_dir = dataset_dir / movie_name
    movie_cache_dir = cache_dir / movie_name
    
    print(f"  Output directory: {movie_output_dir}")
    print(f"  Cache directory: {movie_cache_dir}")
    if start_time > 0 or end_time is not None:
        print(f"  Time range: {start_time}s - {end_time if end_time else 'end'}s")
    
    # Segment the video
    segments = await segment_video(
        video_path=video_path,
        hop=hop,
        win=win,
        min_len=min_len,
        max_len=max_len,
        embedding_purpose=embedding_purpose,
        segment_penalty=segment_penalty,
        clip_dir=movie_cache_dir,
        start_time=start_time,
        end_time=end_time,
        max_concurrent=max_concurrent,
        video_encoder=video_encoder,
    )
    
    print(f"  Generated {len(segments)} segments")
    
    # Split and save segments
    movie_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = split_segments_ffmpeg(
        video_path=video_path,
        segments=segments,
        output_dir=movie_output_dir,
        video_encoder=video_encoder,
    )
    
    print(f"  Saved {len(output_paths)} segment files to {movie_output_dir}")


async def create_dataset(
    dataset_name: str,
    hop: float = 0.5,
    win: float = 1.0,
    min_len: float = 1.0,
    max_len: float = 5.0,
    embedding_purpose: str = "CLUSTERING",
    segment_penalty: float = 0.3,
    max_concurrent: int = 128,
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
) -> None:
    """Create a segmented dataset from all movies in datasets/raw/."""
    # Set up directory structure
    dataset_dir = PROJECT_ROOT / "datasets" / dataset_name
    cache_dir = PROJECT_ROOT / "datasets" / "cache" / dataset_name
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Cache directory: {cache_dir}")
    
    # Find all movie files
    all_movie_files = find_movie_files(RAW_VIDEOS_DIR)
    if not all_movie_files:
        print(f"No movie files found in {RAW_VIDEOS_DIR}")
        return
    
    # Filter to only movies specified in MOVIE_TIME_RANGES
    movie_files: list[Path] = []
    for movie_file in all_movie_files:
        movie_name = get_movie_name(movie_file)
        if movie_name in MOVIE_TIME_RANGES:
            movie_files.append(movie_file)
    
    if not movie_files:
        print(f"No movies specified in MOVIE_TIME_RANGES dictionary.")
        print(f"Found {len(all_movie_files)} movie(s) in {RAW_VIDEOS_DIR}, but none are in the dictionary.")
        print("Available movies:", ", ".join(get_movie_name(f) for f in all_movie_files))
        return
    
    print(f"\nFound {len(movie_files)} movie(s) to process (from MOVIE_TIME_RANGES):")
    for movie_file in movie_files:
        movie_name = get_movie_name(movie_file)
        time_range = MOVIE_TIME_RANGES[movie_name]
        if time_range is None or time_range == (None, None):
            range_str = " (full video)"
        else:
            start, end = time_range
            start_str = f"{start}s" if start is not None else "0s"
            end_str = f"{end}s" if end is not None else "end"
            range_str = f" ({start_str} - {end_str})"
        print(f"  - {movie_name}{range_str}")
    
    # Process each movie
    for movie_file in movie_files:
        movie_name = get_movie_name(movie_file)
        await process_movie(
            video_path=movie_file,
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            movie_name=movie_name,
            hop=hop,
            win=win,
            min_len=min_len,
            max_len=max_len,
            embedding_purpose=embedding_purpose,
            segment_penalty=segment_penalty,
            max_concurrent=max_concurrent,
            video_encoder=video_encoder,
        )
    
    print(f"\n✓ Dataset creation complete: {dataset_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a segmented dataset from raw video files using DP-based segmentation."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the dataset (will be created under datasets/{dataset_name})",
    )
    parser.add_argument(
        "--hop",
        type=float,
        default=0.5,
        help=(
            "Scan hop in seconds (time between consecutive scan centers). "
            "Smaller hops sample more densely and increase clip count."
        ),
    )
    parser.add_argument(
        "--win",
        type=float,
        default=1.0,
        help=(
            "Window length per scan in seconds (clip duration centered on each hop). "
            "Longer windows average more context but increase embedding cost."
        ),
    )
    parser.add_argument(
        "--min-len",
        type=float,
        default=1.0,
        help="Minimum segment length in seconds.",
    )
    parser.add_argument(
        "--max-len",
        type=float,
        default=5.0,
        help="Maximum segment length in seconds.",
    )
    parser.add_argument(
        "--segment-penalty",
        type=float,
        default=0.5,
        help="Optional per-segment penalty added to the SSE cost.",
    )
    parser.add_argument(
        "--embedding-purpose",
        type=str,
        default="CLUSTERING",
        help="Embedding purpose for Nova embeddings.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=128,
        help="Maximum concurrent clip embeddings.",
    )
    parser.add_argument(
        "--video-encoder",
        type=str,
        default=DEFAULT_VIDEO_ENCODER,
        help="Video encoder used for scan clips and output segments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(
        create_dataset(
            dataset_name=args.dataset_name,
            hop=args.hop,
            win=args.win,
            min_len=args.min_len,
            max_len=args.max_len,
            embedding_purpose=args.embedding_purpose,
            segment_penalty=args.segment_penalty,
            max_concurrent=args.max_concurrent,
            video_encoder=args.video_encoder,
        )
    )


if __name__ == "__main__":
    main()

