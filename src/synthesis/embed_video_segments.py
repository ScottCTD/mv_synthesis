from __future__ import annotations

import argparse
import asyncio
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from synthesis.config import (
    DEFAULT_DATASET_ROOT,
    VIDEO_SEGMENTS_COLLECTION,
    VIDEO_VIBE_CARDS_COLLECTION,
)
from synthesis.db import QdrantStore
from synthesis.ffmpeg_utils import get_video_duration
from synthesis.lyrics_io import path_to_relative
from synthesis.nova2_lite_model import Nova2LiteModel
from synthesis.nova_embedding_model import Nova2OmniEmbeddings


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def collect_video_segments(all_video_path: Path) -> dict[str, Path]:
    video_id_to_path: dict[str, Path] = {}
    videos = sorted(os.listdir(all_video_path))
    for video in videos:
        if "." in video:
            continue
        segments = sorted(os.listdir(all_video_path / video))
        for segment in segments:
            if not segment.endswith(".mp4") and not segment.endswith(".mkv"):
                continue
            segment_path = all_video_path / video / segment
            segment_name = segment_path.stem
            segment_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{video}-{segment_name}"))
            video_id_to_path[segment_id] = segment_path
    return video_id_to_path


async def embed_and_store_segment(
    video_id_to_path: dict[str, Path],
    segment_id: str,
    embed_model: Nova2OmniEmbeddings,
    vision_model: Nova2LiteModel,
    store: QdrantStore,
    dataset_root: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        segment_path = video_id_to_path[segment_id]
        if store.point_exists(VIDEO_SEGMENTS_COLLECTION, segment_id):
            print(f"Skipping for segment path = {segment_path} because exists")
            return
        try:
            segment_duration = get_video_duration(segment_path)
            assert (
                segment_duration is not None
            ), f"Segment duration not found for segment {segment_path}"
            result = await embed_model.embed_video(
                str(segment_path),
                embedding_purpose="GENERIC_INDEX",
                embedding_mode="AUDIO_VIDEO_SEPARATE",
            )
            video_embedding = None
            for embedding in result["embeddings"]:
                if embedding["embedding_type"] == "VIDEO":
                    video_embedding = embedding["embedding"]
                    break
            assert (
                video_embedding is not None
            ), f"Video embedding not found for segment {segment_path}"

            vibe_card = await vision_model.generate_vibe_card_frames(str(segment_path), fps=3, max_frames=10, reasoning_effort="medium")
            vibe_card_embedding = await embed_model.embed_text(
                vibe_card, embedding_purpose="GENERIC_INDEX", truncation_mode="END"
            )
            vibe_card_embedding = vibe_card_embedding["embeddings"][0]["embedding"]

            relative_path = path_to_relative(segment_path, dataset_root)
            base_payload = {
                "segment_id": segment_id,
                "segment_path": str(relative_path),
                "duration": segment_duration,
            }

            store.upsert_vector(
                collection_name=VIDEO_SEGMENTS_COLLECTION,
                point_id=segment_id,
                vector=video_embedding,
                payload=base_payload,
            )
            store.upsert_vector(
                collection_name=VIDEO_VIBE_CARDS_COLLECTION,
                point_id=segment_id,
                vector=vibe_card_embedding,
                payload={
                    **base_payload,
                    "vibe_card": vibe_card,
                },
            )
        except Exception as e:
            print(f"Error embedding and storing segment {segment_path}: {e}")
            traceback.print_exc()


async def main(
    db_url: str,
    all_video_path: Path,
    dataset_root: Path,
    limit: Optional[int] = None,
    max_concurrent: int = 10,
    enable_filter: bool = False,
    max_duration_seconds: float = 30.0,
    max_size_mb: float = 50.0,
) -> None:
    if not all_video_path.exists():
        raise FileNotFoundError(f"videos dir not found: {all_video_path}")
    # Create directory if it's a local path (not a URL)
    if not db_url.startswith("http://") and not db_url.startswith("https://"):
        Path(db_url).mkdir(parents=True, exist_ok=True)
    store = QdrantStore(db_url=db_url)
    store.ensure_collections([VIDEO_SEGMENTS_COLLECTION, VIDEO_VIBE_CARDS_COLLECTION])

    embed_model = Nova2OmniEmbeddings()
    vision_model = Nova2LiteModel()
    video_id_to_path = collect_video_segments(all_video_path)

    # Filter segments by size and duration (if enabled)
    segment_ids = list(video_id_to_path.keys())
    filtered_segment_ids = []
    skipped_count = 0
    
    if enable_filter:
        for segment_id in tqdm(segment_ids, desc="Filtering segments"):
            segment_path = video_id_to_path[segment_id]
            duration = get_video_duration(segment_path)
            file_size_mb = get_file_size_mb(segment_path)
            
            if duration is not None and duration > max_duration_seconds:
                print(f"Skipping {segment_path} because of size limit (duration: {duration:.2f}s > {max_duration_seconds}s)")
                skipped_count += 1
                continue
            
            if file_size_mb > max_size_mb:
                print(f"Skipping {segment_path} because of size limit (size: {file_size_mb:.2f}MB > {max_size_mb}MB)")
                skipped_count += 1
                continue
            
            filtered_segment_ids.append(segment_id)
    else:
        filtered_segment_ids = segment_ids
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} segments due to size limits")
    
    if limit is not None:
        filtered_segment_ids = filtered_segment_ids[:limit]
        print(
            f"Limiting processing to first {limit} segments (out of {len(filtered_segment_ids)} after filtering)"
        )
    else:
        print(f"Processing {len(filtered_segment_ids)} segments (after filtering {skipped_count} segments)")

    try:
        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        
        # Create a progress bar that we can update dynamically
        pbar = tqdm(
            total=len(filtered_segment_ids),
            desc="Processing video segments",
            unit="segment"
        )
        
        tasks = [
            asyncio.create_task(
                embed_and_store_segment(
                    video_id_to_path,
                    segment_id,
                    embed_model,
                    vision_model,
                    store,
                    dataset_root,
                    semaphore,
                )
            )
            for segment_id in filtered_segment_ids
        ]
        
        # Update progress bar as tasks complete
        for future in asyncio.as_completed(tasks):
            await future
            # Update progress bar with current total cost
            total_cost = vision_model.costs["total_cost"] + embed_model.costs["total_cost"]
            pbar.set_postfix_str(f"Cost: ${total_cost:.6f}")
            pbar.update(1)
        
        pbar.close()

        total_cost = vision_model.costs["total_cost"] + embed_model.costs["total_cost"]
        print(f"\nTotal cost: ${total_cost:.6f}")
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed video segments and store them in a Qdrant database"
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Dataset root directory (db_path defaults to dataset_root/db, all_video_path defaults to dataset_root/videos)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="http://localhost:6333",
        help="URL to remote Qdrant server (e.g., http://localhost:6333) or path to local Qdrant database directory (e.g., /path/to/db). Defaults to http://localhost:6333.",
    )
    parser.add_argument(
        "--all-video-path",
        type=Path,
        default=None,
        help="Path to the directory containing video segments (defaults to dataset_root/videos)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of segments to process (e.g., --limit 5)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum number of concurrent embedding tasks (default: 10)",
    )
    parser.add_argument(
        "--enable-filter",
        action="store_true",
        help="Enable filtering by duration and file size (default: disabled)",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds (default: 30.0, only used with --enable-filter)",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=50.0,
        help="Maximum segment file size in MB (default: 50.0, only used with --enable-filter)",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root
    db_url = args.db_url
    all_video_path = args.all_video_path or (dataset_root / "videos")
    asyncio.run(
        main(
            db_url=db_url,
            all_video_path=all_video_path,
            dataset_root=dataset_root,
            limit=args.limit,
            max_concurrent=args.max_concurrent,
            enable_filter=args.enable_filter,
            max_duration_seconds=args.max_duration_seconds,
            max_size_mb=args.max_size_mb,
        )
    )
