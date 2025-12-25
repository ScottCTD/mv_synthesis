from __future__ import annotations

import argparse
import asyncio
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm

try:
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
except ImportError:
    from config import (
        DEFAULT_DATASET_ROOT,
        VIDEO_SEGMENTS_COLLECTION,
        VIDEO_VIBE_CARDS_COLLECTION,
    )
    from db import QdrantStore
    from ffmpeg_utils import get_video_duration
    from lyrics_io import path_to_relative
    from nova2_lite_model import Nova2LiteModel
    from nova_embedding_model import Nova2OmniEmbeddings


def collect_video_segments(all_video_path: Path) -> dict[str, Path]:
    video_id_to_path: dict[str, Path] = {}
    videos = os.listdir(all_video_path)
    for video in videos:
        if "." in video:
            continue
        for segment in os.listdir(all_video_path / video):
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
) -> None:
    segment_path = video_id_to_path[segment_id]
    if store.point_exists(VIDEO_SEGMENTS_COLLECTION, segment_id):
        print(f"Skipping for segment path = {segment_path} because exists")
        return
    try:
        segment_duration = get_video_duration(segment_path)
        assert segment_duration is not None, (
            f"Segment duration not found for segment {segment_path}"
        )
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
        assert video_embedding is not None, (
            f"Video embedding not found for segment {segment_path}"
        )

        vibe_card = await vision_model.generate_vibe_card(str(segment_path))
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
    db_path: Path,
    all_video_path: Path,
    dataset_root: Path,
    limit: Optional[int] = None,
) -> None:
    if not all_video_path.exists():
        raise FileNotFoundError(f"videos dir not found: {all_video_path}")
    db_path.mkdir(parents=True, exist_ok=True)
    store = QdrantStore(db_path)
    store.ensure_collections(
        [VIDEO_SEGMENTS_COLLECTION, VIDEO_VIBE_CARDS_COLLECTION]
    )

    embed_model = Nova2OmniEmbeddings()
    vision_model = Nova2LiteModel()
    video_id_to_path = collect_video_segments(all_video_path)

    segment_ids = list(video_id_to_path.keys())
    if limit is not None:
        segment_ids = segment_ids[:limit]
        print(
            f"Limiting processing to first {limit} segments (out of {len(video_id_to_path)} total)"
        )

    try:
        tasks = [
            embed_and_store_segment(
                video_id_to_path,
                segment_id,
                embed_model,
                vision_model,
                store,
                dataset_root,
            )
            for segment_id in segment_ids
        ]
        await tqdm.gather(
            *tasks, desc="Processing video segments", total=len(segment_ids)
        )

        total_cost = vision_model.costs["total_cost"] + embed_model.costs["total_cost"]
        print(f"\nTotal cost: ${total_cost:.6f}")
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed video segments and store them in a Qdrant database"
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the Qdrant database directory",
    )
    parser.add_argument(
        "all_video_path",
        type=Path,
        help="Path to the directory containing video segments",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root for storing relative paths (defaults to parent of videos dir).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of segments to process (e.g., --limit 5)",
    )
    args = parser.parse_args()
    dataset_root = args.dataset_root or args.all_video_path.parent or DEFAULT_DATASET_ROOT
    asyncio.run(
        main(args.db_path, args.all_video_path, dataset_root, args.limit)
    )
