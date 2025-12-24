import argparse
import asyncio
import os
import traceback
import uuid
from pathlib import Path
from typing import Optional

from nova_embedding_model import Nova2OmniEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm.asyncio import tqdm

from nova2_lite_model import Nova2LiteModel


def collect_video_segments(all_video_path: Path) -> dict[str, Path]:
    video_id_to_path = {}
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
    db_client: QdrantClient,
):
    try:
        segment_path = video_id_to_path[segment_id]
        result = await embed_model.embed_video(
            segment_path,
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

        vibe_card = vision_model.generate_video_vibe_card(str(segment_path))
        vibe_card_embedding = await embed_model.embed_text(
            vibe_card, embedding_purpose="GENERIC_INDEX", truncation_mode="END"
        )
        vibe_card_embedding = vibe_card_embedding["embeddings"][0]["embedding"]

        db_client.upsert(
            collection_name="segment_video_embeddings",
            points=[
                PointStruct(
                    id=segment_id,
                    vector=video_embedding,
                    payload={"segment_path": str(segment_path)},
                ),
            ],
        )
        db_client.upsert(
            collection_name="segment_vibe_card_embeddings",
            points=[
                PointStruct(
                    id=segment_id,
                    vector=vibe_card_embedding,
                    payload={"segment_path": str(segment_path), "vibe_card": vibe_card},
                ),
            ],
        )
    except Exception as e:
        print(f"Error embedding and storing segment {segment_path}: {e}")
        traceback.print_exc()


async def main(db_path: Path, all_video_path: Path, limit: Optional[int] = None):
    db_client = QdrantClient(path=str(db_path))
    # Create collections if they don't already exist
    for collection_name in ["segment_video_embeddings", "segment_vibe_card_embeddings"]:
        try:
            db_client.get_collection(collection_name)
            print(f"Collection '{collection_name}' already exists, skipping creation.")
        except Exception:
            # Collection doesn't exist, create it
            db_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
            )
            print(f"Created collection '{collection_name}'.")

    embed_model = Nova2OmniEmbeddings()
    vision_model = Nova2LiteModel()
    video_id_to_path = collect_video_segments(all_video_path)
    
    # Limit the number of segments if specified
    segment_ids = list(video_id_to_path.keys())
    if limit is not None:
        segment_ids = segment_ids[:limit]
        print(f"Limiting processing to first {limit} segments (out of {len(video_id_to_path)} total)")
    
    tasks = [
        embed_and_store_segment(
            video_id_to_path, segment_id, embed_model, vision_model, db_client
        )
        for segment_id in segment_ids
    ]
    await tqdm.gather(
        *tasks, desc="Processing video segments", total=len(segment_ids)
    )
    
    # Get costs from models and print total
    total_cost = vision_model.costs["total_cost"] + embed_model.costs["total_cost"]
    print(f"\nTotal cost: ${total_cost:.6f}")


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
        "--limit",
        type=int,
        default=None,
        help="Limit the number of segments to process (e.g., --limit 5 to process only the first 5 segments)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.db_path, args.all_video_path, args.limit))
