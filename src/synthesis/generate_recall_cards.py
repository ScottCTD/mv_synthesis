"""
Generate recall cards for segments stored in the local Qdrant DB.

Reads segment metadata from the `video-vibe_cards` collection under `datasets/ds1/db`
and runs Nova2LiteModel.generate_recall_card on each segment file (optionally limited).

Usage:
  python notebooks/generate_recall_cards.py --dataset-root datasets/ds1 --limit 5

NOTE: Each call hits Bedrock and incurs cost. Use --limit to sample safely.
"""

import argparse
import asyncio
import re
from pathlib import Path
from typing import Dict, Optional

from config import VIDEO_VIBE_CARDS_COLLECTION
from db import QdrantStore
from nova2_lite_model import Nova2LiteModel
from tqdm.asyncio import tqdm


def _parse_recall_card_from_text(raw: str) -> Dict[str, str]:
    """
    Fallback parser for recall card outputs formatted like:
    <tools>
      <__function=recall_card_extractor>
        <__parameter=Noun_Key_Words>cat, dog</__parameter>
        <__parameter=Verb_Key_Words>run, jump</__parameter>
      </__function>
    Handles minor formatting issues and returns a dict.
    """
    params: Dict[str, str] = {}
    # First try well-formed parameter blocks.
    for key, val in re.findall(
        r"<__parameter=([A-Za-z0-9_]+)>(.*?)</__parameter>", raw, flags=re.DOTALL
    ):
        params[key.strip()] = val.strip()
    if params:
        return params

    # Fallback: line-based parsing to handle missing closing tags.
    for line in raw.splitlines():
        if "<__parameter=" not in line:
            continue
        try:
            _, rest = line.split("<__parameter=", 1)
            key, remainder = rest.split(">", 1)
            key = key.strip()
            value = remainder.split("</__parameter", 1)[0].strip()
            if key and value:
                params[key] = value
        except ValueError:
            continue
    return params


async def _process_point(
    point,
    videos_root: Path,
    model: Nova2LiteModel,
    store: QdrantStore,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        payload = point.payload or {}
        rel_path = payload.get("segment_path")
        if not rel_path:
            print(f"Skipping {point.id}: missing segment_path")
            return None

        media_path = videos_root / rel_path
        if not media_path.exists():
            print(f"Skipping {point.id}: file missing at {media_path}")
            return None

        recall_card = await model.generate_recall_card(str(media_path))

        parsed_recall_card = _parse_recall_card_from_text(recall_card)
        # Persist the recall card onto the existing payload for this point.
        store.client.set_payload(
            collection_name=VIDEO_VIBE_CARDS_COLLECTION,
            payload={"recall_card": parsed_recall_card},
            points=[point.id],
        )
        return point.id, media_path, parsed_recall_card


async def generate_recall_cards(
    dataset_root: Path,
    start_index: int,
    limit: Optional[int],
    max_concurrent: int,
) -> None:
    db_path = dataset_root / "db"
    videos_root = dataset_root

    if not db_path.exists():
        raise FileNotFoundError(f"Qdrant DB not found: {db_path}")
    if not videos_root.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_root}")

    store = QdrantStore(db_path)
    model = Nova2LiteModel()

    try:
        # Collect points (iterate scroll pages until limit reached or exhausted)
        collected = []
        next_page = None
        seen = 0
        remaining = limit
        # Scroll until we've skipped start_index and filled limit (if provided)
        while True:
            page_limit = 1000
            points, next_page = store.client.scroll(
                collection_name=VIDEO_VIBE_CARDS_COLLECTION,
                limit=page_limit,
                with_vectors=False,
                with_payload=True,
                offset=next_page,
            )
            if not points:
                break

            for p in points:
                if seen < start_index:
                    seen += 1
                    continue
                collected.append(p)
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        break
            if remaining is not None and remaining <= 0:
                break
            if next_page is None:
                break

        if not collected:
            print("No points found in video-vibe_cards.")
            return

        semaphore = asyncio.Semaphore(max(1, max_concurrent))
        tasks = [
            _process_point(p, videos_root, model, store, semaphore) for p in collected
        ]

        results = await tqdm.gather(
            *tasks,
            desc="Generating recall cards",
            total=len(tasks),
        )

        for res in results:
            if res is None:
                continue
            point_id, media_path, recall_card = res
            print(f"\n=== {point_id} ===")
            print(f"file: {media_path}")
            print(recall_card)

        print(f"\nTotal Bedrock cost: ${model.costs['total_cost']:.4f}")
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate recall cards for video segments"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/ds1"),
        help="Dataset root containing db/ and videos/ (default: datasets/ds1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of points to process (default: all).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Number of points to skip before processing (default: 0).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent model calls (default: 10)",
    )
    args = parser.parse_args()
    asyncio.run(
        generate_recall_cards(
            args.dataset_root,
            args.start_index,
            args.limit,
            args.max_concurrent,
        )
    )
