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
import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

from tqdm.asyncio import tqdm

from synthesis.config import VIDEO_VIBE_CARDS_COLLECTION
from synthesis.db import QdrantStore
from synthesis.nova2_lite_model import Nova2LiteModel

logger = logging.getLogger(__name__)


async def _process_point(
    point,
    videos_root: Path,
    model: Nova2LiteModel,
    store: QdrantStore,
    semaphore: asyncio.Semaphore,
    max_retries: int,
):
    async with semaphore:
        payload = point.payload or {}
        rel_path = payload.get("segment_path")
        if not rel_path:
            logger.warning("Skipping %s: missing segment_path", point.id)
            return None

        media_path = videos_root / rel_path
        if not media_path.exists():
            logger.warning("Skipping %s: file missing at %s",
                           point.id, media_path)
            return None

        rc_raw = payload.get("recall_card")
        should_rerun, reason = _needs_rerun(rc_raw)
        if not should_rerun:
            return point.id, media_path, rc_raw

        logger.warning(
            "Regenerating recall_card for %s (%s): %s. Existing value: %s",
            point.id,
            rel_path,
            reason,
            rc_raw,
        )

        attempts = 0
        recall_card = rc_raw
        while attempts < max_retries:
            attempts += 1
            recall_card = await model.generate_recall_card(str(media_path))
            valid, _ = _needs_rerun(recall_card)
            if not valid:
                logger.info(
                    "Recall card regenerated successfully for %s on attempt %d",
                    point.id,
                    attempts,
                )
                break
            logger.warning(
                "Regenerated recall_card still invalid for %s on attempt %d; retrying...",
                point.id,
                attempts,
            )

        # Persist the last recall card (valid or not) for traceability.
        store.client.set_payload(
            collection_name=VIDEO_VIBE_CARDS_COLLECTION,
            payload={"recall_card": recall_card},
            points=[point.id],
        )
        return point.id, media_path, recall_card


def _parse_recall_card(rc) -> Dict[str, str]:
    if rc is None:
        return {}
    if isinstance(rc, dict):
        return rc
    if isinstance(rc, str):
        try:
            parsed = json.loads(rc)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _is_valid_token(token: str) -> bool:
    """
    Accepts single keywords or short phrases (e.g., "brick wall") but rejects long sentences.
    """
    if not token:
        return False
    words = token.strip().split()
    if not 1 <= len(words) <= 3:
        return False
    return all(re.match(r"^[A-Za-z0-9_\-]+$", w) for w in words)


def _field_invalid(value: str) -> bool:
    if not isinstance(value, str):
        return True
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        return True
    return any(not _is_valid_token(t) for t in tokens)


def _needs_rerun(rc_raw) -> tuple[bool, str]:
    if rc_raw in (None, "", {}):
        return True, "recall_card missing or empty"
    rc = _parse_recall_card(rc_raw)
    noun = rc.get("Noun_Key_Words")
    verb = rc.get("Verb_Key_Words")
    if noun is None or verb is None:
        return True, "missing Noun_Key_Words or Verb_Key_Words"
    if _field_invalid(noun) or _field_invalid(verb):
        return True, "invalid Noun_Key_Words or Verb_Key_Words content"
    return False, ""


async def generate_recall_cards(
    dataset_root: Path,
    start_index: int,
    limit: Optional[int],
    max_concurrent: int,
    max_retries: int,
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
            _process_point(p, videos_root, model, store,
                           semaphore, max_retries)
            for p in collected
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
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.INFO: "\033[92m",     # green
            logging.WARNING: "\033[93m",  # yellow
            logging.ERROR: "\033[91m",    # red
        }
        RESET = "\033[0m"

        def format(self, record):
            color = self.COLORS.get(record.levelno, "")
            prefix = f"{color}[{record.levelname.lower()}]{self.RESET}"
            record.msg = f"{prefix} {record.msg}"
            return super().format(record)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ColorFormatter("%(asctime)s %(message)s"))

    from datetime import datetime

    log_dir = Path(__file__).resolve().parents[2] / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d-%H%M")
    logfile = log_dir / f"generate_recall_cards-{timestamp}.log"
    file_handler = logging.FileHandler(
        logfile, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[
                        stream_handler, file_handler])
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts to regenerate an invalid recall card (default: 3)",
    )
    args = parser.parse_args()
    asyncio.run(
        generate_recall_cards(
            args.dataset_root,
            args.start_index,
            args.limit,
            args.max_concurrent,
            args.max_retries,
        )
    )
