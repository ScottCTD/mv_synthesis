#!/usr/bin/env python3
"""
Validation script to check if all video segments have corresponding vibe cards
in a Qdrant database.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from synthesis.config import VIDEO_SEGMENTS_COLLECTION, VIDEO_VIBE_CARDS_COLLECTION
    from synthesis.db import QdrantStore
except ImportError:
    from config import VIDEO_SEGMENTS_COLLECTION, VIDEO_VIBE_CARDS_COLLECTION
    from db import QdrantStore


def get_all_segment_ids(store: QdrantStore) -> set[str]:
    """Get all segment IDs from the video-segments collection."""
    segment_ids = set()
    offset = None
    
    while True:
        result = store.client.scroll(
            collection_name=VIDEO_SEGMENTS_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        
        points, next_offset = result
        
        for point in points:
            segment_ids.add(str(point.id))
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    return segment_ids


def get_all_vibe_card_ids(store: QdrantStore) -> set[str]:
    """Get all segment IDs from the video-vibe_cards collection."""
    vibe_card_ids = set()
    offset = None
    
    while True:
        result = store.client.scroll(
            collection_name=VIDEO_VIBE_CARDS_COLLECTION,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        
        points, next_offset = result
        
        for point in points:
            vibe_card_ids.add(str(point.id))
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    return vibe_card_ids


def get_segment_path(store: QdrantStore, segment_id: str) -> str:
    """Get the segment path for a given segment ID."""
    points = store.client.retrieve(
        collection_name=VIDEO_SEGMENTS_COLLECTION,
        ids=[segment_id],
        with_vectors=False,
        with_payload=True,
    )
    if points and points[0].payload:
        return points[0].payload.get("segment_path", "N/A")
    return "N/A"


def validate_db(db_path: Path) -> bool:
    """Validate that all video segments have corresponding vibe cards.
    
    Returns:
        True if validation passes, False otherwise.
    """
    if not db_path.exists():
        print(f"Error: Database path does not exist: {db_path}", file=sys.stderr)
        return False
    
    print(f"Connecting to Qdrant database at: {db_path}")
    store = QdrantStore(db_path)
    
    try:
        # Check if collections exist
        try:
            collection_info = store.client.get_collection(VIDEO_SEGMENTS_COLLECTION)
            video_segments_count = collection_info.points_count
            print(f"Found {video_segments_count} video segments")
        except Exception as e:
            print(f"Error: Collection '{VIDEO_SEGMENTS_COLLECTION}' not found: {e}", file=sys.stderr)
            return False
        
        try:
            collection_info = store.client.get_collection(VIDEO_VIBE_CARDS_COLLECTION)
            vibe_cards_count = collection_info.points_count
            print(f"Found {vibe_cards_count} vibe cards")
        except Exception as e:
            print(f"Error: Collection '{VIDEO_VIBE_CARDS_COLLECTION}' not found: {e}", file=sys.stderr)
            return False
        
        # Get all segment IDs
        print("\nCollecting segment IDs...")
        segment_ids = get_all_segment_ids(store)
        print(f"Found {len(segment_ids)} unique segment IDs")
        
        # Get all vibe card IDs
        print("Collecting vibe card IDs...")
        vibe_card_ids = get_all_vibe_card_ids(store)
        print(f"Found {len(vibe_card_ids)} unique vibe card IDs")
        
        # Find missing vibe cards
        missing_vibe_cards = segment_ids - vibe_card_ids
        
        if missing_vibe_cards:
            print(f"\n❌ Validation FAILED: {len(missing_vibe_cards)} video segments are missing vibe cards:")
            print("\nMissing vibe cards:")
            for segment_id in sorted(missing_vibe_cards):
                segment_path = get_segment_path(store, segment_id)
                print(f"  - Segment ID: {segment_id}")
                print(f"    Path: {segment_path}")
            return False
        else:
            print(f"\n✅ Validation PASSED: All {len(segment_ids)} video segments have corresponding vibe cards")
            
            # Also check for orphaned vibe cards (vibe cards without segments)
            orphaned_vibe_cards = vibe_card_ids - segment_ids
            if orphaned_vibe_cards:
                print(f"\n⚠️  Warning: {len(orphaned_vibe_cards)} vibe cards exist without corresponding video segments")
                print("Orphaned vibe cards:")
                for segment_id in sorted(orphaned_vibe_cards):
                    print(f"  - Segment ID: {segment_id}")
            
            return True
    
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that all video segments have corresponding vibe cards in a Qdrant database"
    )
    parser.add_argument(
        "db_path",
        type=Path,
        help="Path to the Qdrant database directory (e.g., datasets/ds1/db)",
    )
    args = parser.parse_args()
    
    success = validate_db(args.db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

