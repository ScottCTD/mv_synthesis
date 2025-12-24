#!/usr/bin/env python3
"""
Debug script to print all points in datasets/ds1/db_test Qdrant database.
"""

from pathlib import Path

from qdrant_client import QdrantClient


def print_collection_points(client: QdrantClient, collection_name: str) -> None:
    """Print all points from a collection, focusing on segment path and vibe card."""
    print(f"\n{'='*80}")
    print(f"Collection: {collection_name}")
    print(f"{'='*80}")
    
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Points count: {collection_info.points_count}\n")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return
    
    # Scroll through all points
    offset = None
    total_points = 0
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,  # Fetch 100 points at a time
            offset=offset,
            with_payload=True,
            with_vectors=False,  # Don't need vectors for this output
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        for point in points:
            total_points += 1
            payload = point.payload or {}
            segment_path = payload.get("segment_path", "N/A")
            vibe_card = payload.get("vibe_card", None)
            
            print(f"\n[Point #{total_points}]")
            print(f"Segment Path: {segment_path}")
            if vibe_card:
                print(f"Vibe Card:\n{vibe_card}")
            else:
                print("Vibe Card: N/A")
            print("-" * 80)
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    print(f"\nTotal points printed: {total_points}")
    print(f"{'='*80}\n")


def main() -> None:
    """Main function to print all points from db_test."""
    db_path = Path(__file__).resolve().parents[1] / "datasets" / "ds1" / "db_test"
    
    if not db_path.exists():
        print(f"Error: Database path does not exist: {db_path}")
        return
    
    print(f"Connecting to Qdrant database at: {db_path}")
    client = QdrantClient(path=str(db_path))
    
    # Get all collections
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        print(f"Found collections: {collection_names}\n")
    except Exception as e:
        print(f"Error getting collections: {e}")
        return
    
    # Print points from each collection
    for collection_name in collection_names:
        print_collection_points(client, collection_name)


if __name__ == "__main__":
    main()

