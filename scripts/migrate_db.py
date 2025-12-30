from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct

SRC_PATH = "/Users/scottcui/projects/mv_synthesis/datasets/ds2/db"
DST_URL = "http://localhost:6333"
BATCH = 256

src = QdrantClient(path=SRC_PATH)      # local-mode (SQLite-backed)
dst = QdrantClient(url=DST_URL)        # server in Docker

cols = [c.name for c in src.get_collections().collections]
print("Found collections:", cols)

for name in cols:
    info = src.get_collection(name)

    # Recreate collection on server with the same vectors config
    vectors_config = info.config.params.vectors

    try:
        dst.delete_collection(name)
    except Exception:
        pass

    dst.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
    )

    # Copy points in batches
    offset = None
    total = 0
    while True:
        res = src.scroll(
            collection_name=name,
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        # scroll() can return either (points, next_offset) or an object with fields
        if isinstance(res, tuple):
            points, offset = res
        else:
            points, offset = res.points, res.next_page_offset

        if not points:
            break

        # Convert Record objects to PointStruct objects
        point_structs = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in points
        ]

        dst.upsert(collection_name=name, points=point_structs)
        total += len(points)

        if offset is None:
            break

    print(f"Migrated {name}: {total} points")

print("Done.")
