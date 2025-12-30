from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from synthesis.config import EMBEDDING_DIM


class QdrantStore:
    def __init__(self, db_url: str):
        """
        Initialize QdrantStore with either a local path or a remote URL.
        
        Args:
            db_url: URL to remote Qdrant server (e.g., "http://localhost:6333") 
                    or path to local Qdrant database directory (e.g., "/path/to/db").
                    Automatically detects whether it's a URL or a path.
        """
        # Check if it's a URL (starts with http:// or https://)
        if db_url.startswith("http://") or db_url.startswith("https://"):
            self.client = QdrantClient(url=db_url)
        else:
            # Treat as local path
            self.client = QdrantClient(path=db_url)

    def close(self) -> None:
        self.client.close()

    def ensure_collection(
        self,
        collection_name: str,
        size: int = EMBEDDING_DIM,
        distance: Distance = Distance.COSINE,
    ) -> None:
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=size, distance=distance),
            )

    def ensure_collections(self, collection_names: Iterable[str]) -> None:
        for name in collection_names:
            self.ensure_collection(name)

    def upsert_vector(
        self,
        collection_name: str,
        point_id: str,
        vector: list[float],
        payload: Optional[dict] = None,
    ) -> None:
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload or {},
                )
            ],
        )

    def query_points(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ):
        return self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=with_payload,
        )

    def point_exists(self, collection_name: str, point_id: str) -> bool:
        """Check if a point ID exists in the specified collection."""
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=False,
            with_payload=False,
        )
        return len(points) > 0

    def retrieve_vector(self, collection_name: str, point_id: str) -> list[float]:
        points = self.client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True,
            with_payload=False,
        )
        if not points:
            raise ValueError(f"Vector not found: {collection_name} id={point_id}")
        vector = points[0].vector
        if isinstance(vector, dict):
            vector = vector.get("default")
        if vector is None:
            raise ValueError(f"Vector missing: {collection_name} id={point_id}")
        return list(vector)
