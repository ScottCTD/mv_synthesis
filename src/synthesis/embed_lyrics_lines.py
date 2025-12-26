from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm

from synthesis.config import (
    DEFAULT_DATASET_ROOT,
    LYRICS_AUDIO_COLLECTION,
    LYRICS_AUGMENTED_QUERY_COLLECTION,
    LYRICS_TEXT_AUGMENTED_QUERY_COLLECTION,
    LYRICS_TEXT_COLLECTION,
)
from synthesis.db import QdrantStore
from synthesis.lyrics_io import build_song_from_clips, resolve_path, write_song
from synthesis.nova2_lite_model import Nova2LiteModel
from synthesis.nova_embedding_model import Nova2OmniEmbeddings


def build_line_id(song_name: str, lyric_text: str, occurrence: int) -> str:
    normalized = lyric_text.strip()
    key = f"{song_name}-{normalized}-{occurrence}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


async def embed_and_store_line(
    song_name: str,
    line,
    occurrence: int,
    augmented_query: str,
    embed_model: Nova2OmniEmbeddings,
    store: QdrantStore,
    dataset_root: Path,
    semaphore: asyncio.Semaphore,
    embed_audio: bool,
) -> None:
    async with semaphore:
        line.augmented_query = augmented_query
        line_id = build_line_id(song_name, line.text, occurrence)
        line.embedding_id = line_id

        audio_path = resolve_path(line.audio_path, dataset_root)
        text_embedding = await embed_model.embed_text(
            line.text, embedding_purpose="VIDEO_RETRIEVAL"
        )
        aug_embedding = await embed_model.embed_text(
            line.augmented_query, embedding_purpose="VIDEO_RETRIEVAL"
        )
        combined_query = f"{line.text}\n{line.augmented_query}"
        combined_embedding = await embed_model.embed_text(
            combined_query, embedding_purpose="VIDEO_RETRIEVAL"
        )
        audio_vector = None
        if embed_audio:
            audio_result = await embed_model.embed_audio(
                str(audio_path), embedding_purpose="AUDIO_RETRIEVAL"
            )
            audio_vector = audio_result["embeddings"][0]["embedding"]

        payload = {
            "song_name": song_name,
            "line_index": line.index,
            "start": line.start,
            "end": line.end,
            "duration": line.duration,
            "text": line.text,
            "augmented_query": line.augmented_query,
            "text_path": str(line.text_path),
            "audio_path": str(line.audio_path),
            "embedding_id": line.embedding_id,
        }

        store.upsert_vector(
            collection_name=LYRICS_TEXT_COLLECTION,
            point_id=line_id,
            vector=text_embedding["embeddings"][0]["embedding"],
            payload=payload,
        )
        store.upsert_vector(
            collection_name=LYRICS_AUGMENTED_QUERY_COLLECTION,
            point_id=line_id,
            vector=aug_embedding["embeddings"][0]["embedding"],
            payload=payload,
        )
        store.upsert_vector(
            collection_name=LYRICS_TEXT_AUGMENTED_QUERY_COLLECTION,
            point_id=line_id,
            vector=combined_embedding["embeddings"][0]["embedding"],
            payload=payload,
        )
        if embed_audio:
            if audio_vector is None:
                raise RuntimeError(f"Missing audio embedding for {audio_path}")
            store.upsert_vector(
                collection_name=LYRICS_AUDIO_COLLECTION,
                point_id=line_id,
                vector=audio_vector,
                payload=payload,
            )


async def main(
    dataset_root: Path,
    song_name: str,
    db_path: Path,
    concurrency: int,
    limit: Optional[int],
    skip_rewrite: bool,
    embed_audio: bool,
) -> None:
    song_dir = dataset_root / "songs" / song_name
    lyrics_dir = song_dir / "clips_and_lyrics"
    if not lyrics_dir.exists():
        raise FileNotFoundError(f"lyrics dir not found: {lyrics_dir}")

    song = build_song_from_clips(
        song_name, lyrics_dir, dataset_root, include_empty=True
    )
    if not song.lyrics_lines:
        raise ValueError(f"No lyric lines found in {lyrics_dir}")
    if limit is not None:
        song.lyrics_lines = song.lyrics_lines[:limit]

    occurrences: list[int] = []
    seen: dict[str, int] = {}
    for line in song.lyrics_lines:
        key = line.text.strip()
        seen[key] = seen.get(key, 0) + 1
        occurrences.append(seen[key])

    if skip_rewrite:
        augmented_queries = [line.text for line in song.lyrics_lines]
        rewrite_cost = 0.0
    else:
        rewrite_model = Nova2LiteModel()
        augmented_queries = await rewrite_model.generate_augmented_queries_for_lines(
            [line.text for line in song.lyrics_lines]
        )
        rewrite_cost = rewrite_model.costs["total_cost"]

    if len(augmented_queries) != len(song.lyrics_lines):
        raise ValueError(
            "Augmented query count mismatch: "
            f"{len(augmented_queries)} vs {len(song.lyrics_lines)}"
        )

    db_path.mkdir(parents=True, exist_ok=True)
    embed_model = Nova2OmniEmbeddings()
    store = QdrantStore(db_path)
    collections = [
        LYRICS_TEXT_COLLECTION,
        LYRICS_AUGMENTED_QUERY_COLLECTION,
        LYRICS_TEXT_AUGMENTED_QUERY_COLLECTION,
    ]
    if embed_audio:
        collections.append(LYRICS_AUDIO_COLLECTION)
    store.ensure_collections(collections)

    try:
        semaphore = asyncio.Semaphore(max(1, concurrency))
        tasks = [
            embed_and_store_line(
                song_name,
                line,
                occurrence,
                augmented_query,
                embed_model,
                store,
                dataset_root,
                semaphore,
                embed_audio,
            )
            for line, occurrence, augmented_query in zip(
                song.lyrics_lines, occurrences, augmented_queries
            )
        ]
        await tqdm.gather(*tasks, desc="Processing lyrics lines", total=len(tasks))

        lyrics_json_path = song_dir / "lyrics_lines.json"
        write_song(song, lyrics_json_path)

        total_cost = rewrite_cost + embed_model.costs["total_cost"]
        print(f"\nTotal cost: ${total_cost:.6f}")
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed lyrics lines and store them in a Qdrant database"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root containing songs/ and db/ directories.",
    )
    parser.add_argument(
        "--song-name",
        required=True,
        help="Song name under the dataset songs directory.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to the Qdrant database directory (defaults to dataset_root/db).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=96,
        help="Concurrent embedding requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of lyric lines to process.",
    )
    parser.add_argument(
        "--skip-rewrite",
        action="store_true",
        help="Skip query rewrite and use raw lyrics as augmented queries.",
    )
    parser.add_argument(
        "--embed-audio",
        action="store_true",
        help="Embed lyric audio clips into the lyrics-audio collection.",
    )
    args = parser.parse_args()

    db_path = args.db_path or (args.dataset_root / "db")
    asyncio.run(
        main(
            dataset_root=args.dataset_root,
            song_name=args.song_name,
            db_path=db_path,
            concurrency=args.concurrency,
            limit=args.limit,
            skip_rewrite=args.skip_rewrite,
            embed_audio=args.embed_audio,
        )
    )
