from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm

from synthesis.config import (
    DEFAULT_DATASET_ROOT,
    LYRICS_COLLECTIONS,
    PROJECT_ROOT,
)
from synthesis.db import QdrantStore
from synthesis.lyrics_io import (
    build_song_from_clips,
    load_song,
    resolve_path,
    write_song,
)
from synthesis.nova2_lite_model import Nova2LiteModel
from synthesis.nova_embedding_model import Nova2OmniEmbeddings


def build_line_id(song_name: str, lyric_text: str, occurrence: int) -> str:
    normalized = lyric_text.strip()
    key = f"{song_name}-{normalized}-{occurrence}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


TEXT_EMBED_PURPOSES = {
    "text_video": "VIDEO_RETRIEVAL",
    "text_text": "TEXT_RETRIEVAL",
}
AUDIO_EMBED_PURPOSES = {
    "audio_video": "VIDEO_RETRIEVAL",
    "audio_text": "TEXT_RETRIEVAL",
}
TEXT_QUERY_SOURCES = ("text", "augment", "combined")


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
    overwrite_embeddings: bool = False,
) -> None:
    async with semaphore:
        line.augmented_query = augmented_query
        line_id = build_line_id(song_name, line.text, occurrence)
        line.embedding_id = line_id

        audio_path = resolve_path(line.audio_path, dataset_root)
        
        text_queries = {
            "text": line.text,
            "augment": line.augmented_query,
            "combined": f"{line.text}\n{line.augmented_query}",
        }

        # Check which collections already have this point (skip if overwrite_embeddings is True)
        if overwrite_embeddings:
            text_exists = {
                source: {purpose: False for purpose in TEXT_EMBED_PURPOSES}
                for source in TEXT_QUERY_SOURCES
            }
            audio_exists = {purpose: False for purpose in AUDIO_EMBED_PURPOSES}
        else:
            text_exists = {}
            for source in TEXT_QUERY_SOURCES:
                text_exists[source] = {}
                for purpose in TEXT_EMBED_PURPOSES:
                    collection_name = LYRICS_COLLECTIONS[source][purpose]
                    text_exists[source][purpose] = store.point_exists(
                        collection_name, line_id
                    )
            if embed_audio:
                audio_exists = {
                    purpose: store.point_exists(
                        LYRICS_COLLECTIONS["audio"][purpose], line_id
                    )
                    for purpose in AUDIO_EMBED_PURPOSES
                }
            else:
                audio_exists = {purpose: True for purpose in AUDIO_EMBED_PURPOSES}
        
        # Skip if all required collections already have this point
        if (
            all(all(purpose_exists.values()) for purpose_exists in text_exists.values())
            and all(audio_exists.values())
        ):
            return

        # Only embed what's missing (or everything if overwrite_embeddings is True)
        text_embeddings: dict[tuple[str, str], list[float]] = {}
        for source, query_text in text_queries.items():
            for purpose_key, embedding_purpose in TEXT_EMBED_PURPOSES.items():
                if text_exists[source][purpose_key] and not overwrite_embeddings:
                    continue
                result = await embed_model.embed_text(
                    query_text, embedding_purpose=embedding_purpose
                )
                text_embeddings[(source, purpose_key)] = result["embeddings"][0]["embedding"]
        
        audio_embeddings: dict[str, list[float]] = {}
        if embed_audio:
            for purpose_key, embedding_purpose in AUDIO_EMBED_PURPOSES.items():
                if audio_exists[purpose_key] and not overwrite_embeddings:
                    continue
                audio_result = await embed_model.embed_audio(
                    str(audio_path), embedding_purpose=embedding_purpose
                )
                audio_embeddings[purpose_key] = audio_result["embeddings"][0]["embedding"]

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

        for source in TEXT_QUERY_SOURCES:
            for purpose_key in TEXT_EMBED_PURPOSES:
                if text_exists[source][purpose_key] and not overwrite_embeddings:
                    continue
                vector = text_embeddings.get((source, purpose_key))
                if vector is None:
                    raise RuntimeError(
                        f"Missing {purpose_key} embedding for {source} line {line_id}"
                    )
                store.upsert_vector(
                    collection_name=LYRICS_COLLECTIONS[source][purpose_key],
                    point_id=line_id,
                    vector=vector,
                    payload=payload,
                )
        
        if embed_audio:
            for purpose_key in AUDIO_EMBED_PURPOSES:
                if audio_exists[purpose_key] and not overwrite_embeddings:
                    continue
                vector = audio_embeddings.get(purpose_key)
                if vector is None:
                    raise RuntimeError(
                        f"Missing {purpose_key} embedding for {audio_path}"
                    )
                store.upsert_vector(
                    collection_name=LYRICS_COLLECTIONS["audio"][purpose_key],
                    point_id=line_id,
                    vector=vector,
                    payload=payload,
                )


async def main(
    dataset_root: Path,
    song_name: str,
    db_url: str,
    concurrency: int = 32,
    limit: Optional[int] = None,
    skip_rewrite: bool = False,
    embed_audio: bool = True,
    overwrite_json: bool = False,
    overwrite_embeddings: bool = False,
    json_only: bool = False,
) -> None:
    song_dir = dataset_root / "songs" / song_name
    lyrics_dir = song_dir / "clips_and_lyrics"
    if not lyrics_dir.exists():
        raise FileNotFoundError(
            f"lyrics dir not found: {lyrics_dir}\n"
            f"Dataset root: {dataset_root}\n"
            f"Song dir: {song_dir}\n"
            f"Make sure the path is correct and the directory exists."
        )

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

    lyrics_json_path = song_dir / "lyrics_lines.json"
    
    # Try to load existing augmented queries from JSON (skip if overwrite_json is True)
    existing_augmented_queries = None
    if not overwrite_json and lyrics_json_path.exists():
        try:
            existing_song = load_song(lyrics_json_path, name_fallback=song_name)
            if len(existing_song.lyrics_lines) == len(song.lyrics_lines):
                # Check if all lines have augmented queries
                if all(
                    line.augmented_query
                    for line in existing_song.lyrics_lines
                    if line.text.strip()
                ):
                    existing_augmented_queries = [
                        line.augmented_query for line in existing_song.lyrics_lines
                    ]
                    print(f"Loaded existing augmented queries from {lyrics_json_path}")
        except Exception as e:
            print(f"Warning: Could not load existing augmented queries: {e}")

    if skip_rewrite:
        if existing_augmented_queries:
            augmented_queries = existing_augmented_queries
        else:
            augmented_queries = [line.text for line in song.lyrics_lines]
        rewrite_cost = 0.0
    else:
        if existing_augmented_queries and not overwrite_json:
            print("Using existing augmented queries from JSON (use --overwrite-json to regenerate)")
            augmented_queries = existing_augmented_queries
            rewrite_cost = 0.0
        else:
            rewrite_model = Nova2LiteModel()
            augmented_queries = await rewrite_model.generate_augmented_queries(
                [line.text for line in song.lyrics_lines]
            )
            rewrite_cost = rewrite_model.costs["total_cost"]

    if len(augmented_queries) != len(song.lyrics_lines):
        raise ValueError(
            "Augmented query count mismatch: "
            f"{len(augmented_queries)} vs {len(song.lyrics_lines)}"
        )

    # Assign augmented queries to lines and generate embedding IDs
    for line, occurrence, augmented_query in zip(song.lyrics_lines, occurrences, augmented_queries):
        line.augmented_query = augmented_query
        line.embedding_id = build_line_id(song_name, line.text, occurrence)
    
    write_song(song, lyrics_json_path)
    print(f"Augmented queries written to {lyrics_json_path}")

    # If json_only is True, skip all embedding and database operations
    if json_only:
        print(f"\nTotal cost: ${rewrite_cost:.6f}")
        return

    # Create directory if it's a local path (not a URL)
    if not db_url.startswith("http://") and not db_url.startswith("https://"):
        Path(db_url).mkdir(parents=True, exist_ok=True)
    embed_model = Nova2OmniEmbeddings()
    store = QdrantStore(db_url=db_url)
    collections = []
    for source in TEXT_QUERY_SOURCES:
        for purpose_key in TEXT_EMBED_PURPOSES:
            collections.append(LYRICS_COLLECTIONS[source][purpose_key])
    if embed_audio:
        for purpose_key in AUDIO_EMBED_PURPOSES:
            collections.append(LYRICS_COLLECTIONS["audio"][purpose_key])
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
                overwrite_embeddings,
            )
            for line, occurrence, augmented_query in zip(
                song.lyrics_lines, occurrences, augmented_queries
            )
        ]
        await tqdm.gather(*tasks, desc="Processing lyrics lines", total=len(tasks))

        # Update JSON file with embedding IDs after embedding completes
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
        "dataset_root",
        type=Path,
        help="Dataset root containing songs/ and db/ directories.",
    )
    parser.add_argument(
        "song_name",
        help="Song name under the dataset songs directory.",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default="http://localhost:6333",
        help="URL to remote Qdrant server (e.g., http://localhost:6333) or path to local Qdrant database directory (e.g., /path/to/db). Defaults to http://localhost:6333.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
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
        "--no-embed-audio",
        action="store_false",
        dest="embed_audio",
        help="Skip embedding lyric audio clips into the lyrics-audio collection.",
    )
    parser.add_argument(
        "--overwrite-json",
        action="store_true",
        help="Regenerate augmented queries and overwrite the JSON file, ignoring existing JSON file.",
    )
    parser.add_argument(
        "--overwrite-embeddings",
        action="store_true",
        help="Overwrite existing embeddings in the database, regenerating all embeddings even if they already exist.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only generate the JSON file with augmented queries, skip all embedding and database operations.",
    )
    args = parser.parse_args()
    
    # Resolve dataset_root: if relative, check if it's in datasets/ first
    dataset_root = args.dataset_root
    if not dataset_root.is_absolute():
        # First try datasets/{name}
        datasets_path = PROJECT_ROOT / "datasets" / dataset_root
        if datasets_path.exists():
            dataset_root = datasets_path
        else:
            # Otherwise resolve relative to PROJECT_ROOT
            dataset_root = PROJECT_ROOT / dataset_root
    dataset_root = dataset_root.resolve()
    
    db_url = args.db_url
    asyncio.run(
        main(
            dataset_root=dataset_root,
            song_name=args.song_name,
            db_url=db_url,
            concurrency=args.concurrency,
            limit=args.limit,
            skip_rewrite=args.skip_rewrite,
            embed_audio=args.embed_audio,
            overwrite_json=args.overwrite_json,
            overwrite_embeddings=args.overwrite_embeddings,
            json_only=args.json_only,
        )
    )
