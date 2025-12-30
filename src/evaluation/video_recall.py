"""
Query the video vibe-card collection with a keyword string and report matches
against stored recall cards.

Example:
  python -m evaluation.video_recall --query "cat, mouse, chase" --top-k 10
"""

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Dict, Set

from synthesis.config import VIDEO_VIBE_CARDS_COLLECTION
from synthesis.db import QdrantStore
from synthesis.nova_embedding_model import Nova2OmniEmbeddings


def _parse_recall_card(payload: dict) -> Dict[str, str]:
    rc = payload.get("recall_card")
    if rc is None:
        return {}
    if isinstance(rc, str):
        try:
            return json.loads(rc)
        except Exception:
            return {}
    if isinstance(rc, dict):
        return rc
    return {}


def _split_terms(value: str) -> Set[str]:
    return {w.strip().lower() for w in value.split(",") if w.strip()}


def _contains_any_term(query_terms: Set[str], recall_card: Dict[str, str]) -> bool:
    if not query_terms:
        return False
    noun_terms = _split_terms(recall_card.get("Noun_Key_Words", ""))
    verb_terms = _split_terms(recall_card.get("Verb_Key_Words", ""))
    rc_terms = noun_terms | verb_terms
    return bool(rc_terms & query_terms)


async def search_videos(dataset_root: Path, query_terms: Set[str], top_k: int):
    store = QdrantStore(dataset_root / "db")
    embed_model = Nova2OmniEmbeddings()
    try:
        embed_res = await embed_model.embed_text(
            ", ".join(sorted(query_terms)),
            embedding_purpose="VIDEO_RETRIEVAL"
        )
        query_vec = embed_res["embeddings"][0]["embedding"]
        results = store.query_points(
            collection_name=VIDEO_VIBE_CARDS_COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        rows = []
        matched_top_k = 0
        matched_flags = []
        for point in results.points:
            payload = point.payload or {}
            recall_card = _parse_recall_card(payload)
            matched = _contains_any_term(query_terms, recall_card)
            if matched:
                matched_top_k += 1
            matched_flags.append(bool(matched))
            rows.append({
                "id": point.id,
                "score": point.score,
                "path": payload.get("segment_path"),
                "recall_card": recall_card,
                "matched": matched,
            })
        # Count total matching videos in the collection for recall denominator.
        total_matching = 0
        next_page = None
        while True:
            points, next_page = store.client.scroll(
                collection_name=VIDEO_VIBE_CARDS_COLLECTION,
                limit=1000,
                with_vectors=False,
                with_payload=True,
                offset=next_page,
            )
            if not points:
                break
            for p in points:
                rc = _parse_recall_card(p.payload or {})
                if _contains_any_term(query_terms, rc):
                    total_matching += 1
            if next_page is None:
                break

        # Metrics: precision@k, recall@k, hit@k, nDCG@k (binary relevance)
        hit_at_k = 1.0 if matched_top_k > 0 else 0.0
        dcg = 0.0
        for idx, rel in enumerate(matched_flags):
            if rel:
                dcg += 1.0 / math.log2(idx + 2)
        ideal_rels = min(total_matching, top_k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        summary = {
            "matched_top_k": matched_top_k,
            "top_k": top_k,
            "total_matching": total_matching,
            "precision@k": matched_top_k / top_k if top_k else 0.0,
            "recall@k": (matched_top_k / total_matching) if total_matching else 0.0,
            "hit@k": hit_at_k,
            "ndcg@k": ndcg,
        }
        return rows, summary
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(
        description="Query video vibe cards and evaluate matches against recall cards."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/datasets/ds1"),
        help="Dataset root containing db/ (default: datasets/ds1)",
    )
    parser.add_argument(
        "--query",
        dest="queries",
        nargs="+",
        required=True,
        help="One or more keyword strings (space separated, e.g., --query cat mouse chase)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results to return (default: 10)",
    )
    args = parser.parse_args()

    query_terms = {q.strip().lower() for q in args.queries if q.strip()}
    rows, summary = asyncio.run(search_videos(
        args.dataset_root, query_terms, args.top_k))
    for r in rows:
        print(f"\nID: {r['id']}  score={r['score']:.4f}")
        print(f"path: {r['path']}")
        print(f"recall_card: {r['recall_card']}")
        print(f"matched={r['matched']}")

    print("\n--- Summary ---")
    print(f"Matched in top-{summary['top_k']}: {summary['matched_top_k']}")
    print(f"Total matching in corpus: {summary['total_matching']}")
    print(f"Precision@{summary['top_k']}: {summary['precision@k']:.3f}")
    print(f"Recall@{summary['top_k']}: {summary['recall@k']:.3f}")
    print(f"Hit@{summary['top_k']}: {summary['hit@k']:.3f}")
    print(f"nDCG@{summary['top_k']}: {summary['ndcg@k']:.3f}")


if __name__ == "__main__":
    main()
