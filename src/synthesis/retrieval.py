from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from synthesis.config import (
    EMBEDDING_DIM,
    VIDEO_SEGMENTS_COLLECTION,
    VIDEO_VIBE_CARDS_COLLECTION,
)
from synthesis.db import QdrantStore
from synthesis.models import Candidate


@dataclass
class RetrievalResults:
    video_candidates: list[Candidate]
    vibe_candidates: list[Candidate]
    merged_candidates: list[Candidate]


@dataclass
class FusedRetrievalResults:
    tv_candidates: list[Candidate]
    tvc_candidates: list[Candidate]
    av_candidates: list[Candidate]
    avc_candidates: list[Candidate]
    merged_candidates: list[Candidate]


def _normalize_hits(hits: Iterable) -> list:
    if hasattr(hits, "points"):
        return list(hits.points)
    return list(hits)


def _random_vector(dim: int) -> list[float]:
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]


def _extract_hit(hit) -> tuple[Optional[str], dict, Optional[float]]:
    if isinstance(hit, dict):
        payload = hit.get("payload", {}) or {}
        score = hit.get("score")
        hit_id = hit.get("id")
    else:
        payload = getattr(hit, "payload", {}) or {}
        score = getattr(hit, "score", None)
        hit_id = getattr(hit, "id", None)
    segment_id = payload.get("segment_id") or (str(hit_id) if hit_id is not None else None)
    return segment_id, payload, score


def _candidate_from_hit(hit, score_field: str) -> Candidate:
    segment_id, payload, score = _extract_hit(hit)
    segment_path = payload.get("segment_path")
    duration = payload.get("duration")
    vibe_card = payload.get("vibe_card")
    candidate = Candidate(
        segment_id=segment_id,
        segment_path=Path(segment_path) if segment_path else None,
        duration=duration,
        vibe_card=vibe_card,
    )
    if score_field == "video":
        candidate.video_score = score
    else:
        candidate.vibe_card_score = score
    return candidate


def _merge_candidates_multi(*candidate_lists: list[Candidate]) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            if candidate.segment_id is None:
                continue
            existing = merged.get(candidate.segment_id)
            if existing is None:
                merged[candidate.segment_id] = candidate
                continue
            if existing.segment_path is None and candidate.segment_path is not None:
                existing.segment_path = candidate.segment_path
            if existing.duration is None and candidate.duration is not None:
                existing.duration = candidate.duration
            if existing.vibe_card is None and candidate.vibe_card is not None:
                existing.vibe_card = candidate.vibe_card
            if existing.video_score is None and candidate.video_score is not None:
                existing.video_score = candidate.video_score
            if existing.vibe_card_score is None and candidate.vibe_card_score is not None:
                existing.vibe_card_score = candidate.vibe_card_score
    return list(merged.values())


def _merge_candidates(
    video_candidates: list[Candidate],
    vibe_candidates: list[Candidate],
) -> list[Candidate]:
    return _merge_candidates_multi(video_candidates, vibe_candidates)


def query_video_candidates(
    store: QdrantStore,
    query_vector: list[float],
    top_k: int,
) -> list[Candidate]:
    hits = store.query_points(
        collection_name=VIDEO_SEGMENTS_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [_candidate_from_hit(hit, "video") for hit in _normalize_hits(hits)]


def query_vibe_candidates(
    store: QdrantStore,
    query_vector: list[float],
    top_k: int,
) -> list[Candidate]:
    hits = store.query_points(
        collection_name=VIDEO_VIBE_CARDS_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [_candidate_from_hit(hit, "vibe") for hit in _normalize_hits(hits)]


def retrieve_candidates(
    store: QdrantStore,
    query_vector: list[float],
    top_k: int,
) -> RetrievalResults:
    video_candidates = query_video_candidates(store, query_vector, top_k)
    vibe_candidates = query_vibe_candidates(store, query_vector, top_k)
    merged = _merge_candidates(video_candidates, vibe_candidates)
    return RetrievalResults(
        video_candidates=video_candidates,
        vibe_candidates=vibe_candidates,
        merged_candidates=merged,
    )


def retrieve_random_candidates(
    store: QdrantStore,
    top_k: int,
) -> RetrievalResults:
    if top_k <= 0:
        return RetrievalResults(
            video_candidates=[],
            vibe_candidates=[],
            merged_candidates=[],
        )
    query_vector = _random_vector(EMBEDDING_DIM)
    video_candidates = query_video_candidates(store, query_vector, top_k)
    vibe_candidates = query_vibe_candidates(store, query_vector, top_k)
    merged = _merge_candidates(video_candidates, vibe_candidates)
    return RetrievalResults(
        video_candidates=video_candidates,
        vibe_candidates=vibe_candidates,
        merged_candidates=merged,
    )


def retrieve_fused_candidates(
    store: QdrantStore,
    text_video_vector: list[float],
    text_vibe_vector: list[float],
    audio_video_vector: list[float],
    audio_vibe_vector: list[float],
    top_k: int,
) -> FusedRetrievalResults:
    tv_candidates = query_video_candidates(store, text_video_vector, top_k)
    tvc_candidates = query_vibe_candidates(store, text_vibe_vector, top_k)
    av_candidates = query_video_candidates(store, audio_video_vector, top_k)
    avc_candidates = query_vibe_candidates(store, audio_vibe_vector, top_k)
    merged = _merge_candidates_multi(
        tv_candidates, tvc_candidates, av_candidates, avc_candidates
    )
    return FusedRetrievalResults(
        tv_candidates=tv_candidates,
        tvc_candidates=tvc_candidates,
        av_candidates=av_candidates,
        avc_candidates=avc_candidates,
        merged_candidates=merged,
    )
