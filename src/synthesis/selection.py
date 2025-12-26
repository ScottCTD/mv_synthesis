from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from synthesis.models import Candidate
from synthesis.retrieval import RetrievalResults


@dataclass
class SelectionResult:
    candidate: Optional[Candidate]
    strategy: str


def select_top_video(video_candidates: list[Candidate]) -> Optional[Candidate]:
    if not video_candidates:
        return None
    return max(
        video_candidates,
        key=lambda cand: cand.video_score if cand.video_score is not None else float("-inf"),
    )


def select_top_video_duration(
    video_candidates: list[Candidate],
    target_duration: float,
) -> Optional[Candidate]:
    if not video_candidates:
        return None
    if target_duration <= 0:
        return select_top_video(video_candidates)
    with_duration = [cand for cand in video_candidates if cand.duration is not None]
    longer_or_equal = [
        cand for cand in with_duration if cand.duration >= target_duration
    ]
    if longer_or_equal:
        return max(
            longer_or_equal,
            key=lambda cand: cand.video_score
            if cand.video_score is not None
            else float("-inf"),
        )
    if with_duration:
        return max(
            with_duration,
            key=lambda cand: cand.duration
            if cand.duration is not None
            else float("-inf"),
        )
    return select_top_video(video_candidates)


def select_top_vibe(vibe_candidates: list[Candidate]) -> Optional[Candidate]:
    if not vibe_candidates:
        return None
    return max(
        vibe_candidates,
        key=lambda cand: cand.vibe_card_score
        if cand.vibe_card_score is not None
        else float("-inf"),
    )


def select_top_vibe_duration(
    vibe_candidates: list[Candidate],
    target_duration: float,
) -> Optional[Candidate]:
    if not vibe_candidates:
        return None
    if target_duration <= 0:
        return select_top_vibe(vibe_candidates)
    with_duration = [cand for cand in vibe_candidates if cand.duration is not None]
    longer_or_equal = [
        cand for cand in with_duration if cand.duration >= target_duration
    ]
    if longer_or_equal:
        return max(
            longer_or_equal,
            key=lambda cand: cand.vibe_card_score
            if cand.vibe_card_score is not None
            else float("-inf"),
        )
    if with_duration:
        return max(
            with_duration,
            key=lambda cand: cand.duration
            if cand.duration is not None
            else float("-inf"),
        )
    return select_top_vibe(vibe_candidates)


def select_intersection(
    video_candidates: list[Candidate],
    vibe_candidates: list[Candidate],
) -> Optional[Candidate]:
    if not video_candidates:
        return None
    vibe_ids = {cand.segment_id for cand in vibe_candidates if cand.segment_id}
    for cand in video_candidates:
        if cand.segment_id and cand.segment_id in vibe_ids:
            return cand
    return video_candidates[0]


def select_candidate(
    results: RetrievalResults,
    strategy: str,
    target_duration: float,
) -> SelectionResult:
    if strategy == "top_vibe":
        candidate = select_top_vibe(results.vibe_candidates)
    elif strategy == "top_vibe_duration":
        candidate = select_top_vibe_duration(results.vibe_candidates, target_duration)
    elif strategy == "top_video_duration":
        candidate = select_top_video_duration(results.video_candidates, target_duration)
    elif strategy == "top_video":
        candidate = select_top_video(results.video_candidates)
    elif strategy == "intersection":
        candidate = select_intersection(results.video_candidates, results.vibe_candidates)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    return SelectionResult(candidate=candidate, strategy=strategy)
