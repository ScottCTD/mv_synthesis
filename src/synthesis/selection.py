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


def _score_value(candidate: Candidate, attr: str) -> float:
    score = getattr(candidate, attr)
    return score if score is not None else float("-inf")


def _select_by_score(
    candidates: list[Candidate],
    score_attr: str,
) -> Optional[Candidate]:
    if not candidates:
        return None
    return max(candidates, key=lambda cand: _score_value(cand, score_attr))


def _select_by_duration_window(
    candidates: list[Candidate],
    target_duration: float,
    duration_threshold: float,
    score_attr: str,
) -> Optional[Candidate]:
    if not candidates:
        return None
    if target_duration <= 0 or duration_threshold <= 0:
        return _select_by_score(candidates, score_attr)
    with_duration = [cand for cand in candidates if cand.duration is not None]
    if not with_duration:
        return _select_by_score(candidates, score_attr)
    min_duration = max(0.0, target_duration - duration_threshold)
    max_duration = target_duration + duration_threshold
    in_range = [
        cand
        for cand in with_duration
        if cand.duration is not None
        and min_duration <= cand.duration <= max_duration
    ]
    if in_range:
        return max(in_range, key=lambda cand: _score_value(cand, score_attr))
    return min(
        with_duration,
        key=lambda cand: (
            abs(cand.duration - target_duration),
            -_score_value(cand, score_attr),
        ),
    )


def select_top_video_duration(
    video_candidates: list[Candidate],
    target_duration: float,
    duration_threshold: float = 0.5,
) -> Optional[Candidate]:
    return _select_by_duration_window(
        video_candidates,
        target_duration,
        duration_threshold,
        "video_score",
    )


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
    duration_threshold: float = 0.5,
) -> Optional[Candidate]:
    return _select_by_duration_window(
        vibe_candidates,
        target_duration,
        duration_threshold,
        "vibe_card_score",
    )


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
    duration_threshold: float = 0.5,
) -> SelectionResult:
    if strategy == "top_vibe":
        candidate = select_top_vibe(results.vibe_candidates)
    elif strategy == "top_vibe_duration":
        candidate = select_top_vibe_duration(
            results.vibe_candidates,
            target_duration,
            duration_threshold,
        )
    elif strategy == "top_video_duration":
        candidate = select_top_video_duration(
            results.video_candidates,
            target_duration,
            duration_threshold,
        )
    elif strategy == "top_video":
        candidate = select_top_video(results.video_candidates)
    elif strategy == "intersection":
        candidate = select_intersection(results.video_candidates, results.vibe_candidates)
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
    return SelectionResult(candidate=candidate, strategy=strategy)
