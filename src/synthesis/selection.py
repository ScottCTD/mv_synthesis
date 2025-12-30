from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from synthesis.models import Candidate
from synthesis.retrieval import FusedRetrievalResults, RetrievalResults


@dataclass
class SelectionResult:
    candidate: Optional[Candidate]
    strategy: str
    details: Optional[dict] = None


@dataclass(frozen=True)
class FusedSelectionConfig:
    top_k: int
    anti_repeat_window: int
    tau: float
    lambda_weight: float
    w_tv: float
    w_tvc: float
    w_av: float
    w_avc: float


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


def _rank_scores(
    candidates: list[Candidate],
    top_k: int,
) -> dict[str, float]:
    if top_k <= 0:
        return {}
    if top_k == 1:
        return {
            cand.segment_id: 1.0
            for cand in candidates[:1]
            if cand.segment_id is not None
        }
    denominator = top_k - 1
    scores: dict[str, float] = {}
    for rank, cand in enumerate(candidates[:top_k], start=1):
        if cand.segment_id is None:
            continue
        scores[cand.segment_id] = 1.0 - (rank - 1) / denominator
    return scores


def _fused_score(
    segment_id: str,
    tv_scores: dict[str, float],
    tvc_scores: dict[str, float],
    av_scores: dict[str, float],
    avc_scores: dict[str, float],
    config: FusedSelectionConfig,
) -> tuple[float, dict]:
    tv = tv_scores.get(segment_id, 0.0)
    tvc = tvc_scores.get(segment_id, 0.0)
    av = av_scores.get(segment_id, 0.0)
    avc = avc_scores.get(segment_id, 0.0)
    hits = sum(1 for score in (tv, tvc, av, avc) if score >= config.tau)
    synergy = max(0, hits - 1) ** 2
    fused = (
        config.w_tv * tv
        + config.w_tvc * tvc
        + config.w_av * av
        + config.w_avc * avc
        + config.lambda_weight * synergy
    )
    details = {
        "tv": tv,
        "tvc": tvc,
        "av": av,
        "avc": avc,
        "m": hits,
        "synergy": synergy,
        "score": fused,
    }
    return fused, details


def _select_fused_with_duration(
    candidates: list[Candidate],
    target_duration: float,
    duration_threshold: float,
    scores: dict[str, float],
) -> Optional[Candidate]:
    if not candidates:
        return None

    def score_for(candidate: Candidate) -> float:
        if candidate.segment_id is None:
            return float("-inf")
        return scores.get(candidate.segment_id, float("-inf"))

    if target_duration <= 0 or duration_threshold <= 0:
        return max(candidates, key=score_for)

    with_duration = [cand for cand in candidates if cand.duration is not None]
    if not with_duration:
        return max(candidates, key=score_for)

    min_duration = max(0.0, target_duration - duration_threshold)
    max_duration = target_duration + duration_threshold
    in_range = [
        cand
        for cand in with_duration
        if cand.duration is not None
        and min_duration <= cand.duration <= max_duration
    ]
    if in_range:
        return max(in_range, key=score_for)

    return min(
        with_duration,
        key=lambda cand: (
            abs(cand.duration - target_duration),
            -score_for(cand),
        ),
    )


def select_fused_candidate(
    results: FusedRetrievalResults,
    config: FusedSelectionConfig,
    target_duration: float,
    duration_threshold: float,
    recent_ids: Iterable[str],
) -> SelectionResult:
    candidates = [cand for cand in results.merged_candidates if cand.segment_id]
    if not candidates:
        return SelectionResult(candidate=None, strategy="fused_rank")

    tv_scores = _rank_scores(results.tv_candidates, config.top_k)
    tvc_scores = _rank_scores(results.tvc_candidates, config.top_k)
    av_scores = _rank_scores(results.av_candidates, config.top_k)
    avc_scores = _rank_scores(results.avc_candidates, config.top_k)

    fused_scores: dict[str, float] = {}
    fused_details: dict[str, dict] = {}
    for cand in candidates:
        segment_id = cand.segment_id
        if segment_id is None:
            continue
        fused, details = _fused_score(
            segment_id,
            tv_scores,
            tvc_scores,
            av_scores,
            avc_scores,
            config,
        )
        fused_scores[segment_id] = fused
        fused_details[segment_id] = details

    recent_set = set(recent_ids) if config.anti_repeat_window != 0 else set()
    filtered = [
        cand for cand in candidates if cand.segment_id and cand.segment_id not in recent_set
    ]
    used_anti_repeat = True
    if not filtered:
        filtered = candidates
        used_anti_repeat = False

    selected = _select_fused_with_duration(
        filtered,
        target_duration,
        duration_threshold,
        fused_scores,
    )
    if selected is None or selected.segment_id is None:
        return SelectionResult(candidate=None, strategy="fused_rank")

    details = fused_details.get(selected.segment_id, {})
    details["anti_repeat_applied"] = used_anti_repeat
    details["anti_repeat_window"] = config.anti_repeat_window
    details["duration_target"] = target_duration
    details["duration_threshold"] = duration_threshold
    return SelectionResult(candidate=selected, strategy="fused_rank", details=details)


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
