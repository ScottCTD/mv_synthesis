from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from synthesis.ffmpeg_utils import get_video_duration, render_clip
from synthesis.lyrics_io import resolve_path
from synthesis.models import Candidate


@dataclass(frozen=True)
class PostprocessPlan:
    input_path: Path
    output_path: Path
    start_offset: float
    duration: float
    pad_black: bool
    clip_duration: Optional[float]
    speed_factor: float
    note: str


def build_postprocess_plan(
    candidate: Candidate,
    target_duration: float,
    output_path: Path,
    dataset_root: Path,
) -> Optional[PostprocessPlan]:
    if candidate.segment_path is None:
        return None
    input_path = resolve_path(candidate.segment_path, dataset_root)
    clip_duration = candidate.duration
    if clip_duration is None:
        clip_duration = get_video_duration(input_path)
    start_offset = 0.0
    pad_black = False
    speed_factor = 1.0
    note = "as_is"

    if clip_duration is not None and target_duration > 0:
        if clip_duration > 0:
            raw_speed = clip_duration / target_duration
            if abs(raw_speed - 1.0) > 1e-3:
                speed_factor = raw_speed
                if speed_factor > 1.0:
                    note = "speed_up"
                else:
                    note = "slow_down"

    return PostprocessPlan(
        input_path=input_path,
        output_path=output_path,
        start_offset=start_offset,
        duration=target_duration,
        pad_black=pad_black,
        clip_duration=clip_duration,
        speed_factor=speed_factor,
        note=note,
    )


def render_postprocess_plan(plan: PostprocessPlan, video_encoder: str) -> None:
    """Render a postprocess plan."""
    render_clip(
        input_path=plan.input_path,
        output_path=plan.output_path,
        start_offset=plan.start_offset,
        duration=plan.duration,
        pad_black=plan.pad_black,
        speed_factor=plan.speed_factor,
        video_encoder=video_encoder,
    )
