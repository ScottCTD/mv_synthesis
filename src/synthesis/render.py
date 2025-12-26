from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Iterable

from synthesis.ffmpeg_utils import (
    add_background_music,
    burn_subtitles,
    concat_clips,
    write_concat_list,
    write_srt,
)
from synthesis.postprocess import PostprocessPlan, render_postprocess_plan
from synthesis.models import LyricsLine


def render_clips_parallel(
    plans: list[PostprocessPlan], workers: int, video_encoder: str
) -> None:
    if not plans:
        return
    max_workers = max(1, workers)
    if max_workers == 1:
        for plan in plans:
            render_postprocess_plan(plan, video_encoder)
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(render_postprocess_plan, plan, video_encoder)
            for plan in plans
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def stitch_video(
    clip_paths: Iterable[Path],
    song_audio: Path,
    lyrics_lines: Iterable[LyricsLine],
    output_dir: Path,
    video_encoder: str,
) -> Path:
    concat_list = output_dir / "concat_list.txt"
    write_concat_list(clip_paths, concat_list)

    stitched_video = output_dir / "mv_video.mp4"
    concat_clips(concat_list, stitched_video, video_encoder=video_encoder)

    final_video = output_dir / "mv_with_music.mp4"
    add_background_music(stitched_video, song_audio, final_video)

    subtitle_path = output_dir / "lyrics.srt"
    write_srt(lyrics_lines, subtitle_path)

    subtitled_video = output_dir / "mv_with_music_subtitled.mp4"
    burn_subtitles(
        final_video,
        subtitle_path,
        subtitled_video,
        video_encoder=video_encoder,
    )

    return subtitled_video
