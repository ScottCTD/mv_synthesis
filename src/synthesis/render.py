from __future__ import annotations

import concurrent.futures
import subprocess
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

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
    # Ensure output directory exists
    if plans:
        plans[0].output_path.parent.mkdir(parents=True, exist_ok=True)
    max_workers = max(1, workers)
    if max_workers == 1:
        for plan in tqdm(plans, desc="Rendering clips", unit="clip"):
            try:
                render_postprocess_plan(plan, video_encoder)
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr if e.stderr else "No error message available"
                raise RuntimeError(
                    f"FFmpeg failed with exit code {e.returncode}.\n"
                    f"Command: {' '.join(e.cmd)}\n"
                    f"Error: {error_msg}"
                ) from e
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(render_postprocess_plan, plan, video_encoder)
            for plan in plans
        ]
        with tqdm(total=len(futures), desc="Rendering clips", unit="clip") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except subprocess.CalledProcessError as e:
                    error_msg = e.stderr if e.stderr else "No error message available"
                    raise RuntimeError(
                        f"FFmpeg failed with exit code {e.returncode}.\n"
                        f"Command: {' '.join(e.cmd)}\n"
                        f"Error: {error_msg}"
                    ) from e
                pbar.update(1)


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
    with tqdm(total=3, desc="Stitching video", unit="step") as pbar:
        concat_clips(concat_list, stitched_video, video_encoder=video_encoder)
        pbar.update(1)

        final_video = output_dir / "mv_with_music.mp4"
        add_background_music(stitched_video, song_audio, final_video)
        pbar.update(1)

        subtitle_path = output_dir / "lyrics.srt"
        write_srt(lyrics_lines, subtitle_path)

        subtitled_video = output_dir / "mv_with_music_subtitled.mp4"
        burn_subtitles(
            final_video,
            subtitle_path,
            subtitled_video,
            video_encoder=video_encoder,
        )
        pbar.update(1)

    return subtitled_video
