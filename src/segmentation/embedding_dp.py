"""DP-based video segmentation with Nova embeddings.

hop controls how often we sample the timeline (scan centers), while win controls
the clip duration around each center used for embeddings. Smaller hop increases
the number of clips; larger win increases contextual averaging and cost.
"""

import argparse
import asyncio
import json
import math
import sys
import tempfile
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scenedetect import split_video_ffmpeg
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.video_splitter import default_formatter
from tqdm import tqdm

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from synthesis.ffmpeg_utils import get_video_duration, resolve_video_encoder, run_ffmpeg
from synthesis.nova_embedding_model import Nova2OmniEmbeddings

SPLIT_TIME_BASE_FPS = 1000.0
OUTPUT_TEMPLATE = "$SCENE_NUMBER-$START_TIME-$END_TIME.mp4"
DEFAULT_VIDEO_ENCODER = "libx264"


def build_scan_times(start: float, end: float, hop: float) -> list[float]:
    if hop <= 0:
        raise ValueError("hop must be positive")
    if end < start:
        raise ValueError("end must be >= start")
    duration = end - start
    if duration < 0:
        raise ValueError("duration must be non-negative")
    steps = int(math.floor((duration + 1e-6) / hop))
    return [start + (i * hop) for i in range(steps + 1)]


def l2_normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _get_encoder_params(video_encoder: str) -> list[str]:
    """Get encoder-specific parameters for FFmpeg."""
    params = []
    if video_encoder in ("h264_nvenc", "hevc_nvenc"):
        # NVENC requires preset and rate control parameters
        params.extend(["-preset", "p4", "-rc", "vbr", "-cq", "23"])
    elif video_encoder == "libx264":
        params.extend(["-preset", "veryfast", "-crf", "22"])
    return params


def extract_clip(
    video_path: Path,
    output_path: Path,
    center: float,
    length: float,
    video_encoder: str,
) -> None:
    start = max(0.0, center - (length / 2.0))
    duration = max(0.0, length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video_path)]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        video_encoder,
    ]
    cmd.extend(_get_encoder_params(video_encoder))
    cmd += [
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def _select_video_embedding(result: dict) -> list[float]:
    embeddings = result.get("embeddings", [])
    if not embeddings:
        raise ValueError("No embeddings returned from model.")
    for embedding in embeddings:
        if embedding.get("embedding_type") == "VIDEO":
            return embedding["embedding"]
    raise ValueError("VIDEO embedding not found in result.")


def _embedding_cache_path(clip_dir: Path, index: int) -> Path:
    return clip_dir / "embeddings" / f"scan_{index:06d}.npz"


async def compute_embeddings(
    video_path: Path,
    times: list[float],
    win: float,
    embed_model: Nova2OmniEmbeddings,
    clip_dir: Path,
    embedding_purpose: str,
    max_concurrent: int,
    video_encoder: str,
) -> tuple[list[float], np.ndarray]:
    semaphore = asyncio.Semaphore(max(1, max_concurrent))
    embedding_cache_dir = clip_dir / "embeddings"
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    max_retries = 5

    async def embed_with_retry(clip_path: Path) -> dict:
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await embed_model.embed_video(
                    str(clip_path),
                    embedding_purpose=embedding_purpose,
                    embedding_mode="AUDIO_VIDEO_SEPARATE",
                )
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries - 1:
                    break
                delay = 2**attempt
                await asyncio.sleep(delay)
        raise last_exc if last_exc is not None else RuntimeError("Embedding failed.")

    async def embed_one(
        index: int, center: float
    ) -> tuple[int, Optional[np.ndarray], Optional[str]]:
        clip_path = clip_dir / f"scan_{index:06d}.mp4"
        cache_path = _embedding_cache_path(clip_dir, index)
        if cache_path.exists():
            try:
                with np.load(cache_path) as cached:
                    return index, cached["embedding"], None
            except Exception as exc:
                print(
                    f"Failed to load cached embedding {cache_path}: {exc}",
                    file=sys.stderr,
                )
        try:
            async with semaphore:
                await asyncio.to_thread(
                    extract_clip, video_path, clip_path, center, win, video_encoder
                )
                result = await embed_with_retry(clip_path)
            embedding = _select_video_embedding(result)
            embedding = l2_normalize(embedding)
            try:
                np.savez_compressed(cache_path, embedding=embedding)
            except Exception as exc:
                print(
                    f"Failed to cache embedding {cache_path}: {exc}",
                    file=sys.stderr,
                )
            return index, embedding, None
        except Exception as exc:
            return index, None, f"{type(exc).__name__}: {exc}"

    tasks = [asyncio.create_task(embed_one(index, center)) for index, center in enumerate(times)]
    results: list[tuple[int, Optional[np.ndarray], Optional[str]]] = []
    progress = tqdm(total=len(tasks), desc="Embedding scans")
    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            progress.set_postfix_str(f"cost=${embed_model.costs['total_cost']:.6f}")
            progress.update(1)
    finally:
        progress.close()
    results.sort(key=lambda item: item[0])
    failures = [(index, error) for index, _, error in results if error]
    if failures:
        print(
            f"Skipping {len(failures)} scans due to embedding errors.",
            file=sys.stderr,
        )
        for index, error in failures[:10]:
            print(f"  scan_{index:06d}.mp4 -> {error}", file=sys.stderr)
        if len(failures) > 10:
            print("  ...", file=sys.stderr)
    kept = [(times[index], embedding) for index, embedding, error in results if not error]
    if not kept:
        raise ValueError("All embedding scans failed.")
    kept_times, kept_embeddings = zip(*kept)
    return list(kept_times), np.vstack(kept_embeddings)


def _build_prefix_sums(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prefix_sum = np.zeros((embeddings.shape[0] + 1, embeddings.shape[1]), dtype=np.float64)
    prefix_sum[1:] = np.cumsum(embeddings, axis=0)
    sq_norms = np.einsum("ij,ij->i", embeddings, embeddings)
    prefix_sum_sqnorm = np.zeros(embeddings.shape[0] + 1, dtype=np.float64)
    prefix_sum_sqnorm[1:] = np.cumsum(sq_norms, axis=0)
    return prefix_sum, prefix_sum_sqnorm


def _segment_cost(
    prefix_sum: np.ndarray,
    prefix_sum_sqnorm: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    count = end_idx - start_idx + 1
    summed = prefix_sum[end_idx + 1] - prefix_sum[start_idx]
    summed_sq = prefix_sum_sqnorm[end_idx + 1] - prefix_sum_sqnorm[start_idx]
    mean = summed / count
    cost = float(summed_sq - count * np.dot(mean, mean))
    return max(cost, 0.0)


def build_scene_list(
    segments: list[tuple[float, float]],
    fps: float = SPLIT_TIME_BASE_FPS,
) -> list[tuple[FrameTimecode, FrameTimecode]]:
    scene_list: list[tuple[FrameTimecode, FrameTimecode]] = []
    for start, end in segments:
        if end <= start:
            raise ValueError(f"Invalid segment [{start}, {end}].")
        scene_list.append(
            (
                FrameTimecode(timecode=start, fps=fps),
                FrameTimecode(timecode=end, fps=fps),
            )
        )
    return scene_list


def _build_split_args(video_encoder: str) -> str:
    """Build FFmpeg arguments for video splitting with encoder-specific parameters."""
    base_args = "-map 0:v:0 -map 0:a? -map 0:s? -c:v"
    
    if video_encoder == DEFAULT_VIDEO_ENCODER:
        return (
            f"{base_args} libx264 "
            "-preset veryfast -crf 22 -c:a aac"
        )
    elif video_encoder in ("h264_nvenc", "hevc_nvenc"):
        # NVENC requires preset and rate control parameters
        return (
            f"{base_args} {video_encoder} "
            "-preset p4 -rc vbr -cq 23 -c:a aac"
        )
    return f"{base_args} {video_encoder} -c:a aac"


def split_segments_ffmpeg(
    video_path: Path,
    segments: list[tuple[float, float]],
    output_dir: Optional[Path],
    video_encoder: str,
) -> list[Path]:
    video_encoder, warning = resolve_video_encoder(video_encoder)
    if warning:
        print(warning, file=sys.stderr)
    scene_list = build_scene_list(segments)
    if not scene_list:
        return []
    output_paths: list[Path] = []
    base_formatter = default_formatter(OUTPUT_TEMPLATE)

    def formatter(video, scene) -> str:
        rel_path = Path(base_formatter(video, scene))
        output_paths.append(rel_path)
        return str(rel_path)

    split_video_ffmpeg(
        input_video_path=str(video_path),
        scene_list=scene_list,
        output_dir=output_dir,
        output_file_template=OUTPUT_TEMPLATE,
        show_progress=True,
        show_output=False,
        formatter=formatter,
        arg_override=_build_split_args(video_encoder),
    )
    if output_dir is None:
        return output_paths
    return [output_dir / path for path in output_paths]


def segment_dp(
    times: list[float],
    embeddings: np.ndarray,
    min_len: float,
    max_len: float,
    segment_penalty: float = 0.0,
) -> list[tuple[float, float]]:
    if min_len < 0 or max_len <= 0:
        raise ValueError("min_len must be >= 0 and max_len must be > 0")
    if min_len > max_len:
        raise ValueError("min_len cannot exceed max_len")
    if len(times) != embeddings.shape[0]:
        raise ValueError("times and embeddings must have matching lengths")

    prefix_sum, prefix_sum_sqnorm = _build_prefix_sums(embeddings)
    n = len(times)
    dp = np.full(n + 1, np.inf, dtype=np.float64)
    prev = np.full(n + 1, -1, dtype=np.int64)
    dp[0] = 0.0

    for i in range(1, n + 1):
        end_idx = i - 1
        t_end = times[end_idx]
        j_min = bisect_left(times, t_end - max_len - 1e-9)
        j_max = bisect_right(times, t_end - min_len + 1e-9) - 1
        j_min = max(j_min, 0)
        j_max = min(j_max, end_idx)
        if j_min > j_max:
            continue
        for j in range(j_min, j_max + 1):
            if not math.isfinite(dp[j]):
                continue
            cost = _segment_cost(prefix_sum, prefix_sum_sqnorm, j, end_idx)
            candidate = dp[j] + cost + segment_penalty
            if candidate < dp[i]:
                dp[i] = candidate
                prev[i] = j

    if not math.isfinite(dp[n]):
        raise ValueError("No feasible segmentation found for the given constraints.")

    boundaries: list[int] = []
    i = n
    while i > 0 and prev[i] != -1:
        j = int(prev[i])
        boundaries.append(j)
        i = j
    boundaries.reverse()

    segments: list[tuple[float, float]] = []
    for idx, start_idx in enumerate(boundaries):
        if idx + 1 < len(boundaries):
            end_idx = boundaries[idx + 1] - 1
        else:
            end_idx = n - 1
        segments.append((times[start_idx], times[end_idx]))
    return segments


async def segment_video(
    video_path: Path,
    hop: float,
    win: float,
    min_len: float,
    max_len: float,
    embedding_purpose: str = "CLUSTERING",
    segment_penalty: float = 0.0,
    clip_dir: Optional[Path] = None,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    max_concurrent: int = 4,
    video_encoder: str = DEFAULT_VIDEO_ENCODER,
) -> list[tuple[float, float]]:
    duration = get_video_duration(video_path)
    if duration is None:
        raise ValueError(f"Unable to determine duration for {video_path}")
    video_encoder, warning = resolve_video_encoder(video_encoder)
    if warning:
        print(warning, file=sys.stderr)
    if start_time < 0:
        raise ValueError("start_time must be >= 0")
    if end_time is None:
        end_time = duration
    if end_time > duration:
        end_time = duration
    if end_time <= start_time:
        raise ValueError("end_time must be > start_time")
    times = build_scan_times(start_time, end_time, hop)
    embed_model = Nova2OmniEmbeddings()

    if clip_dir is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            kept_times, embeddings = await compute_embeddings(
                video_path=video_path,
                times=times,
                win=win,
                embed_model=embed_model,
                clip_dir=Path(tmp_dir),
                embedding_purpose=embedding_purpose,
                max_concurrent=max_concurrent,
                video_encoder=video_encoder,
            )
    else:
        clip_dir.mkdir(parents=True, exist_ok=True)
        kept_times, embeddings = await compute_embeddings(
            video_path=video_path,
            times=times,
            win=win,
            embed_model=embed_model,
            clip_dir=clip_dir,
            embedding_purpose=embedding_purpose,
            max_concurrent=max_concurrent,
            video_encoder=video_encoder,
        )

    return segment_dp(
        times=kept_times,
        embeddings=embeddings,
        min_len=min_len,
        max_len=max_len,
        segment_penalty=segment_penalty,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment a video with DP over Nova embedding scan points."
    )
    parser.add_argument("video_path", type=Path, help="Path to the input video.")
    parser.add_argument(
        "--hop",
        type=float,
        default=0.5,
        help=(
            "Scan hop in seconds (time between consecutive scan centers). "
            "Smaller hops sample more densely and increase clip count."
        ),
    )
    parser.add_argument(
        "--win",
        type=float,
        default=1.0,
        help=(
            "Window length per scan in seconds (clip duration centered on each hop). "
            "Longer windows average more context but increase embedding cost."
        ),
    )
    parser.add_argument(
        "--min-len", type=float, default=1.0, help="Minimum segment length in seconds."
    )
    parser.add_argument(
        "--max-len", type=float, default=5.0, help="Maximum segment length in seconds."
    )
    parser.add_argument(
        "--segment-penalty",
        type=float,
        default=0.3,
        help="Optional per-segment penalty added to the SSE cost.",
    )
    parser.add_argument(
        "--embedding-purpose",
        type=str,
        default="CLUSTERING",
        help="Embedding purpose for Nova embeddings.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=128,
        help="Maximum concurrent clip embeddings.",
    )
    parser.add_argument(
        "--video-encoder",
        type=str,
        default=DEFAULT_VIDEO_ENCODER,
        help="Video encoder used for scan clips and output segments.",
    )
    parser.add_argument(
        "--clip-dir",
        type=Path,
        default=None,
        help="Directory to store extracted clips (defaults to output-dir/cache when set).",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds to segment within the video.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=None,
        help="End time in seconds to segment within the video (default: video end).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print clip count and total clip seconds without embedding.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for segments or output clip paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write final split segments as mp4 files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_dir is None and args.output_dir is not None:
        args.clip_dir = args.output_dir / "cache"
    if args.dry_run:
        duration = get_video_duration(args.video_path)
        if duration is None:
            raise ValueError(f"Unable to determine duration for {args.video_path}")
        if args.start < 0:
            raise ValueError("start must be >= 0")
        end_time = duration if args.end is None else min(args.end, duration)
        if end_time <= args.start:
            raise ValueError("end must be > start")
        times = build_scan_times(args.start, end_time, args.hop)
        num_clips = len(times)
        total_clip_seconds = num_clips * args.win
        # Cost calculation based on Nova2OmniEmbeddings pricing:
        # AUDIO_VIDEO_SEPARATE mode charges for both video ($0.0007/sec) and audio ($0.00014/sec)
        video_cost_per_second = 0.0007
        audio_cost_per_second = 0.00014
        cost_per_clip_second = video_cost_per_second + audio_cost_per_second
        estimated_cost = total_clip_seconds * cost_per_clip_second
        summary = {
            "duration_seconds": duration,
            "start_seconds": args.start,
            "end_seconds": end_time,
            "range_seconds": end_time - args.start,
            "hop_seconds": args.hop,
            "win_seconds": args.win,
            "num_clips": num_clips,
            "total_clip_seconds": total_clip_seconds,
            "estimated_cost_usd": round(estimated_cost, 6),
            "cost_per_clip_second": cost_per_clip_second,
        }
        print(json.dumps(summary, indent=2))
        return
    segments = asyncio.run(
        segment_video(
            video_path=args.video_path,
            hop=args.hop,
            win=args.win,
            min_len=args.min_len,
            max_len=args.max_len,
            embedding_purpose=args.embedding_purpose,
            segment_penalty=args.segment_penalty,
            clip_dir=args.clip_dir,
            start_time=args.start,
            end_time=args.end,
            max_concurrent=args.max_concurrent,
            video_encoder=args.video_encoder,
        )
    )
    if args.output_dir is not None:
        output_paths = split_segments_ffmpeg(
            video_path=args.video_path,
            segments=segments,
            output_dir=args.output_dir,
            video_encoder=args.video_encoder,
        )
        payload = [str(path) for path in output_paths]
    else:
        payload = [[start, end] for start, end in segments]
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
