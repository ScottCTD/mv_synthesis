import subprocess
from pathlib import Path
from typing import Iterable, Optional


def run_ffmpeg(cmd: list[str]) -> None:
    """Run an ffmpeg command list and raise if the subprocess fails."""
    subprocess.run(cmd, check=True)


def get_video_duration(video_path: Path) -> Optional[float]:
    """Get video duration in seconds using ffprobe. Returns None on error."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except (
        subprocess.CalledProcessError,
        ValueError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return None


def render_clip(
    input_path: Path,
    output_path: Path,
    start_offset: float,
    duration: float,
    pad_black: bool,
) -> None:
    cmd = ["ffmpeg", "-y"]
    if start_offset > 0:
        cmd += ["-ss", f"{start_offset:.3f}"]
    cmd += ["-i", str(input_path)]
    filters = []
    if pad_black:
        input_duration = get_video_duration(input_path)
        if input_duration is not None and duration > input_duration:
            pad_duration = duration - input_duration
            filters.append(
                f"tpad=stop_mode=add:stop_duration={pad_duration:.3f}:color=black"
            )
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def convert_audio_for_embedding(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def wrap_audio_in_video(
    input_path: Path,
    output_path: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s={width}x{height}:r={fps}",
        "-i",
        str(input_path),
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def write_concat_list(paths: Iterable[Path], list_path: Path) -> None:
    lines = [f"file '{path.as_posix()}'" for path in paths]
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def concat_clips(list_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def add_background_music(video_path: Path, audio_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    run_ffmpeg(cmd)


def format_srt_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def write_srt(lines: Iterable, srt_path: Path) -> None:
    blocks: list[str] = []
    index = 1
    for line in lines:
        if not getattr(line, "text", ""):
            continue
        start = format_srt_timestamp(line.start)
        end = format_srt_timestamp(line.end)
        blocks.append(f"{index}\n{start} --> {end}\n{line.text}\n")
        index += 1
    srt_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")


def escape_subtitles_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\").replace(":", "\\:")


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> None:
    subtitle_filter = f"subtitles={escape_subtitles_path(srt_path)}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    run_ffmpeg(cmd)
