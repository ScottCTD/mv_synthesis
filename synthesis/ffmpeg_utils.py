import subprocess
from pathlib import Path


def get_video_duration(video_path: Path) -> float | None:
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
