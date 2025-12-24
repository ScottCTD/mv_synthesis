import json
from pathlib import Path
from typing import Dict, List, Tuple

from scenedetect import ContentDetector, FrameTimecode, detect

TIME_BASE_FPS = 1  # 1 fps time base so get_seconds returns the stored seconds


def save_scenes(cache_path: Path, scenes: List[Tuple[FrameTimecode, FrameTimecode]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, List[Dict[str, float]]] = {
        "scenes": [
            {
                "start_seconds": start.get_seconds(),
                "end_seconds": end.get_seconds(),
                "start_timecode": str(start),
                "end_timecode": str(end),
                "duration_seconds": end.get_seconds() - start.get_seconds(),
            }
            for start, end in scenes
        ]
    }
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2)


def load_or_detect_scenes(
    video_path: str, detector: ContentDetector, cache_dir: Path
) -> Tuple[List[Tuple[FrameTimecode, FrameTimecode]], Path]:
    base_name = Path(video_path).stem
    cache_path = cache_dir / f"{base_name}_scenes.json"

    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as cache_file:
            data = json.load(cache_file)
        scenes = [
            (
                FrameTimecode(timecode=entry["start_seconds"], fps=TIME_BASE_FPS),
                FrameTimecode(timecode=entry["end_seconds"], fps=TIME_BASE_FPS),
            )
            for entry in data.get("scenes", [])
        ]
    else:
        scenes = detect(video_path, detector)
        save_scenes(cache_path, scenes)

    return scenes, cache_path
