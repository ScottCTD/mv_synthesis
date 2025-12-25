from pathlib import Path
from typing import Iterable

from scenedetect import ContentDetector, split_video_ffmpeg
from tqdm import tqdm

from .scene_cache import load_or_detect_scenes
from .scene_stats import build_duration_report


def process_videos(
    paths: Iterable[str],
    root_folder: str,
    output_root: Path,
    detector: ContentDetector,
    split: bool = True,
) -> None:
    cache_dir = output_root / "scene_lists"
    split_dir_root = output_root / "output_scenes"

    for index, path in tqdm(enumerate(paths), total=len(paths)):
        video_path = f"{root_folder}/{path}"

        scenes, cache_path = load_or_detect_scenes(
            video_path, detector, cache_dir)
        durations = [end.get_seconds() - start.get_seconds()
                     for start, end in scenes]

        print(f"\nVideo: {path}")
        print(f"Scene cache: {cache_path}")
        print(build_duration_report(durations))

        if not split:
            print("Splitting disabled; scenes cached only.")
            continue

        output_dir = split_dir_root / str(index)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_video_ffmpeg(
            input_video_path=video_path,
            scene_list=scenes,
            output_dir=str(output_dir),
            output_file_template="$VIDEO_NAME-Scene-$SCENE_NUMBER.mkv",
            show_progress=True,
        )
