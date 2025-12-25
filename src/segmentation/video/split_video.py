import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scenedetect import ContentDetector

from segmentation.video.scene_runner import process_videos


DEFAULT_THRESHOLD = 37
DEFAULT_MIN_SCENE_LEN = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and optionally split video scenes.")
    parser.add_argument("--threshold", type=int,
                        default=DEFAULT_THRESHOLD, help="ContentDetector threshold.")
    parser.add_argument("--min-scene-len", type=int,
                        default=DEFAULT_MIN_SCENE_LEN, help="Minimum scene length.")
    parser.add_argument("--split", dest="split", action="store_true",
                        help="Enable splitting scenes.", default=True)
    parser.add_argument("--no-split", dest="split",
                        action="store_false", help="Disable splitting scenes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    output_root = workspace_root / "outputs"

    root_folder = "/Users/zhaoxiling/Documents/2025Fall/CSC2508/TomAndJerry/Tom_and_Jerry(1080p_Mixed_x265_HEVC_10bit Mixed_2.0_Ghost)/S1950"
    paths = [
        "E01-Little_Quacker.mkv",
        # "E02-Saturday_Evening_Puss.mkv",
        # "E03-Texas_Tom.mkv",
        # "E04-Jerry_And_The_Lion.mkv",
        # "E05-Safety_Second.mkv",
        # "E06-Tom_and_Jerry_in_the_Hollywood_Bowl.mkv",
        # "E07-The_Framed_Cat.mkv",
        # "E08-Cue_Ball_Cat.mkv",
        # "E09-Casanova_Cat.mkv",
        # "E10-Jerry_And_The_Goldfish.mkv",
    ]

    detector = ContentDetector(
        threshold=args.threshold, min_scene_len=args.min_scene_len)
    process_videos(paths=paths, root_folder=root_folder,
                   output_root=output_root, detector=detector, split=args.split)


if __name__ == "__main__":
    main()
