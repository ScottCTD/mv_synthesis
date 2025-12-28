from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIO_PATH = "/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/datasets/ds1/songs/countingStars/countingStars.mp3"
LYRIC_PATH = "/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/datasets/ds1/songs/countingStars/countingStars.txt"
OUT_DIR = PROJECT_ROOT / "datasets" / "ds1" / \
    "songs" / "countingStars" / "clips_and_lyrics"
OUT_EXT = "wav"
AAC_BITRATE = "192k"  # used only when OUT_EXT != "wav"
