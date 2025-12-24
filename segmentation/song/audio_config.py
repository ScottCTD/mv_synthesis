from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_PATH = "/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/inputs/countingStars.mp3"
LYRIC_PATH = "/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/inputs/countingStars.txt"
OUT_DIR = PROJECT_ROOT / "outputs" / "song"
OUT_EXT = "wav"
AAC_BITRATE = "192k"  # used only when OUT_EXT != "wav"
