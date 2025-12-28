from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SONG_NAME = "countingStars"
AUDIO_PATH = f"/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/datasets/ds1/songs/{SONG_NAME}/{SONG_NAME}.mp3"
LYRIC_PATH = f"/Users/zhaoxiling/Documents/2025Fall/CSC2508/mv_synthesis/datasets/ds1/songs/{SONG_NAME}/{SONG_NAME}.txt"
OUT_DIR = PROJECT_ROOT / "datasets" / "ds1" / \
    "songs" / SONG_NAME / "clips_and_lyrics"
OUT_EXT = "wav"
AAC_BITRATE = "192k"  # used only when OUT_EXT != "wav"
