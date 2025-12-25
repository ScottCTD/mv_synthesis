import json
import os
import subprocess
from typing import Dict, List

from audio_config import AAC_BITRATE, AUDIO_PATH, LYRIC_PATH, OUT_DIR, OUT_EXT


def parse_lyric_file(lyric_path) -> dict[float, str]:
    def mmssxx_to_seconds(s):
        mm, ss = s.split(":")
        res = int(mm)*60 + float(ss)
        return res
    lyric_time_dict = {}
    with open(lyric_path) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('['):
                time_tag = line[1:9]
                time_tag = mmssxx_to_seconds(time_tag)
                lyric_text = line[11:].strip()
                lyric_time_dict[time_tag] = lyric_text
    return lyric_time_dict


def sh(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def ffprobe_duration(path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]).decode().strip()
    return float(out)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    total = ffprobe_duration(AUDIO_PATH)

    lyrics = parse_lyric_file(LYRIC_PATH)

    # Sort and round timestamps to avoid float artifacts.
    times = sorted({round(float(t), 3) for t in lyrics.keys()})

    manifest = []

    for i, start in enumerate(times):
        end = times[i + 1] if i + 1 < len(times) else total
        if end <= start:
            continue

        # Resolve lyric (handles tiny float mismatches).
        lyric = lyrics.get(start)
        if lyric is None:
            for k, v in lyrics.items():
                if abs(float(k) - start) <= 1e-3:
                    lyric = v
                    break
        lyric = lyric or ""

        clip_base = f"{i+1:04d}_{start:07.3f}-{end:07.3f}"
        clip_path = os.path.join(OUT_DIR, f"{clip_base}.{OUT_EXT}")
        txt_path = os.path.join(OUT_DIR, f"{clip_base}.txt")

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(lyric)

        # Accurate cutting (re-encode). If you need fastest (keyframe-limited) cuts, use -c copy.
        if OUT_EXT.lower() == "wav":
            cmd = [
                "ffmpeg", "-y",
                "-i", AUDIO_PATH,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-vn",
                "-acodec", "pcm_s16le",
                clip_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y",
                "-i", AUDIO_PATH,
                "-ss", f"{start:.3f}",
                "-to", f"{end:.3f}",
                "-vn",
                "-c:a", "aac",
                "-b:a", AAC_BITRATE,
                "-movflags", "+faststart",
                clip_path,
            ]

        sh(cmd)

        manifest.append({
            "index": i + 1,
            "start": float(f"{start:.3f}"),
            "end": float(f"{end:.3f}"),
            "clip": clip_path,
            "lyric_txt": txt_path,
            "lyric": lyric,
        })

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Done. Wrote {len(manifest)} clips to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
