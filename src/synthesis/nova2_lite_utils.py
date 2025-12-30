import re

import cv2

_LRC_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}\.\d{3}\]\s*")


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def normalize_lyric_lines(full_lyrics: str) -> list[str]:
    """
    Convert raw lyrics (often LRC format) into a list of non-empty lyric lines.
    - Strips leading LRC timestamps like "[00:12.345]".
    - Drops empty lines.
    """
    lines: list[str] = []
    for raw in full_lyrics.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = _LRC_TIMESTAMP_RE.sub("", s).strip()
        if not s:
            continue
        lines.append(s)
    return lines


def _extract_converse_text(response: dict) -> str:
    text_response = ""
    content_list = response.get("output", {}).get("message", {}).get("content", []) or []
    for content in content_list:
        if "text" in content:
            text_response += content["text"]
    return text_response


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:05.2f}"


def _sample_video_frames(
    video_path: str,
    max_frames: int = 8,
    fps: float | None = None,
) -> tuple[list[bytes], list[str]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    frames: list[bytes] = []
    timestamps: list[str] = []
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        target_fps = float(fps) if fps is not None else native_fps

        if total_frames > 0:
            if target_fps and target_fps > 0 and native_fps > 0:
                step = max(1, int(round(native_fps / target_fps)))
                indices = list(range(0, total_frames, step))
            elif total_frames <= max_frames:
                indices = list(range(total_frames))
            elif max_frames == 1:
                indices = [total_frames // 2]
            else:
                indices = [
                    int(round(i * (total_frames - 1) / (max_frames - 1)))
                    for i in range(max_frames)
                ]

            if len(indices) > max_frames and max_frames > 0:
                indices = [
                    indices[int(round(i * (len(indices) - 1) / (max_frames - 1)))]
                    for i in range(max_frames)
                ]

            for frame_index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                ok, buffer = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                frames.append(buffer.tobytes())
                if native_fps > 0:
                    timestamps.append(_format_timestamp(frame_index / native_fps))
        else:
            step = int(round(native_fps)) if native_fps else 30
            if target_fps and target_fps > 0 and native_fps > 0:
                step = max(1, int(round(native_fps / target_fps)))
            frame_index = 0
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    ok, buffer = cv2.imencode(".jpg", frame)
                    if not ok:
                        frame_index += 1
                        continue
                    frames.append(buffer.tobytes())
                    if native_fps > 0:
                        timestamps.append(_format_timestamp(frame_index / native_fps))
                frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {video_path}")

    return frames, timestamps
