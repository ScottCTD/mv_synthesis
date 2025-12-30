# MV Synthesis Human Eval Web

Minimal two-column comparison UI for MV synthesis methods, served by a tiny Python server.

## Quick start
- From repo root: `python3 web/server.py`
- Open `http://localhost:8000`
- Or use the helper script: `bash web/start.sh`

## Configure data roots
Edit `web/config.json`:
- `dataset_root`: default `datasets/ds2` (expects `songs/{song}/{song}.mp3` and `videos/...`)
- `outputs_root`: default `outputs` (expects `outputs/{method}/{song}/retrieval_manifest.json`)
- `results_root`: default `results` (where comparisons are saved)
- `prefer_rendered`: if `true`, use `postprocess.output_path` when present
- `default_left_method` / `default_right_method`: optional method names to preselect
- `anonymize_methods`: if `true`, show Method 1/2 labels and disable method selection
- `hide_method_select`: legacy flag; treated the same as `anonymize_methods`

## URL params (optional)
Open with query params to preselect:
`http://localhost:8000/?song=countingStars&left=methodA&right=methodB`

## UI behavior
- The start screen collects username and song; method pair selection is shown unless anonymized.
- Top block shows song audio + current lyric line + line audio.
- Username is required to enable playback and voting.
- Left/right columns show the selected clip first, then `video_candidates` and `vibe_candidates`.
- Bottom bar has `I like left`, `I like right`, `Tie`, `Next`.
- `Play all` starts playback once for the lyric line audio and all videos.
- Keyboard shortcuts: `ArrowLeft`, `ArrowRight`, `T` (tie), `N` (next/skip).

## Results format
Results are saved as JSON (not JSONL) per song/method pair in `results/`.
File name: `{username}-{methodA}-{methodB}-{song}.json` (methods are sorted alphabetically).

Each comparison entry includes:
- `decision`: `left`, `right`, `tie`, or `skip`
- `winner`: method name or `null` for tie/skip
- `line_index`, `line_position`, `lyric_text`, `line_audio_url`
- full `entry` payloads for left/right methods (selected + candidates)

## Notes
- The server is stdlib-only and supports range requests for video playback.
- If manifests or assets are missing, the UI displays an empty state for that section.
