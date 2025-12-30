# mv_synthesis

This project builds music videos by matching lyric lines to video clips via embeddings.

## Dataset Layout

```
datasets/
  ds1/
    songs/
      <song_name>/
        clips_and_lyrics/
          0001_000.000-003.850.txt
          0001_000.000-003.850.wav
          ...
        <song>.mp3
        <song>.txt
        lyrics_lines.json
    videos/
      <video_name>/
        <segment>.mp4
        ...
    db/
```

Collections:
- `video-segments`
- `video-vibe_cards`
- `lyrics-{text_video,text_text}-{text,augment,combined}`
- `lyrics-{audio_video,audio_text}-audio` (optional)

## Synthesis Pipeline (High-Level)

1. **Embed video segments** into `video-segments` and `video-vibe_cards`.
2. **Embed lyric lines** into `lyrics-{text_video,text_text}-{text,augment,combined}` (and optionally `lyrics-{audio_video,audio_text}-audio`).
3. **Synthesize MV** by retrieving candidates for each lyric line, selecting a clip, post-processing, and stitching.

## Usage

### 1) Embed video segments

```
python synthesis/embed_video_segments.py \
  datasets/ds1/db \
  datasets/ds1/videos \
  --dataset-root datasets/ds1
```

### 2) Embed lyric lines (pre-process)

This builds `lyrics_lines.json` in the song directory and writes embeddings to Qdrant.

```
python synthesis/embed_lyrics_lines.py \
  --dataset-root datasets/ds1 \
  --song-name counting_stars
```

To include audio embeddings (note: audio-only embedding may fail depending on provider constraints):

```
python synthesis/embed_lyrics_lines.py \
  --dataset-root datasets/ds1 \
  --song-name counting_stars \
  --embed-audio
```

### 3) Synthesize MV

```
python synthesis/synthesize_pipeline.py \
  --dataset-root datasets/ds1 \
  --song-name counting_stars \
  --top-k 5 \
  --selection-strategy top_vibe
```

Outputs:
- `outputs/synthesis/<song>/mv_with_music_subtitled.mp4`
- `outputs/synthesis/<song>/retrieval_manifest.json`

## Notes

- The pipeline reads lyric timing from `clips_and_lyrics/*.txt` so any intro/intermezzo lines live there.
- `lyrics_lines.json` is the cached, processed representation of lyric lines for re-use.
- By default, retrieval uses `lyrics-text_video-combined` embeddings. Use `--query-source text` or `--query-source text_augmented` to switch.
- Selection strategies include `top_vibe_duration` and `top_video_duration` (prefer top score when duration >= lyric duration, otherwise choose the longest shorter clip).
*** End Patch

## Experiments

```
python src/synthesis/synthesize_pipeline.py ds2 countingStars \
  --run-name random \
  --query-source random \
  --selection-strategy top_video_duration
```

```
python src/synthesis/synthesize_pipeline.py ds2 countingStars \
  --run-name text_video \
  --query-source text \
  --selection-strategy top_video_duration
```

```
python src/synthesis/synthesize_pipeline.py ds2 countingStars \
  --run-name av \
  --selection-strategy fused_rank \
  --fused-text-source combined \
  --fused-weight-tv 0.0 \
  --fused-weight-tvc 0.0 \
  --fused-weight-av 1.0 \
  --fused-weight-avc 0.0 \

python src/synthesis/synthesize_pipeline.py ds2 sunshine \
  --run-name fused_rank \
  --selection-strategy fused_rank \
  --fused-text-source combined \
  --fused-weight-tv 0.5 \
  --fused-weight-tvc 0.4 \
  --fused-weight-av 0.1 \
  --fused-weight-avc 0.0 \


python src/synthesis/synthesize_pipeline.py ds2 happy \
  --run-name fused_rank \
  --selection-strategy fused_rank \
  --fused-text-source combined \
  --fused-weight-tv 0.5 \
  --fused-weight-tvc 0.5 \
  --fused-weight-av 0.0 \
  --fused-weight-avc 0.0 \
  --fused-anti-repeat 100 \

python src/synthesis/synthesize_pipeline.py ds2 payphone \
  --run-name fused_rank \
  --selection-strategy fused_rank \
  --fused-text-source combined \
  --fused-weight-tv 0.5 \
  --fused-weight-tvc 0.3 \
  --fused-weight-av 0.1 \
  --fused-weight-avc 0.1 \
  --fused-anti-repeat 100 \

python src/synthesis/synthesize_pipeline.py ds2 firework \
  --run-name fused_rank \
  --selection-strategy fused_rank \
  --fused-text-source combined \
  --fused-weight-tv 0.5 \
  --fused-weight-tvc 0.5 \
  --fused-weight-av 0.0 \
  --fused-weight-avc 0.0 \
  --fused-anti-repeat 100 \
```