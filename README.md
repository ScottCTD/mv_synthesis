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
- `lyrics-text`
- `lyrics-augmented_query`
- `lyrics-audio` (optional)

## Synthesis Pipeline (High-Level)

1. **Embed video segments** into `video-segments` and `video-vibe_cards`.
2. **Embed lyric lines** into `lyrics-text` and `lyrics-augmented_query` (and optionally `lyrics-audio`).
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
- By default, retrieval uses `lyrics-augmented_query` embeddings. Use `--query-source text` to switch.
*** End Patch
