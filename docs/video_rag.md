# Video RAG

`apps/video_rag.py` indexes sampled video frames with CLIP embeddings so a text query can retrieve
the most relevant moments in local videos. It is intended for visual frame retrieval, not speech
transcription or full video understanding.

## Install

From a source checkout, install the optional video dependencies:

```bash
uv sync --extra video
```

For a standalone environment, install OpenCV and Pillow alongside LEANN:

```bash
pip install leann opencv-python-headless pillow
```

## Build

```bash
python -m apps.video_rag --video-dir ./videos --index-dir ./video_index --force-rebuild
```

By default, LEANN samples one frame every five seconds and caps each video at 120 sampled frames.
Use `--sample-interval-seconds` for denser or sparser sampling and `--max-frames-per-video -1` to
disable the per-video cap.

```bash
python -m apps.video_rag \
  --video-dir ./videos \
  --sample-interval-seconds 2 \
  --max-frames-per-video 300 \
  --index-dir ./video_index \
  --force-rebuild
```

## Search

```bash
python -m apps.video_rag \
  --video-dir ./videos \
  --index-dir ./video_index \
  --query "slide with latency chart"
```

The index stores one passage per sampled frame. Each passage includes the video path, relative path,
frame index, timestamp, frame size, video duration, and embedding model. Passage IDs are stable
`video-frame-<hash>` values derived from the relative video path, frame index, and timestamp.

## Notes

- Supported default extensions are `.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, and `.m4v`.
- Queries search CLIP text embeddings against precomputed CLIP frame embeddings. The app disables
  LEANN embedding recomputation at query time because stored frame descriptions cannot recreate the
  original pixels.
- Codec support depends on the installed OpenCV build and platform media libraries.
