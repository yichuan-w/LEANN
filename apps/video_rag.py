#!/usr/bin/env python3
"""
CLIP Video RAG Application.

This application indexes sampled video frames with CLIP embeddings so users can
search local videos using text queries.

Usage:
    python -m apps.video_rag --video-dir ./my_videos --query "whiteboard diagram"
    python -m apps.video_rag --video-dir ./my_videos --sample-interval-seconds 2
"""

import argparse
import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from leann.api import LeannChat
from leann.registry import register_project_directory
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from apps.base_rag_example import BaseRAGExample, create_rag_session

DEFAULT_VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"]


@dataclass(frozen=True)
class VideoFrameRecord:
    """A sampled RGB frame and the metadata needed to index it."""

    image: Image.Image
    frame_index: int
    timestamp_seconds: float
    fps: float
    width: int
    height: int
    duration_seconds: float


def normalize_video_extensions(extensions: Iterable[str]) -> set[str]:
    """Normalize video extensions to lowercase dot-prefixed suffixes."""
    normalized = set()
    for extension in extensions:
        stripped = extension.strip().lower()
        if not stripped:
            continue
        normalized.add(stripped if stripped.startswith(".") else f".{stripped}")
    return normalized


def discover_video_files(video_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """Return video files in deterministic relative-path order."""
    if not video_dir.exists():
        raise ValueError(f"Video directory does not exist: {video_dir}")
    if not video_dir.is_dir():
        raise ValueError(f"Video path is not a directory: {video_dir}")

    allowed_extensions = normalize_video_extensions(extensions)
    if not allowed_extensions:
        raise ValueError("At least one video extension must be provided.")

    video_files = [
        path
        for path in video_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    return sorted(
        video_files,
        key=lambda path: (
            path.relative_to(video_dir).as_posix().lower(),
            path.relative_to(video_dir).as_posix(),
        ),
    )


def video_frame_passage_id(
    video_dir: Path, video_path: Path, frame_index: int, timestamp_ms: int
) -> str:
    """Create a stable passage ID for a sampled frame."""
    relative_path = video_path.relative_to(video_dir).as_posix()
    key = f"{relative_path}:{frame_index}:{timestamp_ms}"
    digest = hashlib.sha256(key.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"video-frame-{digest[:16]}"


def video_frame_text(
    video_path: Path, video_dir: Path, timestamp_seconds: float, frame_index: int
) -> str:
    """Create the text payload stored alongside each sampled frame vector."""
    relative_path = video_path.relative_to(video_dir).as_posix()
    return (
        f"Video frame: {video_path.name}\n"
        f"Relative path: {relative_path}\n"
        f"Timestamp: {timestamp_seconds:.3f}s\n"
        f"Frame index: {frame_index}\n"
        f"Path: {video_path}"
    )


def video_frame_metadata(
    video_path: Path,
    video_dir: Path,
    embedding_model: str,
    frame_index: int,
    timestamp_seconds: float,
    fps: float,
    width: int,
    height: int,
    duration_seconds: float,
) -> dict[str, Any]:
    """Create JSON-serializable metadata for an indexed video frame."""
    relative_path = video_path.relative_to(video_dir).as_posix()
    timestamp_ms = round(timestamp_seconds * 1000)
    stat = video_path.stat()
    return {
        "id": video_frame_passage_id(video_dir, video_path, frame_index, timestamp_ms),
        "source": str(video_path),
        "video_path": str(video_path),
        "video_name": video_path.name,
        "video_dir": str(video_dir),
        "relative_path": relative_path,
        "extension": video_path.suffix.lower(),
        "media_type": "video_frame",
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "timestamp_ms": timestamp_ms,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
        "size_bytes": stat.st_size,
        "embedding_model": embedding_model,
    }


class VideoRAG(BaseRAGExample):
    """RAG application for text-to-video-frame search using CLIP embeddings."""

    embedding_model_default = "clip-ViT-L-14"
    embedding_mode_default = "sentence-transformers"
    max_items_default = -1

    def __init__(self):
        super().__init__(
            name="Video RAG",
            description="RAG application for searching sampled video frames with CLIP embeddings",
            default_index_name="video_index",
        )
        self._video_data: list[dict[str, Any]] = []

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add video-specific arguments."""
        video_group = parser.add_argument_group("Video Parameters")
        video_group.add_argument(
            "--video-dir",
            type=str,
            required=True,
            help="Directory containing videos to index",
        )
        video_group.add_argument(
            "--video-extensions",
            type=str,
            nargs="+",
            default=DEFAULT_VIDEO_EXTENSIONS,
            help="Video file extensions to process (default: .mp4 .mov .mkv .avi .webm .m4v)",
        )
        video_group.add_argument(
            "--sample-interval-seconds",
            type=float,
            default=5.0,
            help="Seconds between sampled frames (default: 5.0)",
        )
        video_group.add_argument(
            "--max-frames-per-video",
            type=int,
            default=120,
            help="Maximum sampled frames per video; use -1 for all sampled frames (default: 120)",
        )
        video_group.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for CLIP frame embedding generation (default: 32)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load videos, generate CLIP frame embeddings, and return index records."""
        self._video_data = self._load_videos_and_embeddings(args)
        return [
            {"text": entry["text"], "metadata": entry["metadata"]} for entry in self._video_data
        ]

    def _load_videos_and_embeddings(self, args) -> list[dict[str, Any]]:
        """Sample video frames, encode them with CLIP, and attach metadata."""
        video_dir = Path(args.video_dir).expanduser().resolve()
        self._validate_video_args(args)

        print(f"Loading videos from {video_dir}...")
        video_files = discover_video_files(video_dir, args.video_extensions)
        if not video_files:
            raise ValueError(
                f"No videos found in {video_dir} with extensions {args.video_extensions}"
            )

        print(f"Found {len(video_files)} videos")
        if args.max_items > 0:
            video_files = video_files[: args.max_items]
            print(f"Processing {len(video_files)} videos (limited by --max-items)")

        embedding_model = getattr(args, "embedding_model", self.embedding_model_default)
        print("Loading CLIP model...")
        model = self._load_clip_model(embedding_model)

        print("Sampling frames and generating embeddings...")
        video_data: list[dict[str, Any]] = []
        batch_frames: list[Image.Image] = []
        batch_context: list[tuple[Path, VideoFrameRecord]] = []

        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                for frame_record in self._extract_video_frames(
                    video_path=video_path,
                    sample_interval_seconds=args.sample_interval_seconds,
                    max_frames_per_video=args.max_frames_per_video,
                ):
                    batch_frames.append(frame_record.image)
                    batch_context.append((video_path, frame_record))
                    if len(batch_frames) >= args.batch_size:
                        pending_frames = batch_frames
                        pending_context = batch_context
                        batch_frames = []
                        batch_context = []
                        video_data.extend(
                            self._encode_frame_batch(
                                model,
                                pending_frames,
                                pending_context,
                                video_dir,
                                embedding_model,
                            )
                        )
            except Exception as exc:
                print(f"Failed to process {video_path}: {exc}")
                continue

        if batch_frames:
            video_data.extend(
                self._encode_frame_batch(
                    model,
                    batch_frames,
                    batch_context,
                    video_dir,
                    embedding_model,
                )
            )

        if not video_data:
            raise ValueError("No frames were extracted from the provided videos.")

        print(f"Processed {len(video_data)} sampled video frames")
        return video_data

    def _validate_video_args(self, args) -> None:
        if args.sample_interval_seconds <= 0:
            raise ValueError("--sample-interval-seconds must be greater than 0")
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")
        if args.max_frames_per_video == 0 or args.max_frames_per_video < -1:
            raise ValueError("--max-frames-per-video must be -1 or greater than 0")

    def _load_clip_model(self, embedding_model: str):
        """Load the configured CLIP encoder. Kept injectable for tests."""
        return SentenceTransformer(embedding_model)

    def _extract_video_frames(
        self,
        video_path: Path,
        sample_interval_seconds: float,
        max_frames_per_video: int,
    ) -> Iterator[VideoFrameRecord]:
        """Extract sampled RGB frames from one video using OpenCV."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError(
                "Video frame extraction requires OpenCV. Install it with "
                "`pip install opencv-python-headless` or `uv sync --extra video`."
            )

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError("OpenCV could not open the video file.")

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

            if fps <= 0 or frame_count <= 0:
                raise RuntimeError("OpenCV did not report a usable frame count and frame rate.")

            stride = max(1, round(sample_interval_seconds * fps))
            sampled = 0
            for frame_index in range(0, frame_count, stride):
                if max_frames_per_video > 0 and sampled >= max_frames_per_video:
                    break
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb).convert("RGB")
                sampled += 1
                yield VideoFrameRecord(
                    image=image,
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    fps=fps,
                    width=width,
                    height=height,
                    duration_seconds=duration_seconds,
                )
        finally:
            capture.release()

    def _encode_frame_batch(
        self,
        model,
        frames: list[Image.Image],
        frame_context: list[tuple[Path, VideoFrameRecord]],
        video_dir: Path,
        embedding_model: str,
    ) -> list[dict[str, Any]]:
        """Encode one sampled-frame batch and attach stable IDs/metadata."""
        embeddings = model.encode(
            frames,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=len(frames),
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.shape[0] != len(frame_context):
            raise RuntimeError(
                f"CLIP encoder returned {embedding_array.shape[0]} embeddings for "
                f"{len(frame_context)} sampled frames."
            )

        records = []
        for (video_path, frame_record), embedding in zip(frame_context, embedding_array):
            metadata = video_frame_metadata(
                video_path=video_path,
                video_dir=video_dir,
                embedding_model=embedding_model,
                frame_index=frame_record.frame_index,
                timestamp_seconds=frame_record.timestamp_seconds,
                fps=frame_record.fps,
                width=frame_record.width,
                height=frame_record.height,
                duration_seconds=frame_record.duration_seconds,
            )
            records.append(
                {
                    "id": metadata["id"],
                    "text": video_frame_text(
                        video_path,
                        video_dir,
                        frame_record.timestamp_seconds,
                        frame_record.frame_index,
                    ),
                    "metadata": metadata,
                    "embedding": embedding.astype(np.float32),
                }
            )
        return records

    async def build_index(self, args, texts: list[dict[str, Any]]) -> str:
        """Build an index using pre-computed CLIP frame embeddings."""
        from leann.api import LeannBuilder

        if not self._video_data or len(self._video_data) != len(texts):
            raise RuntimeError("No video data found. Make sure load_data() ran successfully.")

        print("Building LEANN index with CLIP video-frame embeddings...")
        embedding_model = getattr(args, "embedding_model", self.embedding_model_default)
        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=embedding_model,
            embedding_mode=self.embedding_mode_default,
            is_recompute=False,
            distance_metric="cosine",
            graph_degree=args.graph_degree,
            complexity=args.build_complexity,
            is_compact=not args.no_compact,
        )

        for data in self._video_data:
            builder.add_text(text=data["text"], metadata=data["metadata"])

        ids = [data["id"] for data in self._video_data]
        embeddings = np.array([data["embedding"] for data in self._video_data], dtype=np.float32)

        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
        builder.build_index_from_arrays(index_path, ids, embeddings)
        register_project_directory(Path.cwd())
        print(f"Index built successfully at {index_path}")
        return index_path

    def _llm_kwargs(self, args) -> dict[str, Any]:
        """Return optional LLM generation kwargs supported by the shared examples."""
        llm_kwargs = {}
        if hasattr(args, "thinking_budget") and args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget
        return llm_kwargs

    def _create_video_chat(self, args, index_path: str) -> LeannChat:
        """Create a chat wrapper for a video-frame index."""
        return LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=(
                "You are a helpful assistant that answers questions about sampled video "
                "frames indexed with CLIP embeddings."
            ),
            complexity=args.search_complexity,
        )

    async def run_interactive_chat(self, args, index_path: str):
        """Run interactive chat with the video-frame index."""
        chat = self._create_video_chat(args, index_path)
        session = create_rag_session(
            app_name=self.name.lower().replace(" ", "_"), data_description=self.name
        )

        def handle_query(query: str):
            response = chat.ask(
                query,
                top_k=args.top_k,
                complexity=args.search_complexity,
                recompute_embeddings=False,
                llm_kwargs=self._llm_kwargs(args),
            )
            print(f"\nAssistant: {response}\n")

        session.run_interactive_loop(handle_query)

    async def run_single_query(self, args, index_path: str, query: str):
        """Run a single text-to-video-frame query without recomputing embeddings."""
        chat = self._create_video_chat(args, index_path)

        print(f"\n[Query]: \033[36m{query}\033[0m")
        response = chat.ask(
            query,
            top_k=args.top_k,
            complexity=args.search_complexity,
            recompute_embeddings=False,
            llm_kwargs=self._llm_kwargs(args),
        )
        print(f"\n[Response]: \033[36m{response}\033[0m")


def main():
    """Main entry point for the video RAG application."""
    import asyncio

    app = VideoRAG()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
