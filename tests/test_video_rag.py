import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from apps import video_rag
from apps.video_rag import VideoFrameRecord, VideoRAG


def _write_video_placeholder(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real video; tests mock decoding")


def test_discover_video_files_is_deterministic_and_case_insensitive(tmp_path):
    video_dir = tmp_path / "videos"
    _write_video_placeholder(video_dir / "b" / "two.MP4")
    _write_video_placeholder(video_dir / "A.mov")
    (video_dir / "notes.txt").write_text("not a video", encoding="utf-8")

    files = video_rag.discover_video_files(video_dir, ["mp4", ".mov"])

    assert [path.relative_to(video_dir).as_posix() for path in files] == ["A.mov", "b/two.MP4"]


def test_video_frame_metadata_has_stable_id_and_frame_fields(tmp_path):
    video_dir = tmp_path / "videos"
    video_path = video_dir / "nested" / "clip.MP4"
    _write_video_placeholder(video_path)

    metadata = video_rag.video_frame_metadata(
        video_path=video_path,
        video_dir=video_dir,
        embedding_model="clip-test",
        frame_index=42,
        timestamp_seconds=1.4,
        fps=30.0,
        width=1920,
        height=1080,
        duration_seconds=10.0,
    )

    assert metadata["id"] == video_rag.video_frame_passage_id(video_dir, video_path, 42, 1400)
    assert metadata["relative_path"] == "nested/clip.MP4"
    assert metadata["extension"] == ".mp4"
    assert metadata["media_type"] == "video_frame"
    assert metadata["timestamp_ms"] == 1400
    assert metadata["frame_index"] == 42
    assert metadata["width"] == 1920
    assert metadata["height"] == 1080
    assert metadata["embedding_model"] == "clip-test"


def test_load_videos_uses_fake_encoder_and_respects_limits(tmp_path):
    video_dir = tmp_path / "videos"
    _write_video_placeholder(video_dir / "first.mp4")
    _write_video_placeholder(video_dir / "second.mp4")

    class FakeModel:
        def encode(self, frames, **kwargs):
            assert kwargs["normalize_embeddings"] is True
            return np.array([[1.0, 0.0] for _ in frames], dtype=np.float32)

    class TestVideoRAG(VideoRAG):
        def _load_clip_model(self, embedding_model: str):
            return FakeModel()

        def _extract_video_frames(
            self,
            video_path,
            sample_interval_seconds: float,
            max_frames_per_video: int,
        ) -> Iterator[VideoFrameRecord]:
            assert sample_interval_seconds == 2.0
            for frame_index in range(3):
                if max_frames_per_video > 0 and frame_index >= max_frames_per_video:
                    break
                yield VideoFrameRecord(
                    image=Image.new("RGB", (2, 2), (frame_index, 0, 0)),
                    frame_index=frame_index,
                    timestamp_seconds=float(frame_index * 2),
                    fps=30.0,
                    width=2,
                    height=2,
                    duration_seconds=6.0,
                )

    app = TestVideoRAG()

    records = app._load_videos_and_embeddings(
        SimpleNamespace(
            video_dir=str(video_dir),
            video_extensions=[".mp4"],
            max_items=1,
            max_frames_per_video=2,
            sample_interval_seconds=2.0,
            batch_size=8,
            embedding_model="clip-test",
        )
    )

    assert len(records) == 2
    assert records[0]["id"] == records[0]["metadata"]["id"]
    assert records[0]["metadata"]["embedding_model"] == "clip-test"
    assert records[0]["embedding"].dtype == np.float32
    assert records[1]["metadata"]["frame_index"] == 1


@pytest.mark.parametrize(
    ("sample_interval_seconds", "batch_size", "max_frames_per_video", "message"),
    [
        (0.0, 8, 2, "sample-interval-seconds"),
        (1.0, 0, 2, "batch-size"),
        (1.0, 8, 0, "max-frames-per-video"),
        (1.0, 8, -2, "max-frames-per-video"),
    ],
)
def test_load_videos_rejects_invalid_args(
    tmp_path, sample_interval_seconds, batch_size, max_frames_per_video, message
):
    video_dir = tmp_path / "videos"
    _write_video_placeholder(video_dir / "first.mp4")
    app = VideoRAG()

    with pytest.raises(ValueError, match=message):
        app._load_videos_and_embeddings(
            SimpleNamespace(
                video_dir=str(video_dir),
                video_extensions=[".mp4"],
                max_items=-1,
                max_frames_per_video=max_frames_per_video,
                sample_interval_seconds=sample_interval_seconds,
                batch_size=batch_size,
                embedding_model="clip-test",
            )
        )


def test_build_index_uses_stable_ids_and_precomputed_arrays(tmp_path, monkeypatch):
    calls = {}

    class FakeBuilder:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.texts = []

        def add_text(self, text, metadata=None):
            self.texts.append((text, metadata))

        def build_index_from_arrays(self, index_path, ids, embeddings):
            calls["build"] = {
                "index_path": index_path,
                "ids": ids,
                "embeddings": embeddings,
                "texts": self.texts,
            }

    monkeypatch.setattr(
        video_rag, "register_project_directory", lambda path: calls.setdefault("registered", path)
    )
    monkeypatch.setattr("leann.api.LeannBuilder", FakeBuilder)

    app = VideoRAG()
    app._video_data = [
        {
            "id": "video-frame-alpha",
            "text": "Video frame: alpha.mp4",
            "metadata": {"id": "video-frame-alpha", "relative_path": "alpha.mp4"},
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
        }
    ]

    index_path = asyncio.run(
        app.build_index(
            SimpleNamespace(
                index_dir=str(tmp_path / "index"),
                backend_name="hnsw",
                graph_degree=16,
                build_complexity=32,
                no_compact=False,
                embedding_model="clip-test",
            ),
            cast(list[dict[str, Any]], [{"text": "Video frame: alpha.mp4"}]),
        ),
    )

    assert index_path.endswith("video_index.leann")
    assert calls["init"]["embedding_model"] == "clip-test"
    assert calls["init"]["embedding_mode"] == "sentence-transformers"
    assert calls["init"]["is_recompute"] is False
    assert calls["init"]["distance_metric"] == "cosine"
    assert calls["build"]["ids"] == ["video-frame-alpha"]
    assert calls["build"]["embeddings"].shape == (1, 2)
    assert calls["build"]["texts"] == [
        ("Video frame: alpha.mp4", {"id": "video-frame-alpha", "relative_path": "alpha.mp4"})
    ]


def test_single_query_disables_recompute_for_video_index(monkeypatch):
    calls = {}

    class FakeChat:
        def __init__(self, index_path, **kwargs):
            calls["init"] = {"index_path": index_path, **kwargs}

        def ask(self, query, **kwargs):
            calls["ask"] = {"query": query, **kwargs}
            return "found frame"

    monkeypatch.setattr(video_rag, "LeannChat", FakeChat)

    app = VideoRAG()
    args = SimpleNamespace(
        llm="simulated",
        llm_model=None,
        llm_host=None,
        llm_api_key=None,
        llm_api_base=None,
        search_complexity=64,
        top_k=3,
        thinking_budget=None,
    )

    asyncio.run(app.run_single_query(args, "video_index.leann", "whiteboard"))

    assert calls["init"]["index_path"] == "video_index.leann"
    assert calls["ask"]["query"] == "whiteboard"
    assert calls["ask"]["top_k"] == 3
    assert calls["ask"]["complexity"] == 64
    assert calls["ask"]["recompute_embeddings"] is False
