#!/usr/bin/env python3
"""
CLIP Image RAG Application

This application enables RAG (Retrieval-Augmented Generation) on images using CLIP embeddings.
You can index a directory of images and search them using text queries.

Usage:
    python -m apps.image_rag --image-dir ./my_images/ --query "a sunset over mountains"
    python -m apps.image_rag --image-dir ./my_images/ --interactive
"""

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from leann.api import LeannChat
from leann.registry import register_project_directory
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from apps.base_rag_example import BaseRAGExample, create_rag_session

DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]


def normalize_image_extensions(extensions: Iterable[str]) -> set[str]:
    """Normalize image extensions to lowercase dot-prefixed suffixes."""
    normalized = set()
    for extension in extensions:
        stripped = extension.strip().lower()
        if not stripped:
            continue
        normalized.add(stripped if stripped.startswith(".") else f".{stripped}")
    return normalized


def discover_image_files(image_dir: Path, extensions: Iterable[str]) -> list[Path]:
    """Return image files in deterministic relative-path order."""
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise ValueError(f"Image path is not a directory: {image_dir}")

    allowed_extensions = normalize_image_extensions(extensions)
    if not allowed_extensions:
        raise ValueError("At least one image extension must be provided.")

    image_files = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
    return sorted(image_files, key=lambda path: path.relative_to(image_dir).as_posix().lower())


def image_passage_id(image_dir: Path, image_path: Path) -> str:
    """Create a stable passage ID from the image path relative to the indexed directory."""
    relative_path = image_path.relative_to(image_dir).as_posix()
    digest = hashlib.sha256(relative_path.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"image-{digest[:16]}"


def image_record_text(image_path: Path, image_dir: Path) -> str:
    """Create the text payload stored alongside each image vector."""
    relative_path = image_path.relative_to(image_dir).as_posix()
    return f"Image: {image_path.name}\nRelative path: {relative_path}\nPath: {image_path}"


def image_record_metadata(
    image_path: Path, image_dir: Path, embedding_model: str
) -> dict[str, Any]:
    """Create JSON-serializable metadata for an indexed image."""
    relative_path = image_path.relative_to(image_dir).as_posix()
    stat = image_path.stat()
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "id": image_passage_id(image_dir, image_path),
        "source": str(image_path),
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_dir": str(image_dir),
        "relative_path": relative_path,
        "extension": image_path.suffix.lower(),
        "size_bytes": stat.st_size,
        "width": width,
        "height": height,
        "media_type": "image",
        "embedding_model": embedding_model,
    }


class ImageRAG(BaseRAGExample):
    """
    RAG application for images using CLIP embeddings.

    This class provides a complete RAG pipeline for image data, including
    CLIP embedding generation, indexing, and text-based image search.
    """

    def __init__(self):
        super().__init__(
            name="Image RAG",
            description="RAG application for images using CLIP embeddings",
            default_index_name="image_index",
        )
        # Override default embedding model to use CLIP
        self.embedding_model_default = "clip-ViT-L-14"
        self.embedding_mode_default = "sentence-transformers"
        self._image_data: list[dict] = []

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add image-specific arguments."""
        image_group = parser.add_argument_group("Image Parameters")
        image_group.add_argument(
            "--image-dir",
            type=str,
            required=True,
            help="Directory containing images to index",
        )
        image_group.add_argument(
            "--image-extensions",
            type=str,
            nargs="+",
            default=DEFAULT_IMAGE_EXTENSIONS,
            help="Image file extensions to process (default: .jpg .jpeg .png .gif .bmp .webp)",
        )
        image_group.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for CLIP embedding generation (default: 32)",
        )

    async def load_data(self, args) -> list[dict[str, Any]]:
        """Load images, generate CLIP embeddings, and return text descriptions."""
        self._image_data = self._load_images_and_embeddings(args)
        return [entry["text"] for entry in self._image_data]

    def _load_images_and_embeddings(self, args) -> list[dict]:
        """Helper to process images and produce embeddings/metadata."""
        image_dir = Path(args.image_dir).expanduser().resolve()
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be greater than 0")

        print(f"📸 Loading images from {image_dir}...")

        image_files = discover_image_files(image_dir, args.image_extensions)

        if not image_files:
            raise ValueError(
                f"No images found in {image_dir} with extensions {args.image_extensions}"
            )

        print(f"✅ Found {len(image_files)} images")

        # Limit if max_items is set
        if args.max_items > 0:
            image_files = image_files[: args.max_items]
            print(f"📊 Processing {len(image_files)} images (limited by --max-items)")

        # Load CLIP model
        print("🔍 Loading CLIP model...")
        embedding_model = getattr(args, "embedding_model", self.embedding_model_default)
        model = self._load_clip_model(embedding_model)

        # Process images and generate embeddings
        print("🖼️  Processing images and generating embeddings...")
        image_data = []
        batch_images = []
        batch_paths = []

        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                with Image.open(image_path) as raw_image:
                    image = raw_image.convert("RGB").copy()
                batch_images.append(image)
                batch_paths.append(image_path)

                # Process in batches
                if len(batch_images) >= args.batch_size:
                    pending_images = batch_images
                    pending_paths = batch_paths
                    batch_images = []
                    batch_paths = []
                    image_data.extend(
                        self._encode_image_batch(
                            model, pending_images, pending_paths, image_dir, embedding_model
                        )
                    )

            except Exception as e:
                print(f"⚠️  Failed to process {image_path}: {e}")
                continue

        # Process remaining images
        if batch_images:
            image_data.extend(
                self._encode_image_batch(
                    model, batch_images, batch_paths, image_dir, embedding_model
                )
            )

        print(f"✅ Processed {len(image_data)} images")
        return image_data

    def _load_clip_model(self, embedding_model: str):
        """Load the configured CLIP encoder. Kept injectable for tests."""
        return SentenceTransformer(embedding_model)

    def _encode_image_batch(
        self,
        model,
        images: list[Image.Image],
        image_paths: list[Path],
        image_dir: Path,
        embedding_model: str,
    ) -> list[dict[str, Any]]:
        """Encode one image batch and attach stable IDs/metadata."""
        embeddings = model.encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=len(images),
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.shape[0] != len(image_paths):
            raise RuntimeError(
                f"CLIP encoder returned {embedding_array.shape[0]} embeddings for "
                f"{len(image_paths)} images."
            )

        records = []
        for img_path, embedding in zip(image_paths, embedding_array):
            metadata = image_record_metadata(img_path, image_dir, embedding_model)
            records.append(
                {
                    "id": metadata["id"],
                    "text": image_record_text(img_path, image_dir),
                    "metadata": metadata,
                    "embedding": embedding.astype(np.float32),
                }
            )
        return records

    async def build_index(self, args, texts: list[dict[str, Any]]) -> str:
        """Build index using pre-computed CLIP embeddings."""
        from leann.api import LeannBuilder

        if not self._image_data or len(self._image_data) != len(texts):
            raise RuntimeError("No image data found. Make sure load_data() ran successfully.")

        print("🔨 Building LEANN index with CLIP embeddings...")
        embedding_model = getattr(args, "embedding_model", self.embedding_model_default)
        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=embedding_model,
            embedding_mode=self.embedding_mode_default,
            is_recompute=False,
            distance_metric="cosine",
            graph_degree=args.graph_degree,
            build_complexity=args.build_complexity,
            is_compact=not args.no_compact,
        )

        for text, data in zip(texts, self._image_data):
            builder.add_text(text=text, metadata=data["metadata"])

        ids = [data["id"] for data in self._image_data]
        embeddings = np.array([data["embedding"] for data in self._image_data], dtype=np.float32)

        index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
        builder.build_index_from_arrays(index_path, ids, embeddings)
        register_project_directory(Path.cwd())
        print(f"✅ Index built successfully at {index_path}")
        return index_path

    def _llm_kwargs(self, args) -> dict[str, Any]:
        """Return optional LLM generation kwargs supported by the shared examples."""
        llm_kwargs = {}
        if hasattr(args, "thinking_budget") and args.thinking_budget:
            llm_kwargs["thinking_budget"] = args.thinking_budget
        return llm_kwargs

    def _create_image_chat(self, args, index_path: str) -> LeannChat:
        """Create a chat wrapper for an image index."""
        return LeannChat(
            index_path,
            llm_config=self.get_llm_config(args),
            system_prompt=(
                "You are a helpful assistant that answers questions about images indexed "
                "with CLIP embeddings."
            ),
            complexity=args.search_complexity,
        )

    async def run_interactive_chat(self, args, index_path: str):
        """Run interactive chat with the image index."""
        chat = self._create_image_chat(args, index_path)
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
        """Run a single text-to-image query without recomputing image embeddings."""
        chat = self._create_image_chat(args, index_path)

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
    """Main entry point for the image RAG application."""
    import asyncio

    app = ImageRAG()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
