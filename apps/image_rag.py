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
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from apps.base_rag_example import BaseRAGExample


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
            default=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
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
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")

        print(f"📸 Loading images from {image_dir}...")

        # Find all image files
        image_files = []
        for ext in args.image_extensions:
            image_files.extend(image_dir.rglob(f"*{ext}"))
            image_files.extend(image_dir.rglob(f"*{ext.upper()}"))

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
        model = SentenceTransformer(self.embedding_model_default)

        # Process images and generate embeddings
        print("🖼️  Processing images and generating embeddings...")
        image_data = []
        batch_images = []
        batch_paths = []

        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                image = Image.open(image_path).convert("RGB")
                batch_images.append(image)
                batch_paths.append(image_path)

                # Process in batches
                if len(batch_images) >= args.batch_size:
                    embeddings = model.encode(
                        batch_images,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=args.batch_size,
                        show_progress_bar=False,
                    )

                    for img_path, embedding in zip(batch_paths, embeddings):
                        image_data.append(
                            {
                                "text": f"Image: {img_path.name}\nPath: {img_path}",
                                "metadata": {
                                    "image_path": str(img_path),
                                    "image_name": img_path.name,
                                    "image_dir": str(image_dir),
                                },
                                "embedding": embedding.astype(np.float32),
                            }
                        )

                    batch_images = []
                    batch_paths = []

            except Exception as e:
                print(f"⚠️  Failed to process {image_path}: {e}")
                continue

        # Process remaining images
        if batch_images:
            embeddings = model.encode(
                batch_images,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=len(batch_images),
                show_progress_bar=False,
            )

            for img_path, embedding in zip(batch_paths, embeddings):
                image_data.append(
                    {
                        "text": f"Image: {img_path.name}\nPath: {img_path}",
                        "metadata": {
                            "image_path": str(img_path),
                            "image_name": img_path.name,
                            "image_dir": str(image_dir),
                        },
                        "embedding": embedding.astype(np.float32),
                    }
                )

        print(f"✅ Processed {len(image_data)} images")
        return image_data

    async def build_index(self, args, texts: list[dict[str, Any]]) -> str:
        """Build index using pre-computed CLIP embeddings."""
        from leann.api import LeannBuilder

        if not self._image_data or len(self._image_data) != len(texts):
            raise RuntimeError("No image data found. Make sure load_data() ran successfully.")

        print("🔨 Building LEANN index with CLIP embeddings...")
        builder = LeannBuilder(
            backend_name=args.backend_name,
            embedding_model=self.embedding_model_default,
            embedding_mode=self.embedding_mode_default,
            is_recompute=False,
            distance_metric="cosine",
            graph_degree=args.graph_degree,
            build_complexity=args.build_complexity,
            is_compact=not args.no_compact,
        )

        for text, data in zip(texts, self._image_data):
            builder.add_text(text=text, metadata=data["metadata"])

        ids = [str(i) for i in range(len(self._image_data))]
        embeddings = np.array([data["embedding"] for data in self._image_data], dtype=np.float32)

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump((ids, embeddings), f)
            pkl_path = f.name

        try:
            index_path = str(Path(args.index_dir) / f"{self.default_index_name}.leann")
            builder.build_index_from_embeddings(index_path, pkl_path)
            print(f"✅ Index built successfully at {index_path}")
            return index_path
        finally:
            Path(pkl_path).unlink()


def main():
    """Main entry point for the image RAG application."""
    import asyncio

    app = ImageRAG()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
