"""
LAION Multimodal Benchmark Setup Script
Downloads LAION subset and builds LEANN index with sentence embeddings
"""

import argparse
import asyncio
import io
import json
import os
import pickle
import time
from pathlib import Path

import aiohttp
import numpy as np
from datasets import load_dataset
from leann import LeannBuilder
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class LAIONSetup:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "laion_images"
        self.metadata_file = self.data_dir / "laion_metadata.jsonl"

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)

    async def download_single_image(self, session, sample_data, semaphore, progress_bar):
        """Download a single image asynchronously"""
        async with semaphore:  # Limit concurrent downloads
            try:
                image_url = sample_data["url"]
                image_path = sample_data["image_path"]

                # Skip if already exists
                if os.path.exists(image_path):
                    progress_bar.update(1)
                    return sample_data

                async with session.get(image_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.read()

                        # Verify it's a valid image
                        try:
                            img = Image.open(io.BytesIO(content))
                            img = img.convert("RGB")
                            img.save(image_path, "JPEG")
                            progress_bar.update(1)
                            return sample_data
                        except Exception:
                            progress_bar.update(1)
                            return None  # Skip invalid images
                    else:
                        progress_bar.update(1)
                        return None

            except Exception:
                progress_bar.update(1)
                return None

    def download_laion_subset(self, num_samples: int = 1000):
        """Download LAION subset from HuggingFace datasets with async parallel downloading"""
        print(f"📥 Downloading LAION subset ({num_samples} samples)...")

        # Load LAION-400M subset from HuggingFace
        print("🤗 Loading from HuggingFace datasets...")
        dataset = load_dataset("laion/laion400m", split="train", streaming=True)

        # Collect sample metadata first (fast)
        print("📋 Collecting sample metadata...")
        candidates = []
        for sample in dataset:
            if len(candidates) >= num_samples * 3:  # Get 3x more candidates in case some fail
                break

            image_url = sample.get("url", "")
            caption = sample.get("caption", "")

            if not image_url or not caption:
                continue

            image_filename = f"laion_{len(candidates):06d}.jpg"
            image_path = self.images_dir / image_filename

            candidate = {
                "id": f"laion_{len(candidates):06d}",
                "url": image_url,
                "caption": caption,
                "image_path": str(image_path),
                "width": sample.get("original_width", 512),
                "height": sample.get("original_height", 512),
                "similarity": sample.get("similarity", 0.0),
            }
            candidates.append(candidate)

        print(
            f"📊 Collected {len(candidates)} candidates, downloading {num_samples} in parallel..."
        )

        # Download images in parallel
        async def download_batch():
            semaphore = asyncio.Semaphore(20)  # Limit to 20 concurrent downloads
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            timeout = aiohttp.ClientTimeout(total=30)

            progress_bar = tqdm(total=len(candidates[: num_samples * 2]), desc="Downloading images")

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                tasks = []
                for candidate in candidates[: num_samples * 2]:  # Try 2x more than needed
                    task = self.download_single_image(session, candidate, semaphore, progress_bar)
                    tasks.append(task)

                # Wait for all downloads
                results = await asyncio.gather(*tasks, return_exceptions=True)
                progress_bar.close()

                # Filter successful downloads
                successful = [r for r in results if r is not None and not isinstance(r, Exception)]
                return successful[:num_samples]

        # Run async download
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            samples = loop.run_until_complete(download_batch())
        finally:
            loop.close()

        # Save metadata
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        print(f"✅ Downloaded {len(samples)} real LAION samples with async parallel downloading")
        return samples

    def generate_clip_image_embeddings(self, samples: list[dict]):
        """Generate CLIP image embeddings for downloaded images"""
        print("🔍 Generating CLIP image embeddings...")

        # Load sentence-transformers CLIP (ViT-L/14, 768-dim) for image embeddings
        # This single model can encode both images and text.
        model = SentenceTransformer("clip-ViT-L-14")

        embeddings = []
        valid_samples = []

        for sample in tqdm(samples, desc="Processing images"):
            try:
                # Load image
                image_path = sample["image_path"]
                image = Image.open(image_path).convert("RGB")

                # Encode image to 768-dim embedding via sentence-transformers (normalized)
                vec = model.encode(
                    [image],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=1,
                    show_progress_bar=False,
                )[0]
                embeddings.append(vec.astype(np.float32))
                valid_samples.append(sample)

            except Exception as e:
                print(f"  ⚠️ Failed to process {sample['id']}: {e}")
                # Skip invalid images

        embeddings = np.array(embeddings, dtype=np.float32)

        # Save embeddings
        embeddings_file = self.data_dir / "clip_image_embeddings.npy"
        np.save(embeddings_file, embeddings)
        print(f"✅ Generated {len(embeddings)} image embeddings, shape: {embeddings.shape}")

        return embeddings, valid_samples

    def build_faiss_baseline(
        self, embeddings: np.ndarray, samples: list[dict], output_dir: str = "baseline"
    ):
        """Build FAISS flat baseline using CLIP image embeddings"""
        print("🔨 Building FAISS Flat baseline...")

        from leann_backend_hnsw import faiss

        os.makedirs(output_dir, exist_ok=True)
        baseline_path = os.path.join(output_dir, "faiss_flat.index")
        metadata_path = os.path.join(output_dir, "metadata.pkl")

        if os.path.exists(baseline_path) and os.path.exists(metadata_path):
            print(f"✅ Baseline already exists at {baseline_path}")
            return baseline_path

        # Extract image IDs (must be present)
        if not samples or "id" not in samples[0]:
            raise KeyError("samples missing 'id' field for FAISS baseline")
        image_ids: list[str] = [str(sample["id"]) for sample in samples]

        print(f"📐 Embedding shape: {embeddings.shape}")
        print(f"📄 Processing {len(image_ids)} images")

        # Build FAISS flat index
        print("🏗️ Building FAISS IndexFlatIP...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Add embeddings to flat index
        embeddings_f32 = embeddings.astype(np.float32)
        index.add(embeddings_f32.shape[0], faiss.swig_ptr(embeddings_f32))

        # Save index and metadata
        faiss.write_index(index, baseline_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(image_ids, f)

        print(f"✅ FAISS baseline saved to {baseline_path}")
        print(f"✅ Metadata saved to {metadata_path}")
        print(f"📊 Total vectors: {index.ntotal}")

        return baseline_path

    def create_leann_passages(self, samples: list[dict]):
        """Create LEANN-compatible passages from LAION data"""
        print("📝 Creating LEANN passages...")

        passages_file = self.data_dir / "laion_passages.jsonl"

        with open(passages_file, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                passage = {
                    "id": sample["id"],
                    "text": sample["caption"],  # Use caption as searchable text
                    "metadata": {
                        "image_url": sample["url"],
                        "image_path": sample.get("image_path", ""),
                        "width": sample["width"],
                        "height": sample["height"],
                        "similarity": sample["similarity"],
                        "image_index": i,  # Index for embedding lookup
                    },
                }
                f.write(json.dumps(passage) + "\n")

        print(f"✅ Created {len(samples)} passages")
        return passages_file

    def build_compact_index(
        self, passages_file: Path, embeddings: np.ndarray, index_path: str, backend: str = "hnsw"
    ):
        """Build compact LEANN index with CLIP embeddings (recompute=True, compact=True)"""
        print(f"🏗️ Building compact LEANN index with {backend} backend...")

        start_time = time.time()

        # Save CLIP embeddings (npy) and also a pickle with (ids, embeddings)
        npy_path = self.data_dir / "clip_image_embeddings.npy"
        np.save(npy_path, embeddings)
        print(f"💾 Saved CLIP embeddings to {npy_path}")

        # Prepare ids in the same order as passages_file (matches embeddings order)
        ids: list[str] = []
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ids.append(str(rec["id"]))

        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"IDs count ({len(ids)}) does not match embeddings ({embeddings.shape[0]})."
            )

        pkl_path = self.data_dir / "clip_image_embeddings.pkl"
        with open(pkl_path, "wb") as pf:
            pickle.dump((ids, embeddings.astype(np.float32)), pf)
        print(f"💾 Saved (ids, embeddings) pickle to {pkl_path}")

        # Initialize builder - compact with recompute
        # Note: For multimodal case, we need to handle embeddings differently
        # Let's try using sentence-transformers mode but with custom embeddings
        builder = LeannBuilder(
            backend_name=backend,
            # Use CLIP text encoder (ViT-L/14) to match image space (768-dim)
            embedding_model="clip-ViT-L-14",
            embedding_mode="sentence-transformers",
            # HNSW params (or forwarded to chosen backend)
            graph_degree=32,
            complexity=64,
            # Compact/pruned with recompute at query time
            is_recompute=True,
            is_compact=True,
            distance_metric="cosine",  # CLIP uses normalized vectors; cosine is appropriate
            num_threads=4,
        )

        # Add passages (text + metadata)
        print("📚 Adding passages...")
        self._add_passages_with_embeddings(builder, passages_file, embeddings)

        print(f"🔨 Building compact index at {index_path} from precomputed embeddings...")
        builder.build_index_from_embeddings(index_path, str(pkl_path))

        build_time = time.time() - start_time
        print(f"✅ Compact index built in {build_time:.2f}s")

        # Analyze index size
        self._analyze_index_size(index_path)

        return index_path

    def build_non_compact_index(
        self, passages_file: Path, embeddings: np.ndarray, index_path: str, backend: str = "hnsw"
    ):
        """Build non-compact LEANN index with CLIP embeddings (recompute=False, compact=False)"""
        print(f"🏗️ Building non-compact LEANN index with {backend} backend...")

        start_time = time.time()

        # Ensure embeddings are saved (npy + pickle)
        npy_path = self.data_dir / "clip_image_embeddings.npy"
        if not npy_path.exists():
            np.save(npy_path, embeddings)
            print(f"💾 Saved CLIP embeddings to {npy_path}")
        # Prepare ids in same order as passages_file
        ids: list[str] = []
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ids.append(str(rec["id"]))
        if len(ids) != embeddings.shape[0]:
            raise ValueError(
                f"IDs count ({len(ids)}) does not match embeddings ({embeddings.shape[0]})."
            )
        pkl_path = self.data_dir / "clip_image_embeddings.pkl"
        if not pkl_path.exists():
            with open(pkl_path, "wb") as pf:
                pickle.dump((ids, embeddings.astype(np.float32)), pf)
            print(f"💾 Saved (ids, embeddings) pickle to {pkl_path}")

        # Initialize builder - non-compact without recompute
        builder = LeannBuilder(
            backend_name=backend,
            embedding_model="clip-ViT-L-14",
            embedding_mode="sentence-transformers",
            graph_degree=32,
            complexity=64,
            is_recompute=False,  # Store embeddings (no recompute needed)
            is_compact=False,  # Store full index (not pruned)
            distance_metric="cosine",
            num_threads=4,
        )

        # Add passages - embeddings will be loaded from file
        print("📚 Adding passages...")
        self._add_passages_with_embeddings(builder, passages_file, embeddings)

        print(f"🔨 Building non-compact index at {index_path} from precomputed embeddings...")
        builder.build_index_from_embeddings(index_path, str(pkl_path))

        build_time = time.time() - start_time
        print(f"✅ Non-compact index built in {build_time:.2f}s")

        # Analyze index size
        self._analyze_index_size(index_path)

        return index_path

    def _add_passages_with_embeddings(self, builder, passages_file: Path, embeddings: np.ndarray):
        """Helper to add passages with pre-computed CLIP embeddings"""
        with open(passages_file, encoding="utf-8") as f:
            for line in tqdm(f, desc="Adding passages"):
                if line.strip():
                    passage = json.loads(line)

                    # Add image metadata - LEANN will handle embeddings separately
                    # Note: We store image metadata and caption text for searchability
                    # Important: ensure passage ID in metadata matches vector ID
                    builder.add_text(
                        text=passage["text"],  # Image caption for searchability
                        metadata={**passage["metadata"], "id": passage["id"]},
                    )

    def _analyze_index_size(self, index_path: str):
        """Analyze index file sizes"""
        print("📏 Analyzing index sizes...")

        index_path = Path(index_path)
        index_dir = index_path.parent
        index_name = index_path.name  # e.g., laion_index.leann
        index_prefix = index_path.stem  # e.g., laion_index

        files = [
            (f"{index_prefix}.index", ".index", "core"),
            (f"{index_name}.meta.json", ".meta.json", "core"),
            (f"{index_name}.ids.txt", ".ids.txt", "core"),
            (f"{index_name}.passages.jsonl", ".passages.jsonl", "passages"),
            (f"{index_name}.passages.idx", ".passages.idx", "passages"),
        ]

        def _fmt_size(bytes_val: int) -> str:
            if bytes_val < 1024:
                return f"{bytes_val} B"
            kb = bytes_val / 1024
            if kb < 1024:
                return f"{kb:.1f} KB"
            mb = kb / 1024
            if mb < 1024:
                return f"{mb:.2f} MB"
            gb = mb / 1024
            return f"{gb:.2f} GB"

        total_index_only_mb = 0.0
        total_all_mb = 0.0
        for filename, label, group in files:
            file_path = index_dir / filename
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                print(f"  {label}: {_fmt_size(size_bytes)}")
                size_mb = size_bytes / (1024 * 1024)
                total_all_mb += size_mb
                if group == "core":
                    total_index_only_mb += size_mb
            else:
                print(f"  {label}: (missing)")
        print(f"  Total (index only, exclude passages): {total_index_only_mb:.2f} MB")
        print(f"  Total (including passages): {total_all_mb:.2f} MB")

    def create_evaluation_queries(self, samples: list[dict], num_queries: int = 200):
        """Create evaluation queries from captions"""
        print(f"📝 Creating {num_queries} evaluation queries...")

        # Sample random captions as queries
        import random

        random.seed(42)  # For reproducibility

        query_samples = random.sample(samples, min(num_queries, len(samples)))

        queries_file = self.data_dir / "evaluation_queries.jsonl"
        with open(queries_file, "w", encoding="utf-8") as f:
            for sample in query_samples:
                query = {
                    "id": sample["id"],
                    "query": sample["caption"],
                    "ground_truth_id": sample["id"],  # For potential recall evaluation
                }
                f.write(json.dumps(query) + "\n")

        print(f"✅ Created {len(query_samples)} evaluation queries")
        return queries_file


def main():
    parser = argparse.ArgumentParser(description="Setup LAION Multimodal Benchmark")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of LAION samples")
    parser.add_argument("--num-queries", type=int, default=50, help="Number of evaluation queries")
    parser.add_argument("--index-path", default="data/laion_index.leann", help="Output index path")
    parser.add_argument(
        "--backend", default="hnsw", choices=["hnsw", "diskann"], help="LEANN backend"
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip LAION dataset download")
    parser.add_argument("--skip-build", action="store_true", help="Skip index building")

    args = parser.parse_args()

    print("🚀 Setting up LAION Multimodal Benchmark")
    print("=" * 50)

    try:
        # Initialize setup
        setup = LAIONSetup(args.data_dir)

        # Step 1: Download LAION subset
        if not args.skip_download:
            print("\n📦 Step 1: Download LAION subset")
            samples = setup.download_laion_subset(args.num_samples)

            # Step 2: Generate CLIP image embeddings
            print("\n🔍 Step 2: Generate CLIP image embeddings")
            embeddings, valid_samples = setup.generate_clip_image_embeddings(samples)

            # Step 3: Create LEANN passages (image metadata with embeddings)
            print("\n📝 Step 3: Create LEANN passages")
            passages_file = setup.create_leann_passages(valid_samples)
        else:
            print("⏭️  Skipping LAION dataset download")
            # Load existing data
            passages_file = setup.data_dir / "laion_passages.jsonl"
            embeddings_file = setup.data_dir / "clip_image_embeddings.npy"

            if not passages_file.exists() or not embeddings_file.exists():
                raise FileNotFoundError(
                    "Passages or embeddings file not found. Run without --skip-download first."
                )

            embeddings = np.load(embeddings_file)
            print(f"📊 Loaded {len(embeddings)} embeddings from {embeddings_file}")

        # Step 4: Build LEANN indexes (both compact and non-compact)
        if not args.skip_build:
            print("\n🏗️ Step 4: Build LEANN indexes with CLIP image embeddings")

            # Build compact index (production mode - small, recompute required)
            compact_index_path = args.index_path
            print(f"Building compact index: {compact_index_path}")
            setup.build_compact_index(passages_file, embeddings, compact_index_path, args.backend)

            # Build non-compact index (comparison mode - large, fast search)
            non_compact_index_path = args.index_path.replace(".leann", "_noncompact.leann")
            print(f"Building non-compact index: {non_compact_index_path}")
            setup.build_non_compact_index(
                passages_file, embeddings, non_compact_index_path, args.backend
            )

            # Step 5: Build FAISS flat baseline
            print("\n🔨 Step 5: Build FAISS flat baseline")
            if not args.skip_download:
                baseline_path = setup.build_faiss_baseline(embeddings, valid_samples)
            else:
                # Load valid_samples from passages file for FAISS baseline
                valid_samples = []
                with open(passages_file, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            passage = json.loads(line)
                            valid_samples.append({"id": passage["id"], "caption": passage["text"]})
                baseline_path = setup.build_faiss_baseline(embeddings, valid_samples)

            # Step 6: Create evaluation queries
            print("\n📝 Step 6: Create evaluation queries")
            queries_file = setup.create_evaluation_queries(valid_samples, args.num_queries)
        else:
            print("⏭️  Skipping index building")
            baseline_path = "data/baseline/faiss_index.bin"
            queries_file = setup.data_dir / "evaluation_queries.jsonl"

        print("\n🎉 Setup completed successfully!")
        print("📊 Summary:")
        if not args.skip_download:
            print(f"  Downloaded samples: {len(samples)}")
            print(f"  Valid samples with embeddings: {len(valid_samples)}")
        else:
            print(f"  Loaded {len(embeddings)} embeddings")

        if not args.skip_build:
            print(f"  Compact index: {compact_index_path}")
            print(f"  Non-compact index: {non_compact_index_path}")
            print(f"  FAISS baseline: {baseline_path}")
            print(f"  Queries: {queries_file}")

            print("\n🔧 Next steps:")
            print(f"  Run evaluation: python evaluate_laion.py --index {compact_index_path}")
            print(f"  Or compare with: python evaluate_laion.py --index {non_compact_index_path}")
        else:
            print("  Skipped building indexes")

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
