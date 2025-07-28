#!/usr/bin/env python3
"""
Basic functionality tests for CI pipeline.
These tests verify that the built packages work correctly.
"""

import sys
import numpy as np
from pathlib import Path


def test_imports():
    """Test that all packages can be imported."""
    print("Testing package imports...")

    try:
        import leann

        print("✅ leann imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import leann: {e}")
        return False

    try:
        import leann_backend_hnsw

        print("✅ leann_backend_hnsw imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import leann_backend_hnsw: {e}")
        return False

    try:
        import leann_backend_diskann

        print("✅ leann_backend_diskann imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import leann_backend_diskann: {e}")
        return False

    # Test C++ extensions
    try:
        from leann_backend_hnsw import faiss

        print("✅ FAISS loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load FAISS: {e}")
        return False

    try:
        import leann_backend_diskann.diskann_backend

        print("✅ DiskANN loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load DiskANN: {e}")
        return False

    return True


def test_hnsw_basic():
    """Test basic HNSW functionality."""
    print("\nTesting HNSW basic functionality...")

    try:
        from leann.api import LeannBuilder

        # Test with small random data
        data = np.random.rand(100, 768).astype(np.float32)
        texts = [f"Text {i}" for i in range(100)]

        builder = LeannBuilder(
            backend_name="hnsw",
            embedding_model="facebook/contriever",
            embedding_mode="sentence-transformers",
            dimensions=768,
            M=16,
            efConstruction=200,
        )

        # Build in-memory index
        index = builder.build_memory_index(data, texts)
        print("✅ HNSW index built successfully")

        # Test search
        results = index.search(["test query"], top_k=5)
        print(f"✅ Search completed, found {len(results[0])} results")

        return True
    except Exception as e:
        print(f"❌ HNSW test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_diskann_basic():
    """Test basic DiskANN functionality."""
    print("\nTesting DiskANN basic functionality...")

    try:
        from leann.api import LeannBuilder
        import tempfile
        import shutil

        # Test with small random data
        data = np.random.rand(100, 768).astype(np.float32)
        texts = [f"Text {i}" for i in range(100)]

        # Create temporary directory for index
        temp_dir = tempfile.mkdtemp()
        index_path = str(Path(temp_dir) / "test.diskann")

        try:
            builder = LeannBuilder(
                backend_name="diskann",
                embedding_model="facebook/contriever",
                embedding_mode="sentence-transformers",
                dimensions=768,
                num_neighbors=32,
                search_list_size=50,
            )

            # Build disk index
            builder.build_index(index_path, texts=texts, embeddings=data)
            print("✅ DiskANN index built successfully")

            # Test search
            from leann.api import LeannSearcher

            searcher = LeannSearcher(index_path)
            results = searcher.search(["test query"], top_k=5)
            print(f"✅ DiskANN search completed, found {len(results[0])} results")

            return True
        finally:
            # Clean up
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        print(f"❌ DiskANN test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running CI Basic Functionality Tests")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test HNSW
    if not test_hnsw_basic():
        all_passed = False

    # Test DiskANN
    if not test_diskann_basic():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
