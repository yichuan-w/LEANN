"""
Test examples from README.md to ensure documentation is accurate.
"""

import os
import platform
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_readme_basic_example(backend_name):
    """Test the basic example from README.md with both backends."""
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")
    # Skip DiskANN on CI (Linux runners) due to C++ extension memory/hardware constraints
    if os.environ.get("CI") == "true" and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI due to resource constraints and instability")

    # This is the exact code from README (with smaller model for CI)
    from leann import LeannBuilder, LeannChat, LeannSearcher
    from leann.api import SearchResult

    with tempfile.TemporaryDirectory() as temp_dir:
        INDEX_PATH = str(Path(temp_dir) / f"demo_{backend_name}.leann")

        # Build an index
        # In CI, use a smaller model to avoid memory issues
        if os.environ.get("CI") == "true":
            builder = LeannBuilder(
                backend_name=backend_name,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model
                dimensions=384,  # Smaller dimensions
            )
        else:
            builder = LeannBuilder(backend_name=backend_name)
        builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
        builder.add_text("Tung Tung Tung Sahur called—they need their banana-crocodile hybrid back")
        builder.build_index(INDEX_PATH)

        # Verify index was created
        # The index path should be a directory containing index files
        index_dir = Path(INDEX_PATH).parent
        assert index_dir.exists()
        # Check that index files were created
        index_files = list(index_dir.glob(f"{Path(INDEX_PATH).stem}.*"))
        assert len(index_files) > 0

        # Search
        searcher = LeannSearcher(INDEX_PATH)
        results = searcher.search("fantastical AI-generated creatures", top_k=1)

        # Verify search results
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score != float("-inf"), (
            f"should return valid scores, got {results[0].score}"
        )
        # The second text about banana-crocodile should be more relevant
        assert "banana" in results[0].text or "crocodile" in results[0].text

        # Ensure we cleanup background embedding server
        searcher.cleanup()

        # Chat with your data (using simulated LLM to avoid external dependencies)
        chat = LeannChat(INDEX_PATH, llm_config={"type": "simulated"})
        response = chat.ask("How much storage does LEANN save?", top_k=1)

        # Verify chat works
        assert isinstance(response, str)
        assert len(response) > 0
        # Cleanup chat resources
        chat.cleanup()


def test_readme_imports():
    """Test that the imports shown in README work correctly."""
    # These are the imports shown in README
    from leann import LeannBuilder, LeannChat, LeannSearcher

    # Verify they are the correct types
    assert callable(LeannBuilder)
    assert callable(LeannSearcher)
    assert callable(LeannChat)


def test_backend_options():
    """Test different backend options mentioned in documentation."""
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    from leann import LeannBuilder

    with tempfile.TemporaryDirectory() as temp_dir:
        # Use smaller model in CI to avoid memory issues
        if os.environ.get("CI") == "true":
            model_args = {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
            }
        else:
            model_args = {}

        # Test HNSW backend (as shown in README)
        hnsw_path = str(Path(temp_dir) / "test_hnsw.leann")
        builder_hnsw = LeannBuilder(backend_name="hnsw", **model_args)
        builder_hnsw.add_text("Test document for HNSW backend")
        builder_hnsw.build_index(hnsw_path)
        assert Path(hnsw_path).parent.exists()
        assert len(list(Path(hnsw_path).parent.glob(f"{Path(hnsw_path).stem}.*"))) > 0

        # Test DiskANN backend (mentioned as available option)
        diskann_path = str(Path(temp_dir) / "test_diskann.leann")
        builder_diskann = LeannBuilder(backend_name="diskann", **model_args)
        builder_diskann.add_text("Test document for DiskANN backend")
        builder_diskann.build_index(diskann_path)
        assert Path(diskann_path).parent.exists()
        assert len(list(Path(diskann_path).parent.glob(f"{Path(diskann_path).stem}.*"))) > 0


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_llm_config_simulated(backend_name):
    """Test simulated LLM configuration option with both backends."""
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    # Skip DiskANN tests in CI due to hardware requirements
    if os.environ.get("CI") == "true" and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI - requires specific hardware and large memory")

    from leann import LeannBuilder, LeannChat

    with tempfile.TemporaryDirectory() as temp_dir:
        # Build a simple index
        index_path = str(Path(temp_dir) / f"test_{backend_name}.leann")
        # Use smaller model in CI to avoid memory issues
        if os.environ.get("CI") == "true":
            builder = LeannBuilder(
                backend_name=backend_name,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=384,
            )
        else:
            builder = LeannBuilder(backend_name=backend_name)
        builder.add_text("Test document for LLM testing")
        builder.build_index(index_path)

        # Test simulated LLM config
        llm_config = {"type": "simulated"}
        chat = LeannChat(index_path, llm_config=llm_config)
        response = chat.ask("What is this document about?", top_k=1)

        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.skip(reason="Requires HF model download and may timeout")
def test_llm_config_hf():
    """Test HuggingFace LLM configuration option."""
    from leann import LeannBuilder, LeannChat

    pytest.importorskip("transformers")  # Skip if transformers not installed

    with tempfile.TemporaryDirectory() as temp_dir:
        # Build a simple index
        index_path = str(Path(temp_dir) / "test.leann")
        builder = LeannBuilder(backend_name="hnsw")
        builder.add_text("Test document for LLM testing")
        builder.build_index(index_path)

        # Test HF LLM config
        llm_config = {"type": "hf", "model": "Qwen/Qwen3-0.6B"}
        chat = LeannChat(index_path, llm_config=llm_config)
        response = chat.ask("What is this document about?", top_k=1)

        assert isinstance(response, str)
        assert len(response) > 0
