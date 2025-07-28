"""
Test examples from README.md to ensure documentation is accurate.
"""

import tempfile
from pathlib import Path

import pytest


def test_readme_basic_example():
    """Test the basic example from README.md."""
    # This is the exact code from README
    from leann import LeannBuilder, LeannChat, LeannSearcher

    with tempfile.TemporaryDirectory() as temp_dir:
        INDEX_PATH = str(Path(temp_dir) / "demo.leann")

        # Build an index
        builder = LeannBuilder(backend_name="hnsw")
        builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
        builder.add_text("Tung Tung Tung Sahur called—they need their banana-crocodile hybrid back")
        builder.build_index(INDEX_PATH)

        # Verify index was created
        assert Path(INDEX_PATH).exists()

        # Search
        searcher = LeannSearcher(INDEX_PATH)
        results = searcher.search("fantastical AI-generated creatures", top_k=1)

        # Verify search results
        assert len(results) > 0
        assert len(results[0]) == 1  # top_k=1
        # The second text about banana-crocodile should be more relevant
        assert "banana" in results[0][0].text or "crocodile" in results[0][0].text

        # Chat with your data (using simulated LLM to avoid external dependencies)
        chat = LeannChat(INDEX_PATH, llm_config={"type": "simulated"})
        response = chat.ask("How much storage does LEANN save?", top_k=1)

        # Verify chat works
        assert isinstance(response, str)
        assert len(response) > 0


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
    from leann import LeannBuilder

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test HNSW backend (as shown in README)
        hnsw_path = str(Path(temp_dir) / "test_hnsw.leann")
        builder_hnsw = LeannBuilder(backend_name="hnsw")
        builder_hnsw.add_text("Test document for HNSW backend")
        builder_hnsw.build_index(hnsw_path)
        assert Path(hnsw_path).exists()

        # Test DiskANN backend (mentioned as available option)
        diskann_path = str(Path(temp_dir) / "test_diskann.leann")
        builder_diskann = LeannBuilder(backend_name="diskann")
        builder_diskann.add_text("Test document for DiskANN backend")
        builder_diskann.build_index(diskann_path)
        assert Path(diskann_path).exists()


@pytest.mark.parametrize("llm_type", ["simulated", "hf"])
def test_llm_config_options(llm_type):
    """Test different LLM configuration options shown in documentation."""
    from leann import LeannBuilder, LeannChat

    if llm_type == "hf":
        pytest.importorskip("transformers")  # Skip if transformers not installed

    with tempfile.TemporaryDirectory() as temp_dir:
        # Build a simple index
        index_path = str(Path(temp_dir) / "test.leann")
        builder = LeannBuilder(backend_name="hnsw")
        builder.add_text("Test document for LLM testing")
        builder.build_index(index_path)

        # Test LLM config
        if llm_type == "simulated":
            llm_config = {"type": "simulated"}
        else:  # hf
            llm_config = {"type": "hf", "model": "Qwen/Qwen3-0.6B"}

        chat = LeannChat(index_path, llm_config=llm_config)
        response = chat.ask("What is this document about?", top_k=1)

        assert isinstance(response, str)
        assert len(response) > 0
