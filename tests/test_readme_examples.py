"""
Test examples from README.md to ensure documentation is accurate.
"""

import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import pytest

TEST_EMBEDDING_MODEL = "test-deterministic-embeddings"
TEST_EMBEDDING_DIMENSIONS = 8


def _deterministic_embeddings(
    chunks,
    model_name,
    mode="sentence-transformers",
    use_server=True,
    port=None,
    is_build=False,
    provider_options=None,
):
    del model_name, mode, use_server, port, is_build, provider_options

    embeddings = []
    for chunk in chunks:
        text = str(chunk).lower()
        vector = np.zeros(TEST_EMBEDDING_DIMENSIONS, dtype=np.float32)
        if any(term in text for term in ("fantastical", "banana", "crocodile")):
            vector[0] = 1.0
        elif any(term in text for term in ("storage", "leann", "saves")):
            vector[1] = 1.0
        else:
            vector[2] = 1.0
        embeddings.append(vector)
    return np.vstack(embeddings)


def _deterministic_direct_embeddings(
    chunks,
    model_name,
    mode="sentence-transformers",
    is_build=False,
    provider_options=None,
):
    return _deterministic_embeddings(
        chunks,
        model_name,
        mode=mode,
        use_server=False,
        is_build=is_build,
        provider_options=provider_options,
    )


@pytest.fixture
def deterministic_embeddings(monkeypatch):
    """Keep README example tests offline and deterministic in CI."""
    monkeypatch.setattr("leann.api.compute_embeddings", _deterministic_embeddings)
    monkeypatch.setattr(
        "leann.embedding_compute.compute_embeddings",
        _deterministic_direct_embeddings,
    )


def _test_builder_kwargs(backend_name):
    kwargs = {
        "backend_name": backend_name,
        "embedding_model": TEST_EMBEDDING_MODEL,
        "dimensions": TEST_EMBEDDING_DIMENSIONS,
    }
    if backend_name == "hnsw":
        kwargs.update({"is_recompute": False, "is_compact": False})
    return kwargs


def _skip_if_backend_unavailable(backend_name):
    from leann.api import get_registered_backends

    if backend_name not in get_registered_backends():
        pytest.skip(f"Backend {backend_name!r} is not installed")


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_readme_basic_example(backend_name, deterministic_embeddings):
    """Test the basic example from README.md with both backends."""
    _skip_if_backend_unavailable(backend_name)
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")
    # Skip DiskANN on CI (Linux runners) due to C++ extension memory/hardware constraints
    if os.environ.get("CI") == "true" and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI due to resource constraints and instability")

    # Exercise the README flow without depending on live model downloads in CI.
    from leann import LeannBuilder, LeannChat, LeannSearcher
    from leann.api import SearchResult

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        INDEX_PATH = str(Path(temp_dir) / f"demo_{backend_name}.leann")

        builder = LeannBuilder(**_test_builder_kwargs(backend_name))
        builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
        builder.add_text("Tung Tung Tung Sahur called—they need their banana-crocodile hybrid back")
        builder.build_index(INDEX_PATH)

        index_dir = Path(INDEX_PATH).parent
        assert index_dir.exists()
        index_files = list(index_dir.glob(f"{Path(INDEX_PATH).stem}.*"))
        assert len(index_files) > 0

        with LeannSearcher(INDEX_PATH, recompute_embeddings=False, enable_warmup=False) as searcher:
            results = searcher.search("fantastical AI-generated creatures", top_k=1)

            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].score != float("-inf"), (
                f"should return valid scores, got {results[0].score}"
            )
            assert "banana" in results[0].text or "crocodile" in results[0].text

        chat = LeannChat(
            INDEX_PATH,
            llm_config={"type": "simulated"},
            recompute_embeddings=False,
        )
        response = chat.ask(
            "How much storage does LEANN save?",
            top_k=1,
            recompute_embeddings=False,
        )

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


def test_backend_options(deterministic_embeddings):
    """Test different backend options mentioned in documentation."""
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    from leann import LeannBuilder

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        is_ci = os.environ.get("CI") == "true"

        hnsw_path = str(Path(temp_dir) / "test_hnsw.leann")
        builder_hnsw = LeannBuilder(**_test_builder_kwargs("hnsw"))
        builder_hnsw.add_text("Test document for HNSW backend")
        builder_hnsw.build_index(hnsw_path)
        assert Path(hnsw_path).parent.exists()
        assert len(list(Path(hnsw_path).parent.glob(f"{Path(hnsw_path).stem}.*"))) > 0

        if is_ci:
            pytest.skip(
                "Skip DiskANN portion in CI - small datasets trigger MKL parameter "
                "errors and pytest-timeout thread kills cause segfaults on Windows"
            )
        _skip_if_backend_unavailable("diskann")

        diskann_path = str(Path(temp_dir) / "test_diskann.leann")
        builder_diskann = LeannBuilder(**_test_builder_kwargs("diskann"))
        builder_diskann.add_text("Test document for DiskANN backend")
        builder_diskann.build_index(diskann_path)
        assert Path(diskann_path).parent.exists()
        assert len(list(Path(diskann_path).parent.glob(f"{Path(diskann_path).stem}.*"))) > 0


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_llm_config_simulated(backend_name, deterministic_embeddings):
    """Test simulated LLM configuration option with both backends."""
    _skip_if_backend_unavailable(backend_name)
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if os.environ.get("CI") == "true" and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    # Skip DiskANN tests in CI due to hardware requirements
    if os.environ.get("CI") == "true" and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI - requires specific hardware and large memory")

    from leann import LeannBuilder, LeannChat

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / f"test_{backend_name}.leann")
        builder = LeannBuilder(**_test_builder_kwargs(backend_name))
        builder.add_text("Test document for LLM testing")
        builder.build_index(index_path)

        llm_config = {"type": "simulated"}
        chat = LeannChat(index_path, llm_config=llm_config)
        response = chat.ask("What is this document about?", top_k=1, recompute_embeddings=False)

        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.skip(reason="Requires HF model download and may timeout")
def test_llm_config_hf():
    """Test HuggingFace LLM configuration option."""
    from leann import LeannBuilder, LeannChat

    pytest.importorskip("transformers")  # Skip if transformers not installed

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
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
