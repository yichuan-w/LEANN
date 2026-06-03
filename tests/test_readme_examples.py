"""
Test examples from README.md to ensure documentation is accurate.
"""

import os
import platform
import tempfile
from pathlib import Path

import pytest

CI_EMBEDDING_DIMENSIONS = 4


def _is_ci() -> bool:
    return os.environ.get("CI") == "true"


def _install_ci_embeddings(monkeypatch):
    """Use deterministic embeddings in CI so docs tests do not depend on model downloads."""
    if not _is_ci():
        return

    import leann.api as leann_api
    import leann.embedding_compute as embedding_compute
    import numpy as np

    def fake_compute_embeddings(
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
            if (
                "fantastical" in text
                or "ai-generated" in text
                or "banana" in text
                or "crocodile" in text
            ):
                embeddings.append([1.0, 0.0, 0.0, 0.0])
            elif "storage" in text or "97%" in text or "saves" in text:
                embeddings.append([0.0, 1.0, 0.0, 0.0])
            elif "llm" in text or "testing" in text:
                embeddings.append([0.0, 0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 0.0, 1.0])
        return np.asarray(embeddings, dtype=np.float32)

    monkeypatch.setattr(leann_api, "compute_embeddings", fake_compute_embeddings)
    monkeypatch.setattr(embedding_compute, "compute_embeddings", fake_compute_embeddings)


def _ci_builder_kwargs() -> dict:
    if not _is_ci():
        return {}
    return {
        "embedding_model": "ci/deterministic-test-embedding",
        "dimensions": CI_EMBEDDING_DIMENSIONS,
        "is_compact": False,
        "is_recompute": False,
    }


def _ci_searcher_kwargs() -> dict:
    if not _is_ci():
        return {}
    return {"enable_warmup": False, "recompute_embeddings": False}


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_readme_basic_example(backend_name, monkeypatch):
    """Test the basic example from README.md with both backends."""
    _install_ci_embeddings(monkeypatch)
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if _is_ci() and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")
    # Skip DiskANN on CI (Linux runners) due to C++ extension memory/hardware constraints
    if _is_ci() and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI due to resource constraints and instability")

    # This is the exact code from README (with smaller model for CI)
    from leann import LeannBuilder, LeannChat, LeannSearcher
    from leann.api import SearchResult

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        INDEX_PATH = str(Path(temp_dir) / f"demo_{backend_name}.leann")

        if _is_ci():
            builder = LeannBuilder(
                backend_name=backend_name,
                **_ci_builder_kwargs(),
            )
        else:
            builder = LeannBuilder(backend_name=backend_name)
        builder.add_text("LEANN saves 97% storage compared to traditional vector databases.")
        builder.add_text("Tung Tung Tung Sahur called—they need their banana-crocodile hybrid back")
        builder.build_index(INDEX_PATH)

        index_dir = Path(INDEX_PATH).parent
        assert index_dir.exists()
        index_files = list(index_dir.glob(f"{Path(INDEX_PATH).stem}.*"))
        assert len(index_files) > 0

        with LeannSearcher(INDEX_PATH, **_ci_searcher_kwargs()) as searcher:
            results = searcher.search(
                "fantastical AI-generated creatures",
                top_k=1,
            )

            assert len(results) > 0
            assert isinstance(results[0], SearchResult)
            assert results[0].score != float("-inf"), (
                f"should return valid scores, got {results[0].score}"
            )
            assert "banana" in results[0].text or "crocodile" in results[0].text

        chat = LeannChat(
            INDEX_PATH,
            llm_config={"type": "simulated"},
            **_ci_searcher_kwargs(),
        )
        response = chat.ask(
            "How much storage does LEANN save?",
            top_k=1,
            recompute_embeddings=not _is_ci(),
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


def test_backend_options(monkeypatch):
    """Test different backend options mentioned in documentation."""
    _install_ci_embeddings(monkeypatch)
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if _is_ci() and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    from leann import LeannBuilder

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        hnsw_path = str(Path(temp_dir) / "test_hnsw.leann")
        if _is_ci():
            builder_hnsw = LeannBuilder(
                backend_name="hnsw",
                **_ci_builder_kwargs(),
            )
        else:
            builder_hnsw = LeannBuilder(
                backend_name="hnsw",
                embedding_model="facebook/contriever",
            )
        builder_hnsw.add_text("Test document for HNSW backend")
        builder_hnsw.build_index(hnsw_path)
        assert Path(hnsw_path).parent.exists()
        assert len(list(Path(hnsw_path).parent.glob(f"{Path(hnsw_path).stem}.*"))) > 0

        if _is_ci():
            pytest.skip(
                "Skip DiskANN portion in CI - small datasets trigger MKL parameter "
                "errors and pytest-timeout thread kills cause segfaults on Windows"
            )

        diskann_path = str(Path(temp_dir) / "test_diskann.leann")
        builder_diskann = LeannBuilder(
            backend_name="diskann",
            embedding_model="facebook/contriever",
        )
        builder_diskann.add_text("Test document for DiskANN backend")
        builder_diskann.build_index(diskann_path)
        assert Path(diskann_path).parent.exists()
        assert len(list(Path(diskann_path).parent.glob(f"{Path(diskann_path).stem}.*"))) > 0


@pytest.mark.parametrize("backend_name", ["hnsw", "diskann"])
def test_llm_config_simulated(backend_name, monkeypatch):
    """Test simulated LLM configuration option with both backends."""
    _install_ci_embeddings(monkeypatch)
    # Skip on macOS CI due to MPS environment issues with all-MiniLM-L6-v2
    if _is_ci() and platform.system() == "Darwin":
        pytest.skip("Skipping on macOS CI due to MPS environment issues with all-MiniLM-L6-v2")

    # Skip DiskANN tests in CI due to hardware requirements
    if _is_ci() and backend_name == "diskann":
        pytest.skip("Skip DiskANN tests in CI - requires specific hardware and large memory")

    from leann import LeannBuilder, LeannChat

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        index_path = str(Path(temp_dir) / f"test_{backend_name}.leann")
        if _is_ci():
            builder = LeannBuilder(
                backend_name=backend_name,
                **_ci_builder_kwargs(),
            )
        else:
            builder = LeannBuilder(backend_name=backend_name)
        builder.add_text("Test document for LLM testing")
        builder.build_index(index_path)

        llm_config = {"type": "simulated"}
        chat = LeannChat(index_path, llm_config=llm_config, **_ci_searcher_kwargs())
        response = chat.ask(
            "What is this document about?",
            top_k=1,
            recompute_embeddings=not _is_ci(),
        )

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
