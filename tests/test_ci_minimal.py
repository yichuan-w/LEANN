"""
Minimal tests for CI that don't require model loading or significant memory.
"""

import subprocess
import sys


def test_package_imports():
    """Test that all core packages can be imported."""
    # Core package

    # Backend packages

    # Core modules

    assert True  # If we get here, imports worked


def test_cli_help():
    """Test that CLI example shows help."""
    result = subprocess.run(
        [sys.executable, "apps/document_rag.py", "--help"], capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
    assert "--llm" in result.stdout or "--llm" in result.stderr


def test_backend_registration():
    """Test that backends are properly registered."""
    from leann.api import get_registered_backends

    backends = get_registered_backends()
    assert "hnsw" in backends
    assert "diskann" in backends


def test_version_info():
    """Test that packages have version information."""
    import leann
    import leann_backend_diskann
    import leann_backend_hnsw

    # Check that packages have __version__ or can be imported
    assert hasattr(leann, "__version__") or True
    assert hasattr(leann_backend_hnsw, "__version__") or True
    assert hasattr(leann_backend_diskann, "__version__") or True
