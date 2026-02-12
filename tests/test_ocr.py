"""
Tests for OCR-based PDF support (issue #158).

These tests verify that:
1. extract_pdf_text_with_pymupdf default behavior is unchanged (use_ocr=False)
2. The --enable-ocr CLI flag is properly registered on the build command

Uses importlib to load the cli module directly, mocking heavy dependencies
that may not be available in the test environment.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_cli_module():
    """Load the cli module directly from file, mocking unavailable dependencies."""
    cli_path = (
        Path(__file__).parent.parent
        / "packages"
        / "leann-core"
        / "src"
        / "leann"
        / "cli.py"
    )

    # Mock heavy dependencies that cli.py imports at module level
    mock_modules = {}
    deps_to_mock = [
        "llama_index", "llama_index.core", "llama_index.core.node_parser",
        "tqdm",
        "leann", "leann.api", "leann.interactive_utils",
        "leann.registry", "leann.settings",
    ]
    for mod_name in deps_to_mock:
        if mod_name not in sys.modules:
            mock_modules[mod_name] = MagicMock()

    saved = {}
    for name, mock in mock_modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mock

    try:
        spec = importlib.util.spec_from_file_location("leann.cli", str(cli_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        # Restore original sys.modules state
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


# Load the module once for all tests
_cli_mod = _load_cli_module()


class TestExtractPdfNoOcrDefault:
    """Verify that extract_pdf_text_with_pymupdf default behavior is unchanged."""

    def test_default_use_ocr_is_false(self):
        """The use_ocr parameter should default to False."""
        fn = _cli_mod.extract_pdf_text_with_pymupdf
        sig = inspect.signature(fn)
        use_ocr_param = sig.parameters.get("use_ocr")
        assert use_ocr_param is not None, (
            "extract_pdf_text_with_pymupdf should accept a use_ocr parameter"
        )
        assert use_ocr_param.default is False, "use_ocr should default to False"

    def test_function_signature_backward_compatible(self):
        """The function should still accept just file_path (backward compatibility)."""
        fn = _cli_mod.extract_pdf_text_with_pymupdf
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        assert params[0] == "file_path", "First parameter should be file_path"
        assert len(params) == 2, "Should have exactly 2 parameters: file_path and use_ocr"

    def test_returns_none_when_fitz_unavailable(self):
        """When PyMuPDF (fitz) is not installed, function returns None."""
        fn = _cli_mod.extract_pdf_text_with_pymupdf
        # Temporarily ensure fitz cannot be imported
        with patch.dict(sys.modules, {"fitz": None}):
            result = fn("nonexistent.pdf")
            assert result is None, "Should return None when fitz is not available"

    def test_returns_none_when_fitz_unavailable_with_ocr_enabled(self):
        """When fitz is unavailable, function returns None even with use_ocr=True."""
        fn = _cli_mod.extract_pdf_text_with_pymupdf
        with patch.dict(sys.modules, {"fitz": None}):
            result = fn("nonexistent.pdf", use_ocr=True)
            assert result is None, "Should return None when fitz is not available, regardless of use_ocr"


class TestOcrFlagAccepted:
    """Verify that the CLI parser accepts the --enable-ocr flag."""

    def test_build_parser_accepts_enable_ocr(self):
        """The build subparser should accept --enable-ocr flag."""
        cli = _cli_mod.LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(
            ["build", "test-index", "--docs", "/tmp/test-docs", "--enable-ocr"]
        )
        assert args.command == "build"
        assert hasattr(args, "enable_ocr"), "build command should have enable_ocr attribute"
        assert args.enable_ocr is True, "--enable-ocr flag should set enable_ocr to True"

    def test_build_parser_enable_ocr_defaults_false(self):
        """Without --enable-ocr, the flag should default to False."""
        cli = _cli_mod.LeannCLI()
        parser = cli.create_parser()

        args = parser.parse_args(
            ["build", "test-index", "--docs", "/tmp/test-docs"]
        )
        assert args.command == "build"
        assert hasattr(args, "enable_ocr"), "build command should have enable_ocr attribute"
        assert args.enable_ocr is False, "enable_ocr should default to False"
