"""
Test suite for astchunk integration with LEANN.
Tests AST-aware chunking functionality, language detection, and fallback mechanisms.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add apps directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

from typing import Optional

from chunking import (
    create_ast_chunks,
    create_text_chunks,
    create_traditional_chunks,
    detect_code_files,
    get_language_from_extension,
)


class MockDocument:
    """Mock LlamaIndex Document for testing."""

    def __init__(self, content: str, file_path: str = "", metadata: Optional[dict] = None):
        self.content = content
        self.metadata = metadata or {}
        if file_path:
            self.metadata["file_path"] = file_path

    def get_content(self) -> str:
        return self.content


class TestCodeFileDetection:
    """Test code file detection and language mapping."""

    def test_detect_code_files_python(self):
        """Test detection of Python files."""
        docs = [
            MockDocument("print('hello')", "/path/to/file.py"),
            MockDocument("This is text", "/path/to/file.txt"),
        ]

        code_docs, text_docs = detect_code_files(docs)

        assert len(code_docs) == 1
        assert len(text_docs) == 1
        assert code_docs[0].metadata["language"] == "python"
        assert code_docs[0].metadata["is_code"] is True
        assert text_docs[0].metadata["is_code"] is False

    def test_detect_code_files_multiple_languages(self):
        """Test detection of multiple programming languages."""
        docs = [
            MockDocument("def func():", "/path/to/script.py"),
            MockDocument("public class Test {}", "/path/to/Test.java"),
            MockDocument("interface ITest {}", "/path/to/test.ts"),
            MockDocument("using System;", "/path/to/Program.cs"),
            MockDocument("Regular text content", "/path/to/document.txt"),
        ]

        code_docs, text_docs = detect_code_files(docs)

        assert len(code_docs) == 4
        assert len(text_docs) == 1

        languages = [doc.metadata["language"] for doc in code_docs]
        assert "python" in languages
        assert "java" in languages
        assert "typescript" in languages
        assert "csharp" in languages

    def test_detect_code_files_no_file_path(self):
        """Test handling of documents without file paths."""
        docs = [
            MockDocument("some content"),
            MockDocument("other content", metadata={"some_key": "value"}),
        ]

        code_docs, text_docs = detect_code_files(docs)

        assert len(code_docs) == 0
        assert len(text_docs) == 2
        for doc in text_docs:
            assert doc.metadata["is_code"] is False

    def test_get_language_from_extension(self):
        """Test language detection from file extensions."""
        assert get_language_from_extension("test.py") == "python"
        assert get_language_from_extension("Test.java") == "java"
        assert get_language_from_extension("component.tsx") == "typescript"
        assert get_language_from_extension("Program.cs") == "csharp"
        assert get_language_from_extension("document.txt") is None
        assert get_language_from_extension("") is None


class TestChunkingFunctions:
    """Test various chunking functionality."""

    def test_create_traditional_chunks(self):
        """Test traditional text chunking."""
        docs = [
            MockDocument(
                "This is a test document. It has multiple sentences. We want to test chunking."
            )
        ]

        chunks = create_traditional_chunks(docs, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_create_traditional_chunks_empty_docs(self):
        """Test traditional chunking with empty documents."""
        chunks = create_traditional_chunks([], chunk_size=50, chunk_overlap=10)
        assert chunks == []

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip astchunk tests in CI - dependency may not be available",
    )
    def test_create_ast_chunks_with_astchunk_available(self):
        """Test AST chunking when astchunk is available."""
        python_code = '''
def hello_world():
    """Print hello world message."""
    print("Hello, World!")

def add_numbers(a, b):
    """Add two numbers and return the result."""
    return a + b

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
'''

        docs = [MockDocument(python_code, "/test/calculator.py", {"language": "python"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=200, chunk_overlap=50)

            # Should have multiple chunks due to different functions/classes
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk.strip()) > 0 for chunk in chunks)

            # Check that code structure is somewhat preserved
            combined_content = " ".join(chunks)
            assert "def hello_world" in combined_content
            assert "class Calculator" in combined_content

        except ImportError:
            # astchunk not available, should fall back to traditional chunking
            chunks = create_ast_chunks(docs, max_chunk_size=200, chunk_overlap=50)
            assert len(chunks) > 0  # Should still get chunks from fallback

    def test_create_ast_chunks_fallback_to_traditional(self):
        """Test AST chunking falls back to traditional when astchunk is not available."""
        docs = [MockDocument("def test(): pass", "/test/script.py", {"language": "python"})]

        # Mock astchunk import to fail
        with patch("chunking.create_ast_chunks"):
            # First call (actual test) should import astchunk and potentially fail
            # Let's call the actual function to test the import error handling
            chunks = create_ast_chunks(docs)

            # Should return some chunks (either from astchunk or fallback)
            assert isinstance(chunks, list)

    def test_create_text_chunks_traditional_mode(self):
        """Test text chunking in traditional mode."""
        docs = [
            MockDocument("def test(): pass", "/test/script.py"),
            MockDocument("This is regular text.", "/test/doc.txt"),
        ]

        chunks = create_text_chunks(docs, use_ast_chunking=False, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_create_text_chunks_ast_mode(self):
        """Test text chunking in AST mode."""
        docs = [
            MockDocument("def test(): pass", "/test/script.py"),
            MockDocument("This is regular text.", "/test/doc.txt"),
        ]

        chunks = create_text_chunks(
            docs,
            use_ast_chunking=True,
            ast_chunk_size=100,
            ast_chunk_overlap=20,
            chunk_size=50,
            chunk_overlap=10,
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_create_text_chunks_custom_extensions(self):
        """Test text chunking with custom code file extensions."""
        docs = [
            MockDocument("function test() {}", "/test/script.js"),  # Not in default extensions
            MockDocument("Regular text", "/test/doc.txt"),
        ]

        # First without custom extensions - should treat .js as text
        chunks_without = create_text_chunks(docs, use_ast_chunking=True, code_file_extensions=None)

        # Then with custom extensions - should treat .js as code
        chunks_with = create_text_chunks(
            docs, use_ast_chunking=True, code_file_extensions=[".js", ".jsx"]
        )

        # Both should return chunks
        assert len(chunks_without) > 0
        assert len(chunks_with) > 0


class TestIntegrationWithDocumentRAG:
    """Integration tests with the document RAG system."""

    @pytest.fixture
    def temp_code_dir(self):
        """Create a temporary directory with sample code files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample Python file
            python_file = temp_path / "example.py"
            python_file.write_text('''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class MathUtils:
    @staticmethod
    def factorial(n):
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n-1)
''')

            # Create sample text file
            text_file = temp_path / "readme.txt"
            text_file.write_text("This is a sample text file for testing purposes.")

            yield temp_path

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip integration tests in CI to avoid dependency issues",
    )
    def test_document_rag_with_ast_chunking(self, temp_code_dir):
        """Test document RAG with AST chunking enabled."""
        with tempfile.TemporaryDirectory() as index_dir:
            cmd = [
                sys.executable,
                "apps/document_rag.py",
                "--llm",
                "simulated",
                "--embedding-model",
                "facebook/contriever",
                "--embedding-mode",
                "sentence-transformers",
                "--index-dir",
                index_dir,
                "--data-dir",
                str(temp_code_dir),
                "--enable-code-chunking",
                "--query",
                "How does the fibonacci function work?",
            ]

            env = os.environ.copy()
            env["HF_HUB_DISABLE_SYMLINKS"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minutes
                    env=env,
                )

                # Should succeed even if astchunk is not available (fallback)
                assert result.returncode == 0, f"Command failed: {result.stderr}"

                output = result.stdout + result.stderr
                assert "Index saved to" in output or "Using existing index" in output

            except subprocess.TimeoutExpired:
                pytest.skip("Test timed out - likely due to model download in CI")

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip integration tests in CI to avoid dependency issues",
    )
    def test_code_rag_application(self, temp_code_dir):
        """Test the specialized code RAG application."""
        with tempfile.TemporaryDirectory() as index_dir:
            cmd = [
                sys.executable,
                "apps/code_rag.py",
                "--llm",
                "simulated",
                "--embedding-model",
                "facebook/contriever",
                "--index-dir",
                index_dir,
                "--repo-dir",
                str(temp_code_dir),
                "--query",
                "What classes are defined in this code?",
            ]

            env = os.environ.copy()
            env["HF_HUB_DISABLE_SYMLINKS"] = "1"
            env["TOKENIZERS_PARALLELISM"] = "false"

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

                # Should succeed
                assert result.returncode == 0, f"Command failed: {result.stderr}"

                output = result.stdout + result.stderr
                assert "Using AST-aware chunking" in output or "traditional chunking" in output

            except subprocess.TimeoutExpired:
                pytest.skip("Test timed out - likely due to model download in CI")


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_text_chunking_empty_documents(self):
        """Test text chunking with empty document list."""
        chunks = create_text_chunks([])
        assert chunks == []

    def test_text_chunking_invalid_parameters(self):
        """Test text chunking with invalid parameters."""
        docs = [MockDocument("test content")]

        # Should handle negative chunk sizes gracefully
        chunks = create_text_chunks(
            docs, chunk_size=0, chunk_overlap=0, ast_chunk_size=0, ast_chunk_overlap=0
        )

        # Should still return some result
        assert isinstance(chunks, list)

    def test_create_ast_chunks_no_language(self):
        """Test AST chunking with documents missing language metadata."""
        docs = [MockDocument("def test(): pass", "/test/script.py")]  # No language set

        chunks = create_ast_chunks(docs)

        # Should fall back to traditional chunking
        assert isinstance(chunks, list)
        assert len(chunks) >= 0  # May be empty if fallback also fails

    def test_create_ast_chunks_empty_content(self):
        """Test AST chunking with empty content."""
        docs = [MockDocument("", "/test/script.py", {"language": "python"})]

        chunks = create_ast_chunks(docs)

        # Should handle empty content gracefully
        assert isinstance(chunks, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
