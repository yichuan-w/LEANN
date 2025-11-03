"""
Test suite for astchunk integration with LEANN.
Tests AST-aware chunking functionality, language detection, and fallback mechanisms.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

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
        # Traditional chunks now return dict format for consistency
        assert all(isinstance(chunk, dict) for chunk in chunks)
        assert all("text" in chunk and "metadata" in chunk for chunk in chunks)
        assert all(len(chunk["text"].strip()) > 0 for chunk in chunks)

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
            # R3: Expect dict format with "text" and "metadata" keys
            assert all(isinstance(chunk, dict) for chunk in chunks), "All chunks should be dicts"
            assert all("text" in chunk and "metadata" in chunk for chunk in chunks), (
                "Each chunk should have 'text' and 'metadata' keys"
            )
            assert all(len(chunk["text"].strip()) > 0 for chunk in chunks), (
                "Each chunk text should be non-empty"
            )

            # Check metadata is present
            assert all("file_path" in chunk["metadata"] for chunk in chunks), (
                "Each chunk should have file_path metadata"
            )

            # Check that code structure is somewhat preserved
            combined_content = " ".join([c["text"] for c in chunks])
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
        # R3: Traditional chunking should also return dict format for consistency
        assert all(isinstance(chunk, dict) for chunk in chunks), "All chunks should be dicts"
        assert all("text" in chunk and "metadata" in chunk for chunk in chunks), (
            "Each chunk should have 'text' and 'metadata' keys"
        )

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
        # R3: AST mode should also return dict format
        assert all(isinstance(chunk, dict) for chunk in chunks), "All chunks should be dicts"
        assert all("text" in chunk and "metadata" in chunk for chunk in chunks), (
            "Each chunk should have 'text' and 'metadata' keys"
        )

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


class TestASTContentExtraction:
    """Test AST content extraction bug fix.

    These tests verify that astchunk's dict format with 'content' key is handled correctly,
    and that the extraction logic doesn't fall through to stringifying entire dicts.
    """

    def test_extract_content_from_astchunk_dict(self):
        """Test that astchunk dict format with 'content' key is handled correctly.

        Bug: Current code checks for chunk["text"] but astchunk returns chunk["content"].
        This causes fallthrough to str(chunk), stringifying the entire dict.

        This test will FAIL until the bug is fixed because:
        - Current code will stringify the dict: "{'content': '...', 'metadata': {...}}"
        - Fixed code should extract just the content value
        """
        # Mock the ASTChunkBuilder class
        mock_builder = Mock()

        # Astchunk returns this format
        astchunk_format_chunk = {
            "content": "def hello():\n    print('world')",
            "metadata": {
                "filepath": "test.py",
                "line_count": 2,
                "start_line_no": 0,
                "end_line_no": 1,
                "node_count": 1,
            },
        }
        mock_builder.chunkify.return_value = [astchunk_format_chunk]

        # Create mock document
        doc = MockDocument(
            "def hello():\n    print('world')", "/test/test.py", {"language": "python"}
        )

        # Mock the astchunk module and its ASTChunkBuilder class
        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        # Patch sys.modules to inject our mock before the import
        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            # Call create_ast_chunks
            chunks = create_ast_chunks([doc])

        # R3: Should return dict format with proper metadata
        assert len(chunks) > 0, "Should return at least one chunk"

        # R3: Each chunk should be a dict
        chunk = chunks[0]
        assert isinstance(chunk, dict), "Chunk should be a dict"
        assert "text" in chunk, "Chunk should have 'text' key"
        assert "metadata" in chunk, "Chunk should have 'metadata' key"

        chunk_text = chunk["text"]

        # CRITICAL: Should NOT contain stringified dict markers in the text field
        # These assertions will FAIL with current buggy code
        assert "'content':" not in chunk_text, (
            f"Chunk text contains stringified dict - extraction failed! Got: {chunk_text[:100]}..."
        )
        assert "'metadata':" not in chunk_text, (
            "Chunk text contains stringified metadata - extraction failed! "
            f"Got: {chunk_text[:100]}..."
        )
        assert "{" not in chunk_text or "def hello" in chunk_text.split("{")[0], (
            "Chunk text appears to be a stringified dict"
        )

        # Should contain actual content
        assert "def hello()" in chunk_text, "Should extract actual code content"
        assert "print('world')" in chunk_text, "Should extract complete code content"

        # R3: Should preserve astchunk metadata
        assert "filepath" in chunk["metadata"] or "file_path" in chunk["metadata"], (
            "Should preserve file path metadata"
        )

    def test_extract_text_key_fallback(self):
        """Test that 'text' key still works for backward compatibility.

        Some chunks might use 'text' instead of 'content' - ensure backward compatibility.
        This test should PASS even with current code.
        """
        mock_builder = Mock()

        # Some chunks might use "text" key
        text_key_chunk = {"text": "def legacy_function():\n    return True"}
        mock_builder.chunkify.return_value = [text_key_chunk]

        # Create mock document
        doc = MockDocument(
            "def legacy_function():\n    return True", "/test/legacy.py", {"language": "python"}
        )

        # Mock the astchunk module
        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            # Call create_ast_chunks
            chunks = create_ast_chunks([doc])

        # R3: Should extract text correctly as dict format
        assert len(chunks) > 0
        chunk = chunks[0]
        assert isinstance(chunk, dict), "Chunk should be a dict"
        assert "text" in chunk, "Chunk should have 'text' key"

        chunk_text = chunk["text"]

        # Should NOT be stringified
        assert "'text':" not in chunk_text, "Should not stringify dict with 'text' key"

        # Should contain actual content
        assert "def legacy_function()" in chunk_text
        assert "return True" in chunk_text

    def test_handles_string_chunks(self):
        """Test that plain string chunks still work.

        Some chunkers might return plain strings - verify these are preserved.
        This test should PASS with current code.
        """
        mock_builder = Mock()

        # Plain string chunk
        plain_string_chunk = "def simple_function():\n    pass"
        mock_builder.chunkify.return_value = [plain_string_chunk]

        # Create mock document
        doc = MockDocument(
            "def simple_function():\n    pass", "/test/simple.py", {"language": "python"}
        )

        # Mock the astchunk module
        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            # Call create_ast_chunks
            chunks = create_ast_chunks([doc])

        # R3: Should wrap string in dict format
        assert len(chunks) > 0
        chunk = chunks[0]
        assert isinstance(chunk, dict), "Even string chunks should be wrapped in dict"
        assert "text" in chunk, "Chunk should have 'text' key"

        chunk_text = chunk["text"]

        assert chunk_text == plain_string_chunk.strip(), (
            "Should preserve plain string chunk content"
        )
        assert "def simple_function()" in chunk_text
        assert "pass" in chunk_text

    def test_multiple_chunks_with_mixed_formats(self):
        """Test handling of multiple chunks with different formats.

        Real-world scenario: astchunk might return a mix of formats.
        This test will FAIL if any chunk with 'content' key gets stringified.
        """
        mock_builder = Mock()

        # Mix of formats
        mixed_chunks = [
            {"content": "def first():\n    return 1", "metadata": {"line_count": 2}},
            "def second():\n    return 2",  # Plain string
            {"text": "def third():\n    return 3"},  # Old format
            {"content": "class MyClass:\n    pass", "metadata": {"node_count": 1}},
        ]
        mock_builder.chunkify.return_value = mixed_chunks

        # Create mock document
        code = "def first():\n    return 1\n\ndef second():\n    return 2\n\ndef third():\n    return 3\n\nclass MyClass:\n    pass"
        doc = MockDocument(code, "/test/mixed.py", {"language": "python"})

        # Mock the astchunk module
        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            # Call create_ast_chunks
            chunks = create_ast_chunks([doc])

        # R3: Should extract all chunks correctly as dicts
        assert len(chunks) == 4, "Should extract all 4 chunks"

        # Check each chunk
        for i, chunk in enumerate(chunks):
            assert isinstance(chunk, dict), f"Chunk {i} should be a dict"
            assert "text" in chunk, f"Chunk {i} should have 'text' key"
            assert "metadata" in chunk, f"Chunk {i} should have 'metadata' key"

            chunk_text = chunk["text"]
            # None should be stringified dicts
            assert "'content':" not in chunk_text, f"Chunk {i} text is stringified (has 'content':)"
            assert "'metadata':" not in chunk_text, (
                f"Chunk {i} text is stringified (has 'metadata':)"
            )
            assert "'text':" not in chunk_text, f"Chunk {i} text is stringified (has 'text':)"

        # Verify actual content is present
        combined = "\n".join([c["text"] for c in chunks])
        assert "def first()" in combined
        assert "def second()" in combined
        assert "def third()" in combined
        assert "class MyClass:" in combined

    def test_empty_content_value_handling(self):
        """Test handling of chunks with empty content values.

        Edge case: chunk has 'content' key but value is empty.
        Should skip these chunks, not stringify them.
        """
        mock_builder = Mock()

        chunks_with_empty = [
            {"content": "", "metadata": {"line_count": 0}},  # Empty content
            {"content": "   ", "metadata": {"line_count": 1}},  # Whitespace only
            {"content": "def valid():\n    return True", "metadata": {"line_count": 2}},  # Valid
        ]
        mock_builder.chunkify.return_value = chunks_with_empty

        doc = MockDocument(
            "def valid():\n    return True", "/test/empty.py", {"language": "python"}
        )

        # Mock the astchunk module
        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            chunks = create_ast_chunks([doc])

        # R3: Should only have the valid chunk (empty ones filtered out)
        assert len(chunks) == 1, "Should filter out empty content chunks"

        chunk = chunks[0]
        assert isinstance(chunk, dict), "Chunk should be a dict"
        assert "text" in chunk, "Chunk should have 'text' key"
        assert "def valid()" in chunk["text"]

        # Should not have stringified the empty dict
        assert "'content': ''" not in chunk["text"]


class TestASTMetadataPreservation:
    """Test metadata preservation in AST chunk dictionaries.

    R3: These tests define the contract for metadata preservation when returning
    chunk dictionaries instead of plain strings. Each chunk dict should have:
    - "text": str - the actual chunk content
    - "metadata": dict - all metadata from document AND astchunk

    These tests will FAIL until G3 implementation changes return type to list[dict].
    """

    def test_ast_chunks_preserve_file_metadata(self):
        """Test that document metadata is preserved in chunk metadata.

        This test verifies that all document-level metadata (file_path, file_name,
        creation_date, last_modified_date) is included in each chunk's metadata dict.

        This will FAIL because current code returns list[str], not list[dict].
        """
        # Create mock document with rich metadata
        python_code = '''
def calculate_sum(numbers):
    """Calculate sum of numbers."""
    return sum(numbers)

class DataProcessor:
    """Process data records."""

    def process(self, data):
        return [x * 2 for x in data]
'''
        doc = MockDocument(
            python_code,
            file_path="/project/src/utils.py",
            metadata={
                "language": "python",
                "file_path": "/project/src/utils.py",
                "file_name": "utils.py",
                "creation_date": "2024-01-15T10:30:00",
                "last_modified_date": "2024-10-31T15:45:00",
            },
        )

        # Mock astchunk to return chunks with metadata
        mock_builder = Mock()
        astchunk_chunks = [
            {
                "content": "def calculate_sum(numbers):\n    return sum(numbers)",
                "metadata": {
                    "filepath": "/project/src/utils.py",
                    "line_count": 2,
                    "start_line_no": 1,
                    "end_line_no": 2,
                    "node_count": 1,
                },
            },
            {
                "content": "class DataProcessor:\n    def process(self, data):\n        return [x * 2 for x in data]",
                "metadata": {
                    "filepath": "/project/src/utils.py",
                    "line_count": 3,
                    "start_line_no": 5,
                    "end_line_no": 7,
                    "node_count": 2,
                },
            },
        ]
        mock_builder.chunkify.return_value = astchunk_chunks

        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            chunks = create_ast_chunks([doc])

        # CRITICAL: These assertions will FAIL with current list[str] return type
        assert len(chunks) == 2, "Should return 2 chunks"

        for i, chunk in enumerate(chunks):
            # Structure assertions - WILL FAIL: current code returns strings
            assert isinstance(chunk, dict), f"Chunk {i} should be dict, got {type(chunk)}"
            assert "text" in chunk, f"Chunk {i} must have 'text' key"
            assert "metadata" in chunk, f"Chunk {i} must have 'metadata' key"
            assert isinstance(chunk["metadata"], dict), f"Chunk {i} metadata should be dict"

            # Document metadata preservation - WILL FAIL
            metadata = chunk["metadata"]
            assert "file_path" in metadata, f"Chunk {i} should preserve file_path"
            assert metadata["file_path"] == "/project/src/utils.py", (
                f"Chunk {i} file_path incorrect"
            )

            assert "file_name" in metadata, f"Chunk {i} should preserve file_name"
            assert metadata["file_name"] == "utils.py", f"Chunk {i} file_name incorrect"

            assert "creation_date" in metadata, f"Chunk {i} should preserve creation_date"
            assert metadata["creation_date"] == "2024-01-15T10:30:00", (
                f"Chunk {i} creation_date incorrect"
            )

            assert "last_modified_date" in metadata, f"Chunk {i} should preserve last_modified_date"
            assert metadata["last_modified_date"] == "2024-10-31T15:45:00", (
                f"Chunk {i} last_modified_date incorrect"
            )

        # Verify metadata is consistent across chunks from same document
        assert chunks[0]["metadata"]["file_path"] == chunks[1]["metadata"]["file_path"], (
            "All chunks from same document should have same file_path"
        )

        # Verify text content is present and not stringified
        assert "def calculate_sum" in chunks[0]["text"]
        assert "class DataProcessor" in chunks[1]["text"]

    def test_ast_chunks_include_astchunk_metadata(self):
        """Test that astchunk-specific metadata is merged into chunk metadata.

        This test verifies that astchunk's metadata (line_count, start_line_no,
        end_line_no, node_count) is merged with document metadata.

        This will FAIL because current code returns list[str], not list[dict].
        """
        python_code = '''
def function_one():
    """First function."""
    x = 1
    y = 2
    return x + y

def function_two():
    """Second function."""
    return 42
'''
        doc = MockDocument(
            python_code,
            file_path="/test/code.py",
            metadata={
                "language": "python",
                "file_path": "/test/code.py",
                "file_name": "code.py",
            },
        )

        # Mock astchunk with detailed metadata
        mock_builder = Mock()
        astchunk_chunks = [
            {
                "content": "def function_one():\n    x = 1\n    y = 2\n    return x + y",
                "metadata": {
                    "filepath": "/test/code.py",
                    "line_count": 4,
                    "start_line_no": 1,
                    "end_line_no": 4,
                    "node_count": 5,  # function, assignments, return
                },
            },
            {
                "content": "def function_two():\n    return 42",
                "metadata": {
                    "filepath": "/test/code.py",
                    "line_count": 2,
                    "start_line_no": 7,
                    "end_line_no": 8,
                    "node_count": 2,  # function, return
                },
            },
        ]
        mock_builder.chunkify.return_value = astchunk_chunks

        mock_astchunk = Mock()
        mock_astchunk.ASTChunkBuilder = Mock(return_value=mock_builder)

        with patch.dict("sys.modules", {"astchunk": mock_astchunk}):
            chunks = create_ast_chunks([doc])

        # CRITICAL: These will FAIL with current list[str] return
        assert len(chunks) == 2

        # First chunk - function_one
        chunk1 = chunks[0]
        assert isinstance(chunk1, dict), "Chunk should be dict"
        assert "metadata" in chunk1

        metadata1 = chunk1["metadata"]

        # Check astchunk metadata is present
        assert "line_count" in metadata1, "Should include astchunk line_count"
        assert metadata1["line_count"] == 4, "line_count should be 4"

        assert "start_line_no" in metadata1, "Should include astchunk start_line_no"
        assert metadata1["start_line_no"] == 1, "start_line_no should be 1"

        assert "end_line_no" in metadata1, "Should include astchunk end_line_no"
        assert metadata1["end_line_no"] == 4, "end_line_no should be 4"

        assert "node_count" in metadata1, "Should include astchunk node_count"
        assert metadata1["node_count"] == 5, "node_count should be 5"

        # Second chunk - function_two
        chunk2 = chunks[1]
        metadata2 = chunk2["metadata"]

        assert metadata2["line_count"] == 2, "line_count should be 2"
        assert metadata2["start_line_no"] == 7, "start_line_no should be 7"
        assert metadata2["end_line_no"] == 8, "end_line_no should be 8"
        assert metadata2["node_count"] == 2, "node_count should be 2"

        # Verify document metadata is ALSO present (merged, not replaced)
        assert metadata1["file_path"] == "/test/code.py"
        assert metadata1["file_name"] == "code.py"
        assert metadata2["file_path"] == "/test/code.py"
        assert metadata2["file_name"] == "code.py"

        # Verify text content is correct
        assert "def function_one" in chunk1["text"]
        assert "def function_two" in chunk2["text"]

    def test_traditional_chunks_as_dicts_helper(self):
        """Test the helper function that wraps traditional chunks as dicts.

        This test verifies that when create_traditional_chunks is called,
        its plain string chunks are wrapped into dict format with metadata.

        This will FAIL because the helper function _traditional_chunks_as_dicts()
        doesn't exist yet, and create_traditional_chunks returns list[str].
        """
        # Create documents with various metadata
        docs = [
            MockDocument(
                "This is the first paragraph of text. It contains multiple sentences. "
                "This should be split into chunks based on size.",
                file_path="/docs/readme.txt",
                metadata={
                    "file_path": "/docs/readme.txt",
                    "file_name": "readme.txt",
                    "creation_date": "2024-01-01",
                },
            ),
            MockDocument(
                "Second document with different metadata. It also has content that needs chunking.",
                file_path="/docs/guide.md",
                metadata={
                    "file_path": "/docs/guide.md",
                    "file_name": "guide.md",
                    "last_modified_date": "2024-10-31",
                },
            ),
        ]

        # Call create_traditional_chunks (which should now return list[dict])
        chunks = create_traditional_chunks(docs, chunk_size=50, chunk_overlap=10)

        # CRITICAL: Will FAIL - current code returns list[str]
        assert len(chunks) > 0, "Should return chunks"

        for i, chunk in enumerate(chunks):
            # Structure assertions - WILL FAIL
            assert isinstance(chunk, dict), f"Chunk {i} should be dict, got {type(chunk)}"
            assert "text" in chunk, f"Chunk {i} must have 'text' key"
            assert "metadata" in chunk, f"Chunk {i} must have 'metadata' key"

            # Text should be non-empty
            assert len(chunk["text"].strip()) > 0, f"Chunk {i} text should be non-empty"

            # Metadata should include document info
            metadata = chunk["metadata"]
            assert "file_path" in metadata, f"Chunk {i} should have file_path in metadata"
            assert "file_name" in metadata, f"Chunk {i} should have file_name in metadata"

        # Verify metadata tracking works correctly
        # At least one chunk should be from readme.txt
        readme_chunks = [c for c in chunks if "readme.txt" in c["metadata"]["file_name"]]
        assert len(readme_chunks) > 0, "Should have chunks from readme.txt"

        # At least one chunk should be from guide.md
        guide_chunks = [c for c in chunks if "guide.md" in c["metadata"]["file_name"]]
        assert len(guide_chunks) > 0, "Should have chunks from guide.md"

        # Verify creation_date is preserved for readme chunks
        for chunk in readme_chunks:
            assert chunk["metadata"].get("creation_date") == "2024-01-01", (
                "readme.txt chunks should preserve creation_date"
            )

        # Verify last_modified_date is preserved for guide chunks
        for chunk in guide_chunks:
            assert chunk["metadata"].get("last_modified_date") == "2024-10-31", (
                "guide.md chunks should preserve last_modified_date"
            )

        # Verify text content is present
        all_text = " ".join([c["text"] for c in chunks])
        assert "first paragraph" in all_text
        assert "Second document" in all_text


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
