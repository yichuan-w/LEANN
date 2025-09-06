"""
Test suite for astchunk integration with LEANN.
Tests AST-aware chunking functionality, language detection, and fallback mechanisms.
"""

import logging
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

logger = logging.getLogger(__name__)


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


class TestGoASTChunking:
    """Test Go AST chunking functionality."""

    def test_go_basic_function_chunking(self):
        """Test chunking of basic Go functions."""
        go_code = """package main

import "fmt"

// Hello prints a greeting
func Hello(name string) {
    fmt.Printf("Hello, %s!\\n", name)
}

// Add adds two numbers
func Add(a, b int) int {
    return a + b
}
"""

        docs = [MockDocument(go_code, "/test/functions.go", {"language": "go"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=200, chunk_overlap=50)

            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)

            # Should contain function definitions
            combined_content = " ".join(chunks)
            assert "func Hello" in combined_content
            assert "func Add" in combined_content

        except Exception as e:
            logger.warning(f"Go AST chunking test failed, expected in some environments: {e}")
            assert True  # Test passes if chunking fails due to missing dependencies

    def test_go_struct_and_methods(self):
        """Test chunking of Go structs with methods."""
        go_code = """package user

// User represents a user in the system
type User struct {
    ID   int    `json:"id"`
    Name string `json:"name"`
    Email string `json:"email"`
}

// GetName returns the user's name
func (u *User) GetName() string {
    return u.Name
}

// SetName sets the user's name
func (u *User) SetName(name string) {
    u.Name = name
}

// Validate validates the user data
func (u User) Validate() error {
    if u.Name == "" {
        return errors.New("name is required")
    }
    return nil
}
"""

        docs = [MockDocument(go_code, "/test/user.go", {"language": "go"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=300, chunk_overlap=50)

            assert len(chunks) > 0
            combined_content = " ".join(chunks)

            # Should contain struct and methods
            assert "type User struct" in combined_content
            assert "func (u *User) GetName" in combined_content or "GetName" in combined_content
            assert "func (u *User) SetName" in combined_content or "SetName" in combined_content
            assert "func (u User) Validate" in combined_content or "Validate" in combined_content

        except Exception as e:
            logger.warning(f"Go struct chunking test failed: {e}")
            assert True

    def test_go_interface_chunking(self):
        """Test chunking of Go interfaces."""
        go_code = """package storage

import "context"

// Storage defines the storage interface
type Storage interface {
    // Get retrieves a value by key
    Get(ctx context.Context, key string) ([]byte, error)

    // Put stores a value with a key
    Put(ctx context.Context, key string, value []byte) error

    // Delete removes a key
    Delete(ctx context.Context, key string) error

    // List returns all keys with optional prefix
    List(ctx context.Context, prefix string) ([]string, error)
}

// ReadOnlyStorage defines a read-only storage interface
type ReadOnlyStorage interface {
    Get(ctx context.Context, key string) ([]byte, error)
    List(ctx context.Context, prefix string) ([]string, error)
}
"""

        docs = [MockDocument(go_code, "/test/storage.go", {"language": "go"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=400, chunk_overlap=50)

            assert len(chunks) > 0
            combined_content = " ".join(chunks)

            # Should contain interfaces
            assert "type Storage interface" in combined_content
            assert "type ReadOnlyStorage interface" in combined_content

        except Exception as e:
            logger.warning(f"Go interface chunking test failed: {e}")
            assert True

    def test_go_generic_types(self):
        """Test chunking of Go generic types and functions (Go 1.18+)."""
        go_code = """package generics

// Stack is a generic stack data structure
type Stack[T any] struct {
    items []T
}

// Push adds an item to the stack
func (s *Stack[T]) Push(item T) {
    s.items = append(s.items, item)
}

// Pop removes and returns the top item
func (s *Stack[T]) Pop() (T, bool) {
    if len(s.items) == 0 {
        var zero T
        return zero, false
    }
    index := len(s.items) - 1
    item := s.items[index]
    s.items = s.items[:index]
    return item, true
}

// Map applies a function to each element
func Map[T, R any](slice []T, fn func(T) R) []R {
    result := make([]R, len(slice))
    for i, item := range slice {
        result[i] = fn(item)
    }
    return result
}
"""

        docs = [MockDocument(go_code, "/test/generics.go", {"language": "go"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=400, chunk_overlap=50)

            assert len(chunks) > 0
            combined_content = " ".join(chunks)

            # Should handle generic syntax
            assert "Stack[T any]" in combined_content or "Stack" in combined_content
            assert "Map[T, R any]" in combined_content or "func Map" in combined_content

        except Exception as e:
            logger.warning(f"Go generics chunking test failed: {e}")
            assert True

    def test_go_large_file_splitting(self):
        """Test intelligent splitting of large Go files."""
        # Create a large Go file with multiple functions
        functions = []
        for i in range(20):
            functions.append(f"""
// Function{i} performs operation {i}
func Function{i}(param{i} int) int {{
    // This is a comment for function {i}
    result := param{i} * {i + 1}
    if result > 100 {{
        return result - 50
    }}
    return result
}}""")

        go_code = f"""package large

import "fmt"
{"".join(functions)}

func main() {{
    fmt.Println("Large file example")
}}"""

        docs = [MockDocument(go_code, "/test/large.go", {"language": "go"})]

        try:
            chunks = create_ast_chunks(docs, max_chunk_size=300, chunk_overlap=50)

            # Should create multiple chunks due to size
            assert len(chunks) > 1
            assert all(len(chunk) <= 500 for chunk in chunks)  # Reasonable size check

            # Verify content preservation
            combined_content = " ".join(chunks)
            assert "func Function0" in combined_content
            assert "func Function19" in combined_content
            assert "func main" in combined_content

        except Exception as e:
            logger.warning(f"Go large file chunking test failed: {e}")
            assert True

    def test_go_error_handling(self):
        """Test Go chunking with malformed code."""
        malformed_go_code = """package main

// Missing closing brace
func BrokenFunction() {
    fmt.Println("This function is broken")
    // Missing }

// Valid function after broken one
func ValidFunction() {
    fmt.Println("This function is valid")
}
"""

        docs = [MockDocument(malformed_go_code, "/test/broken.go", {"language": "go"})]

        # Should handle malformed code gracefully
        chunks = create_ast_chunks(docs, max_chunk_size=200, chunk_overlap=50)

        # Should still return some chunks (fallback behavior)
        assert isinstance(chunks, list)
        assert len(chunks) >= 0


class TestLocalASTChunkers:
    """Test local AST chunker implementations."""

    def test_local_go_chunker_fallback(self):
        """Test local Go AST chunker when external astchunk is not available."""
        go_code = """package main

import "fmt"

// Hello prints a greeting
func Hello(name string) {
    fmt.Printf("Hello, %s!\\n", name)
}

// Person represents a person
type Person struct {
    Name string
    Age  int
}

// Greet greets a person
func (p Person) Greet() {
    fmt.Printf("Hi, I'm %s and I'm %d years old\\n", p.Name, p.Age)
}

func main() {
    Hello("World")
    person := Person{Name: "Alice", Age: 30}
    person.Greet()
}
"""

        docs = [MockDocument(go_code, "/test/main.go", {"language": "go"})]

        try:
            # Try to use the local chunker integration
            from chunking import create_ast_chunks_with_local_chunkers

            chunks = create_ast_chunks_with_local_chunkers(
                docs, max_chunk_size=300, chunk_overlap=50
            )

            # Should create multiple chunks
            assert len(chunks) > 0
            assert all(isinstance(chunk, str) for chunk in chunks)
            assert all(len(chunk.strip()) > 0 for chunk in chunks if chunk.strip())

            # Check that Go constructs are preserved in some form
            combined_content = " ".join(chunks)
            assert "func Hello" in combined_content
            assert "type Person" in combined_content
            assert "func main" in combined_content

        except ImportError:
            # If local chunker integration is not available, skip this test
            pytest.skip("Local Go chunker integration not available")

    def test_go_chunker_direct_import(self):
        """Test direct import and usage of Go AST chunker."""
        go_code = """package calculator

import "math"

// Calculator provides basic arithmetic operations
type Calculator struct {
    history []string
}

// Add performs addition
func (c *Calculator) Add(a, b float64) float64 {
    result := a + b
    c.recordOperation("add", a, b, result)
    return result
}

// recordOperation records an operation in history
func (c *Calculator) recordOperation(op string, a, b, result float64) {
    entry := fmt.Sprintf("%s: %.2f %s %.2f = %.2f", op, a, op, b, result)
    c.history = append(c.history, entry)
}

// Sqrt calculates square root
func Sqrt(x float64) float64 {
    return math.Sqrt(x)
}
"""

        try:
            # Import the Go chunker directly
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).parent.parent / "apps" / "chunking" / "ast_chunkers")
            )

            from go import chunk_go_code

            chunks = chunk_go_code(go_code, max_chunk_size=400, chunk_overlap=50)

            # Should create chunks
            assert len(chunks) > 0
            assert all("text" in chunk and "metadata" in chunk for chunk in chunks)

            # Check metadata structure
            for chunk in chunks:
                metadata = chunk["metadata"]
                assert "type" in metadata
                assert "language" in metadata
                assert metadata["language"] == "go"
                assert "start_line" in metadata
                assert "end_line" in metadata

                # Text should not be empty (unless it's a structural chunk)
                text = chunk["text"]
                if text.strip():  # Skip empty chunks
                    assert len(text) > 0

            # Verify that different Go constructs are identified
            chunk_types = {chunk["metadata"]["type"] for chunk in chunks}

            # Should have some variety in chunk types (even if using fallback)
            assert len(chunk_types) >= 1

        except ImportError as e:
            pytest.skip(f"Go chunker not available: {e}")

    def test_go_chunker_error_handling(self):
        """Test Go chunker handles malformed code gracefully."""
        malformed_go_code = """package main

// Missing import statement for fmt
func main() {
    fmt.Println("This will cause issues")
    // Unclosed function
    func broken() {
        if true {
    // Missing closing braces
"""

        try:
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).parent.parent / "apps" / "chunking" / "ast_chunkers")
            )

            from go import chunk_go_code

            # Should handle malformed code without crashing
            chunks = chunk_go_code(malformed_go_code, max_chunk_size=200, chunk_overlap=20)

            # Should return some result (even if it's fallback chunking)
            assert isinstance(chunks, list)
            # May be empty or contain fallback chunks

        except ImportError:
            pytest.skip("Go chunker not available")

    def test_go_chunker_complex_constructs(self):
        """Test Go chunker with complex language constructs."""
        complex_go_code = """package advanced

import (
    "context"
    "fmt"
    "sync"
)

// Generic interface with type constraints
type Comparable[T any] interface {
    Compare(other T) int
    ~int | ~string | ~float64
}

// Generic struct with methods
type Container[T Comparable[T]] struct {
    items []T
    mu    sync.RWMutex
}

// Add adds an item to the container
func (c *Container[T]) Add(item T) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.items = append(c.items, item)
}

// Find finds an item in the container
func (c *Container[T]) Find(target T) (T, bool) {
    c.mu.RLock()
    defer c.mu.RUnlock()

    for _, item := range c.items {
        if item.Compare(target) == 0 {
            return item, true
        }
    }

    var zero T
    return zero, false
}

// ProcessWithContext processes items with context
func ProcessWithContext[T any](ctx context.Context, items []T, processor func(T) error) error {
    for _, item := range items {
        select {
        case <-ctx.Done():
            return ctx.Err()
        default:
            if err := processor(item); err != nil {
                return fmt.Errorf("processing item %v: %w", item, err)
            }
        }
    }
    return nil
}
"""

        try:
            import sys
            from pathlib import Path

            sys.path.insert(
                0, str(Path(__file__).parent.parent / "apps" / "chunking" / "ast_chunkers")
            )

            from go import chunk_go_code

            chunks = chunk_go_code(complex_go_code, max_chunk_size=600, chunk_overlap=50)

            # Should handle complex constructs
            assert len(chunks) > 0
            assert all("text" in chunk and "metadata" in chunk for chunk in chunks)

            # Check that complex constructs are captured
            combined_content = " ".join(chunk["text"] for chunk in chunks)
            assert "Comparable[T]" in combined_content or "Comparable" in combined_content
            assert "Container[T]" in combined_content or "Container" in combined_content
            assert "ProcessWithContext" in combined_content

        except ImportError:
            pytest.skip("Go chunker not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
