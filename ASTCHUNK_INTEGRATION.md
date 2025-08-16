# AST-Aware Code Chunking Integration

## Overview

This document describes the integration of **astchunk** library into LEANN for AST-aware code chunking. This enhancement provides better semantic understanding of code structure compared to traditional text-based chunking.

## 🚀 Features Added

### 1. Enhanced Chunking Utilities (`apps/chunking/`)
- **AST-aware chunking** for Python, Java, C#, TypeScript files
- **Automatic language detection** from file extensions  
- **Fallback mechanisms** to traditional chunking when AST fails
- **Configurable chunk sizes** and overlap parameters
- **Graceful error handling** for unsupported languages

### 2. Updated Document RAG (`apps/document_rag.py`)
- New `--enable-code-chunking` flag for AST-aware processing
- Backward compatibility with existing workflows
- Automatic detection and processing of code vs text files

### 3. Specialized Code RAG Application (`apps/code_rag.py`)
- Dedicated application for code repository indexing
- Optimized parameters for code understanding
- Automatic exclusion of build directories and cache files
- Support for common code file extensions

### 4. Enhanced CLI Support (`packages/leann-core/src/leann/cli.py`)
- New AST chunking command-line arguments:
  - `--use-ast-chunking`: Enable AST-aware chunking
  - `--ast-chunk-size`: Set AST chunk size (default: 768)
  - `--ast-chunk-overlap`: Set AST chunk overlap (default: 96)
  - `--ast-fallback-traditional`: Enable fallback mode

### 5. Comprehensive Test Suite (`tests/test_astchunk_integration.py`)
- Unit tests for chunking functions
- Integration tests with document RAG
- Error handling and edge case testing
- Mock objects for testing without dependencies

## 🛠️ Installation

### Dependencies Added to `pyproject.toml`
```toml
# AST-aware code chunking dependencies
"astchunk>=0.1.0",
"tree-sitter>=0.20.0", 
"tree-sitter-python>=0.20.0",
"tree-sitter-java>=0.20.0",
"tree-sitter-c-sharp>=0.20.0",
"tree-sitter-typescript>=0.20.0",
```

### Install Dependencies
```bash
# Install LEANN with AST chunking support
uv pip install -e "."

# Or install dependencies manually
uv pip install astchunk tree-sitter tree-sitter-python tree-sitter-java tree-sitter-c-sharp tree-sitter-typescript
```

## 📖 Usage Examples

### 1. Document RAG with Code Chunking
```bash
# Enable AST chunking for mixed content
python -m apps.document_rag --enable-code-chunking --data-dir ./my_project

# Query code and documentation together
python -m apps.document_rag --enable-code-chunking --query "How does the authentication system work?"
```

### 2. Specialized Code RAG
```bash
# Index an entire code repository  
python -m apps.code_rag --repo-dir ./my_codebase

# Index with custom settings
python -m apps.code_rag --repo-dir ./src --ast-chunk-size 1024 --query "Show me the database connection logic"

# Include specific file types
python -m apps.code_rag --include-extensions .py .js .ts --query "Find all API endpoints"
```

### 3. Global CLI with AST Support
```bash
# Build index with AST chunking
leann build my-code-index --docs ./src --use-ast-chunking --ast-chunk-size 512

# Search with traditional CLI
leann search my-code-index "database connection"
leann ask my-code-index --interactive
```

### 4. Advanced Configuration
```bash
# Fine-tune AST chunking parameters
python -m apps.code_rag \
    --repo-dir ./large_project \
    --ast-chunk-size 1024 \
    --ast-chunk-overlap 128 \
    --max-file-size 2000000 \
    --exclude-dirs .git __pycache__ node_modules build dist
```

## 🧪 Testing

### Run Integration Tests
```bash
# Quick integration test (no dependencies required)
python3 test_astchunk_integration_manual.py

# Full test suite
pytest tests/test_astchunk_integration.py -v

# Test with existing document RAG tests
pytest tests/test_document_rag.py::test_document_rag_with_ast_chunking -v
```

### Test Sample Code Files
The integration includes sample code files in `data/code_samples/`:
- `vector_search.py` - Python vector search implementation
- `DataProcessor.java` - Java data processing utilities  
- `text_analyzer.ts` - TypeScript text analysis module
- `ImageProcessor.cs` - C# image processing library

## 🔧 Technical Implementation

### Language Support
| Extension | Language | AST Parser |
|-----------|----------|------------|
| `.py` | Python | tree-sitter-python |
| `.java` | Java | tree-sitter-java |
| `.cs` | C# | tree-sitter-c-sharp |
| `.ts`, `.tsx` | TypeScript | tree-sitter-typescript |
| `.js`, `.jsx` | JavaScript | tree-sitter-typescript |

### Chunking Strategy
1. **File Detection**: Automatically detect code vs text files by extension
2. **Language Mapping**: Map file extensions to AST parsers
3. **AST Chunking**: Use astchunk for structure-aware parsing
4. **Fallback**: Traditional text chunking for unsupported languages
5. **Error Handling**: Graceful degradation on parsing errors

### Key Functions
- `detect_code_files()`: Separate code and text documents
- `create_ast_chunks()`: AST-aware chunking with astchunk
- `create_text_chunks()`: Unified chunking interface
- `get_language_from_extension()`: Language detection

## 🎯 Benefits

### For Developers
- **Better Code Understanding**: Preserve function/class boundaries
- **Improved Search Quality**: Semantic context maintained
- **Language Agnostic**: Support for multiple programming languages
- **Backward Compatible**: Existing workflows continue to work

### For LEANN Users
- **Enhanced Claude Code Integration**: Better context for code assistance
- **Mixed Content Support**: Handle code + documentation seamlessly
- **Optimized Parameters**: Separate chunk sizes for code vs text
- **Flexible Configuration**: Enable/disable per use case

## 🔄 Migration Guide

### Existing Users
No changes required - all existing functionality remains the same.

### To Enable AST Chunking
1. **Document RAG**: Add `--enable-code-chunking` flag
2. **CLI**: Add `--use-ast-chunking` flag  
3. **Code RAG**: Use the new `apps/code_rag.py` application

### Configuration Migration
```bash
# Before (traditional)
python -m apps.document_rag --chunk-size 256

# After (with AST support)  
python -m apps.document_rag --enable-code-chunking --chunk-size 256 --ast-chunk-size 512
```

## 🐛 Troubleshooting

### Common Issues

1. **astchunk not available**
   - Install: `uv pip install astchunk`
   - Falls back to traditional chunking automatically

2. **Tree-sitter parsers missing**
   - Install language-specific parsers: `uv pip install tree-sitter-python`
   - Check supported languages in `CODE_EXTENSIONS`

3. **Large files causing issues**
   - Adjust `--max-file-size` parameter
   - Exclude problematic directories with `--exclude-dirs`

4. **Poor chunking quality**
   - Adjust `--ast-chunk-size` (try 512, 768, 1024)
   - Modify `--ast-chunk-overlap` (try 64, 96, 128)

### Debug Mode
Set environment variable for verbose logging:
```bash
export LEANN_LOG_LEVEL=DEBUG
python -m apps.code_rag --repo-dir ./my_code
```

## 🔮 Future Enhancements

1. **Additional Language Support**: Go, Rust, Swift, Kotlin
2. **Custom AST Rules**: User-defined chunking strategies
3. **Semantic Chunking**: Content-aware chunk boundaries
4. **Performance Optimizations**: Parallel AST processing
5. **IDE Integration**: Direct integration with development environments

## 📚 References

- [astchunk GitHub Repository](https://github.com/yilinjz/astchunk)
- [tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [LEANN Documentation](README.md)
- [cAST Research Paper](https://arxiv.org/abs/your-paper-reference)

---

**Note**: This integration maintains full backward compatibility while adding powerful new capabilities for code-aware RAG applications.