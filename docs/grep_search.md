# LEANN Grep Search Usage Guide

## Overview

LEANN's grep search functionality provides exact text matching for finding specific code patterns, error messages, function names, or exact phrases in your indexed documents.

## Basic Usage

### Simple Grep Search

```python
from leann.api import LeannSearcher

searcher = LeannSearcher("your_index_path")

# Exact text search
results = searcher.search("def authenticate_user", use_grep=True, top_k=5)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.text[:100]}...")
    print("-" * 40)
```

### Comparison: Semantic vs Grep Search

```python
# Semantic search - finds conceptually similar content
semantic_results = searcher.search("machine learning algorithms", top_k=3)

# Grep search - finds exact text matches
grep_results = searcher.search("def train_model", use_grep=True, top_k=3)
```

## When to Use Grep Search

### Use Cases

- **Code Search**: Finding specific function definitions, class names, or variable references
- **Error Debugging**: Locating exact error messages or stack traces
- **Documentation**: Finding specific API endpoints or exact terminology

### Examples

```python
# Find function definitions
functions = searcher.search("def __init__", use_grep=True)

# Find import statements
imports = searcher.search("from sklearn import", use_grep=True)

# Find specific error types
errors = searcher.search("FileNotFoundError", use_grep=True)

# Find TODO comments
todos = searcher.search("TODO:", use_grep=True)

# Find configuration entries
configs = searcher.search("server_port=", use_grep=True)
```

## Technical Details

### How It Works

1. **File Location**: Grep search operates on the raw text stored in `.jsonl` files
2. **Command Execution**: Uses the system `grep` command with case-insensitive search
3. **Result Processing**: Parses JSON lines and extracts text and metadata
4. **Scoring**: Simple frequency-based scoring based on query term occurrences

### Search Process

```
Query: "def train_model" 
  ↓
grep -i -n "def train_model" documents.leann.passages.jsonl
  ↓
Parse matching JSON lines
  ↓
Calculate scores based on term frequency
  ↓
Return top_k results
```

### Scoring Algorithm

```python
# Term frequency in document
score = text.lower().count(query.lower())
```

Results are ranked by score (highest first), with higher scores indicating more occurrences of the search term.

## Error Handling

### Common Issues

#### Grep Command Not Found
```
RuntimeError: grep command not found. Please install grep or use semantic search.
```

**Solution**: Install grep on your system:
- **Ubuntu/Debian**: `sudo apt-get install grep`
- **macOS**: grep is pre-installed
- **Windows**: Use WSL or install grep via Git Bash/MSYS2

#### No Results Found
```python
# Check if your query exists in the raw data
results = searcher.search("your_query", use_grep=True)
if not results:
    print("No exact matches found. Try:")
    print("1. Check spelling and case")
    print("2. Use partial terms")
    print("3. Switch to semantic search")
```

## Complete Example

```python
#!/usr/bin/env python3
"""
Grep Search Example
Demonstrates grep search for exact text matching.
"""

from leann.api import LeannSearcher

def demonstrate_grep_search():
    # Initialize searcher
    searcher = LeannSearcher("my_index")
    
    print("=== Function Search ===")
    functions = searcher.search("def __init__", use_grep=True, top_k=5)
    for i, result in enumerate(functions, 1):
        print(f"{i}. Score: {result.score}")
        print(f"   Preview: {result.text[:60]}...")
        print()
    
    print("=== Error Search ===")
    errors = searcher.search("FileNotFoundError", use_grep=True, top_k=3)
    for result in errors:
        print(f"Content: {result.text.strip()}")
        print("-" * 40)

if __name__ == "__main__":
    demonstrate_grep_search()
```
