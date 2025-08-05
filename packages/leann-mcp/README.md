# LEANN MCP Server

**Transform Claude Code into a RAG-Powered Development Assistant**

This package provides a Model Context Protocol (MCP) server that integrates LEANN's vector search and RAG capabilities directly into Claude Code, enabling intelligent code analysis, documentation Q&A, and knowledge-driven development.

## 🚀 Quick Start

### 1. Install

```bash
# Install dependencies
pip install leann mcp

# Clone or download this package
git clone https://github.com/yichuan-w/LEANN.git
cd LEANN-RAG/packages/leann-mcp
```

### 2. Configure Claude Code

Add to your `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "leann-rag": {
      "command": "python",
      "args": ["/absolute/path/to/leann_mcp_server.py"]
    }
  }
}
```

### 3. Start Using

```bash
# Start Claude Code
claude

# In Claude, use LEANN tools:
# "Build an index from my codebase and help me understand the architecture"
```

## 🛠️ Available Tools

### `leann_build`
Build a vector index from documents or code
```python
leann_build(
    index_name="my-project",
    data_path="./src",
    backend="hnsw",  # or "diskann"
    embedding_model="facebook/contriever"
)
```

### `leann_search`
Search through an index for relevant passages
```python
leann_search(
    query="authentication middleware",
    index_name="my-project",
    top_k=10,
    complexity=64
)
```

### `leann_ask`
Ask questions using RAG with LLM responses
```python
leann_ask(
    question="How does user authentication work?",
    index_name="my-project",
    llm_config={"type": "ollama", "model": "qwen3:7b"}
)
```

### `leann_list_indexes`
List all available indexes

### `leann_delete_index`
Delete an index (with confirmation)

## 💡 Use Cases

### 📚 **Code Understanding**
```
"Build an index from my codebase and explain the authentication flow"
```

### 🔍 **Smart Code Search**
```
"Search for error handling patterns in our API endpoints"
```

### 📖 **Documentation Q&A**
```
"Create an index from our docs and answer: What are the deployment requirements?"
```

### 🏗️ **Architecture Analysis**
```
"Analyze our system architecture and suggest improvements"
```

### 🔧 **Development Assistance**
```
"Based on existing code patterns, help me implement user permissions"
```

## 🎯 Key Features

- **🔌 Zero-Config Integration**: Works out of the box with Claude Code
- **🧠 Smart Indexing**: Automatically handles multiple file formats
- **⚡ High Performance**: LEANN's 97% storage savings + fast search
- **🔄 Real-Time**: Build and query indexes during development
- **🎨 Flexible**: Support for multiple backends and embedding models
- **💬 Conversational**: Natural language interface for complex queries

## 📁 Project Structure

```
packages/leann-mcp/
├── leann_mcp_server.py          # Main MCP server implementation
├── requirements.txt             # Python dependencies
├── package.json                 # NPM package metadata
├── claude-config-examples/      # Configuration examples
│   ├── claude-mcp-config.json   # Basic Claude configuration
│   └── usage-examples.md        # Detailed usage examples
└── README.md                    # This file
```

## 🔧 Advanced Configuration

### Custom Index Directory
```python
# In your environment or server config
DEFAULT_CONFIG = {
    "indexes_dir": "/custom/path/to/indexes",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "backend": "diskann"
}
```

### Hook Integration
Automatically reindex when files change:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write.*\\.(py|js|ts)$",
        "hooks": [{"type": "mcp_call", "server": "leann-rag", "tool": "leann_build"}]
      }
    ]
  }
}
```

### Sub-Agent Templates
Create specialized RAG agents in `.claude/agents/`:

```markdown
---
name: code-analyst
description: Code analysis using LEANN RAG
tools: leann_build, leann_search, leann_ask
---

You are a senior code analyst with access to LEANN RAG.
When analyzing code, always:
1. Build indexes of relevant code sections
2. Search for patterns and anti-patterns
3. Provide evidence-based recommendations
```

## 🚀 Performance & Scaling

- **Small Projects** (<1K files): Use HNSW backend
- **Large Codebases** (>10K files): Use DiskANN backend
- **Memory Usage**: ~100MB per index (vs ~10GB traditional)
- **Build Time**: 2-5 minutes for typical project
- **Search Time**: <100ms for most queries

## 🤝 Contributing

This MCP server is part of the larger LEANN project. See the main README for contribution guidelines.

## 📄 License

MIT License - see the main LEANN project for details.

## 🔗 Links

- [LEANN Main Project](../../README.md)
- [Claude Code Documentation](https://docs.anthropic.com/claude/docs/claude-code)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Usage Examples](claude-config-examples/usage-examples.md)

---

**Built with ❤️ by the LEANN team for the Claude Code community**
