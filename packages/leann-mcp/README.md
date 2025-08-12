# 🔥 LEANN Claude Code Integration

Transform your development workflow with intelligent code assistance using LEANN's semantic search directly in Claude Code.

## Prerequisites

**Step 1:** First, complete the basic LEANN installation following the [📦 Installation guide](../../README.md#installation) in the root README:

```bash
uv venv
source .venv/bin/activate
uv pip install leann
```

**Step 2:** Install LEANN globally for MCP integration:
```bash
uv tool install leann-core
```

This makes the `leann` command available system-wide, which `leann_mcp` requires.

## 🚀 Quick Setup

Add the LEANN MCP server to Claude Code:

```bash
claude mcp add leann-server -- leann_mcp
```

## 🛠️ Available Tools

Once connected, you'll have access to these powerful semantic search tools in Claude Code:

- **`leann_list`** - List all available indexes across your projects
- **`leann_search`** - Perform semantic searches across code and documents
- **`leann_ask`** - Ask natural language questions and get AI-powered answers from your codebase

## 🎯 Quick Start Example

```bash
# Build an index for your project (change to your actual path)
leann build my-project --docs ./

# Start Claude Code
claude
```

## 🚀 Advanced Usage Examples

### Index Entire Git Repository
```bash
# Index all tracked files in your git repository, note right now we will skip submodules, but we can add it back easily if you want
leann build my-repo --docs $(git ls-files) --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw

# Index only specific file types from git
leann build my-python-code --docs $(git ls-files "*.py") --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw
```

### Multiple Directories and Files
```bash
# Index multiple directories
leann build my-codebase --docs ./src ./tests ./docs ./config --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw

# Mix files and directories
leann build my-project --docs ./README.md ./src/ ./package.json ./docs/ --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw

# Specific files only
leann build my-configs --docs ./tsconfig.json ./package.json ./webpack.config.js --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw
```

### Advanced Git Integration
```bash
# Index recently modified files
leann build recent-changes --docs $(git diff --name-only HEAD~10..HEAD) --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw

# Index files matching pattern
leann build frontend --docs $(git ls-files "*.tsx" "*.ts" "*.jsx" "*.js") --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw

# Index documentation and config files
leann build docs-and-configs --docs $(git ls-files "*.md" "*.yml" "*.yaml" "*.json" "*.toml") --embedding-mode sentence-transformers --embedding-model all-MiniLM-L6-v2 --backend hnsw
```


**Try this in Claude Code:**
```
Help me understand this codebase. List available indexes and search for authentication patterns.
```

<p align="center">
  <img src="../../assets/claude_code_leann.png" alt="LEANN in Claude Code" width="80%">
</p>


## 🧠 How It Works

The integration consists of three key components working seamlessly together:

- **`leann`** - Core CLI tool for indexing and searching (installed globally via `uv tool install`)
- **`leann_mcp`** - MCP server that wraps `leann` commands for Claude Code integration
- **Claude Code** - Calls `leann_mcp`, which executes `leann` commands and returns intelligent results

## 📁 File Support

LEANN understands **30+ file types** including:
- **Programming**: Python, JavaScript, TypeScript, Java, Go, Rust, C++, C#
- **Data**: SQL, YAML, JSON, CSV, XML
- **Documentation**: Markdown, TXT, PDF
- **And many more!**

## 💾 Storage & Organization

- **Project indexes**: Stored in `.leann/` directory (just like `.git`)
- **Global registry**: Project tracking at `~/.leann/projects.json`
- **Multi-project support**: Switch between different codebases seamlessly
- **Portable**: Transfer indexes between machines with minimal overhead

## 🗑️ Uninstalling

To remove the LEANN MCP server from Claude Code:

```bash
claude mcp remove leann-server
```
To remove LEANN
```
uv pip uninstall leann leann-backend-hnsw leann-core
```
