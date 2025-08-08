# LEANN Claude Code Integration

Intelligent code assistance using LEANN's vector search directly in Claude Code.

## Prerequisites

First, install LEANN CLI globally:

```bash
uv tool install leann-core
```

This makes the `leann` command available system-wide, which `leann_mcp` requires.

## Quick Setup

Add the LEANN MCP server to Claude Code:

```bash
claude mcp add leann-server -- leann_mcp
```

## Available Tools

- **`leann_list`** - List available indexes across all projects
- **`leann_search`** - Search code and documents with semantic queries
- **`leann_ask`** - Ask questions and get AI-powered answers from your codebase

## Quick Start

```bash
# Build an index for your project
leann build my-project

# Start Claude Code
claude
```

Then in Claude Code:
```
Help me understand this codebase. List available indexes and search for authentication patterns.
```

<p align="center">
  <img src="../../assets/claude_code_leann.png" alt="LEANN in Claude Code" width="80%">
</p>


## How It Works

- **`leann`** - Core CLI tool for indexing and searching (installed globally)
- **`leann_mcp`** - MCP server that wraps `leann` commands for Claude Code integration
- Claude Code calls `leann_mcp`, which executes `leann` commands and returns results

## File Support

Python, JavaScript, TypeScript, Java, Go, Rust, SQL, YAML, JSON, and 30+ more file types.

## Storage

- Project indexes in `.leann/` directory (like `.git`)
- Global project registry at `~/.leann/projects.json`
- Multi-project support built-in

## Removing

```bash
claude mcp remove leann-server
```
