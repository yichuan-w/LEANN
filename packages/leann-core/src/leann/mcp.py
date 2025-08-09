#!/usr/bin/env python3

import json
import subprocess
import sys


def handle_request(request):
    if request.get("method") == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "capabilities": {"tools": {}},
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "leann-mcp", "version": "1.0.0"},
            },
        }

    elif request.get("method") == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "leann_index",
                        "description": """🏗️ Index a codebase for intelligent code search and understanding.

🎯 **When to use**: Before analyzing, modifying, or understanding any codebase
📁 **What it does**: Creates a semantic search index of code files and documentation
⚡ **Why it's useful**: Enables fast, intelligent searches like "authentication logic", "error handling patterns", "API endpoints"

This is your first step for any serious codebase work - think of it as giving yourself superpowers to understand and navigate code.""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Name for the new index. Use descriptive names like 'my-project' or 'backend-api'.",
                                },
                                "docs_path": {
                                    "type": "string",
                                    "description": "Path to the directory containing code/documents to index. Can be relative (e.g., './src') or absolute.",
                                },
                                "force": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Force rebuild of existing index. Use when you want to completely reindex and overwrite existing data.",
                                },
                                "backend": {
                                    "type": "string",
                                    "enum": ["hnsw", "diskann"],
                                    "default": "hnsw",
                                    "description": "Vector index backend: 'hnsw' for balanced performance, 'diskann' for large-scale datasets.",
                                },
                                "embedding_model": {
                                    "type": "string",
                                    "default": "facebook/contriever",
                                    "description": "Embedding model to use. Popular options: 'facebook/contriever', 'sentence-transformers/all-MiniLM-L6-v2'",
                                },
                                "file_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "File extensions to include (e.g., ['.py', '.js', '.ts', '.md']). If not specified, uses default supported types.",
                                },
                                "ignore_patterns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "default": [],
                                    "description": "Patterns to ignore during indexing (e.g., ['node_modules', '__pycache__', '*.tmp', 'dist']). Common patterns are automatically ignored.",
                                },
                            },
                            "required": ["index_name", "docs_path"],
                        },
                    },
                    {
                        "name": "leann_search",
                        "description": """🔍 Search code using natural language - like having a coding assistant who knows your entire codebase!

🎯 **Perfect for**:
- "How does authentication work?" → finds auth-related code
- "Error handling patterns" → locates try-catch blocks and error logic
- "Database connection setup" → finds DB initialization code
- "API endpoint definitions" → locates route handlers
- "Configuration management" → finds config files and usage

💡 **Pro tip**: Use this before making any changes to understand existing patterns and conventions.""",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Name of the LEANN index to search. Use 'leann_list' first to see available indexes.",
                                },
                                "query": {
                                    "type": "string",
                                    "description": "Search query - can be natural language (e.g., 'how to handle errors') or technical terms (e.g., 'async function definition')",
                                },
                                "top_k": {
                                    "type": "integer",
                                    "default": 5,
                                    "minimum": 1,
                                    "maximum": 20,
                                    "description": "Number of search results to return. Use 5-10 for focused results, 15-20 for comprehensive exploration.",
                                },
                                "complexity": {
                                    "type": "integer",
                                    "default": 32,
                                    "minimum": 16,
                                    "maximum": 128,
                                    "description": "Search complexity level. Use 16-32 for fast searches (recommended), 64+ for higher precision when needed.",
                                },
                                "search_mode": {
                                    "type": "string",
                                    "enum": ["fast", "balanced", "precise"],
                                    "default": "balanced",
                                    "description": "Search strategy: 'fast' (~2-5s), 'balanced' (~5-10s), 'precise' (~10-20s). Choose based on time vs accuracy needs.",
                                },
                                "recompute_embeddings": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Recompute embeddings for maximum accuracy. Enable only when precision is more important than speed.",
                                },
                                "file_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Filter results by file types (e.g., ['py', 'js', 'ts']). Searches all indexed file types if not specified.",
                                },
                                "min_score": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "default": 0.0,
                                    "description": "Minimum relevance score threshold (0.0-1.0). Higher values return more relevant but fewer results.",
                                },
                            },
                            "required": ["index_name", "query"],
                        },
                    },
                    {
                        "name": "leann_status",
                        "description": "📊 Check the health and stats of your code indexes - like a medical checkup for your codebase knowledge!",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Optional: Name of specific index to check. If not provided, shows status of all indexes.",
                                }
                            },
                        },
                    },
                    {
                        "name": "leann_clear",
                        "description": "🗑️ Safely delete a code index (with confirmation required). Think of it as 'rm -rf' but for your search indexes - be careful!",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {
                                    "type": "string",
                                    "description": "Name of the index to clear/delete.",
                                },
                                "confirm": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Confirmation flag. Must be set to true to actually perform the deletion.",
                                },
                            },
                            "required": ["index_name"],
                        },
                    },
                    {
                        "name": "leann_list",
                        "description": "📋 Show all your indexed codebases - your personal code library! Use this to see what's available for search.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }

    elif request.get("method") == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        try:
            if tool_name == "leann_index":
                # Validate required parameters
                if not args.get("index_name") or not args.get("docs_path"):
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Error: Both index_name and docs_path are required",
                                }
                            ]
                        },
                    }

                # Validate docs_path exists
                import os

                docs_path = args["docs_path"]
                if not os.path.exists(docs_path):
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Error: Path '{docs_path}' does not exist",
                                }
                            ]
                        },
                    }

                # Build index command
                cmd = [
                    "leann",
                    "build",
                    args["index_name"],
                    "--docs",
                    docs_path,
                    "--backend",
                    args.get("backend", "hnsw"),
                    "--embedding-model",
                    args.get("embedding_model", "facebook/contriever"),
                ]

                # Add force flag if specified
                if args.get("force", False):
                    cmd.append("--force")

                # Add file types if specified (now as array)
                file_types = args.get("file_types")
                if file_types and isinstance(file_types, list):
                    cmd.extend(["--file-types", ",".join(file_types)])

                # Add ignore patterns if specified
                ignore_patterns = args.get("ignore_patterns", [])
                if ignore_patterns and isinstance(ignore_patterns, list):
                    # For now, pass as comma-separated string - CLI can be enhanced later
                    cmd.extend(["--ignore", ",".join(ignore_patterns)])
                result = subprocess.run(cmd, capture_output=True, text=True)

            elif tool_name == "leann_search":
                # Validate required parameters
                if not args.get("index_name") or not args.get("query"):
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Error: Both index_name and query are required",
                                }
                            ]
                        },
                    }

                # Build command with enhanced parameters
                cmd = [
                    "leann",
                    "search",
                    args["index_name"],
                    args["query"],
                    f"--top-k={args.get('top_k', 5)}",
                ]

                # Handle search mode mapping to set complexity and beam width
                search_mode = args.get("search_mode", "balanced")
                if search_mode == "fast":
                    cmd.extend(["--complexity=16", "--beam-width=1"])
                elif search_mode == "precise":
                    cmd.extend(["--complexity=64", "--beam-width=2"])
                else:  # balanced mode
                    complexity = args.get("complexity", 32)
                    cmd.append(f"--complexity={complexity}")

                # Handle recompute embeddings
                if args.get("recompute_embeddings", False):
                    cmd.append("--recompute-embeddings")

                # Handle file types filtering
                file_types = args.get("file_types")
                if file_types and isinstance(file_types, list):
                    # Validate file extensions
                    valid_extensions = []
                    for ext in file_types:
                        if isinstance(ext, str) and ext.strip():
                            clean_ext = ext.strip()
                            if not clean_ext.startswith("."):
                                clean_ext = "." + clean_ext
                            valid_extensions.append(clean_ext)

                    if valid_extensions:
                        cmd.extend(["--filter-extensions", ",".join(valid_extensions)])

                result = subprocess.run(cmd, capture_output=True, text=True)

                # Handle min_score filtering in post-processing if needed
                min_score = args.get("min_score", 0.0)
                if min_score > 0.0 and result.returncode == 0:
                    # Note: This is a basic implementation. For full support,
                    # the CLI would need to return structured data for filtering
                    pass

            elif tool_name == "leann_status":
                if args.get("index_name"):
                    # Check specific index status - for now, we'll use leann list and filter
                    result = subprocess.run(["leann", "list"], capture_output=True, text=True)
                    # We could enhance this to show more detailed status per index
                else:
                    # Show all indexes status
                    result = subprocess.run(["leann", "list"], capture_output=True, text=True)

            elif tool_name == "leann_clear":
                index_name = args["index_name"]
                confirm = args.get("confirm", False)

                if not confirm:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Warning: This will permanently delete index '{index_name}'. To proceed, call this tool again with confirm=true.",
                                }
                            ]
                        },
                    }

                # For clearing, we need to implement this in the CLI
                # For now, we'll return a message explaining the limitation
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Clear functionality for index '{index_name}' is not yet implemented in CLI. You can manually delete the index files in .leann/indexes/{index_name}/",
                            }
                        ]
                    },
                }

            elif tool_name == "leann_list":
                result = subprocess.run(["leann", "list"], capture_output=True, text=True)

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result.stdout
                            if result.returncode == 0
                            else f"Error: {result.stderr}",
                        }
                    ]
                },
            }

        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -1, "message": str(e)},
            }


def main():
    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            response = handle_request(request)
            if response:
                print(json.dumps(response))
                sys.stdout.flush()
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -1, "message": str(e)},
            }
            print(json.dumps(error_response))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
