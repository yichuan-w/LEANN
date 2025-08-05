#!/usr/bin/env python3

import json
import os
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
                        "name": "leann_search",
                        "description": "Search LEANN index",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {"type": "string"},
                                "query": {"type": "string"},
                                "top_k": {"type": "integer", "default": 5},
                            },
                            "required": ["index_name", "query"],
                        },
                    },
                    {
                        "name": "leann_ask",
                        "description": "Ask question using LEANN RAG",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "index_name": {"type": "string"},
                                "question": {"type": "string"},
                            },
                            "required": ["index_name", "question"],
                        },
                    },
                    {
                        "name": "leann_list",
                        "description": "List all LEANN indexes",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }

    elif request.get("method") == "tools/call":
        tool_name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        # Set working directory and environment
        env = os.environ.copy()
        cwd = "/Users/andyl/Projects/LEANN-RAG"

        try:
            if tool_name == "leann_search":
                cmd = [
                    "leann",
                    "search",
                    args["index_name"],
                    args["query"],
                    "--recompute-embeddings",
                    f"--top-k={args.get('top_k', 5)}",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)

            elif tool_name == "leann_ask":
                cmd = f'echo "{args["question"]}" | leann ask {args["index_name"]} --recompute-embeddings --llm ollama --model qwen3:8b'
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env
                )

            elif tool_name == "leann_list":
                result = subprocess.run(
                    ["leann", "list"], capture_output=True, text=True, cwd=cwd, env=env
                )

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
