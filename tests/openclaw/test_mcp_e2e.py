"""
End-to-end test for the MCP server: build a real index, then invoke
handle_request(tools/call → leann_search) and verify JSON results come back
through the full MCP → subprocess → leann CLI pipeline.

Requires the embedding model and `leann` on PATH (satisfied by uv run pytest).
Marked 'slow'.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "leann-core" / "src"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("CI") == "true", reason="Skip in CI — needs model download"),
    pytest.mark.skipif(not shutil.which("leann"), reason="leann CLI not on PATH"),
]


@pytest.fixture(scope="module")
def mcp_index(tmp_path_factory):
    """Build a real LEANN index once for the entire module."""
    from leann.cli import LeannCLI

    tmp = tmp_path_factory.mktemp("mcp_e2e")
    fixtures = Path(__file__).parent / "fixtures"
    docs = tmp / "docs"
    shutil.copytree(fixtures, docs)

    cli = LeannCLI()
    cli.indexes_dir = tmp / ".leann" / "indexes"
    cli.indexes_dir.mkdir(parents=True)

    parser = cli.create_parser()
    args = parser.parse_args(
        [
            "build",
            "openclaw-memory",
            "--docs",
            str(docs),
            "--backend-name",
            "hnsw",
            "--no-compact",
            "--embedding-model",
            "all-MiniLM-L6-v2",
            "--embedding-mode",
            "sentence-transformers",
        ]
    )
    asyncio.get_event_loop().run_until_complete(cli.build_index(args))

    index_dir = cli.indexes_dir / "openclaw-memory"
    assert (index_dir / "documents.leann.meta.json").exists(), "Index build failed"
    return index_dir


def _project_root(mcp_index):
    """The temp dir that contains .leann/indexes/ — used as cwd for subprocess.

    mcp_index = tmp/.leann/indexes/openclaw-memory
    We need tmp (3 parents up) so LeannCLI finds .leann/indexes/.
    """
    return str(mcp_index.parent.parent.parent)


def test_mcp_search_via_subprocess(mcp_index):
    """Invoke `leann search --json` as a subprocess (same as MCP server does)."""
    leann_bin = shutil.which("leann")
    assert leann_bin is not None, "leann not found on PATH"

    cmd = [
        leann_bin,
        "search",
        "openclaw-memory",
        "gRPC migration benchmarks",
        "--top-k=3",
        "--complexity=32",
        "--non-interactive",
        "--json",
    ]
    cwd = _project_root(mcp_index)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, f"leann search failed (cwd={cwd}): {result.stderr[:500]}"
    assert result.stdout.strip(), f"No output from leann search (stderr: {result.stderr[:500]})"

    results = json.loads(result.stdout)
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("text" in r and "score" in r for r in results)
    assert all(isinstance(r["score"], float) for r in results), "score must be native float"


def test_mcp_search_relevance(mcp_index):
    """Search for 'payment gateway' should return relevant content."""
    leann_bin = shutil.which("leann")
    assert leann_bin is not None, "leann not found on PATH"

    cmd = [
        leann_bin,
        "search",
        "openclaw-memory",
        "payment gateway timeout hotfix",
        "--top-k=3",
        "--complexity=32",
        "--non-interactive",
        "--json",
    ]
    cwd = _project_root(mcp_index)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    assert result.returncode == 0, f"leann search failed (cwd={cwd}): {result.stderr[:500]}"
    assert result.stdout.strip(), f"No output from leann search (stderr: {result.stderr[:500]})"

    results = json.loads(result.stdout)
    top_text = results[0]["text"].lower()
    assert any(kw in top_text for kw in ["payment", "gateway", "hotfix", "timeout"])


def test_mcp_list(mcp_index):
    """leann list should show the openclaw-memory index."""
    leann_bin = shutil.which("leann")
    assert leann_bin is not None, "leann not found on PATH"

    result = subprocess.run(
        [leann_bin, "list"],
        capture_output=True,
        text=True,
        cwd=_project_root(mcp_index),
    )
    assert result.returncode == 0, f"leann list failed: {result.stderr[:300]}"
    assert "openclaw-memory" in result.stdout


def test_mcp_stdio_protocol(mcp_index):
    """Spawn the MCP server as a subprocess, send JSON-RPC via stdin, read responses."""
    leann_mcp_bin = shutil.which("leann_mcp")
    if not leann_mcp_bin:
        pytest.skip("leann_mcp not on PATH")
    assert leann_mcp_bin is not None

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    ]
    stdin_data = "\n".join(json.dumps(r) for r in requests) + "\n"

    proc = subprocess.run(
        [leann_mcp_bin],
        input=stdin_data,
        capture_output=True,
        text=True,
        timeout=10,
    )

    lines = [line for line in proc.stdout.strip().split("\n") if line.strip()]
    assert len(lines) >= 2, f"Expected 2 responses, got {len(lines)}: {proc.stdout[:300]}"

    init_resp = json.loads(lines[0])
    assert init_resp["id"] == 1
    assert init_resp["result"]["serverInfo"]["name"] == "leann-mcp"

    list_resp = json.loads(lines[1])
    assert list_resp["id"] == 2
    tool_names = {t["name"] for t in list_resp["result"]["tools"]}
    assert "leann_search" in tool_names
    assert "leann_list" in tool_names
