"""End-to-end integration test: real OpenClaw instance → memory formation → LEANN indexing → search.

Requirements:
    - Docker running with the openclaw-leann-test container (see docker-compose.yml)
    - Ollama running on host with a model available (e.g. qwen3:8b)
    - LEANN installed in the current virtualenv

Run:
    uv run pytest tests/openclaw/test_openclaw_e2e.py -m integration -v --timeout=600
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

DOCKER_CONTAINER = "openclaw-leann-test"
OPENCLAW_BIN = "/app/node_modules/.pnpm/node_modules/.bin/openclaw"
WORKSPACE_DIR = Path(__file__).parent / "docker-data" / "workspace"


def _docker_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", DOCKER_CONTAINER],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _openclaw_agent(message: str, timeout: int = 300) -> dict:
    """Send a message through OpenClaw's full agent CLI and return parsed JSON."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            DOCKER_CONTAINER,
            OPENCLAW_BIN,
            "agent",
            "--agent",
            "main",
            "-m",
            message,
            "--json",
            "--timeout",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )
    assert result.returncode == 0, f"openclaw agent failed: {result.stderr[:500]}"
    return json.loads(result.stdout)


def _leann_cmd(
    args: list[str], cwd: str | None = None, timeout: int = 60
) -> subprocess.CompletedProcess[str]:
    leann_bin = shutil.which("leann")
    assert leann_bin is not None, "leann not found on PATH"
    resolved_cwd: str = cwd if cwd is not None else str(Path(__file__).parent.parent.parent)
    return subprocess.run(
        [leann_bin, *args],
        capture_output=True,
        text=True,
        cwd=resolved_cwd,
        timeout=timeout,
    )


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def openclaw_ready():
    """Ensure OpenClaw Docker container is running."""
    if not _docker_running():
        pytest.skip(f"Docker container '{DOCKER_CONTAINER}' not running")


@pytest.fixture(scope="module")
def openclaw_memory(openclaw_ready):
    """Send messages to OpenClaw and wait for memory file creation.

    Returns the path to the workspace memory directory.
    """
    memory_dir = WORKSPACE_DIR / "memory"
    memory_md = WORKSPACE_DIR / "MEMORY.md"

    already_has_memory = memory_dir.exists() and any(memory_dir.glob("*.md"))
    if already_has_memory and memory_md.exists():
        return memory_dir

    today = time.strftime("%Y-%m-%d")

    _openclaw_agent(
        f"Save to memory/{today}.md: I am TestUser working on LEANN, "
        "a vector database with 97% storage compression via graph-pruned "
        "recomputation. Today I fixed numpy float32 JSON serialization "
        "and C++ printf stdout pollution in FAISS. Stack: Python 3.11, "
        "uv, Cursor IDE, macOS.",
    )

    _openclaw_agent(
        "Update MEMORY.md with my profile: TestUser, developer of LEANN. "
        "LEANN backends: HNSW (FAISS), DiskANN, IVF. "
        "OpenClaw integration via MCP — leann_search and leann_list tools. "
        "Preferred stack: Python 3.11, uv, Cursor, macOS.",
    )

    assert memory_dir.exists(), "OpenClaw did not create memory/ directory"
    md_files = list(memory_dir.glob("*.md"))
    assert len(md_files) > 0, "No .md files in memory/"

    return memory_dir


@pytest.fixture(scope="module")
def leann_index(openclaw_memory):
    """Build a LEANN index over real OpenClaw memory files."""
    index_name = "openclaw-e2e-test"
    memory_md = WORKSPACE_DIR / "MEMORY.md"

    docs_args = ["--docs", str(openclaw_memory)]
    if memory_md.exists():
        docs_args.extend([str(memory_md)])

    result = _leann_cmd(
        [
            "build",
            index_name,
            *docs_args,
            "--embedding-mode",
            "ollama",
            "--embedding-model",
            "nomic-embed-text",
        ],
        timeout=120,
    )
    assert result.returncode == 0, f"leann build failed: {result.stderr[:500]}"

    yield index_name

    _leann_cmd(["remove", index_name], timeout=10)


class TestOpenClawMemoryFormation:
    """Verify OpenClaw actually creates memory files."""

    def test_memory_dir_exists(self, openclaw_memory):
        assert openclaw_memory.exists()

    def test_daily_log_created(self, openclaw_memory):
        md_files = list(openclaw_memory.glob("*.md"))
        assert len(md_files) > 0
        content = md_files[0].read_text(encoding="utf-8")
        assert len(content) > 50, "Daily log is too short to be real"

    def test_daily_log_content(self, openclaw_memory):
        md_files = list(openclaw_memory.glob("*.md"))
        content = md_files[0].read_text(encoding="utf-8").lower()
        assert any(kw in content for kw in ["leann", "testuser", "vector"]), (
            f"Daily log missing expected keywords: {content[:200]}"
        )

    def test_memory_md_exists(self, openclaw_memory):
        memory_md = WORKSPACE_DIR / "MEMORY.md"
        assert memory_md.exists(), "MEMORY.md not created by OpenClaw"

    def test_memory_md_content(self, openclaw_memory):
        memory_md = WORKSPACE_DIR / "MEMORY.md"
        if not memory_md.exists():
            pytest.skip("MEMORY.md not created")
        content = memory_md.read_text(encoding="utf-8").lower()
        assert any(kw in content for kw in ["leann", "hnsw", "mcp"]), (
            f"MEMORY.md missing expected keywords: {content[:200]}"
        )


class TestLeannIndexOnRealMemory:
    """Verify LEANN can index and search real OpenClaw memory."""

    def test_index_built(self, leann_index):
        result = _leann_cmd(["list"])
        assert result.returncode == 0
        assert leann_index in result.stdout

    def test_search_returns_results(self, leann_index):
        result = _leann_cmd(
            [
                "search",
                leann_index,
                "vector database storage compression",
                "--top-k=3",
                "--non-interactive",
                "--json",
            ]
        )
        assert result.returncode == 0, f"search failed: {result.stderr[:300]}"
        results = json.loads(result.stdout)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_search_relevance_bug_fix(self, leann_index):
        result = _leann_cmd(
            [
                "search",
                leann_index,
                "numpy float32 JSON serialization bug",
                "--top-k=3",
                "--non-interactive",
                "--json",
            ]
        )
        assert result.returncode == 0
        results = json.loads(result.stdout)
        assert len(results) > 0
        top_text = results[0]["text"].lower()
        assert any(kw in top_text for kw in ["numpy", "float", "json", "serializ"]), (
            f"Top result not relevant: {top_text[:200]}"
        )

    def test_search_relevance_project_info(self, leann_index):
        result = _leann_cmd(
            [
                "search",
                leann_index,
                "what backends does LEANN support",
                "--top-k=3",
                "--non-interactive",
                "--json",
            ]
        )
        assert result.returncode == 0
        results = json.loads(result.stdout)
        assert len(results) > 0
        all_text = " ".join(r["text"].lower() for r in results)
        assert any(kw in all_text for kw in ["hnsw", "diskann", "backend"]), (
            f"Results don't mention backends: {all_text[:300]}"
        )

    def test_search_score_is_native_float(self, leann_index):
        result = _leann_cmd(
            [
                "search",
                leann_index,
                "LEANN project",
                "--top-k=3",
                "--non-interactive",
                "--json",
            ]
        )
        assert result.returncode == 0
        results = json.loads(result.stdout)
        assert all(isinstance(r["score"], float) for r in results), (
            "score must be native float, not numpy.float32"
        )

    def test_search_metadata_has_file_path(self, leann_index):
        result = _leann_cmd(
            [
                "search",
                leann_index,
                "TestUser developer",
                "--top-k=1",
                "--non-interactive",
                "--json",
            ]
        )
        assert result.returncode == 0
        results = json.loads(result.stdout)
        assert len(results) > 0
        meta = results[0].get("metadata", {})
        assert "file_name" in meta
        assert meta["file_name"].endswith(".md")
