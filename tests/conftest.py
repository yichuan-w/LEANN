"""Minimal test session cleanup to prevent hanging background servers.

This keeps the test suite simple while ensuring any stray embedding server
processes are terminated at session start and end.
"""

import subprocess


def _kill_embedding_servers() -> None:
    patterns = [
        "hnsw_embedding_server",
        "diskann_embedding_server",
        "embedding_server",
    ]
    for pat in patterns:
        try:
            subprocess.run(["pkill", "-9", "-f", pat], timeout=2, capture_output=True)
        except Exception:
            pass


def pytest_sessionstart(session):
    _kill_embedding_servers()


def pytest_sessionfinish(session, exitstatus):
    _kill_embedding_servers()
