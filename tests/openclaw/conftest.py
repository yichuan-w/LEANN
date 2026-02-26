"""Shared fixtures for OpenClaw integration tests."""

import json
import shutil
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def memory_fixtures(tmp_path):
    """Copy memory fixture files into a temp directory and return the path."""
    dest = tmp_path / "memory_docs"
    shutil.copytree(FIXTURES_DIR, dest)
    return dest


@pytest.fixture
def leann_index_dir(tmp_path):
    """Provide a clean temporary directory for LEANN indexes."""
    idx_dir = tmp_path / ".leann" / "indexes"
    idx_dir.mkdir(parents=True)
    return idx_dir


@pytest.fixture
def skill_dir():
    """Return the path to the leann-memory skill directory."""
    return Path(__file__).parent.parent.parent / "skills" / "leann-memory"


@pytest.fixture
def claw_manifest(skill_dir):
    """Load and return the parsed claw.json manifest."""
    manifest_path = skill_dir / "claw.json"
    assert manifest_path.exists(), f"claw.json not found at {manifest_path}"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
