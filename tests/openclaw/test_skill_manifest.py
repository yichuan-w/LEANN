"""Validate the ClawHub skill manifest and instructions."""


def test_claw_json_required_fields(claw_manifest):
    """claw.json must contain all fields required by ClawHub."""
    required = {"name", "version", "description", "author", "license", "permissions", "entry"}
    missing = required - claw_manifest.keys()
    assert not missing, f"Missing required fields: {missing}"


def test_claw_json_name(claw_manifest):
    assert claw_manifest["name"] == "leann-memory"


def test_claw_json_permissions(claw_manifest):
    """Skill requires shell permission to invoke the leann CLI."""
    assert "shell" in claw_manifest["permissions"]


def test_claw_json_entry_exists(skill_dir, claw_manifest):
    """The entry file referenced in claw.json must exist."""
    entry = skill_dir / claw_manifest["entry"]
    assert entry.exists(), f"Entry file {entry} does not exist"


def test_claw_json_tags(claw_manifest):
    """Should include relevant tags for discoverability."""
    tags = set(claw_manifest.get("tags", []))
    assert "memory" in tags
    assert "search" in tags


def test_claw_json_models(claw_manifest):
    """Should declare compatible model families."""
    models = claw_manifest.get("models", [])
    assert len(models) >= 1, "Must declare at least one compatible model"


def test_instructions_contains_build_command(skill_dir, claw_manifest):
    """instructions.md should tell the agent how to build an index."""
    instructions = (skill_dir / claw_manifest["entry"]).read_text(encoding="utf-8")
    assert "leann build" in instructions


def test_instructions_contains_search_command(skill_dir, claw_manifest):
    """instructions.md should tell the agent how to search."""
    instructions = (skill_dir / claw_manifest["entry"]).read_text(encoding="utf-8")
    assert "leann search" in instructions
    assert "--json" in instructions


def test_instructions_contains_install_check(skill_dir, claw_manifest):
    """instructions.md should have a prerequisite check for leann installation."""
    instructions = (skill_dir / claw_manifest["entry"]).read_text(encoding="utf-8")
    assert "which leann" in instructions or "leann --version" in instructions


def test_readme_exists(skill_dir):
    """README.md should exist for human-facing documentation."""
    readme = skill_dir / "README.md"
    assert readme.exists(), "README.md missing from skill directory"
