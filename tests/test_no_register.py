import json

import pytest
from leann.registry import register_project_directory


@pytest.fixture
def project_dir(tmp_path):
    proj = tmp_path / "proj"
    (proj / ".leann" / "indexes" / "dummy").mkdir(parents=True)
    return proj


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def test_no_register_env_set_skips_registration(project_dir, fake_home, monkeypatch):
    # Arrange
    monkeypatch.setenv("LEANN_NO_REGISTER", "1")

    # Act
    register_project_directory(project_dir)

    # Assert
    assert not (fake_home / ".leann" / "projects.json").exists()
    assert not (fake_home / ".leann").exists()


def test_env_unset_registers_project(project_dir, fake_home, monkeypatch):
    """Guard test: proves the fixture is registrable when the switch is off."""
    # Arrange
    monkeypatch.delenv("LEANN_NO_REGISTER", raising=False)

    # Act
    register_project_directory(project_dir)

    # Assert
    registry_file = fake_home / ".leann" / "projects.json"
    assert registry_file.exists()
    projects = json.loads(registry_file.read_text())
    assert str(project_dir.resolve()) in projects
