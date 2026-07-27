import asyncio
import json
import sys

import pytest
from leann import registry
from leann.cli import LeannCLI
from leann.registry import iter_index_meta_files


def test_index_discovery_finds_metadata_within_default_depth(tmp_path):
    meta_file = tmp_path / "project" / "data" / "sample.leann.meta.json"
    meta_file.parent.mkdir(parents=True)
    meta_file.touch()

    assert list(iter_index_meta_files(tmp_path)) == [meta_file]


def test_index_discovery_does_not_descend_past_max_depth(tmp_path):
    shallow_meta = tmp_path / "one" / "shallow.leann.meta.json"
    shallow_meta.parent.mkdir()
    shallow_meta.touch()
    deep_meta = tmp_path / "one" / "two" / "deep.leann.meta.json"
    deep_meta.parent.mkdir()
    deep_meta.touch()

    assert list(iter_index_meta_files(tmp_path, max_depth=1)) == [shallow_meta]


@pytest.mark.parametrize("excluded_dir", [".git", ".venv", "node_modules", "Library"])
def test_index_discovery_skips_large_irrelevant_directories(tmp_path, excluded_dir):
    hidden_meta = tmp_path / excluded_dir / "hidden.leann.meta.json"
    hidden_meta.parent.mkdir()
    hidden_meta.touch()

    assert list(iter_index_meta_files(tmp_path)) == []


def test_index_discovery_rejects_negative_max_depth(tmp_path):
    with pytest.raises(ValueError, match="max_depth must be non-negative"):
        list(iter_index_meta_files(tmp_path, max_depth=-1))


def test_list_parser_accepts_custom_max_depth():
    args = LeannCLI().create_parser().parse_args(["list", "--max-depth", "5"])

    assert args.max_depth == 5


def test_list_parser_rejects_negative_max_depth(capsys):
    with pytest.raises(SystemExit):
        LeannCLI().create_parser().parse_args(["list", "--max-depth", "-1"])

    assert "must be non-negative" in capsys.readouterr().err


def test_cli_project_discovery_respects_max_depth(tmp_path):
    shallow_meta = tmp_path / "shallow" / "index.leann.meta.json"
    shallow_meta.parent.mkdir()
    shallow_meta.touch()
    deep_meta = tmp_path / "one" / "two" / "index.leann.meta.json"
    deep_meta.parent.mkdir(parents=True)
    deep_meta.touch()

    indexes = LeannCLI()._discover_indexes_in_project(tmp_path, max_depth=1)

    assert [index["name"] for index in indexes] == ["shallow"]


def test_list_command_passes_max_depth_to_discovery(monkeypatch):
    cli = LeannCLI()
    received = {}

    def capture_max_depth(max_depth):
        received["max_depth"] = max_depth

    monkeypatch.setattr(cli, "list_indexes", capture_max_depth)
    monkeypatch.setattr(sys, "argv", ["leann", "list", "--max-depth", "7"])

    asyncio.run(cli.run())

    assert received == {"max_depth": 7}


def test_project_registration_does_not_scan_beyond_default_depth(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    deep_meta = (
        project / "one" / "two" / "three" / "four" / "five" / "six" / "index.leann.meta.json"
    )
    deep_meta.parent.mkdir(parents=True)
    deep_meta.touch()
    monkeypatch.setattr(registry.Path, "home", classmethod(lambda cls: home))

    registry.register_project_directory(project)

    assert not (home / ".leann" / "projects.json").exists()


def test_project_registration_accepts_custom_max_depth(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    deep_meta = (
        project / "one" / "two" / "three" / "four" / "five" / "six" / "index.leann.meta.json"
    )
    deep_meta.parent.mkdir(parents=True)
    deep_meta.touch()
    monkeypatch.setattr(registry.Path, "home", classmethod(lambda cls: home))

    registry.register_project_directory(project, max_depth=6)

    registry_file = home / ".leann" / "projects.json"
    assert json.loads(registry_file.read_text()) == [str(project.resolve())]


def test_registered_app_project_is_listed(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "registered-project"
    meta_file = project / "app-index" / "documents.leann.meta.json"
    meta_file.parent.mkdir(parents=True)
    meta_file.touch()
    current = tmp_path / "current-project"
    current.mkdir()
    monkeypatch.setattr(registry.Path, "home", classmethod(lambda cls: home))
    registry.register_project_directory(project)
    monkeypatch.chdir(current)

    LeannCLI().list_indexes()

    output = capsys.readouterr().out
    assert "registered-project" in output
    assert "app-index" in output


def test_registered_app_project_respects_requested_max_depth(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "registered-project"
    meta_file = (
        project / "one" / "two" / "three" / "four" / "five" / "six" / "deep-app.leann.meta.json"
    )
    meta_file.parent.mkdir(parents=True)
    meta_file.touch()
    registry_file = home / ".leann" / "projects.json"
    registry_file.parent.mkdir()
    registry_file.write_text(json.dumps([str(project.resolve())]))
    current = tmp_path / "current-project"
    current.mkdir()
    monkeypatch.setattr(registry.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(current)

    LeannCLI().list_indexes(max_depth=6)

    output = capsys.readouterr().out
    assert "registered-project" in output
    assert "six" in output


def test_registered_ancestor_does_not_hide_current_app_index(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    registered_ancestor = tmp_path / "registered-ancestor"
    current = registered_ancestor / "current-project"
    meta_file = current / "current-app" / "documents.leann.meta.json"
    meta_file.parent.mkdir(parents=True)
    meta_file.touch()
    registry_file = home / ".leann" / "projects.json"
    registry_file.parent.mkdir()
    registry_file.write_text(json.dumps([str(registered_ancestor.resolve())]))
    monkeypatch.setattr(registry.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(current)

    LeannCLI().list_indexes()

    current_section = capsys.readouterr().out.split("Other Projects", maxsplit=1)[0]
    assert "current-app" in current_section


def test_cli_format_index_discovery_is_not_limited_by_app_scan_depth(tmp_path):
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").touch()

    indexes = LeannCLI()._discover_indexes_in_project(tmp_path, max_depth=0)

    assert [(index["name"], index["type"]) for index in indexes] == [("sample", "cli")]
