import asyncio
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


def test_cli_format_index_discovery_is_not_limited_by_app_scan_depth(tmp_path):
    index_dir = tmp_path / ".leann" / "indexes" / "sample"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").touch()

    indexes = LeannCLI()._discover_indexes_in_project(tmp_path, max_depth=0)

    assert [(index["name"], index["type"]) for index in indexes] == [("sample", "cli")]
