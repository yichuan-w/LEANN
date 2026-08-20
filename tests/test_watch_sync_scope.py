"""Regression tests for leann watch sync scope (#345)."""

from leann.cli import LeannCLI
from leann.sync import FileSynchronizer


def test_resolve_sync_scope_keeps_loose_files_separate(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    readme = repo / "README.md"
    readme.write_text("# hello", encoding="utf-8")

    cli = LeannCLI()
    directories, files = cli._resolve_sync_scope([str(src), str(readme)])

    assert directories == [str(src.resolve())]
    assert files == [str(readme.resolve())]


def test_watch_scope_does_not_scan_sibling_media(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src"
    assets = repo / "assets"
    src.mkdir(parents=True)
    assets.mkdir()
    readme = repo / "README.md"
    readme.write_text("# hello", encoding="utf-8")
    (src / "main.py").write_text("print('ok')", encoding="utf-8")
    (assets / "icon.png").write_bytes(b"png")

    synchronizers = [
        FileSynchronizer(
            root_dir=str(src),
            include_extensions=[".py", ".md"],
            snapshot_path=str(tmp_path / "sync_src.pickle"),
            auto_load=False,
        ),
        FileSynchronizer(
            explicit_files=[str(readme.resolve())],
            include_extensions=[".py", ".md"],
            snapshot_path=str(tmp_path / "sync_readme.pickle"),
            auto_load=False,
        ),
    ]

    hashed_paths: set[str] = set()
    for fs in synchronizers:
        hashed_paths.update(fs.generate_file_hashes().keys())

    assert str((src / "main.py").resolve()) in hashed_paths
    assert str(readme.resolve()) in hashed_paths
    assert str((assets / "icon.png").resolve()) not in hashed_paths


def test_mixed_txt_and_bin_directory_skips_bin_without_crash(tmp_path):
    """Same dir with .txt and .bin: hash only text, ignore binary (review #377)."""
    docs = tmp_path / "docs"
    docs.mkdir()
    txt = docs / "notes.txt"
    bin_file = docs / "payload.bin"
    txt.write_text("hello", encoding="utf-8")
    bin_file.write_bytes(bytes(range(256)))

    fs = FileSynchronizer(
        root_dir=str(docs),
        include_extensions=[".txt"],
        snapshot_path=str(tmp_path / "sync.pickle"),
        auto_load=False,
    )

    hashes = fs.generate_file_hashes()
    assert set(hashes.keys()) == {str(txt.resolve())}
    assert str(bin_file.resolve()) not in hashes

    fs.create_snapshot()
    fs2 = FileSynchronizer(
        root_dir=str(docs),
        include_extensions=[".txt"],
        snapshot_path=str(tmp_path / "sync.pickle"),
    )
    added, removed, modified = fs2.detect_changes()
    assert not added and not removed and not modified


def test_watch_tick_survives_corrupt_sync_roots(tmp_path, monkeypatch, capsys):
    # Arrange: a registered index whose sync_roots.json is corrupt
    monkeypatch.chdir(tmp_path)
    from leann.cli import LeannCLI

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (index_dir / "sync_roots.json").write_text(
        '{"directories": ["' + str(docs) + '"], "files": [], "sync_key": null', encoding="utf-8"
    )  # truncated JSON
    monkeypatch.setattr(cli, "_resolve_index_for_watch", lambda name: {"index_dir": index_dir})

    # Act: must not raise — a watch tick skips, it doesn't kill the daemon
    added, removed, modified = cli._watch_check_changes("idx")

    # Assert
    assert (added, removed, modified) == (set(), set(), set())


def test_watch_tick_survives_corrupt_snapshot(tmp_path, monkeypatch, capsys):
    # Arrange: valid scope but corrupt snapshot pickle
    monkeypatch.chdir(tmp_path)
    import hashlib as _hashlib
    import json as _json

    from leann.cli import LeannCLI

    cli = LeannCLI()
    index_dir = tmp_path / ".leann" / "indexes" / "idx"
    index_dir.mkdir(parents=True)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha", encoding="utf-8")
    (index_dir / "sync_roots.json").write_text(
        _json.dumps({"directories": [str(docs)], "files": []}), encoding="utf-8"
    )
    tag = _hashlib.sha256(str(docs).encode()).hexdigest()[:12]
    (index_dir / f"sync_{tag}.pickle").write_bytes(b"not a pickle")
    monkeypatch.setattr(cli, "_resolve_index_for_watch", lambda name: {"index_dir": index_dir})

    # Act / Assert
    added, removed, modified = cli._watch_check_changes("idx")
    assert (added, removed, modified) == (set(), set(), set())
    assert "watch tick skipped" in capsys.readouterr().out
