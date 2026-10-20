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
