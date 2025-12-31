"""
Test CLI update command parsing and validation.
"""

from leann.cli import LeannCLI


def test_cli_update_accepts_required_args(tmp_path, monkeypatch):
    """Test that update command parser accepts required arguments."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["update", "my-index", "--docs", "./new-documents"])

    assert args.command == "update"
    assert args.index_name == "my-index"
    assert args.docs == ["./new-documents"]


def test_cli_update_accepts_multiple_docs(tmp_path, monkeypatch):
    """Test that update command accepts multiple document paths."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["update", "my-index", "--docs", "./docs1", "./docs2", "./file.txt"])

    assert args.command == "update"
    assert args.index_name == "my-index"
    assert args.docs == ["./docs1", "./docs2", "./file.txt"]


def test_cli_update_accepts_chunking_options(tmp_path, monkeypatch):
    """Test that update command accepts chunking options."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(
        [
            "update",
            "my-index",
            "--docs",
            "./docs",
            "--doc-chunk-size",
            "512",
            "--doc-chunk-overlap",
            "64",
            "--code-chunk-size",
            "1024",
            "--code-chunk-overlap",
            "100",
        ]
    )

    assert args.command == "update"
    assert args.index_name == "my-index"
    assert args.doc_chunk_size == 512
    assert args.doc_chunk_overlap == 64
    assert args.code_chunk_size == 1024
    assert args.code_chunk_overlap == 100


def test_cli_update_accepts_file_types(tmp_path, monkeypatch):
    """Test that update command accepts file type filters."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(
        ["update", "my-index", "--docs", "./docs", "--file-types", ".py,.js,.ts"]
    )

    assert args.command == "update"
    assert args.index_name == "my-index"
    assert args.file_types == ".py,.js,.ts"


def test_cli_update_accepts_ast_chunking(tmp_path, monkeypatch):
    """Test that update command accepts AST chunking options."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(
        [
            "update",
            "my-index",
            "--docs",
            "./docs",
            "--use-ast-chunking",
            "--ast-chunk-size",
            "400",
            "--ast-chunk-overlap",
            "80",
        ]
    )

    assert args.command == "update"
    assert args.use_ast_chunking is True
    assert args.ast_chunk_size == 400
    assert args.ast_chunk_overlap == 80


def test_cli_update_accepts_include_hidden(tmp_path, monkeypatch):
    """Test that update command accepts include-hidden flag."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["update", "my-index", "--docs", "./docs", "--include-hidden"])

    assert args.command == "update"
    assert args.include_hidden is True


def test_cli_update_default_values(tmp_path, monkeypatch):
    """Test that update command has correct default values."""
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["update", "my-index", "--docs", "./docs"])

    # Check default chunking values
    assert args.doc_chunk_size == 256
    assert args.doc_chunk_overlap == 128
    assert args.code_chunk_size == 512
    assert args.code_chunk_overlap == 50

    # Check default AST values
    assert args.ast_chunk_size == 300
    assert args.ast_chunk_overlap == 64
    assert args.ast_fallback_traditional is True

    # Check default flags
    assert args.include_hidden is False
    assert args.use_ast_chunking is False
