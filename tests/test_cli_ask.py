import asyncio
import json

from leann.cli import LeannCLI


def test_cli_ask_accepts_positional_query(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["ask", "my-docs", "Where are prompts configured?"])

    assert args.command == "ask"
    assert args.index_name == "my-docs"
    assert args.query == "Where are prompts configured?"


def test_cli_ask_parses_metadata_filters_flag():
    cli = LeannCLI()
    parser = cli.create_parser()

    filters_json = '{"chapter": {"<=": 5}, "genre": {"==": "fiction"}}'
    args = parser.parse_args(
        ["ask", "my-docs", "Summarize early chapters", "--metadata-filters", filters_json]
    )

    assert args.command == "ask"
    assert args.metadata_filters == filters_json
    # The raw string parses to the expected dict so downstream consumers can rely on it.
    assert json.loads(args.metadata_filters) == {
        "chapter": {"<=": 5},
        "genre": {"==": "fiction"},
    }


def test_cli_ask_metadata_filters_default_is_none():
    cli = LeannCLI()
    parser = cli.create_parser()

    args = parser.parse_args(["ask", "my-docs", "any query"])

    assert args.metadata_filters is None


def test_cli_ask_rejects_invalid_metadata_filters_json(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()
    # Set up an empty index dir so the existence check passes and we reach the JSON parser.
    index_dir = tmp_path / ".leann" / "indexes" / "my-docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text("{}")

    args = parser.parse_args(["ask", "my-docs", "any query", "--metadata-filters", "not-json"])

    asyncio.run(cli.ask_questions(args))

    captured = capsys.readouterr()
    assert "--metadata-filters is not valid JSON" in captured.out


def test_cli_ask_rejects_non_object_metadata_filters(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    cli = LeannCLI()
    parser = cli.create_parser()
    index_dir = tmp_path / ".leann" / "indexes" / "my-docs"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.leann.meta.json").write_text("{}")

    # A valid JSON value that is not an object (dict) — must be rejected.
    args = parser.parse_args(["ask", "my-docs", "any query", "--metadata-filters", "[1, 2, 3]"])

    asyncio.run(cli.ask_questions(args))

    captured = capsys.readouterr()
    assert "--metadata-filters must be a JSON object" in captured.out
