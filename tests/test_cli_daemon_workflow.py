import argparse
import asyncio
from pathlib import Path
from typing import Any

from leann.cli import LeannCLI


def test_search_passes_daemon_flags_to_searcher(monkeypatch):
    cli = LeannCLI()
    monkeypatch.setattr(
        cli,
        "_resolve_index_path",
        lambda *args, **kwargs: "/tmp/demo/documents.leann",
    )

    captured: dict[str, dict[str, Any]] = {"init": {}, "search": {}}

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            captured["init"] = kwargs

        def search(self, *args, **kwargs):
            captured["search"] = kwargs
            return []

    monkeypatch.setattr("leann.cli.LeannSearcher", DummySearcher)

    args = argparse.Namespace(
        index_name="demo",
        query="hello",
        top_k=3,
        complexity=64,
        beam_width=1,
        prune_ratio=0.0,
        recompute_embeddings=True,
        pruning_strategy="global",
        non_interactive=True,
        show_metadata=False,
        embedding_prompt_template=None,
        use_daemon=True,
        daemon_ttl=222,
        enable_warmup=True,
    )
    asyncio.run(cli.search_documents(args))

    assert captured["init"]["enable_warmup"] is True
    assert captured["init"]["use_daemon"] is True
    assert captured["init"]["daemon_ttl_seconds"] == 222
    assert captured["search"]["recompute_embeddings"] is True


def test_warmup_command_calls_searcher_warmup(monkeypatch):
    cli = LeannCLI()
    monkeypatch.setattr(
        cli,
        "_resolve_index_path",
        lambda *args, **kwargs: "/tmp/demo/documents.leann",
    )

    state: dict[str, int] = {"warmup_called": 0}

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            pass

        def warmup(self):
            state["warmup_called"] += 1

    monkeypatch.setattr("leann.cli.LeannSearcher", DummySearcher)

    args = argparse.Namespace(
        index_name="demo",
        use_daemon=True,
        daemon_ttl=120,
        enable_warmup=True,
    )
    asyncio.run(cli.warmup_index(args))
    assert state["warmup_called"] == 1


def test_daemon_status_filters_by_index(monkeypatch, capsys):
    cli = LeannCLI()
    monkeypatch.setattr(
        cli,
        "_resolve_index_path",
        lambda *args, **kwargs: "/tmp/demo/documents.leann",
    )
    meta_path = str(Path("/tmp/demo/documents.leann.meta.json").resolve())

    monkeypatch.setattr(
        "leann.cli.EmbeddingServerManager.list_daemons",
        lambda: [
            {
                "pid": 101,
                "port": 5557,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"passages_file": meta_path, "model_name": "m1"},
            },
            {
                "pid": 202,
                "port": 5558,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {
                    "passages_file": "/tmp/other/doc.meta.json",
                    "model_name": "m2",
                },
            },
        ],
    )

    args = argparse.Namespace(daemon_command="status", index_name="demo")
    asyncio.run(cli.daemon_command(args))
    out = capsys.readouterr().out
    assert "Active embedding daemons: 1" in out
    assert "pid=101" in out
    assert "pid=202" not in out


def test_daemon_stop_by_index_calls_stop_daemons(monkeypatch):
    cli = LeannCLI()
    monkeypatch.setattr(
        cli,
        "_resolve_index_path",
        lambda *args, **kwargs: "/tmp/demo/documents.leann",
    )
    captured: dict[str, dict[str, Any]] = {"kwargs": {}}

    def fake_stop_daemons(**kwargs):
        captured["kwargs"] = kwargs
        return 1

    monkeypatch.setattr("leann.cli.EmbeddingServerManager.stop_daemons", fake_stop_daemons)

    args = argparse.Namespace(daemon_command="stop", index_name="demo", all=False)
    asyncio.run(cli.daemon_command(args))

    assert captured["kwargs"]["passages_file"].endswith("documents.leann.meta.json")


def test_daemon_start_calls_searcher_warmup(monkeypatch):
    cli = LeannCLI()
    monkeypatch.setattr(
        cli,
        "_resolve_index_path",
        lambda *args, **kwargs: "/tmp/demo/documents.leann",
    )

    state: dict[str, Any] = {"warmup_called": 0, "init_kwargs": {}}

    class DummySearcher:
        def __init__(self, *args, **kwargs):
            state["init_kwargs"] = kwargs

        def warmup(self):
            state["warmup_called"] += 1

    monkeypatch.setattr("leann.cli.LeannSearcher", DummySearcher)

    args = argparse.Namespace(
        daemon_command="start",
        index_name="demo",
        daemon_ttl=88,
        enable_warmup=True,
    )
    asyncio.run(cli.daemon_command(args))

    assert state["warmup_called"] == 1
    assert state["init_kwargs"]["use_daemon"] is True
    assert state["init_kwargs"]["daemon_ttl_seconds"] == 88


def test_daemon_status_all_lists_records(monkeypatch, capsys):
    cli = LeannCLI()
    monkeypatch.setattr(
        "leann.cli.EmbeddingServerManager.list_daemons",
        lambda: [
            {
                "pid": 301,
                "port": 6001,
                "backend_module_name": "leann_backend_hnsw.hnsw_embedding_server",
                "config_signature": {"model_name": "m-a"},
            },
            {
                "pid": 302,
                "port": 6002,
                "backend_module_name": "leann_backend_diskann.diskann_embedding_server",
                "config_signature": {"model_name": "m-b"},
            },
        ],
    )
    args = argparse.Namespace(daemon_command="status", index_name=None)
    asyncio.run(cli.daemon_command(args))
    out = capsys.readouterr().out
    assert "Active embedding daemons: 2" in out
    assert "pid=301" in out
    assert "pid=302" in out


def test_daemon_stop_all_calls_manager(monkeypatch):
    cli = LeannCLI()
    captured = {"called": False}

    def fake_stop_daemons(**kwargs):
        captured["called"] = True
        assert kwargs == {}
        return 2

    monkeypatch.setattr("leann.cli.EmbeddingServerManager.stop_daemons", fake_stop_daemons)
    args = argparse.Namespace(daemon_command="stop", index_name=None, all=True)
    asyncio.run(cli.daemon_command(args))
    assert captured["called"] is True
