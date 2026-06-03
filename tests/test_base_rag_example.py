import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apps import base_rag_example
from apps.base_rag_example import BaseRAGExample


class MinimalRAGExample(BaseRAGExample):
    def __init__(self):
        super().__init__(
            name="Minimal",
            description="Minimal test RAG",
            default_index_name="minimal",
        )

    def _add_specific_arguments(self, parser):
        return None

    async def load_data(self, args):
        return []


def _args(*, no_recompute: bool) -> SimpleNamespace:
    return SimpleNamespace(
        llm="simulated",
        llm_model=None,
        llm_host=None,
        llm_api_base=None,
        llm_api_key=None,
        thinking_budget=None,
        no_recompute=no_recompute,
        search_complexity=17,
        top_k=3,
    )


def _capture_chat_calls(monkeypatch):
    calls = []

    class FakeChat:
        def __init__(self, *args, **kwargs):
            pass

        def ask(self, query, **kwargs):
            calls.append((query, kwargs))
            return "answer"

    monkeypatch.setattr(base_rag_example, "LeannChat", FakeChat)
    return calls


def test_single_query_propagates_no_recompute_to_chat(monkeypatch):
    calls = _capture_chat_calls(monkeypatch)

    asyncio.run(
        MinimalRAGExample().run_single_query(
            _args(no_recompute=True),
            "index.leann",
            "question",
        )
    )

    assert calls == [
        (
            "question",
            {
                "top_k": 3,
                "complexity": 17,
                "recompute_embeddings": False,
                "llm_kwargs": {},
            },
        )
    ]


def test_interactive_query_propagates_recompute_to_chat(monkeypatch):
    calls = _capture_chat_calls(monkeypatch)

    class FakeSession:
        def run_interactive_loop(self, handler):
            handler("interactive question")

    monkeypatch.setattr(
        base_rag_example,
        "create_rag_session",
        lambda app_name, data_description: FakeSession(),
    )

    asyncio.run(
        MinimalRAGExample().run_interactive_chat(
            _args(no_recompute=False),
            "index.leann",
        )
    )

    assert calls == [
        (
            "interactive question",
            {
                "top_k": 3,
                "complexity": 17,
                "recompute_embeddings": True,
                "llm_kwargs": {},
            },
        )
    ]
