from typing import cast

from leann.api import LeannChat, LeannSearcher, SearchResult


class RecordingSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [
            SearchResult(
                id="doc-1",
                score=1.0,
                text="retrieved context",
                metadata={"source": "test"},
            )
        ]


def test_chat_ask_honors_searcher_recompute_default():
    searcher = RecordingSearcher()
    chat = LeannChat(
        "unused.leann",
        llm_config={"type": "simulated"},
        searcher=cast(LeannSearcher, searcher),
    )

    response = chat.ask("question", top_k=2)

    assert "simulated answer" in response
    assert searcher.calls == [
        (
            "question",
            {
                "top_k": 2,
                "complexity": 64,
                "beam_width": 1,
                "prune_ratio": 0.0,
                "recompute_embeddings": None,
                "pruning_strategy": "global",
                "expected_zmq_port": 5557,
                "metadata_filters": None,
                "use_grep": False,
                "vector_weight": 1.0,
                "batch_size": 0,
            },
        )
    ]


def test_chat_ask_can_explicitly_override_recompute():
    searcher = RecordingSearcher()
    chat = LeannChat(
        "unused.leann",
        llm_config={"type": "simulated"},
        searcher=cast(LeannSearcher, searcher),
    )

    chat.ask("question", recompute_embeddings=False)

    assert searcher.calls[0][1]["recompute_embeddings"] is False
