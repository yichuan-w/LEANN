import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks import run_evaluation
from benchmarks.run_evaluation import (
    format_markdown,
    format_summary,
    load_queries,
    main,
    run_retrieval_evaluation,
)


class FakePassageManager:
    def __init__(self, passages):
        self.passages = passages

    def get_passage(self, passage_id):
        return self.passages[passage_id]


class FakeSearcher:
    backend_name = "hnsw"
    embedding_model = "fake-model"
    embedding_mode = "sentence-transformers"
    passage_id_scheme = "content-hash"

    def __init__(self):
        self.passage_manager = FakePassageManager(
            {
                "1": {"text": "alpha document"},
                "2": {"text": "beta document"},
                "3": {"text": "gamma document"},
                "4": {"text": "gamma document"},
            }
        )
        self.calls = []

    def search(self, query, *, top_k, complexity, batch_size):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "complexity": complexity,
                "batch_size": batch_size,
            }
        )
        if query == "alpha":
            return [
                SimpleNamespace(id="1", text="alpha document"),
                SimpleNamespace(id="x", text="unrelated document"),
            ]
        if query == "beta":
            return [
                SimpleNamespace(id="miss-1", text="miss"),
                SimpleNamespace(text="miss"),
            ]
        return [
            SimpleNamespace(id="3", text="gamma document"),
            SimpleNamespace(id="3-copy", text="gamma document"),
        ]


def test_run_retrieval_evaluation_reports_recall_latency_storage(tmp_path):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"query": "alpha"}\n{"query": "beta"}\n', encoding="utf-8")
    ground_truth_file = tmp_path / "truth.json"
    ground_truth_file.write_text(
        json.dumps({"indices": [["1", "2"], ["3", "4"]]}), encoding="utf-8"
    )
    index_path = tmp_path / "documents.leann"
    (tmp_path / "documents.leann.meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "documents.leann.passages.idx").write_bytes(b"idx")
    (tmp_path / "documents.index").write_bytes(b"12345")
    (tmp_path / "other.index").write_bytes(b"not part of this index")

    summary = run_retrieval_evaluation(
        FakeSearcher(),
        ["alpha", "beta", "missing-ground-truth"],
        {"indices": [["1", "2"], ["3", "4"]]},
        index_path=str(index_path),
        dataset_type="fixture",
        queries_file=queries_file,
        ground_truth_file=ground_truth_file,
        num_queries=3,
        top_k=2,
        complexity=64,
        batch_size=0,
        data_source="fixture-source",
        data_revision="fixture-revision",
        queries_sha256=sha256(queries_file.read_bytes()).hexdigest(),
        ground_truth_sha256=sha256(ground_truth_file.read_bytes()).hexdigest(),
    )

    assert summary["schema_version"] == 2
    assert summary["benchmark"] == "retrieval_evaluation"
    assert summary["mode"] == "retrieval_only"
    assert summary["llm_used"] is False
    assert summary["data_source"] == "fixture-source"
    assert summary["data_revision"] == "fixture-revision"
    assert summary["command"] is None
    assert summary["queries_file"] == str(queries_file)
    assert summary["queries_sha256"] == sha256(queries_file.read_bytes()).hexdigest()
    assert summary["ground_truth_file"] == str(ground_truth_file)
    assert summary["ground_truth_sha256"] == sha256(ground_truth_file.read_bytes()).hexdigest()
    assert summary["environment"]["python"]
    assert "platform" in summary["environment"]
    assert "leann_commit" in summary["environment"]
    assert "leann_branch" in summary["environment"]
    assert "leann_dirty" in summary["environment"]
    assert summary["backend_name"] == "hnsw"
    assert summary["query_count"] == 3
    assert summary["recall"]["evaluated_queries"] == 2
    assert summary["recall"]["missing_ground_truth_queries"] == 1
    assert summary["recall"]["recall_at_k"] == 0.25
    assert summary["recall"]["hit_rate_at_k"] == 0.5
    assert summary["recall"]["queries_with_missing_result_ids"] == 1
    assert summary["recall"]["missing_result_id_count"] == 1
    assert summary["recall"]["queries_with_duplicate_result_texts"] == 2
    assert summary["recall"]["queries_with_duplicate_golden_texts"] == 1
    assert summary["average_result_count"] == 2.0
    assert summary["latency_ms"]["median"] >= 0.0
    assert summary["storage"]["bytes"] == 10
    assert {Path(path).name for path in summary["storage"]["files"]} == {
        "documents.index",
        "documents.leann.meta.json",
        "documents.leann.passages.idx",
    }
    assert summary["per_query"][0]["result_ids"] == ["1", "x"]
    assert summary["per_query"][0]["golden_ids"] == ["1", "2"]
    assert summary["per_query"][1]["result_ids"] == ["miss-1"]
    assert summary["per_query"][1]["missing_result_id_count"] == 1
    assert summary["per_query"][1]["golden_duplicate_text_count"] == 1
    assert summary["per_query"][2]["result_duplicate_text_count"] == 1

    markdown = format_markdown(summary)
    assert markdown.startswith("# LEANN Retrieval Evaluation")
    assert "Data source: fixture-source" in markdown
    assert "Data revision: fixture-revision" in markdown
    assert "Command: unavailable" in markdown
    assert "Queries SHA256:" in markdown
    assert "Ground truth SHA256:" in markdown
    assert "Recall@2: 0.2500" in markdown
    assert "Queries with missing result IDs: 1" in markdown
    assert "Queries with duplicate golden text: 1" in markdown
    assert json.loads(format_summary(summary, "json"))["dataset_type"] == "fixture"


def test_run_retrieval_evaluation_leaves_hashes_unset_for_in_memory_inputs(tmp_path):
    summary = run_retrieval_evaluation(
        FakeSearcher(),
        ["alpha"],
        {"indices": [["1"]]},
        index_path=str(tmp_path / "documents.leann"),
        dataset_type="fixture",
        queries_file="memory://queries",
        ground_truth_file="memory://truth",
        num_queries=1,
        top_k=1,
        complexity=16,
        batch_size=0,
    )

    assert summary["queries_sha256"] is None
    assert summary["ground_truth_sha256"] is None
    markdown = format_markdown(summary)
    assert "Queries SHA256: `unavailable`" in markdown
    assert "Ground truth SHA256: `unavailable`" in markdown


def test_run_retrieval_evaluation_p95_matches_shared_timing_rule(tmp_path, monkeypatch):
    class SequencedSearcher(FakeSearcher):
        def search(self, query, *, top_k, complexity, batch_size):
            return [SimpleNamespace(id=str(query), text=f"doc-{query}")]

    queries = [str(index) for index in range(100)]
    golden_results = {"indices": [[str(index)] for index in range(100)]}
    searcher = SequencedSearcher()
    searcher.passage_manager = FakePassageManager(
        {str(index): {"text": f"doc-{index}"} for index in range(100)}
    )

    ticks = iter(timestamp for index in range(100) for timestamp in (0.0, float(index) / 1000.0))

    def fake_perf_counter() -> float:
        return next(ticks)

    monkeypatch.setattr(run_evaluation.time, "perf_counter", fake_perf_counter)
    summary = run_retrieval_evaluation(
        searcher,
        queries,
        golden_results,
        index_path=str(tmp_path / "documents.leann"),
        dataset_type="fixture",
        queries_file="memory://queries",
        ground_truth_file="memory://truth",
        num_queries=100,
        top_k=1,
        complexity=16,
        batch_size=0,
    )

    assert summary["latency_ms"]["p95"] == 94.0


def test_load_queries_skips_blank_lines_and_requires_query(tmp_path):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"query": "alpha"}\n\n{"query": "beta"}\n', encoding="utf-8")
    assert load_queries(queries_file) == ["alpha", "beta"]

    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"question": "alpha"}\n', encoding="utf-8")
    try:
        load_queries(bad_file)
    except ValueError as exc:
        assert "missing string query" in str(exc)
    else:
        raise AssertionError("query rows without query should fail clearly")


def test_download_data_if_needed_refreshes_incomplete_data_root(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "README.md").write_text("metadata only", encoding="utf-8")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    run_evaluation.download_data_if_needed(data_root)

    assert len(calls) == 1
    assert calls[0]["repo_id"] == "LEANN-RAG/leann-rag-evaluation-data"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["local_dir"] == data_root
    assert "indices/**" in calls[0]["allow_patterns"]
    assert "queries/**" in calls[0]["allow_patterns"]
    assert "ground_truth/**" in calls[0]["allow_patterns"]
    assert "is incomplete" in capsys.readouterr().out


def test_download_data_if_needed_skips_complete_data_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    (data_root / "queries").mkdir(parents=True)
    (data_root / "queries" / "nq_open.jsonl").write_text('{"query": "alpha"}\n')
    (data_root / "ground_truth").mkdir()
    (data_root / "indices" / "dpr").mkdir(parents=True)
    (data_root / "indices" / "dpr" / "dpr.index").write_bytes(b"index")

    def fail_snapshot_download(**kwargs):
        raise AssertionError("complete evaluation data should not be downloaded")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_snapshot_download),
    )

    run_evaluation.download_data_if_needed(data_root)


def test_download_data_if_needed_refreshes_empty_indices(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    (data_root / "queries").mkdir(parents=True)
    (data_root / "queries" / "nq_open.jsonl").write_text('{"query": "alpha"}\n')
    (data_root / "ground_truth").mkdir()
    (data_root / "indices").mkdir()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    run_evaluation.download_data_if_needed(data_root)

    assert len(calls) == 1
    assert "indices/**" in calls[0]["allow_patterns"]
    assert "is incomplete" in capsys.readouterr().out


def test_download_data_if_needed_refreshes_artifact_free_indices(tmp_path, monkeypatch, capsys):
    data_root = tmp_path / "data"
    (data_root / "queries").mkdir(parents=True)
    (data_root / "queries" / "nq_open.jsonl").write_text('{"query": "alpha"}\n')
    (data_root / "ground_truth").mkdir()
    (data_root / "indices" / "dpr").mkdir(parents=True)
    (data_root / "indices" / "dpr" / "README.md").write_text("metadata only", encoding="utf-8")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    run_evaluation.download_data_if_needed(data_root)

    assert len(calls) == 1
    assert "indices/**" in calls[0]["allow_patterns"]
    assert "is incomplete" in capsys.readouterr().out


def test_main_writes_artifacts_and_does_not_run_llm_by_default(tmp_path, monkeypatch, capsys):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"query": "alpha"}\n', encoding="utf-8")
    ground_truth_file = tmp_path / "truth.json"
    ground_truth_file.write_text(json.dumps({"indices": [["1"]]}), encoding="utf-8")
    index_path = tmp_path / "documents.leann"
    (tmp_path / "documents.leann.meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(run_evaluation, "download_data_if_needed", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_evaluation, "LeannSearcher", lambda index_path: FakeSearcher())

    def fail_chat(*args, **kwargs):
        raise AssertionError("LeannChat must be opt-in for retrieval-only evaluation")

    monkeypatch.setattr(run_evaluation, "LeannChat", fail_chat)
    json_output = tmp_path / "benchmark-results" / "retrieval.json"
    markdown_output = tmp_path / "benchmark-results" / "retrieval.md"

    main(
        [
            str(index_path),
            "--dataset",
            "fixture",
            "--queries-file",
            str(queries_file),
            "--ground-truth",
            str(ground_truth_file),
            "--num-queries",
            "1",
            "--top-k",
            "1",
            "--complexity",
            "16",
            "--data-source",
            "fixture-source",
            "--data-revision",
            "fixture-revision",
            "--format",
            "markdown",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("INFO: Detected dataset type: fixture")
    assert "# LEANN Retrieval Evaluation" in captured.out
    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["llm_used"] is False
    assert summary["data_source"] == "fixture-source"
    assert summary["data_revision"] == "fixture-revision"
    assert summary["command"].startswith("benchmarks/run_evaluation.py ")
    assert "--data-source fixture-source" in summary["command"]
    markdown = markdown_output.read_text(encoding="utf-8")
    assert "```bash\nbenchmarks/run_evaluation.py " in markdown
    assert summary["queries_sha256"] == sha256(queries_file.read_bytes()).hexdigest()
    assert summary["ground_truth_sha256"] == sha256(ground_truth_file.read_bytes()).hexdigest()
    assert markdown.startswith("# LEANN Retrieval Evaluation")


def test_main_rejects_invalid_args_and_output_overwrites(tmp_path, monkeypatch, capsys):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"query": "alpha"}\n', encoding="utf-8")
    ground_truth_file = tmp_path / "truth.json"
    ground_truth_file.write_text(json.dumps({"indices": [["1"]]}), encoding="utf-8")
    monkeypatch.setattr(run_evaluation, "download_data_if_needed", lambda *args, **kwargs: None)

    for args in (
        ["index", "--num-queries", "0"],
        ["index", "--top-k", "0"],
        ["index", "--complexity", "0"],
        ["index", "--batch-size", "-1"],
        [
            "index",
            "--queries-file",
            str(queries_file),
            "--ground-truth",
            str(ground_truth_file),
            "--json-output",
            str(queries_file),
        ],
    ):
        try:
            main(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"invalid args should fail: {args}")

    captured = capsys.readouterr()
    assert "--num-queries must be greater than 0" in captured.err
    assert "--top-k must be greater than 0" in captured.err
    assert "--complexity must be greater than 0" in captured.err
    assert "--batch-size must be greater than or equal to 0" in captured.err
    assert "summary output path must not overwrite an input file" in captured.err


def test_main_rejects_empty_queries_file(tmp_path, monkeypatch, capsys):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text("\n", encoding="utf-8")
    ground_truth_file = tmp_path / "truth.json"
    ground_truth_file.write_text(json.dumps({"indices": []}), encoding="utf-8")

    monkeypatch.setattr(run_evaluation, "download_data_if_needed", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_evaluation, "LeannSearcher", lambda index_path: FakeSearcher())

    try:
        main(
            [
                "index",
                "--queries-file",
                str(queries_file),
                "--ground-truth",
                str(ground_truth_file),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("empty query files should fail clearly")

    assert "queries file did not contain any queries" in capsys.readouterr().err
