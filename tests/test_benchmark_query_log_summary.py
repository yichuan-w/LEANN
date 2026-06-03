import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.summarize_query_log import (
    format_markdown,
    format_summary,
    load_ground_truth,
    load_query_log,
    main,
    summarize_query_log,
)


def test_summarize_query_log_reports_recall_latency_and_storage(tmp_path):
    query_log = tmp_path / "queries.jsonl"
    query_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "query": "alpha",
                        "duration_ms": 10.0,
                        "search_mode": "vector",
                        "backend_name": "hnsw",
                        "results": [
                            {"id": "doc-a", "score": 0.9},
                            {"id": "doc-b", "score": 0.8},
                        ],
                    }
                ),
                json.dumps(
                    {
                        "query": "beta",
                        "duration_ms": 20.0,
                        "search_mode": "hybrid",
                        "backend_name": "hnsw",
                        "results": [{"id": "doc-c", "score": 0.7}],
                    }
                ),
                json.dumps(
                    {
                        "query": "gamma",
                        "duration_ms": 100.0,
                        "results": [{"score": 0.1}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ground_truth = tmp_path / "truth.json"
    ground_truth.write_text(
        json.dumps({"alpha": ["doc-a", "doc-z"], "beta": ["doc-x"]}),
        encoding="utf-8",
    )
    index_path = tmp_path / "documents.leann"
    (tmp_path / "documents.leann.meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "documents.leann.passages.idx").write_bytes(b"idx")
    (tmp_path / "documents.index").write_bytes(b"12345")
    (tmp_path / "documents2.index").write_bytes(b"not part of this index")

    summary = summarize_query_log(
        load_query_log(query_log),
        ground_truth=load_ground_truth(ground_truth),
        k=1,
        index_paths=[index_path],
        data_source="fixture-source",
        data_revision="fixture-revision",
    )

    assert summary["schema_version"] == 2
    assert summary["benchmark"] == "query_log_summary"
    assert summary["data_source"] == "fixture-source"
    assert summary["data_revision"] == "fixture-revision"
    assert summary["command"] is None
    assert summary["environment"]["python"]
    assert "leann_commit" in summary["environment"]
    assert "leann_branch" in summary["environment"]
    assert "leann_dirty" in summary["environment"]
    assert summary["query_count"] == 3
    assert summary["average_result_count"] == 1.0
    assert summary["records_with_missing_result_ids"] == 1
    assert summary["missing_result_id_count"] == 1
    assert summary["records_missing_latency"] == 0
    assert summary["records_missing_search_mode"] == 1
    assert summary["records_missing_backend_name"] == 1
    assert summary["search_modes"] == {"vector": 1, "hybrid": 1, "unknown": 1}
    assert summary["backends"] == {"hnsw": 2, "unknown": 1}
    assert summary["latency_ms"]["mean"] == 130.0 / 3.0
    assert summary["latency_ms"]["p95"] == 20.0
    assert summary["recall"]["evaluated_queries"] == 2
    assert summary["recall"]["missing_queries"] == 1
    assert summary["recall"]["recall_at_k"] == 0.25
    assert summary["recall"]["hit_rate_at_k"] == 0.5
    assert summary["storage"]["total_bytes"] == 10
    assert {Path(path).name for path in summary["storage"]["indexes"][0]["files"]} == {
        "documents.index",
        "documents.leann.meta.json",
        "documents.leann.passages.idx",
    }

    markdown = format_markdown(summary)
    assert "Data source: fixture-source" in markdown
    assert "Data revision: fixture-revision" in markdown
    assert "Command: unavailable" in markdown
    assert "Recall@1: 0.250" in markdown
    assert "Latency ms: mean=43.333, median=20.000, p95=20.000" in markdown
    assert "Records with missing result IDs: 1" in markdown
    assert "Missing result ID count: 1" in markdown
    assert "Records missing backend name: 1" in markdown


def test_load_ground_truth_accepts_jsonl_rows(tmp_path):
    path = tmp_path / "truth.jsonl"
    path.write_text(
        json.dumps({"query": "alpha", "relevant_ids": ["doc-a"]}) + "\n",
        encoding="utf-8",
    )

    assert load_ground_truth(path) == {"alpha": {"doc-a"}}


def test_format_summary_rejects_unknown_format():
    try:
        format_summary({"query_count": 0}, "html")
    except ValueError as exc:
        assert "unsupported output format" in str(exc)
    else:
        raise AssertionError("unknown summary formats should fail clearly")


def test_summarize_query_log_reports_empty_ground_truth_explicitly():
    summary = summarize_query_log([{"query": "alpha", "results": []}], ground_truth={}, k=1)

    assert summary["recall"] == {
        "evaluated_queries": 0,
        "missing_queries": 1,
        "recall_at_k": 0.0,
        "hit_rate_at_k": 0.0,
    }


def test_summarize_query_log_distinguishes_empty_results_from_missing_ids():
    summary = summarize_query_log(
        [
            {"query": "empty", "results": []},
            {"query": "partial", "results": [{"id": "doc-a"}, {"score": 0.1}]},
            {"query": "bad", "results": {"id": "doc-b"}},
            {"query": "absent"},
        ],
        k=1,
    )

    assert summary["query_count"] == 4
    assert summary["average_result_count"] == 0.25
    assert summary["records_with_missing_result_ids"] == 2
    assert summary["missing_result_id_count"] == 2


def test_summarize_query_log_p95_uses_lower_nearest_rank_sample():
    records = [
        {"query": f"q-{index}", "duration_ms": float(index), "results": []} for index in range(100)
    ]

    summary = summarize_query_log(records, k=1)

    assert summary["latency_ms"]["p95"] == 94.0


def test_main_writes_json_and_markdown_artifacts_while_preserving_stdout(tmp_path, capsys):
    query_log = tmp_path / "queries.jsonl"
    query_log.write_text(
        json.dumps(
            {
                "query": "alpha",
                "duration_ms": 12.0,
                "search_mode": "vector",
                "backend_name": "hnsw",
                "results": [{"id": "doc-a", "score": 0.9}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    json_output = tmp_path / "artifacts" / "summary.json"
    markdown_output = tmp_path / "artifacts" / "summary.md"

    main(
        [
            str(query_log),
            "--format",
            "markdown",
            "--data-source",
            "fixture-source",
            "--data-revision",
            "fixture-revision",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("# LEANN Query Log Summary")
    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["query_count"] == 1
    assert summary["data_source"] == "fixture-source"
    assert summary["data_revision"] == "fixture-revision"
    assert summary["command"].startswith("benchmarks/summarize_query_log.py ")
    assert "--data-source fixture-source" in summary["command"]
    markdown = markdown_output.read_text(encoding="utf-8")
    assert markdown.startswith("# LEANN Query Log Summary")
    assert "```bash\nbenchmarks/summarize_query_log.py " in markdown
    assert "Queries: 1" in markdown


def test_main_rejects_invalid_k_and_query_log_overwrite(tmp_path, capsys):
    query_log = tmp_path / "queries.jsonl"
    query_log.write_text('{"query": "alpha", "results": []}\n', encoding="utf-8")
    ground_truth = tmp_path / "truth.json"
    ground_truth.write_text(json.dumps({"alpha": ["doc-a"]}), encoding="utf-8")

    for args in (
        [str(query_log), "--k", "0"],
        [str(query_log), "--json-output", str(query_log)],
        [
            str(query_log),
            "--ground-truth",
            str(ground_truth),
            "--markdown-output",
            str(ground_truth),
        ],
        [
            str(query_log),
            "--json-output",
            str(tmp_path / "same"),
            "--markdown-output",
            str(tmp_path / "same"),
        ],
    ):
        try:
            main(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"invalid args should fail: {args}")

    captured = capsys.readouterr()
    assert "--k must be greater than 0" in captured.err
    assert "must not overwrite an input file" in captured.err
    assert "JSON and Markdown output paths must be different" in captured.err
