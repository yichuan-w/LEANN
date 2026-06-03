import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.bm25_diskann_baselines import run_bm25, run_diskann
from benchmarks.storage import directory_storage

REPO_ROOT = Path(__file__).parent.parent
BASELINE_DIR = REPO_ROOT / "benchmarks" / "bm25_diskann_baselines"


def test_bm25_latency_report_uses_shared_observed_percentiles():
    report = run_bm25.latency_report(
        [float(index) for index in range(100)],
        total_searches=100,
        total_time=20.0,
    )

    assert report == {
        "queries": 100,
        "avg_s": 49.5,
        "p50_s": 49.5,
        "p90_s": 89.0,
        "p95_s": 94.0,
        "p99_s": 98.0,
        "min_s": 0.0,
        "max_s": 99.0,
        "total_time_s": 20.0,
        "qps": 5.0,
    }


def test_bm25_latency_report_handles_empty_latencies():
    report = run_bm25.latency_report([], total_searches=0, total_time=0.0)

    assert report["avg_s"] == 0.0
    assert report["p95_s"] == 0.0
    assert report["qps"] == 0.0


def test_bm25_benchmark_report_includes_provenance(tmp_path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "alpha"}\n', encoding="utf-8")
    index_dir = tmp_path / "bm25"
    index_dir.mkdir()
    (index_dir / "segments").write_bytes(b"abc")
    latency = run_bm25.latency_report([1.0, 2.0], total_searches=2, total_time=4.0)

    report = run_bm25.benchmark_report(
        latency=latency,
        queries_file=queries,
        index_dir=index_dir,
        k=10,
        k1=0.9,
        b=0.4,
        warmup=5,
        fetch_docs=False,
        requested_query_count=100,
        data_source="fixture-source",
        data_revision="fixture-revision",
        command="benchmarks/bm25_diskann_baselines/run_bm25.py --report report.json",
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "bm25_baseline_latency"
    assert report["data_source"] == "fixture-source"
    assert report["data_revision"] == "fixture-revision"
    assert report["command"].startswith("benchmarks/bm25_diskann_baselines/run_bm25.py")
    assert report["queries_file"] == str(queries)
    assert report["queries_sha256"] == sha256(queries.read_bytes()).hexdigest()
    assert report["index_dir"] == str(index_dir.resolve())
    assert report["storage"] == {
        "path": str(index_dir.resolve()),
        "exists": True,
        "bytes": 3,
        "file_count": 1,
    }
    assert report["query_count"] == 2
    assert report["requested_query_count"] == 100
    assert report["timing_scope"] == "search_only"
    assert report["settings"] == {
        "k": 10,
        "k1": 0.9,
        "b": 0.4,
        "warmup": 5,
        "fetch_docs": False,
    }
    assert report["latency_s"]["mean"] == 1.5
    assert report["latency_s"]["min"] == 1.0
    assert report["latency_s"]["max"] == 2.0
    assert report["latency_s"]["total"] == 4.0
    assert "leann_commit" in report["environment"]


def test_bm25_write_json_report(tmp_path):
    report_path = tmp_path / "reports" / "bm25.json"

    run_bm25.write_json_report(report_path, {"schema_version": 1, "benchmark": "bm25"})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "benchmark": "bm25",
    }


def test_diskann_latency_report_uses_shared_timing_stats():
    report = run_diskann.latency_report([float(index) for index in range(100)])

    assert report == {
        "queries": 100,
        "avg_s": 49.5,
        "p50_s": 49.5,
        "p90_s": 89.0,
        "p95_s": 94.0,
        "p99_s": 98.0,
        "min_s": 0.0,
        "max_s": 99.0,
        "total_time_s": 4950.0,
        "qps": 1.0 / 49.5,
    }


def test_diskann_latency_report_handles_empty_latencies():
    report = run_diskann.latency_report([])

    assert report == {
        "queries": 0,
        "avg_s": 0.0,
        "p50_s": 0.0,
        "p90_s": 0.0,
        "p95_s": 0.0,
        "p99_s": 0.0,
        "min_s": 0.0,
        "max_s": 0.0,
        "total_time_s": 0,
        "qps": 0.0,
    }


def test_diskann_benchmark_report_includes_provenance(tmp_path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "alpha"}\n', encoding="utf-8")
    index_dir = tmp_path / "diskann"
    index_dir.mkdir()
    (index_dir / "ann").write_bytes(b"abc")
    (index_dir / "ann.tags").write_bytes(b"de")
    latency = run_diskann.latency_report([1.0, 2.0])

    report = run_diskann.benchmark_report(
        latency=latency,
        queries_file=queries,
        index_dir=index_dir,
        index_prefix="ann",
        top_k=10,
        complexity=62,
        threads=1,
        beam_width=1,
        cache_mechanism=2,
        num_nodes_to_cache=0,
        requested_query_count=200,
        data_source="fixture-source",
        data_revision="fixture-revision",
        command="benchmarks/bm25_diskann_baselines/run_diskann.py --report report.json",
    )

    assert report["schema_version"] == 1
    assert report["benchmark"] == "diskann_baseline_latency"
    assert report["data_source"] == "fixture-source"
    assert report["data_revision"] == "fixture-revision"
    assert report["command"].startswith("benchmarks/bm25_diskann_baselines/run_diskann.py")
    assert report["queries_sha256"] == sha256(queries.read_bytes()).hexdigest()
    assert report["index_dir"] == str(index_dir.resolve())
    assert report["index_prefix"] == "ann"
    assert report["index_prefix_path"] == str(index_dir.resolve() / "ann")
    assert report["storage"] == {
        "path": str(index_dir.resolve()),
        "exists": True,
        "bytes": 5,
        "file_count": 2,
    }
    assert report["query_count"] == 2
    assert report["requested_query_count"] == 200
    assert report["embedding_model"] == "facebook/contriever-msmarco"
    assert report["embedding_in_latency"] is False
    assert report["timing_scope"] == "search_only"
    assert report["settings"]["top_k"] == 10
    assert report["settings"]["complexity"] == 62
    assert report["settings"]["prune_ratio"] == 0.0
    assert report["settings"]["recompute_embeddings"] is False
    assert report["settings"]["batch_recompute"] is False
    assert report["settings"]["dedup_node_dis"] is False
    assert report["latency_s"]["mean"] == 1.5
    assert report["latency_s"]["p90"] == 1.0
    assert report["latency_s"]["p99"] == 1.0
    assert report["latency_s"]["total"] == 3.0
    assert "leann_commit" in report["environment"]


def test_diskann_write_json_report(tmp_path):
    report_path = tmp_path / "reports" / "diskann.json"

    run_diskann.write_json_report(report_path, {"schema_version": 1, "benchmark": "diskann"})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "benchmark": "diskann",
    }


def test_directory_storage_reports_missing_and_nested_directories(tmp_path):
    root = tmp_path / "index"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "a").write_bytes(b"abc")
    (nested / "b").write_bytes(b"de")

    assert directory_storage(root) == {
        "path": str(root.resolve()),
        "exists": True,
        "bytes": 5,
        "file_count": 2,
    }
    assert directory_storage(tmp_path / "missing") == {
        "path": str((tmp_path / "missing").resolve()),
        "exists": False,
        "bytes": 0,
        "file_count": 0,
    }


def test_directory_storage_reports_empty_existing_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert directory_storage(empty) == {
        "path": str(empty.resolve()),
        "exists": True,
        "bytes": 0,
        "file_count": 0,
    }


def test_baseline_reports_must_not_overwrite_inputs(tmp_path, capsys):
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query": "alpha"}\n', encoding="utf-8")
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    for module, args in (
        (run_bm25, ["--queries", str(queries), "--report", str(queries)]),
        (run_diskann, ["--queries-file", str(queries), "--report", str(queries)]),
        (
            run_bm25,
            [
                "--bm25-index",
                str(index_dir),
                "--queries",
                str(queries),
                "--report",
                str(index_dir / "report.json"),
            ],
        ),
        (
            run_diskann,
            [
                "--index-dir",
                str(index_dir),
                "--queries-file",
                str(queries),
                "--report",
                str(index_dir / "report.json"),
            ],
        ),
    ):
        try:
            module.main(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("report path should not overwrite queries")

    captured = capsys.readouterr()
    assert "report path must not overwrite the queries file" in captured.err
    assert "report path must not be inside the index directory" in captured.err


def test_baseline_scripts_help_from_repo_root_and_subdirectory():
    for script_name in ("run_bm25.py", "run_diskann.py"):
        script = BASELINE_DIR / script_name
        for cwd in (REPO_ROOT, BASELINE_DIR):
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "usage:" in result.stdout
            assert "--report" in result.stdout
            assert "--data-source" in result.stdout
            assert "--data-revision" in result.stdout
