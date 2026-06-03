import json
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks import compare_retrieval_backends


class FakePassageManager:
    def __init__(self):
        self.passages = {
            "1": {"text": "alpha document"},
            "2": {"text": "beta document"},
        }

    def get_passage(self, passage_id):
        return self.passages[passage_id]


class FakeSearcher:
    embedding_model = "fake-model"
    embedding_mode = "sentence-transformers"
    passage_id_scheme = "content-hash"

    def __init__(self, index_path):
        self.index_path = str(index_path)
        self.backend_name = "ivf" if "ivf" in self.index_path else "hnsw"
        self.passage_manager = FakePassageManager()

    def search(self, query, *, top_k, complexity, batch_size):
        if self.backend_name == "hnsw":
            text = "alpha document" if query == "alpha" else "beta document"
            result_id = "1" if query == "alpha" else "2"
        elif query == "beta":
            text = "beta document"
            result_id = "2"
        else:
            text = "miss"
            result_id = "miss"
        return [SimpleNamespace(id=result_id, text=text)]


def _write_fixture_files(tmp_path):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text('{"query": "alpha"}\n{"query": "beta"}\n', encoding="utf-8")
    ground_truth_file = tmp_path / "truth.json"
    ground_truth_file.write_text(json.dumps({"indices": [["1"], ["2"]]}), encoding="utf-8")
    for name in ("hnsw", "ivf"):
        index_dir = tmp_path / name
        index_dir.mkdir()
        (index_dir / "documents.leann.meta.json").write_text("{}", encoding="utf-8")
        (index_dir / "documents.index").write_bytes(name.encode())
    return queries_file, ground_truth_file


def _manifest(tmp_path):
    return {
        "dataset": "fixture",
        "data_source": "fixture-source",
        "data_revision": "fixture-revision",
        "queries_file": "queries.jsonl",
        "ground_truth_file": "truth.json",
        "num_queries": 2,
        "top_k": 1,
        "complexity": 16,
        "batch_size": 0,
        "runs": [
            {"name": "hnsw", "backend": "hnsw", "index_path": "hnsw/documents.leann"},
            {"name": "ivf", "backend": "ivf", "index_path": "ivf/documents.leann"},
        ],
    }


def test_run_comparison_uses_identical_settings_and_normalizes_rows(tmp_path, monkeypatch):
    queries_file, ground_truth_file = _write_fixture_files(tmp_path)
    monkeypatch.setattr(compare_retrieval_backends.run_evaluation, "LeannSearcher", FakeSearcher)

    summary = compare_retrieval_backends.run_comparison(_manifest(tmp_path), manifest_dir=tmp_path)

    assert summary["schema_version"] == 1
    assert summary["benchmark"] == "retrieval_backend_comparison"
    assert summary["dataset"] == "fixture"
    assert summary["data_source"] == "fixture-source"
    assert summary["data_revision"] == "fixture-revision"
    assert summary["command"] is None
    assert summary["queries_file"] == str(queries_file)
    assert summary["ground_truth_file"] == str(ground_truth_file)
    assert summary["queries_sha256"] == sha256(queries_file.read_bytes()).hexdigest()
    assert summary["ground_truth_sha256"] == sha256(ground_truth_file.read_bytes()).hexdigest()
    assert summary["requested_query_count"] == 2
    assert summary["top_k"] == 1
    assert summary["complexity"] == 16
    assert summary["batch_size"] == 0
    assert summary["run_count"] == 2
    assert len(summary["evaluations"]) == 2

    hnsw, ivf = summary["runs"]
    assert hnsw["name"] == "hnsw"
    assert hnsw["manifest_backend"] == "hnsw"
    assert hnsw["backend_name"] == "hnsw"
    assert hnsw["recall_at_k"] == 1.0
    assert hnsw["hit_rate_at_k"] == 1.0
    assert hnsw["storage_bytes"] > 0
    assert hnsw["passage_id_scheme"] == "content-hash"
    assert hnsw["missing_result_id_count"] == 0
    assert hnsw["queries_with_duplicate_result_texts"] == 0
    assert hnsw["queries_with_duplicate_golden_texts"] == 0
    assert "leann_commit" in hnsw
    assert ivf["name"] == "ivf"
    assert ivf["backend_name"] == "ivf"
    assert ivf["recall_at_k"] == 0.5
    assert ivf["hit_rate_at_k"] == 0.5
    assert ivf["missing_result_id_count"] == 0
    assert ivf["top_k"] == hnsw["top_k"] == 1
    assert ivf["complexity"] == hnsw["complexity"] == 16

    markdown = compare_retrieval_backends.format_markdown(summary)
    assert markdown.startswith("# LEANN Retrieval Backend Comparison")
    assert "Command: unavailable" in markdown
    assert "Queries SHA256:" in markdown
    assert "Missing result IDs" in markdown
    assert "Duplicate result-text queries" in markdown
    assert "| hnsw | hnsw | content-hash | 1.0000 | 1.0000 | 0 | 0 | 0 |" in markdown
    assert "| ivf | ivf | content-hash | 0.5000 | 0.5000 | 0 | 0 | 0 |" in markdown


def test_comparison_row_preserves_latency_summary():
    summary = {
        "recall": {
            "recall_at_k": 0.75,
            "hit_rate_at_k": 1.0,
            "evaluated_queries": 4,
            "missing_ground_truth_queries": 0,
            "missing_golden_passages": 0,
        },
        "latency_ms": {"mean": 10.0, "median": 9.0, "p95": 14.0, "min": 8.0, "max": 15.0},
        "storage": {"bytes": 123, "files": ["documents.index"]},
        "backend_name": "hnsw",
        "index_path": "documents.leann",
        "dataset_type": "fixture",
        "query_count": 4,
        "requested_query_count": 4,
        "top_k": 3,
        "complexity": 64,
        "batch_size": 0,
        "embedding_model": "fake-model",
        "embedding_mode": "sentence-transformers",
        "passage_id_scheme": "content-hash",
        "environment": {"leann_commit": "abc123", "leann_dirty": False},
    }

    row = compare_retrieval_backends._comparison_row({"name": "hnsw"}, summary)

    assert row["latency_ms"] == summary["latency_ms"]


def test_main_writes_json_and_markdown_artifacts(tmp_path, monkeypatch, capsys):
    _write_fixture_files(tmp_path)
    monkeypatch.setattr(compare_retrieval_backends.run_evaluation, "LeannSearcher", FakeSearcher)
    manifest_path = tmp_path / "comparison.json"
    manifest_path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    json_output = tmp_path / "benchmark-results" / "comparison.json"
    markdown_output = tmp_path / "benchmark-results" / "comparison.md"

    compare_retrieval_backends.main(
        [
            str(manifest_path),
            "--format",
            "markdown",
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    captured = capsys.readouterr()
    assert captured.out.startswith("# LEANN Retrieval Backend Comparison")
    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["run_count"] == 2
    assert summary["runs"][0]["name"] == "hnsw"
    assert summary["command"].startswith("benchmarks/compare_retrieval_backends.py ")
    assert summary["evaluations"][0]["command"] is None
    assert "--json-output" in summary["command"]
    markdown = markdown_output.read_text(encoding="utf-8")
    assert markdown.startswith("# LEANN Retrieval Backend Comparison")
    assert "```bash\nbenchmarks/compare_retrieval_backends.py " in markdown


def test_main_rejects_query_or_ground_truth_output_overwrite(tmp_path, monkeypatch, capsys):
    queries_file, ground_truth_file = _write_fixture_files(tmp_path)
    monkeypatch.setattr(compare_retrieval_backends.run_evaluation, "LeannSearcher", FakeSearcher)
    manifest_path = tmp_path / "comparison.json"
    manifest_path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")

    for args in (
        [str(manifest_path), "--json-output", str(queries_file)],
        [str(manifest_path), "--markdown-output", str(ground_truth_file)],
    ):
        try:
            compare_retrieval_backends.main(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"input overwrite should fail: {args}")

    assert "comparison output path must not overwrite an input file" in capsys.readouterr().err


def test_run_comparison_rejects_manifest_backend_mismatch(tmp_path, monkeypatch):
    _write_fixture_files(tmp_path)
    monkeypatch.setattr(compare_retrieval_backends.run_evaluation, "LeannSearcher", FakeSearcher)
    manifest = _manifest(tmp_path)
    manifest["runs"][1]["backend"] = "hnsw"

    try:
        compare_retrieval_backends.run_comparison(manifest, manifest_dir=tmp_path)
    except ValueError as exc:
        assert "manifest backend for run 'ivf' was 'hnsw' but loaded index reported 'ivf'" in str(
            exc
        )
    else:
        raise AssertionError("manifest backend mismatches should fail loudly")


def test_main_rejects_invalid_manifest_and_output_overwrites(tmp_path, capsys):
    manifest_path = tmp_path / "comparison.json"
    manifest_path.write_text(json.dumps({"dataset": "fixture", "runs": []}), encoding="utf-8")

    for args in (
        [str(manifest_path)],
        [str(manifest_path), "--json-output", str(manifest_path)],
    ):
        try:
            compare_retrieval_backends.main(args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"invalid args should fail: {args}")

    stderr = capsys.readouterr().err
    assert "comparison manifest requires non-empty string field: queries_file" in stderr
    assert "comparison output path must not overwrite an input file" in stderr
