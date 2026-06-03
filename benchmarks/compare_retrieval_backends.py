#!/usr/bin/env python3
"""Run one retrieval benchmark manifest across multiple prebuilt LEANN indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import run_evaluation
from benchmarks.artifact_formatting import command_markdown_lines
from benchmarks.provenance import benchmark_command, environment_metadata, file_sha256


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("comparison manifest must be a JSON object")
    return manifest


def run_comparison(
    manifest: dict[str, Any],
    *,
    manifest_dir: str | Path = ".",
    command: str | None = None,
) -> dict[str, Any]:
    """Evaluate all manifest runs with identical retrieval settings."""
    config = _comparison_config(manifest, Path(manifest_dir))
    queries = run_evaluation.load_queries(config["queries_file"])
    if not queries:
        raise ValueError("queries file did not contain any queries")
    queries_sha256 = file_sha256(config["queries_file"])
    ground_truth_sha256 = file_sha256(config["ground_truth_file"])
    with config["ground_truth_file"].open(encoding="utf-8") as handle:
        golden_results_data = json.load(handle)

    evaluations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for run_config in config["runs"]:
        searcher = run_evaluation.LeannSearcher(str(run_config["index_path"]))
        summary = run_evaluation.run_retrieval_evaluation(
            searcher,
            queries,
            golden_results_data,
            index_path=str(run_config["index_path"]),
            dataset_type=config["dataset"],
            queries_file=config["queries_file"],
            ground_truth_file=config["ground_truth_file"],
            num_queries=config["num_queries"],
            top_k=config["top_k"],
            complexity=config["complexity"],
            batch_size=config["batch_size"],
            data_source=config["data_source"],
            data_revision=config["data_revision"],
            queries_sha256=queries_sha256,
            ground_truth_sha256=ground_truth_sha256,
        )
        _validate_manifest_backend(run_config, summary)
        summary["comparison_run_name"] = run_config["name"]
        if run_config.get("backend"):
            summary["manifest_backend"] = run_config["backend"]
        evaluations.append(summary)
        rows.append(_comparison_row(run_config, summary))

    return {
        "schema_version": 1,
        "benchmark": "retrieval_backend_comparison",
        "dataset": config["dataset"],
        "data_source": config["data_source"],
        "data_revision": config["data_revision"],
        "command": command,
        "queries_file": str(config["queries_file"]),
        "queries_sha256": queries_sha256,
        "ground_truth_file": str(config["ground_truth_file"]),
        "ground_truth_sha256": ground_truth_sha256,
        "requested_query_count": config["num_queries"],
        "top_k": config["top_k"],
        "complexity": config["complexity"],
        "batch_size": config["batch_size"],
        "run_count": len(rows),
        "environment": environment_metadata(),
        "runs": rows,
        "evaluations": evaluations,
    }


def format_summary(summary: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        return format_markdown(summary)
    raise ValueError(f"unsupported output format: {output_format}")


def format_markdown(summary: dict[str, Any]) -> str:
    top_k = summary["top_k"]
    lines = [
        "# LEANN Retrieval Backend Comparison",
        "",
        f"- Dataset: {summary['dataset']}",
        f"- Data source: {summary.get('data_source') or 'unknown'}",
        f"- Data revision: {summary.get('data_revision') or 'unknown'}",
    ]
    lines.extend(command_markdown_lines(summary.get("command")))
    lines.extend(
        [
            f"- Queries file: `{summary['queries_file']}`",
            f"- Queries SHA256: `{summary['queries_sha256']}`",
            f"- Ground truth file: `{summary['ground_truth_file']}`",
            f"- Ground truth SHA256: `{summary['ground_truth_sha256']}`",
            f"- Requested queries: {summary['requested_query_count']}",
            f"- top_k: {top_k}",
            f"- complexity: {summary['complexity']}",
            f"- batch_size: {summary['batch_size']}",
            "",
            f"| Run | Backend | Passage IDs | Recall@{top_k} | Hit rate@{top_k} | "
            "Missing result IDs | Duplicate result-text queries | Duplicate golden-text queries | "
            "Latency median ms | Latency p95 ms | Storage bytes | Index path |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["runs"]:
        lines.append(
            "| {name} | {backend} | {passage_id_scheme} | {recall:.4f} | {hit_rate:.4f} | "
            "{missing_ids} | {duplicate_results} | {duplicate_golden} | {median:.3f} | "
            "{p95:.3f} | {storage_bytes} | `{index_path}` |".format(
                name=_markdown_cell(row["name"]),
                backend=_markdown_cell(row["backend_name"]),
                passage_id_scheme=_markdown_cell(row["passage_id_scheme"]),
                recall=row["recall_at_k"],
                hit_rate=row["hit_rate_at_k"],
                missing_ids=row["missing_result_id_count"],
                duplicate_results=row["queries_with_duplicate_result_texts"],
                duplicate_golden=row["queries_with_duplicate_golden_texts"],
                median=row["latency_ms"]["median"],
                p95=row["latency_ms"]["p95"],
                storage_bytes=row["storage_bytes"],
                index_path=_markdown_cell(row["index_path"]),
            )
        )
    return "\n".join(lines) + "\n"


def write_artifact(path: str | Path, content: str) -> None:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"comparison output path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _comparison_config(manifest: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    dataset = _required_string(manifest, "dataset")
    queries_file = _resolve_path(_required_string(manifest, "queries_file"), manifest_dir)
    ground_truth_file = _resolve_path(_required_string(manifest, "ground_truth_file"), manifest_dir)
    runs = _comparison_runs(manifest.get("runs"), manifest_dir)
    return {
        "dataset": dataset,
        "queries_file": queries_file,
        "ground_truth_file": ground_truth_file,
        "num_queries": _positive_int(manifest, "num_queries", 10),
        "top_k": _positive_int(manifest, "top_k", 3),
        "complexity": _positive_int(manifest, "complexity", 120),
        "batch_size": _non_negative_int(manifest, "batch_size", 0),
        "data_source": manifest.get("data_source") or "LEANN-RAG/leann-rag-evaluation-data",
        "data_revision": manifest.get("data_revision"),
        "runs": runs,
    }


def _comparison_runs(value: Any, manifest_dir: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("comparison manifest must include at least two runs")
    runs: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, run in enumerate(value):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a JSON object")
        name = _required_string(run, "name")
        if name in names:
            raise ValueError(f"duplicate run name: {name}")
        names.add(name)
        runs.append(
            {
                "name": name,
                "index_path": _resolve_path(_required_string(run, "index_path"), manifest_dir),
                "backend": run.get("backend"),
            }
        )
    return runs


def _comparison_row(run_config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    recall = summary["recall"]
    latency = summary["latency_ms"]
    storage = summary["storage"]
    return {
        "name": run_config["name"],
        "manifest_backend": run_config.get("backend"),
        "backend_name": summary["backend_name"],
        "index_path": summary["index_path"],
        "dataset": summary["dataset_type"],
        "query_count": summary["query_count"],
        "requested_query_count": summary["requested_query_count"],
        "top_k": summary["top_k"],
        "complexity": summary["complexity"],
        "batch_size": summary["batch_size"],
        "recall_at_k": recall["recall_at_k"],
        "hit_rate_at_k": recall["hit_rate_at_k"],
        "evaluated_queries": recall["evaluated_queries"],
        "missing_ground_truth_queries": recall["missing_ground_truth_queries"],
        "missing_golden_passages": recall["missing_golden_passages"],
        "queries_with_missing_result_ids": recall.get("queries_with_missing_result_ids", 0),
        "missing_result_id_count": recall.get("missing_result_id_count", 0),
        "queries_with_duplicate_result_texts": recall.get("queries_with_duplicate_result_texts", 0),
        "queries_with_duplicate_golden_texts": recall.get("queries_with_duplicate_golden_texts", 0),
        "latency_ms": latency,
        "storage_bytes": storage["bytes"],
        "storage_file_count": len(storage["files"]),
        "embedding_model": summary["embedding_model"],
        "embedding_mode": summary["embedding_mode"],
        "passage_id_scheme": summary["passage_id_scheme"],
        "leann_commit": summary["environment"].get("leann_commit"),
        "leann_dirty": summary["environment"].get("leann_dirty"),
    }


def _validate_manifest_backend(run_config: dict[str, Any], summary: dict[str, Any]) -> None:
    manifest_backend = run_config.get("backend")
    if manifest_backend and manifest_backend != summary["backend_name"]:
        raise ValueError(
            "manifest backend for run "
            f"'{run_config['name']}' was '{manifest_backend}' but loaded index reported "
            f"'{summary['backend_name']}'"
        )


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"comparison manifest requires non-empty string field: {key}")
    return value


def _positive_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"comparison manifest field {key} must be a positive integer")
    return value


def _non_negative_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"comparison manifest field {key} must be a non-negative integer")
    return value


def _resolve_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _validate_output_paths(
    parser: argparse.ArgumentParser,
    *,
    input_paths: list[str | Path],
    output_paths: list[str | None],
) -> None:
    resolved_inputs = {Path(path).resolve() for path in input_paths}
    seen_outputs: set[Path] = set()
    for output_path in output_paths:
        if not output_path:
            continue
        resolved_output = Path(output_path).resolve()
        if resolved_output in resolved_inputs:
            parser.error("comparison output path must not overwrite an input file")
        if resolved_output in seen_outputs:
            parser.error("JSON and Markdown output paths must be different")
        seen_outputs.add(resolved_output)


def main(argv: list[str] | None = None) -> None:
    command = benchmark_command(__file__, argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="JSON manifest describing the comparison runs.")
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for stdout (default: json).",
    )
    parser.add_argument("--json-output", help="Write the comparison JSON artifact to this file.")
    parser.add_argument(
        "--markdown-output", help="Write the comparison Markdown artifact to this file."
    )
    args = parser.parse_args(argv)
    _validate_output_paths(
        parser,
        input_paths=[args.manifest],
        output_paths=[args.json_output, args.markdown_output],
    )

    try:
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        config = _comparison_config(manifest, manifest_path.parent)
        _validate_output_paths(
            parser,
            input_paths=[config["queries_file"], config["ground_truth_file"]],
            output_paths=[args.json_output, args.markdown_output],
        )
        summary = run_comparison(manifest, manifest_dir=manifest_path.parent, command=command)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    if args.json_output:
        write_artifact(args.json_output, format_summary(summary, "json"))
    if args.markdown_output:
        write_artifact(args.markdown_output, format_summary(summary, "markdown"))
    print(format_summary(summary, args.format), end="")


if __name__ == "__main__":
    main()
