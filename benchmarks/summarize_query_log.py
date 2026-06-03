#!/usr/bin/env python3
"""Summarize LEANN query logs into reproducible benchmark metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.artifact_formatting import command_markdown_lines
from benchmarks.metrics import mean, timing_stats
from benchmarks.provenance import benchmark_command, environment_metadata

GROUND_TRUTH_KEYS = ("relevant_ids", "gold_ids", "expected_ids", "ids")
LATENCY_KEYS = ("duration_ms", "latency_ms", "search_ms")


def load_query_log(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    return records


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    raw_path = Path(path)
    if raw_path.suffix == ".jsonl":
        rows = []
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return _ground_truth_from_rows(rows)

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {
            str(query): {str(item) for item in ids}
            for query, ids in data.items()
            if isinstance(ids, list)
        }
    if isinstance(data, list):
        return _ground_truth_from_rows(data)
    raise ValueError("ground truth must be a JSON object, JSON list, or JSONL records")


def summarize_query_log(
    records: list[dict[str, Any]],
    *,
    ground_truth: dict[str, set[str]] | None = None,
    k: int = 10,
    index_paths: list[str | Path] | None = None,
    data_source: str | None = None,
    data_revision: str | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    result_ids_by_record = [_result_ids(record) for record in records]
    result_counts = [len(ids) for ids in result_ids_by_record]
    result_id_gaps = [_result_id_gap_count(record) for record in records]
    latency_values = [_latency_ms(record) for record in records]
    latency_values = [value for value in latency_values if value is not None]
    summary: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "query_log_summary",
        "data_source": data_source,
        "data_revision": data_revision,
        "command": command,
        "query_count": len(records),
        "k": k,
        "average_result_count": mean(result_counts),
        "records_with_missing_result_ids": sum(1 for count in result_id_gaps if count),
        "missing_result_id_count": sum(result_id_gaps),
        "records_missing_latency": len(records) - len(latency_values),
        "records_missing_search_mode": sum(
            1 for record in records if not record.get("search_mode")
        ),
        "records_missing_backend_name": sum(
            1 for record in records if not record.get("backend_name")
        ),
        "search_modes": dict(
            Counter(str(record.get("search_mode", "unknown")) for record in records)
        ),
        "backends": dict(Counter(str(record.get("backend_name", "unknown")) for record in records)),
    }
    if latency_values:
        summary["latency_ms"] = timing_stats(latency_values)
    if ground_truth is not None:
        summary["recall"] = _recall_summary(records, ground_truth, k=k)

    paths = list(index_paths or [])
    if not paths:
        paths = sorted(
            {
                record["index_path"]
                for record in records
                if isinstance(record.get("index_path"), str) and record["index_path"]
            }
        )
    storage = [_index_storage(path) for path in paths]
    storage = [item for item in storage if item["exists"]]
    if storage:
        summary["storage"] = {
            "indexes": storage,
            "total_bytes": sum(item["bytes"] for item in storage),
        }
    summary["environment"] = environment_metadata()
    return summary


def format_summary(summary: dict[str, Any], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output_format == "markdown":
        return format_markdown(summary)
    raise ValueError(f"unsupported output format: {output_format}")


def format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# LEANN Query Log Summary",
        "",
        f"- Data source: {summary.get('data_source') or 'unknown'}",
        f"- Data revision: {summary.get('data_revision') or 'unknown'}",
    ]
    lines.extend(command_markdown_lines(summary.get("command")))
    lines.extend(
        [
            f"- Queries: {summary['query_count']}",
            f"- k: {summary['k']}",
            f"- Average result count: {summary['average_result_count']:.3f}",
            f"- Search modes: {json.dumps(summary['search_modes'], sort_keys=True)}",
            f"- Backends: {json.dumps(summary['backends'], sort_keys=True)}",
        ]
    )
    if "recall" in summary:
        recall = summary["recall"]
        lines.extend(
            [
                f"- Evaluated recall queries: {recall['evaluated_queries']}",
                f"- Recall@{summary['k']}: {recall['recall_at_k']:.3f}",
                f"- Hit rate@{summary['k']}: {recall['hit_rate_at_k']:.3f}",
            ]
        )
    if "latency_ms" in summary:
        latency = summary["latency_ms"]
        lines.append(
            "- Latency ms: "
            f"mean={latency['mean']:.3f}, median={latency['median']:.3f}, "
            f"p95={latency['p95']:.3f}, min={latency['min']:.3f}, max={latency['max']:.3f}"
        )
    lines.extend(
        [
            f"- Records with missing result IDs: {summary['records_with_missing_result_ids']}",
            f"- Missing result ID count: {summary['missing_result_id_count']}",
            f"- Records missing latency: {summary['records_missing_latency']}",
            f"- Records missing search mode: {summary['records_missing_search_mode']}",
            f"- Records missing backend name: {summary['records_missing_backend_name']}",
        ]
    )
    if "storage" in summary:
        lines.append(f"- Storage bytes: {summary['storage']['total_bytes']}")
    return "\n".join(lines) + "\n"


def write_summary_artifact(path: str | Path, content: str) -> None:
    output_path = Path(path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"summary output path is a directory: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    command = benchmark_command(__file__, argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_log", help="LEANN query log JSONL path")
    parser.add_argument("--ground-truth", help="JSON/JSONL mapping queries to relevant IDs")
    parser.add_argument("--k", type=int, default=10, help="Recall cutoff (default: 10)")
    parser.add_argument(
        "--index-path",
        action="append",
        default=[],
        help="Logical .leann index path to include in storage accounting. May repeat.",
    )
    parser.add_argument(
        "--data-source",
        help="Dataset/source identifier recorded in benchmark summary artifacts.",
    )
    parser.add_argument(
        "--data-revision",
        help="Dataset revision, commit, snapshot, or download date recorded in benchmark summaries.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--json-output",
        help="Write the benchmark summary JSON artifact to this file.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Write the benchmark summary Markdown artifact to this file.",
    )
    args = parser.parse_args(argv)
    if args.k <= 0:
        parser.error("--k must be greater than 0")

    ground_truth = load_ground_truth(args.ground_truth) if args.ground_truth else None
    summary = summarize_query_log(
        load_query_log(args.query_log),
        ground_truth=ground_truth,
        k=args.k,
        index_paths=args.index_path,
        data_source=args.data_source,
        data_revision=args.data_revision,
        command=command,
    )
    query_log_path = Path(args.query_log).resolve()
    protected_inputs = [query_log_path]
    if args.ground_truth:
        protected_inputs.append(Path(args.ground_truth).resolve())
    for output_path in (args.json_output, args.markdown_output):
        if output_path and Path(output_path).resolve() in protected_inputs:
            parser.error("summary output path must not overwrite an input file")
    if (
        args.json_output
        and args.markdown_output
        and Path(args.json_output).resolve() == Path(args.markdown_output).resolve()
    ):
        parser.error("JSON and Markdown output paths must be different")

    if args.json_output:
        write_summary_artifact(args.json_output, format_summary(summary, "json"))
    if args.markdown_output:
        write_summary_artifact(args.markdown_output, format_summary(summary, "markdown"))
    print(format_summary(summary, args.format), end="")


def _ground_truth_from_rows(rows: list[Any]) -> dict[str, set[str]]:
    ground_truth: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        query = row.get("query") or row.get("question")
        if not isinstance(query, str):
            continue
        for key in GROUND_TRUTH_KEYS:
            ids = row.get(key)
            if isinstance(ids, list):
                ground_truth[query] = {str(item) for item in ids}
                break
    return ground_truth


def _recall_summary(
    records: list[dict[str, Any]],
    ground_truth: dict[str, set[str]],
    *,
    k: int,
) -> dict[str, Any]:
    recalls: list[float] = []
    hits = 0
    missing_queries = 0
    for record in records:
        query = record.get("query")
        relevant = ground_truth.get(str(query))
        if not relevant:
            missing_queries += 1
            continue
        returned = set(_result_ids(record)[:k])
        overlap = returned & relevant
        recalls.append(len(overlap) / len(relevant))
        if overlap:
            hits += 1
    return {
        "evaluated_queries": len(recalls),
        "missing_queries": missing_queries,
        "recall_at_k": mean(recalls),
        "hit_rate_at_k": hits / len(recalls) if recalls else 0.0,
    }


def _result_ids(record: dict[str, Any]) -> list[str]:
    results = record.get("results")
    if not isinstance(results, list):
        return []
    ids: list[str] = []
    for result in results:
        if isinstance(result, dict) and result.get("id") is not None:
            ids.append(str(result["id"]))
    return ids


def _result_id_gap_count(record: dict[str, Any]) -> int:
    results = record.get("results")
    if results is None:
        return 0
    if not isinstance(results, list):
        return 1
    return sum(1 for result in results if not isinstance(result, dict) or result.get("id") is None)


def _latency_ms(record: dict[str, Any]) -> float | None:
    for key in LATENCY_KEYS:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _index_storage(index_path: str | Path) -> dict[str, Any]:
    path = Path(index_path)
    parent = path.parent
    logical_name = path.name
    native_stem = path.stem
    files: list[Path] = []
    if parent.exists():
        for candidate in parent.iterdir():
            if not candidate.is_file():
                continue
            if _is_index_artifact(candidate.name, logical_name, native_stem):
                files.append(candidate)
    return {
        "index_path": str(path),
        "exists": bool(files),
        "bytes": sum(file.stat().st_size for file in files),
        "files": [str(file) for file in sorted(files)],
    }


def _is_index_artifact(filename: str, logical_name: str, native_stem: str) -> bool:
    return (
        filename == logical_name
        or filename.startswith(f"{logical_name}.")
        or filename in {f"{native_stem}.index", f"{native_stem}.ids.txt"}
    )


if __name__ == "__main__":
    main()
