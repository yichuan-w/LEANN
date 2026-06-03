import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks.provenance import benchmark_command


def test_benchmark_command_uses_repo_relative_script_path():
    command = benchmark_command(
        Path(__file__).parent.parent / "benchmarks" / "run_evaluation.py",
        ["--dataset", "fixture data"],
    )

    assert command == "benchmarks/run_evaluation.py --dataset 'fixture data'"
    assert shlex.split(command) == [
        "benchmarks/run_evaluation.py",
        "--dataset",
        "fixture data",
    ]


def test_benchmark_command_uses_sys_argv_when_argv_is_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ignored.py", "--format", "markdown"])

    command = benchmark_command(
        Path(__file__).parent.parent / "benchmarks" / "summarize_query_log.py",
        None,
    )

    assert command == "benchmarks/summarize_query_log.py --format markdown"
