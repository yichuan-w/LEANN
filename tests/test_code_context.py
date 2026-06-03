import json

from leann.code_context import (
    CodeContextGraph,
    extract_context_for_file,
    extract_python_code_context,
    extract_python_context,
    metadata_for_line_range,
    module_path_from_file,
)

PYTHON_SOURCE = """\
import os
from collections import defaultdict as dd
from .helpers import normalize


class processor:
    def run(self, items):
        bucket = dd(list)
        return normalize(bucket)


async def fetch(client):
    return await client.get("/status")


def helper(value):
    return os.path.join("root", value)
"""


def test_module_path_from_file_handles_repo_relative_and_init():
    assert module_path_from_file("/repo/src/pkg/mod.py", repo_root="/repo") == "src.pkg.mod"
    assert module_path_from_file("/repo/src/pkg/__init__.py", repo_root="/repo") == "src.pkg"


def test_extract_python_context_records_symbols_imports_calls_and_refs():
    graph = extract_python_context(PYTHON_SOURCE, "/repo/src/pkg/mod.py", repo_root="/repo")
    alias_graph = extract_python_code_context(PYTHON_SOURCE, "/repo/src/pkg/mod.py")

    assert (
        alias_graph.to_dict()
        == extract_python_code_context(PYTHON_SOURCE, "/repo/src/pkg/mod.py").to_dict()
    )

    symbols = {symbol.qualified_name: symbol for symbol in graph.symbols}
    assert set(symbols) == {
        "src.pkg.mod.processor",
        "src.pkg.mod.processor.run",
        "src.pkg.mod.fetch",
        "src.pkg.mod.helper",
    }
    assert symbols["src.pkg.mod.processor"].kind == "class"
    assert symbols["src.pkg.mod.processor.run"].kind == "method"
    assert symbols["src.pkg.mod.fetch"].kind == "async_function"

    imports = {(edge.imported_name, edge.local_name, edge.line) for edge in graph.imports}
    assert imports == {
        ("os", "os", 1),
        ("collections.defaultdict", "dd", 2),
        (".helpers.normalize", "normalize", 3),
    }

    calls = {(edge.caller, edge.callee) for edge in graph.calls}
    assert ("src.pkg.mod.processor.run", "dd") in calls
    assert ("src.pkg.mod.processor.run", "normalize") in calls
    assert ("src.pkg.mod.fetch", "client.get") in calls
    assert ("src.pkg.mod.helper", "os.path.join") in calls

    references = {(ref.scope, ref.name) for ref in graph.references}
    assert ("src.pkg.mod.processor.run", "bucket") in references
    assert ("src.pkg.mod.helper", "os") in references


def test_code_context_graph_round_trips_through_json_dicts():
    graph = extract_python_context(PYTHON_SOURCE, "/repo/src/pkg/mod.py", repo_root="/repo")

    encoded = json.dumps(graph.to_dict())
    restored = CodeContextGraph.from_dict(json.loads(encoded))

    assert restored.to_dict() == graph.to_dict()
    assert CodeContextGraph.from_json(graph.to_json()).to_dict() == graph.to_dict()


def test_metadata_for_line_range_returns_compact_chunk_context():
    graph = extract_python_context(PYTHON_SOURCE, "/repo/src/pkg/mod.py", repo_root="/repo")

    metadata = metadata_for_line_range(graph, start_line=7, end_line=9)

    assert metadata["code_context_version"] == 1
    assert metadata["module_path"] == "src.pkg.mod"
    assert metadata["qualified_name"] == "src.pkg.mod.processor.run"
    assert metadata["symbol"] == "run"
    assert metadata["symbol_kind"] == "method"
    assert metadata["defined_symbols"] == [
        "src.pkg.mod.processor",
        "src.pkg.mod.processor.run",
    ]
    assert {call["callee"] for call in metadata["calls"]} == {"dd", "normalize"}
    assert "bucket" in metadata["referenced_symbols"]
    assert {edge["imported_name"] for edge in metadata["imports"]} == {
        "os",
        "collections.defaultdict",
        ".helpers.normalize",
    }
    assert {edge["local_name"] for edge in metadata["imports"]} == {"os", "dd", "normalize"}
    assert {edge["module"] for edge in metadata["imports"]} == {"os", "collections", ".helpers"}


def test_extract_context_for_file_ignores_non_python_and_invalid_python():
    assert (
        extract_context_for_file("function nope() {}", "app.js", language="typescript").to_dict()[
            "symbols"
        ]
        == []
    )
    assert (
        extract_context_for_file("def broken(:", "bad.py", language="python").to_dict()["symbols"]
        == []
    )
