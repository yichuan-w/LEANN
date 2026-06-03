from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CODE_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SymbolDef:
    name: str
    qualified_name: str
    kind: str
    file_path: str
    module_path: str
    start_line: int
    end_line: int
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportEdge:
    module: str
    name: str | None
    alias: str | None
    file_path: str
    module_path: str
    line: int
    is_from_import: bool

    @property
    def imported_name(self) -> str:
        if self.name:
            return f"{self.module}.{self.name}" if self.module else self.name
        return self.module

    @property
    def local_name(self) -> str:
        if self.alias:
            return self.alias
        if self.name:
            return self.name
        return self.module.split(".")[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CallEdge:
    caller: str
    callee: str
    file_path: str
    module_path: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SymbolRef:
    name: str
    scope: str
    file_path: str
    module_path: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeContextGraph:
    schema_version: int = CODE_CONTEXT_SCHEMA_VERSION
    symbols: list[SymbolDef] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)
    calls: list[CallEdge] = field(default_factory=list)
    references: list[SymbolRef] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbols": [symbol.to_dict() for symbol in self.symbols],
            "imports": [edge.to_dict() for edge in self.imports],
            "calls": [edge.to_dict() for edge in self.calls],
            "references": [ref.to_dict() for ref in self.references],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodeContextGraph:
        return cls(
            schema_version=int(data.get("schema_version", CODE_CONTEXT_SCHEMA_VERSION)),
            symbols=[SymbolDef(**item) for item in data.get("symbols", [])],
            imports=[_import_edge_from_dict(item) for item in data.get("imports", [])],
            calls=[CallEdge(**item) for item in data.get("calls", [])],
            references=[SymbolRef(**item) for item in data.get("references", [])],
        )

    @classmethod
    def from_json(cls, data: str) -> CodeContextGraph:
        return cls.from_dict(json.loads(data))


def module_path_from_file(file_path: str, repo_root: str | None = None) -> str:
    path = Path(file_path)
    if repo_root:
        try:
            path = path.resolve().relative_to(Path(repo_root).resolve())
        except (OSError, ValueError):
            pass
    without_suffix = path.with_suffix("")
    if without_suffix.name == "__init__":
        without_suffix = without_suffix.parent
    ignored_parts = {"", ".", without_suffix.anchor}
    return ".".join(part for part in without_suffix.parts if part not in ignored_parts)


def extract_python_context(
    source: str,
    file_path: str,
    *,
    repo_root: str | None = None,
    module_path: str | None = None,
) -> CodeContextGraph:
    """Extract deterministic Python symbol, import, call, and reference facts."""
    module_path = module_path or module_path_from_file(file_path, repo_root=repo_root)
    tree = ast.parse(source)
    visitor = _PythonContextVisitor(file_path=file_path, module_path=module_path)
    visitor.visit(tree)
    return visitor.graph


def extract_python_code_context(source: str, file_path: str = "<memory>") -> CodeContextGraph:
    """Extract Python code context from source text.

    This first-slice API is intentionally standalone: it parses Python source
    into serializable graph data and does not integrate with indexing/search.
    SyntaxError is allowed to propagate for callers that want strict parsing.
    """
    return extract_python_context(source, file_path)


def extract_python_file_context(file_path: str | Path) -> CodeContextGraph:
    """Read a Python file and extract its code context graph."""
    path = Path(file_path)
    return extract_python_context(path.read_text(encoding="utf-8"), str(path))


def extract_context_for_file(
    source: str,
    file_path: str,
    *,
    language: str | None = None,
    repo_root: str | None = None,
) -> CodeContextGraph:
    """Extract context for one file. Currently implemented for Python only."""
    if language not in (None, "", "python"):
        return CodeContextGraph()
    try:
        return extract_python_context(source, file_path, repo_root=repo_root)
    except SyntaxError:
        return CodeContextGraph()


def metadata_for_line_range(
    graph: CodeContextGraph,
    *,
    start_line: int | None,
    end_line: int | None,
) -> dict[str, Any]:
    """Return compact JSON-serializable code-context metadata for a chunk."""
    metadata: dict[str, Any] = {"code_context_version": graph.schema_version}
    if not graph.symbols and not graph.imports and not graph.calls and not graph.references:
        return metadata

    if start_line is None or end_line is None:
        symbols = graph.symbols
        calls = graph.calls
        references = graph.references
    else:
        symbols = [
            symbol
            for symbol in graph.symbols
            if _ranges_overlap(start_line, end_line, symbol.start_line, symbol.end_line)
        ]
        calls = [call for call in graph.calls if start_line <= call.line <= end_line]
        references = [ref for ref in graph.references if start_line <= ref.line <= end_line]

    metadata["module_path"] = _first_module_path(graph)
    metadata["defined_symbols"] = [symbol.qualified_name for symbol in symbols]
    metadata["symbols"] = [symbol.name for symbol in symbols]
    metadata["imports"] = [_import_metadata(edge) for edge in graph.imports]
    metadata["calls"] = [edge.to_dict() for edge in calls]
    metadata["referenced_symbols"] = sorted({ref.name for ref in references})

    if symbols:
        primary = max(
            symbols,
            key=lambda symbol: (symbol.start_line, len(symbol.qualified_name)),
        )
        metadata["qualified_name"] = primary.qualified_name
        metadata["symbol"] = primary.name
        metadata["symbol_kind"] = primary.kind
    return metadata


class _PythonContextVisitor(ast.NodeVisitor):
    def __init__(self, *, file_path: str, module_path: str):
        self.file_path = file_path
        self.module_path = module_path
        self.graph = CodeContextGraph()
        self._scope: list[str] = []
        self._scope_kinds: list[str] = []

    @property
    def _current_scope(self) -> str:
        return ".".join([self.module_path, *self._scope])

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.graph.imports.append(
                ImportEdge(
                    module=alias.name,
                    name=None,
                    alias=alias.asname,
                    file_path=self.file_path,
                    module_path=self.module_path,
                    line=node.lineno,
                    is_from_import=False,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            self.graph.imports.append(
                ImportEdge(
                    module=module,
                    name=alias.name,
                    alias=alias.asname,
                    file_path=self.file_path,
                    module_path=self.module_path,
                    line=node.lineno,
                    is_from_import=True,
                )
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record_symbol(node, "class")
        self._scope.append(node.name)
        self._scope_kinds.append("class")
        self.generic_visit(node)
        self._scope.pop()
        self._scope_kinds.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, "async_function")

    def visit_Call(self, node: ast.Call) -> None:
        callee = _expr_name(node.func)
        if callee:
            self.graph.calls.append(
                CallEdge(
                    caller=self._current_scope,
                    callee=callee,
                    file_path=self.file_path,
                    module_path=self.module_path,
                    line=node.lineno,
                )
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.graph.references.append(
                SymbolRef(
                    name=node.id,
                    scope=self._current_scope,
                    file_path=self.file_path,
                    module_path=self.module_path,
                    line=node.lineno,
                )
            )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> None:
        if "class" in self._scope_kinds:
            symbol_kind = "async_method" if kind == "async_function" else "method"
        else:
            symbol_kind = kind
        self._record_symbol(node, symbol_kind)
        self._scope.append(node.name)
        self._scope_kinds.append(kind)
        self.generic_visit(node)
        self._scope.pop()
        self._scope_kinds.pop()

    def _record_symbol(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        kind: str,
    ) -> None:
        qualified_name = ".".join([self.module_path, *self._scope, node.name])
        parent = self._current_scope if self._scope else None
        self.graph.symbols.append(
            SymbolDef(
                name=node.name,
                qualified_name=qualified_name,
                kind=kind,
                file_path=self.file_path,
                module_path=self.module_path,
                start_line=node.lineno,
                end_line=getattr(node, "end_lineno", None) or node.lineno,
                parent=parent,
            )
        )


def _expr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _expr_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return _expr_name(node.func)
    if isinstance(node, ast.Subscript):
        return _expr_name(node.value)
    return None


def _import_edge_from_dict(data: dict[str, Any]) -> ImportEdge:
    clean = {
        key: value for key, value in data.items() if key not in {"imported_name", "local_name"}
    }
    return ImportEdge(**clean)


def _first_module_path(graph: CodeContextGraph) -> str:
    if graph.symbols:
        return graph.symbols[0].module_path
    if graph.imports:
        return graph.imports[0].module_path
    if graph.calls:
        return graph.calls[0].module_path
    if graph.references:
        return graph.references[0].module_path
    return ""


def _import_metadata(edge: ImportEdge) -> dict[str, Any]:
    metadata = edge.to_dict()
    metadata["imported_name"] = edge.imported_name
    metadata["local_name"] = edge.local_name
    return metadata


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start <= b_end and b_start <= a_end
