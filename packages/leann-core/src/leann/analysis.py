import logging
import re
from pathlib import Path
from typing import Any, Optional

# Use explicit imports matching astchunk to ensure compatibility
try:
    import tree_sitter as ts
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_python as tspython
    import tree_sitter_typescript as tstypescript

    # Java/C# optional
    try:
        import tree_sitter_java as tsjava
    except ImportError:
        tsjava = None
    try:
        import tree_sitter_c_sharp as tscsharp
    except ImportError:
        tscsharp = None

    from tree_sitter import Language, Parser, Query, QueryCursor

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    ts = None  # type: ignore

# Integration with astchunk (internal library)
try:
    from astchunk import ASTChunkBuilder

    ASTCHUNK_AVAILABLE = True
except ImportError:
    ASTCHUNK_AVAILABLE = False

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Analyzes source code to extract structural metadata and semantic chunks.

    Refined Capabilities (v2):
    1. Static Module Resolution: Resolves `leann.analysis` from file paths.
    2. Concise Skeleton: Compact outline of classes/functions for LLM context.
    3. Context Injection: Enriches chunks with ancestors and global context.
    4. Modern Tree-sitter: Uses 0.23+ bindings.
    """

    def __init__(self, language: str):
        """
        Initialize the analyzer for a specific language.

        Args:
            language: "python", "javascript", "typescript", "tsx", "java", "c_sharp"
        """
        self.language = language
        self.parser = None
        self._language_obj = None

        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. Analysis capabilities limited.")
            return

        try:
            if language == "python":
                self._language_obj = Language(tspython.language())
                self.parser = Parser(self._language_obj)

            elif language in ["javascript", "js", "jsx"]:
                # Use JS parser preference
                self._language_obj = Language(tsjavascript.language())
                self.parser = Parser(self._language_obj)

            elif language in ["typescript", "ts", "tsx"]:
                self._language_obj = Language(tstypescript.language_tsx())
                self.parser = Parser(self._language_obj)

            elif language == "java" and tsjava:
                self._language_obj = Language(tsjava.language())
                self.parser = Parser(self._language_obj)

            elif language == "csharp" and tscsharp:
                self._language_obj = Language(tscsharp.language())
                self.parser = Parser(self._language_obj)

            else:
                logger.warning(f"Unsupported or missing language binding: {language}")

        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter for {language}: {e}", exc_info=True)

    def analyze(self, code: str, file_path: str = "") -> dict[str, Any]:
        """
        Analyze code content and return extracted global metadata.
        """
        result = {
            "imports": [],
            "five_paths": [],
            "module_name": "",
            "is_script": False,
            "skeleton": "",
            "context_block": "",
        }

        if not self.parser or not code.strip():
            return result

        try:
            tree = self.parser.parse(bytes(code, "utf8"))

            # 1. Module Resolution
            result["module_name"] = self._resolve_module_name(file_path)

            # 2. Script Detection
            result["is_script"] = self._is_script(tree, code)

            # 3. Imports Extraction
            imports = self._extract_imports(tree, code)
            result["imports"] = imports
            result["five_paths"] = imports[:5]

            # 4. Skeleton Generation
            result["skeleton"] = self._generate_concise_skeleton(tree, code)

            # 5. Import Resolution (Project Local)
            resolved_imports = {}
            if file_path:
                try:
                    path_obj = Path(file_path).resolve()
                    search_root = path_obj.parent
                    # Crawl up for project root
                    for _ in range(5):
                        if (search_root / "src").exists() or (search_root / ".git").exists():
                            break
                        if search_root.parent == search_root:
                            break
                        search_root = search_root.parent
                    
                    for imp in imports:
                        # Normalize import path
                        # Python: foo.bar -> foo/bar
                        # JS/TS: ./utils -> ./utils, ../foo -> ../foo
                        
                        rel_path = imp
                        is_relative = imp.startswith(".")
                        
                        if self.language == "python":
                             rel_path = imp.replace(".", "/")
                        
                        # Search candidates
                        candidates = []
                        
                        if self.language == "python":
                            candidates.append(search_root / f"{rel_path}.py")
                            candidates.append(search_root / rel_path / "__init__.py")
                        elif self.language in ["javascript", "typescript", "js", "ts", "jsx", "tsx"]:
                            # JS/TS often omit extensions or index.js
                            # If relative, resolve from current file's dir, NOT project root
                            if is_relative:
                                # Resolving relative to the file being analyzed
                                current_dir = path_obj.parent
                                # We need to handle ./ and ../ carefully with pathlib
                                # imp such as './foo' or '../bar'
                                try:
                                    # pathlib join with relative parts works
                                    base_resolve = (current_dir / imp).resolve()
                                    candidates.append(base_resolve.with_suffix(".ts"))
                                    candidates.append(base_resolve.with_suffix(".tsx"))
                                    candidates.append(base_resolve.with_suffix(".js"))
                                    candidates.append(base_resolve.with_suffix(".jsx"))
                                    candidates.append(base_resolve / "index.ts")
                                    candidates.append(base_resolve / "index.js")
                                    # Exact match (if extension was provided)
                                    candidates.append(base_resolve)
                                except Exception:
                                    pass
                            else:
                                # Non-relative imports in JS/TS (e.g. 'react', 'src/components')
                                # Solving 'src/...' aliases is hard without tsconfig, but we can try from search_root
                                candidates.append(search_root / f"{rel_path}.ts")
                                candidates.append(search_root / f"{rel_path}.tsx")
                                candidates.append(search_root / f"{rel_path}.js")
                                candidates.append(search_root / rel_path / "index.ts")
                                candidates.append(search_root / rel_path / "index.js")
                        
                        for cand in candidates:
                            if cand.exists() and cand.is_file():
                                try:
                                    resolved_imports[imp] = str(cand.relative_to(search_root)).replace("\\", "/")
                                    break
                                except ValueError:
                                    # Candidate might be outside search_root (e.g. monorepo sibling)
                                    resolved_imports[imp] = str(cand).replace("\\", "/")
                                    break
                except Exception:
                    pass
            result["resolved_imports"] = resolved_imports

            # 6. Context Block Generation
            context_parts = []
            if result["module_name"]:
                context_parts.append(f"Module: {result['module_name']}")
            elif result["is_script"]:
                context_parts.append("Type: Script / Entry Point")

            if result["five_paths"]:
                context_parts.append("Imports: " + ", ".join(result["five_paths"]))
            
            if resolved_imports:
                res_list = [f"{k} ({v})" for k, v in list(resolved_imports.items())[:5]]
                context_parts.append("Project Imports: " + ", ".join(res_list))

            # [Optimization] We remove result["skeleton"] from the context_block
            # because prepending a full file skeleton to EVERY chunk is extremely
            # VRAM intensive during indexing and often exceeds model token limits.
            # The skeleton is still preserved in the chunk metadata for display.

            if context_parts:
                result["context_block"] = "\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}", exc_info=True)

        return result

    def get_semantic_chunks(
        self, code: str, file_path: str = "", metadata: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Split code into semantic chunks using astchunk.
        Enriches chunks with global metadata context block.
        """
        if not ASTCHUNK_AVAILABLE:
            return []

        if not code.strip():
            return []

        # normalized language for astchunk
        lang_map = {
            "python": "python",
            "java": "java",
            "c_sharp": "csharp",
            "cs": "csharp",
            "typescript": "typescript",
            "ts": "typescript",
            "tsx": "typescript",
            "js": "javascript",  # Explicitly map js to javascript now that we have custom handling
            "javascript": "javascript",
            "jsx": "javascript",
        }

        astchunk_lang = lang_map.get(self.language, self.language)

        repo_metadata = metadata or {}
        repo_metadata.setdefault("filepath", file_path)
        repo_metadata.setdefault("file_path", file_path)
        repo_metadata["total_lines"] = len(code.splitlines())

        try:
            configs = {
                "max_chunk_size": 512,
                "language": astchunk_lang,
                "metadata_template": "default",
                "chunk_overlap": 64,
                "repo_level_metadata": repo_metadata,
                "chunk_expansion": True,
            }

            chunk_builder = ASTChunkBuilder(**configs)
            chunks = chunk_builder.chunkify(code)

            # Get Context Block
            global_analysis = self.analyze(code, file_path)
            context_header = global_analysis.get("context_block", "")

            result_chunks = []
            for chunk in chunks:
                chunk_text = ""
                chunk_meta = {}

                if isinstance(chunk, dict):
                    chunk_text = chunk.get("content", chunk.get("text", ""))
                    chunk_meta = chunk.get("metadata", {})
                else:
                    chunk_text = str(chunk)

                if context_header:
                    # Prepend Context Header
                    # Use a clear separator standard for LLMs
                    chunk_text = f"'''\n{context_header}\n'''\n{chunk_text}"

                final_meta = {**repo_metadata, **chunk_meta}
                # Also store raw analysis fields in metadata for advanced filtering
                final_meta["module_name"] = global_analysis.get("module_name")
                final_meta["imports"] = global_analysis.get("imports", [])
                final_meta["resolved_imports"] = global_analysis.get("resolved_imports", {})
                final_meta["skeleton"] = global_analysis.get("skeleton", "")

                result_chunks.append({"text": chunk_text, "metadata": final_meta})

            # [Safety] Final pass to ensure no chunk exceeds the model's token limit
            # This is critical to prevent VRAM spikes from extremely long context headers
            from .chunking_utils import validate_chunk_token_limits
            texts = [c["text"] for c in result_chunks]
            validated_texts, truncated_count = validate_chunk_token_limits(texts, max_tokens=2048)
            
            if truncated_count > 0:
                logger.info(f"Refined {truncated_count} chunks to stay within 2048 token limit for {file_path}")
                for i, v_text in enumerate(validated_texts):
                    result_chunks[i]["text"] = v_text

            return result_chunks

        except Exception as e:
            logger.error(f"AST Chunking failed for {file_path}: {e}")
            return []

    def _resolve_module_name(self, file_path: str) -> str:
        """
        Resolve logical module name from file path.
        e.g. src/leann/analysis.py -> leann.analysis
        """
        if not file_path:
            return ""

        try:
            path = Path(file_path).resolve()

            # Simple heuristic: crawl up until no __init__.py (for Python)
            # or until package.json (for TS/JS)
            if self.language == "python":
                parts = []
                current = path.parent
                parts.append(path.stem)
                if path.name == "__init__.py":
                    parts = []  # Parent dir is the module name

                # Traverse up
                while current.joinpath("__init__.py").exists():
                    parts.insert(0, current.name)
                    if current == current.parent:
                        break  # Prevent infinite loop at root
                    current = current.parent

                if len(parts) > 0 and parts[-1] != "__init__":
                    return ".".join(parts)

            elif self.language in ["typescript", "javascript", "ts", "js", "tsx", "jsx"]:
                # Find package.json
                current = path.parent
                root = None
                while str(current) != current.root:
                    if current.joinpath("package.json").exists():
                        root = current
                        break
                    current = current.parent

                if root:
                    # Relative path from package root
                    rel = path.relative_to(root)
                    # Convert to module notation (foo/bar)
                    mod = rel.with_suffix("").as_posix()
                    if mod.endswith("/index"):
                        mod = mod[:-6]
                    return mod

        except Exception:
            pass  # Fallback to empty if resolution fails

        return ""

    def _is_script(self, tree, code: str) -> bool:
        """Check if file is an executable script."""
        # Check shebang
        if code.startswith("#!"):
            return True

        # Python: Check for if __name__ == "__main__"
        if self.language == "python":
            if 'if __name__ == "__main__":' in code or "if __name__ == '__main__':" in code:
                return True

        return False

    def _extract_imports(self, tree, code: str) -> list[str]:
        """Extract import paths."""
        imports = []
        root_node = tree.root_node

        if self.language == "python":
            query = Query(
                self._language_obj,
                """
            (import_from_statement
                module_name: (dotted_name) @module
            )
            (import_statement
                name: (dotted_name) @module
            )
            """,
            )
            cursor = QueryCursor(query)
            captures = cursor.captures(root_node)
            seen = set()
            # captures is dict: {"capture_name": [list of nodes]}
            for node in captures.get("module", []):
                text = node.text.decode("utf8")
                if text not in seen:
                    imports.append(text)
                    seen.add(text)

        elif self.language in ["javascript", "typescript", "tsx", "js", "ts", "jsx"]:
            query = Query(
                self._language_obj,
                """
            (import_statement
                source: (string) @source
            )
            (call_expression
                function: (identifier) @func
                arguments: (arguments (string) @arg)
            )
            """,
            )
            cursor = QueryCursor(query)
            captures = cursor.captures(root_node)
            seen = set()
            # Handle ES6 imports
            for node in captures.get("source", []):
                text = node.text.decode("utf8").strip("'").strip('"')
                if text not in seen:
                    imports.append(text)
                    seen.add(text)
            # Handle require() calls
            for node in captures.get("arg", []):
                parent = node.parent.parent
                if parent and parent.type == "call_expression":
                    func = parent.child_by_field_name("function")
                    if func and func.text.decode("utf8") == "require":
                        text = node.text.decode("utf8").strip("'").strip('"')
                        if text not in seen:
                            imports.append(text)
                            seen.add(text)
                            imports.append(text)
                            seen.add(text)

        # Generic: Scan for string literals that look like file paths
        # This covers "JSON config imports" or other dynamic loading
        # Query for all strings
        if self.parser: # Re-use parser logic broadly
            try:
                # Reuse query structure or a simple new query for strings
                # This works for most languages (python, js, ts, java, c# all have 'string' nodes)
                query_str = "(string) @str"
                query = Query(self._language_obj, query_str)
                cursor = QueryCursor(query)
                captures = cursor.captures(root_node)
                
                for node in captures.get("str", []):
                    # Clean quotes
                    raw = node.text.decode("utf8")
                    cleaned = raw.strip("'").strip('"')
                    
                    if not cleaned or "\n" in cleaned or len(cleaned) > 255:
                        continue
                        
                    if cleaned in seen:
                        continue
                        
                    # Heuristic: does it look like a file path?
                    # Contains slash or has extension
                    if "/" in cleaned or "\\" in cleaned or "." in cleaned:
                         imports.append(cleaned)
                         seen.add(cleaned)
            except Exception:
                pass
                
        return imports

    def _generate_concise_skeleton(self, tree, code: str) -> str:
        """Generate a COMPACT skeleton."""
        lines = []
        root_node = tree.root_node

        # Python Query
        if self.language == "python":
            query = Query(
                self._language_obj,
                """
            (function_definition) @func
            (class_definition) @class
            """,
            )
        # JS Query (no interface_declaration)
        elif self.language in ["javascript", "js", "jsx"]:
            query = Query(
                self._language_obj,
                """
            (function_declaration) @func
            (class_declaration) @class
            (method_definition) @method
            """,
            )
        # TS Query (includes interface)
        elif self.language in ["typescript", "tsx", "ts"]:
            query = Query(
                self._language_obj,
                """
            (function_declaration) @func
            (class_declaration) @class
            (interface_declaration) @interface
            (method_definition) @method
            """,
            )
        else:
            return ""

        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)

        # Flatten all captured nodes with their type info
        all_nodes = []
        for capture_name, nodes in captures.items():
            for node in nodes:
                all_nodes.append((node, capture_name))
        # Sort by line number for consistent output
        all_nodes.sort(key=lambda x: x[0].start_point[0])

        for node, _name in all_nodes:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            sig_text = ""
            doc_text = ""

            if self.language == "python":
                body = node.child_by_field_name("body")
                if body:
                    # Signature is everything before body
                    sig_bytes = code.encode("utf8")[node.start_byte : body.start_byte]
                    sig_text = sig_bytes.decode("utf8").strip().rstrip(":")

                    # Extract docstring
                    first_stmt = body.child(0)
                    if first_stmt and first_stmt.type == "expression_statement":
                        expr = first_stmt.child(0)
                        if expr and expr.type == "string":
                            raw_doc = expr.text.decode("utf8").strip("\"'")
                            # Truncate to 1 line, max 80 chars
                            cleaned_doc = re.sub(r"\s+", " ", raw_doc).strip()
                            if len(cleaned_doc) > 60:
                                doc_text = cleaned_doc[:57] + "..."
                            else:
                                doc_text = cleaned_doc
                else:
                    sig_text = node.text.decode("utf8").split("\n")[0]

            elif self.language in ["javascript", "typescript", "tsx", "js", "ts"]:
                body = node.child_by_field_name("body")
                if body:
                    sig_bytes = code.encode("utf8")[node.start_byte : body.start_byte]
                    sig_text = sig_bytes.decode("utf8").strip().rstrip("{")
                else:
                    sig_text = node.text.decode("utf8").split("\n")[0].strip().rstrip("{")

            # Format: signature # L10-20
            line_entry = f"{sig_text}  # L{start_line}-{end_line}"
            lines.append(line_entry)

            if doc_text:
                lines.append(f'  """ {doc_text} """')

        # Remove too many newlines, keep it compact
        return "\n".join(lines)
