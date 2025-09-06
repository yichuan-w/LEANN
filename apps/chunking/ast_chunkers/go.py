"""
Go AST chunker for LEANN's code chunking system.
Provides semantically-aware code chunking for Go language using tree-sitter.
"""

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from tree_sitter import Language, Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import tree_sitter_go

    TREE_SITTER_GO_AVAILABLE = True
except ImportError:
    TREE_SITTER_GO_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GoCodeBlock:
    """Represents a semantic Go code unit for chunking."""

    # Core attributes
    text: str
    block_type: str  # 'package', 'import', 'function', 'method', 'struct', 'interface', 'type', 'var', 'const'
    name: str
    start_line: int
    end_line: int

    # Go-specific attributes
    receiver: Optional[str] = None  # For methods: receiver type
    receiver_pointer: bool = False  # Whether receiver is a pointer
    package_name: Optional[str] = None
    comments: list[str] = None  # Associated comments
    imports: list[str] = None  # Import statements (for package blocks)
    embedded_types: list[str] = None  # For structs with embedded types
    interface_methods: list[str] = None  # For interfaces
    generic_params: list[str] = None  # Generic type parameters

    # Metadata
    complexity_score: int = 0  # Estimated complexity for chunking decisions
    dependencies: set[str] = None  # Referenced types/functions

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.comments is None:
            self.comments = []
        if self.imports is None:
            self.imports = []
        if self.embedded_types is None:
            self.embedded_types = []
        if self.interface_methods is None:
            self.interface_methods = []
        if self.generic_params is None:
            self.generic_params = []
        if self.dependencies is None:
            self.dependencies = set()

    def to_dict(self) -> dict:
        """Convert to dictionary format for LEANN integration."""
        metadata = {
            "type": self.block_type,
            "name": self.name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": "go",
            "complexity_score": self.complexity_score,
        }

        # Add Go-specific metadata
        if self.receiver:
            metadata["receiver"] = self.receiver
            metadata["receiver_pointer"] = self.receiver_pointer
        if self.package_name:
            metadata["package_name"] = self.package_name
        if self.generic_params:
            metadata["generic_params"] = self.generic_params
        if self.embedded_types:
            metadata["embedded_types"] = self.embedded_types
        if self.interface_methods:
            metadata["interface_methods"] = self.interface_methods
        if self.dependencies:
            metadata["dependencies"] = list(self.dependencies)

        return {"text": self.text, "metadata": metadata}


class GoASTChunker:
    """AST-aware chunker for Go code using tree-sitter."""

    def __init__(self, max_chunk_size: int = 512, chunk_overlap: int = 64):
        """
        Initialize Go AST chunker.

        Args:
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.parser = None
        self._init_parser()

    def _init_parser(self) -> None:
        """Initialize tree-sitter parser for Go."""
        if not TREE_SITTER_AVAILABLE:
            logger.error("tree-sitter not available. Cannot parse Go AST.")
            return

        if not TREE_SITTER_GO_AVAILABLE:
            logger.error("tree-sitter-go not available. Cannot parse Go AST.")
            return

        try:
            # Initialize Go language parser
            GO_LANGUAGE = Language(tree_sitter_go.language())
            self.parser = Parser(GO_LANGUAGE)
            logger.debug("Go AST parser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Go parser: {e}")
            self.parser = None

    def _extract_text(self, node, source: bytes) -> str:
        """Extract text content from a tree-sitter node."""
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _get_node_line_range(self, node) -> tuple[int, int]:
        """Get the line range (1-indexed) for a node."""
        return node.start_point[0] + 1, node.end_point[0] + 1

    def _extract_comments_before(self, node, source: bytes, all_comments: list) -> list[str]:
        """Extract comments that precede a node."""
        node_start_line = node.start_point[0]
        preceding_comments = []

        for comment_node in all_comments:
            comment_end_line = comment_node.end_point[0]
            # Consider comments that end within 2 lines before the node
            if comment_end_line < node_start_line and node_start_line - comment_end_line <= 2:
                comment_text = self._extract_text(comment_node, source).strip()
                preceding_comments.append(comment_text)

        return preceding_comments

    def _calculate_complexity(self, node, source: bytes) -> int:
        """Calculate a complexity score for a code block."""
        text = self._extract_text(node, source)

        # Simple complexity heuristics
        complexity = 0
        complexity += text.count("if") * 2
        complexity += text.count("for") * 3
        complexity += text.count("switch") * 2
        complexity += text.count("select") * 3
        complexity += text.count("func") * 1
        complexity += text.count("struct") * 1
        complexity += text.count("interface") * 2
        complexity += text.count("defer") * 1
        complexity += text.count("go ") * 2  # goroutines

        return complexity

    def _extract_receiver_info(self, method_node, source: bytes) -> tuple[Optional[str], bool]:
        """Extract receiver information from a method node."""
        # Look for receiver in method declaration
        for child in method_node.children:
            if child.type == "parameter_list":
                # This is the receiver
                receiver_text = self._extract_text(child, source).strip()
                if receiver_text.startswith("(") and receiver_text.endswith(")"):
                    receiver_text = receiver_text[1:-1].strip()

                # Check if it's a pointer receiver
                is_pointer = receiver_text.startswith("*")
                if is_pointer:
                    receiver_text = receiver_text[1:].strip()

                # Extract type name (skip variable name if present)
                parts = receiver_text.split()
                if len(parts) >= 2:
                    return parts[-1], is_pointer  # Last part is the type
                elif len(parts) == 1:
                    return parts[0], is_pointer
                break

        return None, False

    def _extract_generic_params(self, node, source: bytes) -> list[str]:
        """Extract generic type parameters from a node."""
        generic_params = []

        for child in node.children:
            if child.type == "type_parameter_list":
                param_text = self._extract_text(child, source).strip()
                if param_text.startswith("[") and param_text.endswith("]"):
                    # Parse individual parameters
                    param_content = param_text[1:-1].strip()
                    if param_content:
                        # Split by comma and clean up
                        params = [p.strip() for p in param_content.split(",")]
                        generic_params.extend(params)
                break

        return generic_params

    def _extract_dependencies(self, node, source: bytes) -> set[str]:
        """Extract type/function dependencies from a node."""
        dependencies = set()
        text = self._extract_text(node, source)

        # Simple pattern matching for common Go types and function calls
        # This is a basic implementation - could be enhanced with more sophisticated parsing
        import re

        # Match type references (basic patterns)
        type_patterns = [
            r"\b([A-Z][a-zA-Z0-9_]*)\b",  # CamelCase types
            r"\.([A-Z][a-zA-Z0-9_]*)",  # Package.Type references
        ]

        for pattern in type_patterns:
            matches = re.findall(pattern, text)
            dependencies.update(matches)

        # Remove common built-in types
        builtin_types = {"String", "Int", "Bool", "Error", "Interface", "Struct"}
        dependencies -= builtin_types

        return dependencies

    def _parse_package_block(self, package_node, source: bytes, all_imports: list) -> GoCodeBlock:
        """Parse a package declaration block."""
        package_text = self._extract_text(package_node, source)
        start_line, end_line = self._get_node_line_range(package_node)

        # Extract package name
        package_name = "main"  # default
        for child in package_node.children:
            if child.type == "package_identifier":
                package_name = self._extract_text(child, source).strip()
                break

        # Include relevant imports
        import_texts = []
        for import_node in all_imports:
            import_text = self._extract_text(import_node, source).strip()
            import_texts.append(import_text)

        return GoCodeBlock(
            text=package_text,
            block_type="package",
            name=package_name,
            start_line=start_line,
            end_line=end_line,
            package_name=package_name,
            imports=import_texts,
            complexity_score=1,
        )

    def _parse_function_block(
        self, func_node, source: bytes, all_comments: list, package_name: str
    ) -> GoCodeBlock:
        """Parse a function declaration block."""
        func_text = self._extract_text(func_node, source)
        start_line, end_line = self._get_node_line_range(func_node)
        comments = self._extract_comments_before(func_node, source, all_comments)

        # Extract function name
        func_name = "anonymous"
        for child in func_node.children:
            if child.type == "identifier":
                func_name = self._extract_text(child, source).strip()
                break

        # Check for generic parameters
        generic_params = self._extract_generic_params(func_node, source)

        complexity = self._calculate_complexity(func_node, source)
        dependencies = self._extract_dependencies(func_node, source)

        return GoCodeBlock(
            text=func_text,
            block_type="function",
            name=func_name,
            start_line=start_line,
            end_line=end_line,
            package_name=package_name,
            comments=comments,
            generic_params=generic_params,
            complexity_score=complexity,
            dependencies=dependencies,
        )

    def _parse_method_block(
        self, method_node, source: bytes, all_comments: list, package_name: str
    ) -> GoCodeBlock:
        """Parse a method declaration block."""
        method_text = self._extract_text(method_node, source)
        start_line, end_line = self._get_node_line_range(method_node)
        comments = self._extract_comments_before(method_node, source, all_comments)

        # Extract method name
        method_name = "anonymous"
        for child in method_node.children:
            if child.type == "identifier":
                method_name = self._extract_text(child, source).strip()
                break

        # Extract receiver information
        receiver, is_pointer = self._extract_receiver_info(method_node, source)

        # Check for generic parameters
        generic_params = self._extract_generic_params(method_node, source)

        complexity = self._calculate_complexity(method_node, source)
        dependencies = self._extract_dependencies(method_node, source)

        return GoCodeBlock(
            text=method_text,
            block_type="method",
            name=method_name,
            start_line=start_line,
            end_line=end_line,
            receiver=receiver,
            receiver_pointer=is_pointer,
            package_name=package_name,
            comments=comments,
            generic_params=generic_params,
            complexity_score=complexity,
            dependencies=dependencies,
        )

    def _parse_struct_block(
        self, struct_node, source: bytes, all_comments: list, package_name: str
    ) -> GoCodeBlock:
        """Parse a struct declaration block."""
        struct_text = self._extract_text(struct_node, source)
        start_line, end_line = self._get_node_line_range(struct_node)
        comments = self._extract_comments_before(struct_node, source, all_comments)

        # Extract struct name
        struct_name = "anonymous"
        for child in struct_node.children:
            if child.type == "type_identifier":
                struct_name = self._extract_text(child, source).strip()
                break

        # Extract embedded types
        embedded_types = []
        for child in struct_node.children:
            if child.type == "field_declaration_list":
                for field in child.children:
                    if field.type == "field_declaration":
                        field_text = self._extract_text(field, source).strip()
                        # Simple heuristic: if field has no name, it's embedded
                        if " " not in field_text or field_text.startswith("*"):
                            embedded_types.append(field_text)

        # Check for generic parameters
        generic_params = self._extract_generic_params(struct_node, source)

        complexity = self._calculate_complexity(struct_node, source)
        dependencies = self._extract_dependencies(struct_node, source)

        return GoCodeBlock(
            text=struct_text,
            block_type="struct",
            name=struct_name,
            start_line=start_line,
            end_line=end_line,
            package_name=package_name,
            comments=comments,
            embedded_types=embedded_types,
            generic_params=generic_params,
            complexity_score=complexity,
            dependencies=dependencies,
        )

    def _parse_interface_block(
        self, interface_node, source: bytes, all_comments: list, package_name: str
    ) -> GoCodeBlock:
        """Parse an interface declaration block."""
        interface_text = self._extract_text(interface_node, source)
        start_line, end_line = self._get_node_line_range(interface_node)
        comments = self._extract_comments_before(interface_node, source, all_comments)

        # Extract interface name
        interface_name = "anonymous"
        for child in interface_node.children:
            if child.type == "type_identifier":
                interface_name = self._extract_text(child, source).strip()
                break

        # Extract interface methods
        interface_methods = []
        for child in interface_node.children:
            if child.type == "interface_type":
                for method_child in child.children:
                    if method_child.type == "method_spec":
                        method_text = self._extract_text(method_child, source).strip()
                        interface_methods.append(method_text)

        # Check for generic parameters
        generic_params = self._extract_generic_params(interface_node, source)

        complexity = self._calculate_complexity(interface_node, source)
        dependencies = self._extract_dependencies(interface_node, source)

        return GoCodeBlock(
            text=interface_text,
            block_type="interface",
            name=interface_name,
            start_line=start_line,
            end_line=end_line,
            package_name=package_name,
            comments=comments,
            interface_methods=interface_methods,
            generic_params=generic_params,
            complexity_score=complexity,
            dependencies=dependencies,
        )

    def _parse_ast_nodes(self, source: bytes) -> list[GoCodeBlock]:
        """Parse AST nodes and extract Go code blocks."""
        if not self.parser:
            raise RuntimeError("Go parser not initialized")

        tree = self.parser.parse(source)
        root_node = tree.root_node

        blocks = []
        package_name = "main"  # default

        # First pass: collect all comments and imports
        all_comments = []
        all_imports = []

        def collect_comments_and_imports(node):
            if node.type in ["comment", "line_comment", "block_comment"]:
                all_comments.append(node)
            elif node.type == "import_declaration":
                all_imports.append(node)

            for child in node.children:
                collect_comments_and_imports(child)

        collect_comments_and_imports(root_node)

        # Second pass: extract package name and main blocks
        def extract_blocks(node):
            nonlocal package_name

            try:
                if node.type == "package_clause":
                    # Extract package name for use in other blocks
                    for child in node.children:
                        if child.type == "package_identifier":
                            package_name = self._extract_text(child, source).strip()
                            break

                    # Create package block
                    package_block = self._parse_package_block(node, source, all_imports)
                    blocks.append(package_block)

                elif node.type == "function_declaration":
                    func_block = self._parse_function_block(
                        node, source, all_comments, package_name
                    )
                    blocks.append(func_block)

                elif node.type == "method_declaration":
                    method_block = self._parse_method_block(
                        node, source, all_comments, package_name
                    )
                    blocks.append(method_block)

                elif node.type == "type_declaration":
                    # Check what kind of type declaration
                    for child in node.children:
                        if child.type == "type_spec":
                            for spec_child in child.children:
                                if spec_child.type == "struct_type":
                                    struct_block = self._parse_struct_block(
                                        node, source, all_comments, package_name
                                    )
                                    blocks.append(struct_block)
                                elif spec_child.type == "interface_type":
                                    interface_block = self._parse_interface_block(
                                        node, source, all_comments, package_name
                                    )
                                    blocks.append(interface_block)

                # Recursively process children
                for child in node.children:
                    extract_blocks(child)

            except Exception as e:
                logger.warning(f"Error parsing AST node {node.type}: {e}")

        extract_blocks(root_node)
        return blocks

    def _split_large_blocks(self, blocks: list[GoCodeBlock]) -> list[GoCodeBlock]:
        """Split large blocks that exceed max_chunk_size."""
        result = []

        for block in blocks:
            if len(block.text) <= self.max_chunk_size:
                result.append(block)
                continue

            # Split large block into smaller chunks
            lines = block.text.split("\n")
            current_chunk_lines = []
            current_size = 0
            chunk_num = 1

            for line in lines:
                line_size = len(line) + 1  # +1 for newline

                if current_size + line_size > self.max_chunk_size and current_chunk_lines:
                    # Create a chunk from current lines
                    chunk_text = "\n".join(current_chunk_lines)
                    chunk_name = f"{block.name}_part{chunk_num}"

                    chunk_block = GoCodeBlock(
                        text=chunk_text,
                        block_type=f"{block.block_type}_split",
                        name=chunk_name,
                        start_line=block.start_line,
                        end_line=block.end_line,
                        receiver=block.receiver,
                        receiver_pointer=block.receiver_pointer,
                        package_name=block.package_name,
                        comments=block.comments
                        if chunk_num == 1
                        else [],  # Only first chunk gets comments
                        complexity_score=block.complexity_score // 2,  # Rough estimate
                        dependencies=block.dependencies,
                    )

                    result.append(chunk_block)
                    current_chunk_lines = []
                    current_size = 0
                    chunk_num += 1

                current_chunk_lines.append(line)
                current_size += line_size

            # Handle remaining lines
            if current_chunk_lines:
                chunk_text = "\n".join(current_chunk_lines)
                chunk_name = f"{block.name}_part{chunk_num}" if chunk_num > 1 else block.name

                chunk_block = GoCodeBlock(
                    text=chunk_text,
                    block_type=f"{block.block_type}_split" if chunk_num > 1 else block.block_type,
                    name=chunk_name,
                    start_line=block.start_line,
                    end_line=block.end_line,
                    receiver=block.receiver,
                    receiver_pointer=block.receiver_pointer,
                    package_name=block.package_name,
                    comments=block.comments if chunk_num == 1 else [],
                    complexity_score=block.complexity_score // 2,
                    dependencies=block.dependencies,
                )

                result.append(chunk_block)

        return result

    def parse_go_code(self, source_code: str) -> list[GoCodeBlock]:
        """
        Parse Go source code and extract semantic blocks.

        Args:
            source_code: Go source code as string

        Returns:
            List of GoCodeBlock objects
        """
        if not self.parser:
            logger.error("Go parser not available, cannot parse code")
            return []

        try:
            source_bytes = source_code.encode("utf-8")
            blocks = self._parse_ast_nodes(source_bytes)

            # Split large blocks if necessary
            blocks = self._split_large_blocks(blocks)

            logger.debug(f"Extracted {len(blocks)} Go code blocks")
            return blocks

        except Exception as e:
            logger.error(f"Error parsing Go code: {e}")
            return []


def chunk_go_code(
    source_code: str, max_chunk_size: int = 512, chunk_overlap: int = 64, **kwargs
) -> list[dict]:
    """
    Main entry point for Go code chunking.

    Args:
        source_code: Go source code to chunk
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks (not used in AST chunking)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    if not TREE_SITTER_AVAILABLE or not TREE_SITTER_GO_AVAILABLE:
        logger.error("tree-sitter or tree-sitter-go not available")
        # Fallback to simple line-based chunking
        return _fallback_chunk_go_code(source_code, max_chunk_size)

    try:
        chunker = GoASTChunker(max_chunk_size=max_chunk_size, chunk_overlap=chunk_overlap)
        blocks = chunker.parse_go_code(source_code)

        # Convert to LEANN format
        chunks = [block.to_dict() for block in blocks]

        logger.info(f"Successfully created {len(chunks)} Go AST chunks")
        return chunks

    except Exception as e:
        logger.error(f"Go AST chunking failed: {e}")
        logger.info("Falling back to simple chunking")
        return _fallback_chunk_go_code(source_code, max_chunk_size)


def _fallback_chunk_go_code(source_code: str, max_chunk_size: int) -> list[dict]:
    """
    Fallback chunking for Go code when AST parsing fails.

    Uses simple heuristics to identify Go constructs.
    """
    lines = source_code.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0
    in_function = False
    in_struct = False
    in_interface = False
    brace_depth = 0
    chunk_start_line = 1

    for i, line in enumerate(lines, 1):
        line_size = len(line) + 1
        stripped = line.strip()

        # Track brace depth for structure detection
        brace_depth += line.count("{") - line.count("}")

        # Detect Go constructs
        if stripped.startswith("func "):
            if current_chunk and current_size > 0:
                # End previous chunk
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "type": "code_block",
                            "start_line": chunk_start_line,
                            "end_line": i - 1,
                            "language": "go",
                        },
                    }
                )
                current_chunk = []
                current_size = 0

            in_function = True
            chunk_start_line = i

        elif stripped.startswith("type ") and "struct" in stripped:
            if current_chunk and current_size > 0:
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "type": "code_block",
                            "start_line": chunk_start_line,
                            "end_line": i - 1,
                            "language": "go",
                        },
                    }
                )
                current_chunk = []
                current_size = 0

            in_struct = True
            chunk_start_line = i

        elif stripped.startswith("type ") and "interface" in stripped:
            if current_chunk and current_size > 0:
                chunk_text = "\n".join(current_chunk)
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            "type": "code_block",
                            "start_line": chunk_start_line,
                            "end_line": i - 1,
                            "language": "go",
                        },
                    }
                )
                current_chunk = []
                current_size = 0

            in_interface = True
            chunk_start_line = i

        # Add line to current chunk
        current_chunk.append(line)
        current_size += line_size

        # Check if we should end current chunk
        should_end_chunk = False

        if (
            (in_function or in_struct or in_interface)
            and brace_depth == 0
            and stripped.endswith("}")
        ):
            should_end_chunk = True
            in_function = in_struct = in_interface = False
        elif current_size >= max_chunk_size:
            should_end_chunk = True

        if should_end_chunk and current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunk_type = (
                "function"
                if in_function
                else "struct"
                if in_struct
                else "interface"
                if in_interface
                else "code_block"
            )

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "type": chunk_type,
                        "start_line": chunk_start_line,
                        "end_line": i,
                        "language": "go",
                    },
                }
            )

            current_chunk = []
            current_size = 0
            chunk_start_line = i + 1

    # Handle remaining content
    if current_chunk:
        chunk_text = "\n".join(current_chunk)
        chunks.append(
            {
                "text": chunk_text,
                "metadata": {
                    "type": "code_block",
                    "start_line": chunk_start_line,
                    "end_line": len(lines),
                    "language": "go",
                },
            }
        )

    logger.info(f"Fallback chunking created {len(chunks)} chunks")
    return chunks
