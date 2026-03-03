import logging
import os
import pickle
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Optional

from llama_index.core import SimpleDirectoryReader

logger = logging.getLogger(__name__)


def hash_data(data: str | bytes):
    if isinstance(data, str):
        data = data.encode()
    return sha256(data).hexdigest()


@dataclass
class MerkleTreeNode:
    ## TODO: this merkle tree only has two layer, need to improve if we want to scale to large codebase
    hash: str
    data: str
    children: dict[str, "MerkleTreeNode"] = field(default_factory=dict)
    parent_id: str | None = None


class MerkleTree:
    def __init__(self):
        self.nodes: dict[str, MerkleTreeNode] = {}
        self.root: MerkleTreeNode | None = None

    def add_node(self, data: str, parent_id=None, hash: Optional[str] = None):
        hash = hash_data(data) if hash is None else hash

        node = MerkleTreeNode(hash=hash, data=data, parent_id=parent_id)
        self.nodes[hash] = node

        if parent_id is None:
            self.root = node
        else:
            self.nodes[parent_id].children[hash] = node

        return hash

    def compare_with(self, other: "MerkleTree"):
        """
        Simple comparison of two flat trees. Check the individual file hashes
        only if the root has changed, otherwise return no changes.
        """
        assert self.root is not None and other.root is not None

        if self.root.hash == other.root.hash:
            return [], [], []

        old_files = self.root.children
        new_files = other.root.children

        all_nodes = new_files.keys() | old_files.keys()

        added, removed, modified = [], [], []
        for path in all_nodes:
            if path in new_files and path in old_files:
                if new_files[path].data != old_files[path].data:
                    modified.append(path)
            elif path in new_files and path not in old_files:
                added.append(path)
            else:
                removed.append(path)

        return added, removed, modified


class FileSynchronizer:
    def __init__(
        self,
        root_dir: str,
        ignore_patterns: Optional[list] = None,
        include_extensions: Optional[list] = None,
        auto_load=True,
        snapshot_path: Optional[str] = None,
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("This is not a valid directory")
        self.root_dir = root_dir
        self.ignore_patterns = ignore_patterns
        self.include_extensions = include_extensions
        self._custom_snapshot_path = snapshot_path
        self._pending_tree: Optional[MerkleTree] = None
        self.tree: Optional[MerkleTree] = None
        if auto_load:
            self.load_snapshot()

    def generate_file_hashes(self):
        file_hashes = {}
        try:
            reader = SimpleDirectoryReader(
                self.root_dir,
                recursive=True,
                exclude=self.ignore_patterns,
                required_exts=self.include_extensions,
                exclude_empty=False,
            )
        except ValueError:
            # Empty directory — no files to hash
            return file_hashes
        # print('reader.iter_data() length', len(list(reader.iter_data())))

        for file in reader.iter_data():
            if len(file) > 1:
                # print('file length is greater than 1', file)
                continue  # SimpleDirectoryReader can load more than 1 documents for weird file types e.g. PDFs
            file = file[0]
            try:
                file_hash = hash_data(file.text)
                file_hashes[file.metadata["file_path"]] = file_hash
            except Exception:
                logger.error(f"Cannot hash file {file.metadata['file_path']}")
                continue

        return file_hashes

    def build_merkle_tree(self, file_hashes):
        """
        Build a flat merkle tree suitable for quick checking of file changes.
        """
        tree = MerkleTree()

        sorted_paths = sorted(file_hashes)
        root_data = "".join(path + file_hashes[path] for path in sorted_paths)

        root_id = tree.add_node(root_data)

        for path in sorted_paths:
            tree.add_node(file_hashes[path], parent_id=root_id, hash=path)

        return tree

    def detect_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Detect changes without persisting. Call commit() after successful processing."""
        file_hashes = self.generate_file_hashes()
        new_tree = self.build_merkle_tree(file_hashes)
        self._pending_tree = new_tree

        if self.tree is None:
            return list(file_hashes.keys()), [], []

        return self.tree.compare_with(new_tree)

    def commit(self):
        """Persist the pending snapshot after successful processing."""
        if self._pending_tree is not None:
            self.tree = self._pending_tree
            self._pending_tree = None
            self.save_snapshot()

    def create_snapshot(self):
        """Build and persist a snapshot from the current file state (for initial / forced builds)."""
        file_hashes = self.generate_file_hashes()
        self.tree = self.build_merkle_tree(file_hashes)
        self.save_snapshot()

    def check_for_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Detect and auto-commit changes (convenience wrapper)."""
        changes = self.detect_changes()
        self.commit()
        return changes

    @property
    def snapshot_path(self):
        if self._custom_snapshot_path:
            return self._custom_snapshot_path
        return f"{self.root_dir}.sync_context.pickle"

    def save_snapshot(self):
        assert self.tree is not None

        with open(self.snapshot_path, "wb") as f:
            pickle.dump(self.tree, f)

    def load_snapshot(self):
        try:
            with open(self.snapshot_path, "rb") as f:
                self.tree = pickle.load(f)
        except FileNotFoundError:
            self.tree = None
