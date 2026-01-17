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
    ):
        if not os.path.isdir(root_dir):
            raise ValueError("This is not a valid directory")
        self.root_dir = root_dir
        self.ignore_patterns = ignore_patterns
        self.include_extensions = include_extensions
        if auto_load:
            self.load_snapshot()

    def generate_file_hashes(self):
        file_hashes = {}
        reader = SimpleDirectoryReader(
            self.root_dir,
            recursive=True,
            exclude=self.ignore_patterns,
            required_exts=self.include_extensions,
            exclude_empty=True,
        )

        for file in reader.iter_data():
            if len(file) > 1:
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

    def check_for_changes(self):
        file_hashes = self.generate_file_hashes()
        new_tree = self.build_merkle_tree(file_hashes)

        changes = self.tree.compare_with(new_tree)

        if changes:
            self.tree = new_tree
            self.save_snapshot()

        return changes

    @property
    def snapshot_path(self):
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
            file_hashes = self.generate_file_hashes()
            self.tree = self.build_merkle_tree(file_hashes)
            self.save_snapshot()
            # yooooo
