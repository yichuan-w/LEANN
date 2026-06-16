import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from leann.sync import FileSynchronizer, MerkleTree, hash_data


class TestMerkleTreeCompare(unittest.TestCase):
    def test_no_changes_if_root_hash_same(self):
        tree1 = Mock()
        tree2 = Mock()

        tree1.root = Mock(hash="root_hash")
        tree2.root = Mock(hash="root_hash")

        added, removed, modified = MerkleTree.compare_with(tree1, tree2)

        self.assertEqual(added, [])
        self.assertEqual(removed, [])
        self.assertEqual(modified, [])

    def test_added_removed_modified(self):
        tree1 = Mock()
        tree2 = Mock()

        file_a_new = Mock()
        file_b_new = Mock()
        file_a_old = Mock()
        file_c_old = Mock()

        file_a_new.__eq__ = Mock(return_value=False)

        tree1.root = Mock(
            hash="new_root",
            children={
                "a.txt": file_a_new,
                "b.txt": file_b_new,
            },
        )

        tree2.root = Mock(
            hash="old_root",
            children={
                "a.txt": file_a_old,
                "c.txt": file_c_old,
            },
        )

        added, removed, modified = MerkleTree.compare_with(tree1, tree2)

        self.assertEqual(added, ["c.txt"])
        self.assertEqual(removed, ["b.txt"])
        self.assertEqual(modified, ["a.txt"])


class TestFileSynchronizer(unittest.TestCase):
    def test_generate_file_hashes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "file.txt"
            file_path.write_text("hello world", encoding="utf-8")
            fs = FileSynchronizer(root_dir=temp_dir, auto_load=False)
            result = fs.generate_file_hashes()
            assert result == {str(file_path.resolve()): hash_data(file_path.read_bytes())}

    def test_generate_file_hashes_skips_binary_extensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "code.py").write_text("print('hi')", encoding="utf-8")
            (root / "image.png").write_bytes(b"\x89PNG\r\n")
            fs = FileSynchronizer(
                root_dir=temp_dir,
                include_extensions=[".py"],
                auto_load=False,
            )
            result = fs.generate_file_hashes()
            assert len(result) == 1
            assert str((root / "code.py").resolve()) in result

    def test_generate_file_hashes_explicit_files_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            src = root / "src"
            src.mkdir()
            readme = root / "README.md"
            readme.write_text("# hi", encoding="utf-8")
            (src / "module.py").write_text("x = 1", encoding="utf-8")
            (root / "assets").mkdir()
            (root / "assets" / "icon.png").write_bytes(b"png")

            fs = FileSynchronizer(
                explicit_files=[str(readme.resolve())],
                include_extensions=[".md"],
                auto_load=False,
            )
            result = fs.generate_file_hashes()
            assert set(result.keys()) == {str(readme.resolve())}

    def test_build_merkle_tree(self):
        fs = FileSynchronizer.__new__(FileSynchronizer)

        file_hashes = {
            "a.txt": "hashA",
            "b.txt": "hashB",
        }

        tree = fs.build_merkle_tree(file_hashes)

        assert tree.root is not None
        assert set(tree.root.children.keys()) == {"a.txt", "b.txt"}
        assert tree.root.children["a.txt"].data == "hashA"
        assert tree.root.children["b.txt"].data == "hashB"

        expected_root_data = "a.txt" + "hashA" + "b.txt" + "hashB"
        assert tree.root.hash == hash_data(expected_root_data)

    def test_check_for_changes_detected(self):
        fs = FileSynchronizer.__new__(FileSynchronizer)

        fs.generate_file_hashes = Mock(return_value={"a.txt": "hash"})
        fs.build_merkle_tree = Mock(return_value=Mock())

        old_tree = Mock()
        new_tree = fs.build_merkle_tree.return_value

        old_tree.compare_with.return_value = (["a.txt"], [], [])
        fs.tree = old_tree

        fs.save_snapshot = Mock()

        changes = fs.check_for_changes()

        assert changes == (["a.txt"], [], [])

        fs.build_merkle_tree.assert_called_once_with({"a.txt": "hash"})
        old_tree.compare_with.assert_called_once_with(new_tree)

        fs.save_snapshot.assert_called_once()
        assert fs.tree is new_tree

    def test_touch_no_false_positive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir)
            f = docs / "a.txt"
            f.write_text("hello", encoding="utf-8")
            snapshot = str(docs / "test.pickle")
            fs = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
            fs.detect_changes()
            fs.commit()

            os.utime(f, None)
            fs2 = FileSynchronizer(root_dir=str(docs), snapshot_path=snapshot)
            added, removed, modified = fs2.detect_changes()
            assert not added and not removed and not modified
