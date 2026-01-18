import unittest
from unittest.mock import Mock, patch

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

        # Mock file nodes
        file_a_new = Mock()
        file_b_new = Mock()
        file_a_old = Mock()
        file_c_old = Mock()

        # Equality behavior
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
        fs = FileSynchronizer("/tmp", auto_load=False)

        mock_file = Mock()
        mock_file.text = "hello world"
        mock_file.metadata = {"file_path": "/tmp/file.txt"}

        mock_reader_instance = Mock()
        mock_reader_instance.iter_data.return_value = [
            [mock_file],
        ]

        with patch("leann.sync.SimpleDirectoryReader") as mock_reader:
            mock_reader.return_value = mock_reader_instance

            result = fs.generate_file_hashes()

        assert result == {"/tmp/file.txt": hash_data("hello world")}

    def test_build_merkle_tree(self):
        fs = FileSynchronizer(".", auto_load=False)

        file_hashes = {
            "a.txt": "hashA",
            "b.txt": "hashB",
        }

        tree = fs.build_merkle_tree(file_hashes)

        # Root exists
        assert tree.root is not None

        # Children added correctly
        assert set(tree.root.children.keys()) == {"a.txt", "b.txt"}

        # Child nodes have correct data
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
