"""
Tests for the FAISS backend implementation.
"""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add package paths to sys.path to allow imports
# Assuming we are running from y:\code\leann-mcp\lib\leann-fork
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "leann-backend-faiss" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "packages" / "leann-core" / "src"))

# Mock faiss and numpy before importing backend
# This allows running tests in environments where faiss/numpy are not installed
start_mock_faiss = MagicMock()
sys.modules["faiss"] = start_mock_faiss
sys.modules["numpy"] = MagicMock()

# Mock other heavy dependencies that might be missing
sys.modules["torch"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()
sys.modules["llama_index"] = MagicMock()
sys.modules["llama_index.core"] = MagicMock()
sys.modules["llama_index.core.node_parser"] = MagicMock()

# Mock leann.api to avoid importing heavy dependencies
sys.modules["leann.api"] = MagicMock()

# Re-import numpy for the test file usage (we need actual numpy or a good mock for array creation in tests)
# Actually, if numpy is missing, we can't really run these tests easily as they rely on numpy arrays.
# But let's assume numpy IS available in CI usually, but FAISS is the hard one.
# If numpy is also missing (as seen in debug), we need to handle that.
# Let's try to import numpy, if fails, mock it fully.
try:
    import numpy as np
except ImportError:
    np = MagicMock()
    sys.modules["numpy"] = np

from leann_backend_faiss import FaissBackendBuilder, FaissBackendSearcher  # noqa: E402


class TestFaissBackendBuilder(unittest.TestCase):
    """Tests for FaissBackendBuilder."""

    @patch("leann_backend_faiss.faiss")
    def test_build_cpu_index(self, mock_faiss):
        """Test building a FAISS index on CPU."""
        # Setup mock
        mock_faiss.StandardGpuResources.side_effect = Exception("No GPU")

        # Create mock index
        mock_index = Mock()
        mock_index.is_trained = False
        mock_index.ntotal = 10
        mock_faiss.IndexFlatIP.return_value = mock_index

        # Test data - properly mock shape
        data = MagicMock()
        data.shape = (10, 128)
        data.dtype = np.float32

        ids = [f"id_{i}" for i in range(10)]

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = str(Path(temp_dir) / "test.index")

            builder = FaissBackendBuilder()
            builder.build(data, ids, index_path)

            # Verify interactions
            mock_faiss.IndexFlatIP.assert_called_with(128)
            mock_faiss.normalize_L2.assert_called_once()
            mock_index.train.assert_called_once()
            mock_index.add.assert_called_once()
            mock_faiss.write_index.assert_called_once()

    @patch("leann_backend_faiss.faiss")
    def test_build_gpu_index_large(self, mock_faiss):
        """Test building a large FAISS index (IVF) on GPU."""
        # Setup mock for GPU
        mock_res = Mock()
        mock_faiss.StandardGpuResources.return_value = mock_res

        mock_index_gpu = Mock()
        mock_index_gpu.is_trained = False
        mock_index_gpu.ntotal = 100001

        mock_index_cpu = Mock()

        mock_faiss.index_factory.return_value = mock_index_cpu
        mock_faiss.index_cpu_to_gpu.return_value = mock_index_gpu
        mock_faiss.index_gpu_to_cpu.return_value = mock_index_cpu

        # Test data > 100k
        data_shape = (100001, 128)
        # remove spec=np.ndarray as np is mocked
        data = MagicMock()
        data.shape = data_shape
        data.dtype = np.float32
        data.__len__.return_value = 100001

        ids = ["id"] * 100001

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = str(Path(temp_dir) / "test.index")

            builder = FaissBackendBuilder()
            builder.build(data, ids, index_path)

            # Verify "IVF" path was chosen
            mock_faiss.index_factory.assert_called()
            args, _ = mock_faiss.index_factory.call_args
            assert "IVF" in args[1]

            # Verify GPU storage
            mock_faiss.index_cpu_to_gpu.assert_called()

            # Verify save conversion
            mock_faiss.index_gpu_to_cpu.assert_called()


class TestFaissBackendSearcher(unittest.TestCase):
    """Tests for FaissBackendSearcher."""

    @patch("leann_backend_faiss.faiss")
    def test_search_cpu(self, mock_faiss):
        """Test searching on CPU."""
        # Setup mock
        mock_faiss.StandardGpuResources.side_effect = Exception("No GPU")
        mock_index = Mock()
        mock_faiss.read_index.return_value = mock_index

        # Mock search results: distances, indices
        # 1 query, top_k=2
        # indices must be integer-like for list indexing to work if not mocking full array behavior
        # But we can just mock indices[i][j] to return an int

        mock_distances = MagicMock()
        mock_distances.__getitem__.return_value.__getitem__.side_effect = [0.9, 0.8]

        mock_indices = MagicMock()
        # when accessing [i][j], return 0 then 1
        mock_indices.__getitem__.return_value.__getitem__.side_effect = [0, 1]

        mock_index.search.return_value = (mock_distances, mock_indices)

        # Mock IDs file
        ids = ["doc1", "doc2", "doc3"]

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test.index"
            # create dummy index file (content doesn't matter as we mock read_index)
            index_path.touch()
            # create ids file
            with open(index_path.with_suffix(".ids.pkl"), "wb") as f:
                pickle.dump(ids, f)

            searcher = FaissBackendSearcher(str(index_path))

            # query must have shape
            query = MagicMock()
            query.shape = (1, 128)
            query.dtype = np.float32

            results = searcher.search(query, top_k=2)

            assert len(results["labels"]) == 1
            assert len(results["labels"][0]) == 2
            assert results["labels"][0] == ["doc1", "doc2"]
            assert results["distances"][0] == [0.9, 0.8]

    @patch("leann.api.compute_embeddings")
    @patch("leann_backend_faiss.faiss")
    def test_compute_query_embedding_deadlock_fix(self, mock_faiss, mock_compute_embeddings):
        """Test that compute_query_embedding enforces use_server=False."""
        mock_faiss.StandardGpuResources.side_effect = Exception("No GPU")
        mock_faiss.read_index.return_value = Mock()

        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "test.index"
            index_path.touch()
            with open(index_path.with_suffix(".ids.pkl"), "wb") as f:
                pickle.dump([], f)

            searcher = FaissBackendSearcher(str(index_path))

            searcher.compute_query_embedding("test query")

            # CRITICAL: Verify use_server is False
            mock_compute_embeddings.assert_called_once()
            call_kwargs = mock_compute_embeddings.call_args[1]
            assert call_kwargs.get("use_server") is False
