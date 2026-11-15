"""
Unit tests for Qdrant Multi-Vector late-interaction store.

Tests:
  - Collection creation in both modes ('binary_quantized' and 'fp32_exact')
  - Multi-vector upsert and payload normalization with shape (10, 128)
  - Late-interaction MaxSim search and parameter propagation (rescore, oversampling)
  - MaxSim mathematical correctness
  - Binary quantization configuration
  - Storage inspection, count, and collection deletion
  - Error handling and parameter validation
"""

from __future__ import annotations

import math
import uuid
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.qdrant_store import QdrantMultiVectorStore, _normalize_point_id
from src.retrieval.visual_encoder import VisualEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_qdrant_models():
    """Mock qdrant_client.models classes and enums."""
    models_mock = MagicMock()

    class MockMultiVectorComparator:
        MAX_SIM = "max_sim"

    class MockDistance:
        COSINE = "Cosine"

    class MockMultiVectorConfig:
        def __init__(self, comparator):
            self.comparator = comparator

    class MockVectorParams:
        def __init__(self, size, distance, multivector_config=None):
            self.size = size
            self.distance = distance
            self.multivector_config = multivector_config

    class MockBinaryQuantizationConfig:
        def __init__(self, always_ram=True):
            self.always_ram = always_ram

    class MockBinaryQuantization:
        def __init__(self, binary):
            self.binary = binary

    class MockPointStruct:
        def __init__(self, id, vector, payload=None):
            self.id = id
            self.vector = vector
            self.payload = payload or {}

    class MockQuantizationSearchParams:
        def __init__(self, rescore=True, oversampling=2.0):
            self.rescore = rescore
            self.oversampling = oversampling

    class MockSearchParams:
        def __init__(self, quantization=None):
            self.quantization = quantization

    models_mock.MultiVectorComparator = MockMultiVectorComparator
    models_mock.Distance = MockDistance
    models_mock.MultiVectorConfig = MockMultiVectorConfig
    models_mock.VectorParams = MockVectorParams
    models_mock.BinaryQuantizationConfig = MockBinaryQuantizationConfig
    models_mock.BinaryQuantization = MockBinaryQuantization
    models_mock.PointStruct = MockPointStruct
    models_mock.QuantizationSearchParams = MockQuantizationSearchParams
    models_mock.SearchParams = MockSearchParams

    return models_mock


@pytest.fixture
def mock_qdrant_client():
    """Mock QdrantClient instance with common collection inspection stubs."""
    client = MagicMock()
    client.collection_exists.return_value = False

    # Mock collection info
    mock_info = MagicMock()
    mock_info.points_count = 42
    mock_info.vectors_count = 4200
    mock_info.status = "green"
    client.get_collection.return_value = mock_info

    # Mock count
    mock_count = MagicMock()
    mock_count.count = 42
    client.count.return_value = mock_count

    return client


@pytest.fixture
def store(mock_qdrant_client, mock_qdrant_models):
    """Instantiate QdrantMultiVectorStore with mocked dependencies."""
    with patch("src.retrieval.qdrant_store.QdrantClient", return_value=mock_qdrant_client), \
         patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
        store_instance = QdrantMultiVectorStore(
            collection_name="test_multimodal_docs",
            qdrant_mode="memory",
        )
        store_instance.client = mock_qdrant_client
        return store_instance


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestQdrantMultiVectorStore:

    def test_normalize_point_id(self):
        """Verify deterministic UUID generation and integer preservation."""
        # Integer should remain unchanged
        assert _normalize_point_id(12345) == 12345

        # Valid UUID string should preserve representation
        valid_uuid = str(uuid.uuid4())
        assert _normalize_point_id(valid_uuid) == valid_uuid

        # Arbitrary string hash should produce valid RFC 4122 UUID
        norm_id = _normalize_point_id("page_sha256_hash_123")
        assert isinstance(norm_id, str)
        uuid_obj = uuid.UUID(norm_id)
        assert uuid_obj.version == 5

    def test_create_collection_binary_quantized(self, store, mock_qdrant_models):
        """Test collection creation with binary quantization and MaxSim comparator."""
        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            store.create_collection(quantization_mode="binary_quantized", embedding_dim=128)

            store.client.create_collection.assert_called_once()
            call_kwargs = store.client.create_collection.call_args[1]
            assert call_kwargs["collection_name"] == "test_multimodal_docs"

            vectors_cfg = call_kwargs["vectors_config"]
            assert vectors_cfg.size == 128
            assert vectors_cfg.multivector_config.comparator == "max_sim"

            quant_cfg = call_kwargs["quantization_config"]
            assert quant_cfg is not None
            assert isinstance(quant_cfg, mock_qdrant_models.BinaryQuantization)
            assert quant_cfg.binary.always_ram is True

    def test_create_collection_fp32_exact(self, store, mock_qdrant_models):
        """Test collection creation with exact float32 (no binary quantization)."""
        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            store.create_collection(quantization_mode="fp32_exact", embedding_dim=128)

            call_kwargs = store.client.create_collection.call_args[1]
            assert call_kwargs["quantization_config"] is None
            assert store.quantization_mode == "fp32_exact"

    def test_create_collection_existing_skips_when_no_force(self, store):
        """If collection already exists and force_recreate=False, skip creation."""
        store.client.collection_exists.return_value = True

        store.create_collection(force_recreate=False)
        store.client.create_collection.assert_not_called()
        store.client.delete_collection.assert_not_called()

    def test_create_collection_force_recreate_drops_existing(self, store, mock_qdrant_models):
        """If collection exists and force_recreate=True, drop and recreate."""
        store.client.collection_exists.return_value = True

        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            store.create_collection(force_recreate=True)
            store.client.delete_collection.assert_called_once_with("test_multimodal_docs")
            store.client.create_collection.assert_called_once()

    def test_upsert_pages_shape_10_128(self, store, mock_qdrant_models):
        """Upsert document multi-vectors of shape (10, 128) with metadata."""
        # Create synthetic multi-vector embeddings of shape (10 patches, 128 dims)
        num_patches = 10
        dim = 128
        doc1_emb = [[float(i * 0.01 + j * 0.001) for j in range(dim)] for i in range(num_patches)]
        doc2_emb = [[float(i * 0.02 + j * 0.002) for j in range(dim)] for i in range(num_patches)]

        page_ids = ["doc1_p1", "doc1_p2"]
        embeddings = [doc1_emb, doc2_emb]
        metadata = [{"page_number": 1, "title": "Doc 1"}, {"page_number": 2, "title": "Doc 2"}]

        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            upserted_count = store.upsert_pages(
                page_ids=page_ids,
                embeddings=embeddings,
                metadata_list=metadata,
                batch_size=32,
            )

            assert upserted_count == 2
            store.client.upsert.assert_called_once()
            call_kwargs = store.client.upsert.call_args[1]
            assert call_kwargs["collection_name"] == "test_multimodal_docs"
            assert call_kwargs["wait"] is True

            points = call_kwargs["points"]
            assert len(points) == 2
            assert points[0].payload["page_id"] == "doc1_p1"
            assert points[0].payload["num_patches"] == 10
            assert points[0].payload["title"] == "Doc 1"
            assert len(points[0].vector) == 10
            assert len(points[0].vector[0]) == 128

    def test_upsert_pages_length_mismatch_raises_error(self, store):
        """Verify ValueError is raised if page_ids and embeddings lengths do not match."""
        with pytest.raises(ValueError, match="Mismatch"):
            store.upsert_pages(page_ids=["p1"], embeddings=[])

        with pytest.raises(ValueError, match="metadata_list length must match"):
            store.upsert_pages(
                page_ids=["p1"],
                embeddings=[[[0.1] * 128]],
                metadata_list=[{}, {}],
            )

    def test_search_binary_quantized_with_rescore(self, store, mock_qdrant_models):
        """Verify binary quantized search uses oversampling and rescoring parameters."""
        store.quantization_mode = "binary_quantized"

        # Mock search response
        mock_hit = MagicMock()
        mock_hit.id = "point-1"
        mock_hit.score = 18.75
        mock_hit.payload = {"page_id": "page_alpha", "section": "Summary"}

        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        store.client.query_points.return_value = mock_response

        # Mock query vector of shape (5 tokens, 128 dim)
        query_vector = [[0.05] * 128 for _ in range(5)]

        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            results = store.search(
                query_embedding=query_vector,
                top_k=3,
                rescore=True,
                oversampling=2.5,
            )

            assert len(results) == 1
            assert results[0]["id"] == "point-1"
            assert results[0]["page_id"] == "page_alpha"
            assert math.isclose(results[0]["score"], 18.75)
            assert results[0]["metadata"]["section"] == "Summary"

            store.client.query_points.assert_called_once()
            call_kwargs = store.client.query_points.call_args[1]
            assert call_kwargs["limit"] == 3
            search_params = call_kwargs["search_params"]
            assert search_params.quantization.rescore is True
            assert search_params.quantization.oversampling == 2.5

    def test_search_fp32_exact_no_quantization_params(self, store, mock_qdrant_models):
        """FP32 search does not include quantization search parameters."""
        store.quantization_mode = "fp32_exact"

        mock_hit = MagicMock()
        mock_hit.id = "point-2"
        mock_hit.score = 22.1
        mock_hit.payload = {"page_id": "page_beta"}

        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        store.client.query_points.return_value = mock_response

        query_vector = [[0.1] * 128]

        with patch("src.retrieval.qdrant_store.models", mock_qdrant_models):
            results = store.search(query_embedding=query_vector, top_k=1)

            assert len(results) == 1
            call_kwargs = store.client.query_points.call_args[1]
            assert call_kwargs["search_params"] is None

    def test_maxsim_mathematical_correctness(self):
        """
        Verify late-interaction MaxSim:
          Score(Q, D) = sum_i max_j (E_q^(i) . E_p^(j))
        with known hand-calculated vectors.
        """
        try:
            import numpy as np

            # Query: 2 token vectors of dimension 3
            # q0 = [1.0, 0.0, 0.0]
            # q1 = [0.0, 1.0, 0.0]
            q = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ], dtype=np.float32)

            # Document: 3 patch vectors of dimension 3
            # p0 = [0.8, 0.6, 0.0] -> <q0, p0> = 0.8, <q1, p0> = 0.6
            # p1 = [0.0, 0.9, 0.4] -> <q0, p1> = 0.0, <q1, p1> = 0.9
            # p2 = [0.5, 0.5, 0.0] -> <q0, p2> = 0.5, <q1, p2> = 0.5
            p = np.array([
                [0.8, 0.6, 0.0],
                [0.0, 0.9, 0.4],
                [0.5, 0.5, 0.0],
            ], dtype=np.float32)

            # MaxSim calculation:
            # Token 0 max: max(0.8, 0.0, 0.5) = 0.8
            # Token 1 max: max(0.6, 0.9, 0.5) = 0.9
            # Total MaxSim = 0.8 + 0.9 = 1.7
            expected_score = 1.7
            computed_score = VisualEncoder.compute_maxsim(q, p)
            assert math.isclose(computed_score, expected_score, rel_tol=1e-5)

        except ImportError:
            # Fallback pure python verification of MaxSim formula
            q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            p = [[0.8, 0.6, 0.0], [0.0, 0.9, 0.4], [0.5, 0.5, 0.0]]

            maxsim = 0.0
            for qi in q:
                max_dot = max(sum(qi[k] * pj[k] for k in range(3)) for pj in p)
                maxsim += max_dot

            assert math.isclose(maxsim, 1.7, rel_tol=1e-5)

    def test_get_storage_info(self, store):
        """Verify storage information inspection."""
        info = store.get_storage_info()
        assert info["collection_name"] == "test_multimodal_docs"
        assert info["points_count"] == 42
        assert info["vectors_count"] == 4200
        assert info["status"] == "green"
        assert "disk_size_mb" in info

    def test_count_and_delete(self, store):
        """Test count and delete collection wrappers."""
        assert store.count() == 42
        store.client.count.assert_called_once_with("test_multimodal_docs")

        assert store.delete_collection() is True
        store.client.delete_collection.assert_called_once_with("test_multimodal_docs")

    def test_client_none_raises_runtime_error(self):
        """Ensure clear RuntimeError when QdrantClient is not available."""
        with patch("src.retrieval.qdrant_store.QdrantClient", None):
            store = QdrantMultiVectorStore(qdrant_mode="local")
            store.client = None

            with pytest.raises(RuntimeError, match="QdrantClient is not available"):
                store.create_collection()

            with pytest.raises(RuntimeError, match="QdrantClient is not available"):
                store.upsert_pages(page_ids=["1"], embeddings=[[[0.1] * 128]])

            with pytest.raises(RuntimeError, match="QdrantClient is not available"):
                store.search(query_embedding=[[0.1] * 128])
