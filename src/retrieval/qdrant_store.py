"""
Qdrant Multi-Vector Store for ColQwen2.5 / ColPali document representations.

Supports:
  - Multi-vector late-interaction indexing with MaxSim comparator.
  - Dual quantization modes: 'fp32_exact' vs 'binary_quantized'.
  - Two-stage search with oversampling and rescoring.
  - Footprint / storage inspection (points, vectors, disk bytes).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)


def _normalize_point_id(raw_id: Union[str, int]) -> Union[str, int]:
    """Ensure the ID is a valid Qdrant ID (unsigned integer or RFC 4122 UUID)."""
    if isinstance(raw_id, int):
        return raw_id
    try:
        parsed = uuid.UUID(str(raw_id))
        return str(parsed)
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))


class QdrantMultiVectorStore:
    """
    Manages multi-vector document collections in Qdrant using MaxSim late interaction.
    """

    def __init__(
        self,
        collection_name: str = "multimodal_docs",
        qdrant_mode: str = "local",
        qdrant_url: Optional[str] = None,
        storage_path: str = "./qdrant_storage",
    ) -> None:
        """
        Initialize Qdrant client and target collection configuration.

        Args:
            collection_name: Target Qdrant collection name.
            qdrant_mode: 'local' (embedded on disk), 'memory' (in-RAM), or 'server'/'remote'.
            qdrant_url: URL for remote Qdrant server (e.g. http://localhost:6333).
            storage_path: Local path for disk persistence when mode is 'local'.
        """
        self.collection_name = collection_name
        self.qdrant_mode = qdrant_mode.lower()
        self.qdrant_url = qdrant_url or "http://localhost:6333"
        self.storage_path = storage_path
        self.quantization_mode: str = "binary_quantized"
        self.embedding_dim: int = 128

        self.client = self._init_client()

    def _init_client(self) -> Any:
        """Instantiate QdrantClient based on mode."""
        if QdrantClient is None:
            logger.warning("qdrant-client is not installed; running in stub mode.")
            return None

        if self.qdrant_mode == "memory" or self.storage_path == ":memory:":
            logger.info("Initializing Qdrant client in in-memory mode (:memory:).")
            return QdrantClient(location=":memory:")
        elif self.qdrant_mode in ("server", "remote"):
            logger.info("Connecting to Qdrant server at %s.", self.qdrant_url)
            return QdrantClient(url=self.qdrant_url)
        else:
            path_obj = Path(self.storage_path)
            path_obj.mkdir(parents=True, exist_ok=True)
            logger.info("Initializing local Qdrant client at %s.", path_obj.resolve())
            return QdrantClient(path=str(path_obj))

    def create_collection(
        self,
        quantization_mode: str = "binary_quantized",
        force_recreate: bool = False,
        embedding_dim: int = 128,
    ) -> None:
        """
        Create a multi-vector collection configured for late-interaction MaxSim.

        Args:
            quantization_mode: 'fp32_exact' or 'binary_quantized'.
            force_recreate: If True, drops existing collection before recreating.
            embedding_dim: Embedding dimension per token/patch (default: 128).
        """
        if self.client is None or models is None:
            raise RuntimeError("QdrantClient is not available. Install qdrant-client.")

        self.quantization_mode = quantization_mode
        self.embedding_dim = embedding_dim

        # Multi-vector configuration with MaxSim late interaction
        multivec_config = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
        vectors_config = models.VectorParams(
            size=embedding_dim,
            distance=models.Distance.COSINE,
            multivector_config=multivec_config,
        )

        quantization_config = None
        if quantization_mode == "binary_quantized":
            quantization_config = models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=True)
            )

        exists = self.client.collection_exists(self.collection_name)
        if exists:
            if force_recreate:
                logger.info("Recreating existing collection '%s'.", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(
                    "Collection '%s' already exists. Skipping creation.",
                    self.collection_name,
                )
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
            quantization_config=quantization_config,
        )
        logger.info(
            "Created multi-vector collection '%s' (mode=%s, dim=%d, comparator=MAX_SIM).",
            self.collection_name,
            quantization_mode,
            embedding_dim,
        )

    def upsert_pages(
        self,
        page_ids: List[str],
        embeddings: List[np.ndarray],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 32,
    ) -> int:
        """
        Upsert page multi-vector embeddings and associated metadata into Qdrant.

        Args:
            page_ids: Unique identifiers for each page.
            embeddings: List of (N_patches, 128) numpy arrays.
            metadata_list: Optional list of arbitrary metadata dicts per page.
            batch_size: Number of points to upsert per request.

        Returns:
            Number of points successfully upserted.
        """
        if self.client is None or models is None:
            raise RuntimeError("QdrantClient is not available. Install qdrant-client.")

        if len(page_ids) != len(embeddings):
            raise ValueError(
                f"Mismatch: len(page_ids)={len(page_ids)} vs len(embeddings)={len(embeddings)}"
            )

        if metadata_list is not None and len(metadata_list) != len(page_ids):
            raise ValueError("metadata_list length must match page_ids length.")

        total_upserted = 0
        points: List[models.PointStruct] = []

        for idx, (raw_id, emb) in enumerate(zip(page_ids, embeddings)):
            point_id = _normalize_point_id(raw_id)
            vector_data = emb.tolist() if isinstance(emb, np.ndarray) else emb

            payload: Dict[str, Any] = {
                "page_id": str(raw_id),
                "num_patches": len(vector_data),
            }
            if metadata_list is not None and idx < len(metadata_list):
                payload.update(metadata_list[idx])

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector_data,
                    payload=payload,
                )
            )

            if len(points) >= batch_size:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                )
                total_upserted += len(points)
                points = []

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            total_upserted += len(points)

        logger.info(
            "Upserted %d multi-vector points into collection '%s'.",
            total_upserted,
            self.collection_name,
        )
        return total_upserted

    def search(
        self,
        query_embedding: Union[np.ndarray, List[List[float]]],
        top_k: int = 5,
        rescore: bool = True,
        oversampling: float = 2.0,
        filter_conditions: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-vector late-interaction MaxSim search.

        Args:
            query_embedding: (N_tokens, 128) array or list of token vectors.
            top_k: Number of candidate pages to retrieve.
            rescore: Whether to rescore binary quantized candidates using original vectors.
            oversampling: Multiplier for initial binary candidate pool before rescore.
            filter_conditions: Optional Qdrant filter object.

        Returns:
            List of dicts: [{'id': str, 'page_id': str, 'score': float, 'metadata': dict}, ...]
        """
        if self.client is None or models is None:
            raise RuntimeError("QdrantClient is not available. Install qdrant-client.")

        query_vector = (
            query_embedding.tolist()
            if isinstance(query_embedding, np.ndarray)
            else query_embedding
        )

        search_params = None
        if self.quantization_mode == "binary_quantized":
            search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=rescore,
                    oversampling=oversampling,
                )
            )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                search_params=search_params,
                query_filter=filter_conditions,
                with_payload=True,
            )
            scored_points = response.points
        else:
            scored_points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                search_params=search_params,
                query_filter=filter_conditions,
                with_payload=True,
            )

        formatted_results: List[Dict[str, Any]] = []
        for point in scored_points:
            payload = point.payload or {}
            formatted_results.append(
                {
                    "id": str(point.id),
                    "page_id": str(payload.get("page_id", point.id)),
                    "score": float(point.score),
                    "metadata": payload,
                }
            )

        return formatted_results

    def get_storage_info(self) -> Dict[str, Any]:
        """
        Inspect collection storage footprint, point count, and configuration.

        Returns:
            Dictionary containing metrics on indexed points, vectors, and disk usage.
        """
        if self.client is None:
            return {"error": "QdrantClient is not available."}

        try:
            info = self.client.get_collection(self.collection_name)
        except Exception as exc:
            logger.warning("Could not fetch collection info: %s", exc)
            return {"error": str(exc), "collection_name": self.collection_name}

        disk_bytes = 0
        if self.qdrant_mode == "local" and Path(self.storage_path).exists():
            for root, _, files in os.walk(self.storage_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        disk_bytes += os.path.getsize(fp)
                    except OSError:
                        pass

        disk_mb = round(disk_bytes / (1024 * 1024), 3)

        points_count = getattr(info, "points_count", 0)
        vectors_count = getattr(info, "vectors_count", None) or getattr(
            info, "indexed_vectors_count", 0
        )
        status = getattr(info, "status", "unknown")

        return {
            "collection_name": self.collection_name,
            "mode": self.qdrant_mode,
            "quantization_mode": self.quantization_mode,
            "embedding_dim": self.embedding_dim,
            "points_count": points_count,
            "vectors_count": vectors_count,
            "status": str(status),
            "disk_size_bytes": disk_bytes,
            "disk_size_mb": disk_mb,
        }

    def count(self) -> int:
        """Return total number of points in the collection."""
        if self.client is None:
            return 0
        try:
            res = self.client.count(self.collection_name)
            return res.count
        except Exception:
            return 0

    def delete_collection(self) -> bool:
        """Drop the target collection."""
        if self.client is None:
            return False
        try:
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted collection '%s'.", self.collection_name)
            return True
        except Exception as exc:
            logger.warning("Failed to delete collection: %s", exc)
            return False
