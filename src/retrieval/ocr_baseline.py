"""
OCR + Dense Text RAG baseline retriever.

Uses PyMuPDF (fitz) for text extraction, recursive character/token chunking
(chunk_size=512, overlap=64), BAAI/bge-small-en-v1.5 embeddings via
sentence-transformers, and Qdrant single-vector storage for baseline benchmarking.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None
    models = None

from .visual_encoder import resolve_device

logger = logging.getLogger(__name__)


def _normalize_point_id(raw_id: Union[str, int]) -> Union[str, int]:
    """Ensure ID is a valid unsigned int or RFC 4122 UUID."""
    if isinstance(raw_id, int):
        return raw_id
    try:
        return str(uuid.UUID(str(raw_id)))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))


class OCRTextRetriever:
    """
    Traditional OCR + dense text retrieval baseline.

    Extracts text from PDF documents using PyMuPDF, chunks text recursively,
    encodes chunks using BAAI/bge-small-en-v1.5 (384-dim), and stores single vectors
    in Qdrant for comparative evaluation against ColQwen multi-vector retrieval.
    """

    DEFAULT_EMBEDDER = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        collection_name: str = "ocr_text_baseline",
        model_name: str = DEFAULT_EMBEDDER,
        qdrant_mode: str = "local",
        qdrant_url: Optional[str] = None,
        storage_path: str = "./qdrant_storage",
        device: str = "auto",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        """
        Initialize the OCR text retriever baseline.

        Args:
            collection_name: Qdrant single-vector collection name.
            model_name: Sentence-transformers embedding model ID.
            qdrant_mode: 'local', 'memory', or 'server'.
            qdrant_url: Remote Qdrant server URL.
            storage_path: Local disk storage directory for Qdrant.
            device: 'auto', 'cuda', 'mps', or 'cpu'.
            chunk_size: Target character/token chunk size (default: 512).
            chunk_overlap: Chunk overlap in characters/tokens (default: 64).
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.qdrant_mode = qdrant_mode.lower()
        self.qdrant_url = qdrant_url or "http://localhost:6333"
        self.storage_path = storage_path
        self.device = resolve_device(device)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embedder: Any = None
        self._load_embedder()

        self.client = self._init_qdrant()
        self.create_collection()

    def _load_embedder(self) -> None:
        """Load sentence-transformers dense embedding model."""
        if SentenceTransformer is None:
            logger.warning(
                "sentence-transformers not installed; OCRTextRetriever in stub mode."
            )
            return

        try:
            logger.info(
                "Loading dense embedder '%s' on device %s...",
                self.model_name,
                self.device,
            )
            self.embedder = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            logger.warning("Failed to load SentenceTransformer: %s", exc)

    def _init_qdrant(self) -> Any:
        """Initialize Qdrant client instance."""
        if QdrantClient is None:
            logger.warning("qdrant-client not installed; running in stub mode.")
            return None

        if self.qdrant_mode == "memory" or self.storage_path == ":memory:":
            return QdrantClient(location=":memory:")
        elif self.qdrant_mode in ("server", "remote"):
            return QdrantClient(url=self.qdrant_url)
        else:
            path_obj = Path(self.storage_path)
            path_obj.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(path_obj))

    def create_collection(self, force_recreate: bool = False) -> None:
        """Create single-vector collection in Qdrant with Cosine distance."""
        if self.client is None or models is None:
            return

        vectors_config = models.VectorParams(
            size=self.EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        )

        exists = self.client.collection_exists(self.collection_name)
        if exists:
            if force_recreate:
                logger.info("Recreating collection '%s'.", self.collection_name)
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )
        logger.info(
            "Created dense text collection '%s' (dim=%d, distance=COSINE).",
            self.collection_name,
            self.EMBEDDING_DIM,
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Recursively chunk text with paragraph, sentence, and word boundaries.

        Args:
            text: Raw input text.
            chunk_size: Maximum chunk length in characters.
            overlap: Overlap length between consecutive chunks.

        Returns:
            List of text chunks.
        """
        c_size = chunk_size or self.chunk_size
        c_overlap = overlap or self.chunk_overlap

        text = text.strip()
        if not text:
            return []

        if len(text) <= c_size:
            return [text]

        # Splitting delimiters in descending granularity
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        def _split_recursive(t: str, seps: List[str]) -> List[str]:
            if len(t) <= c_size or not seps:
                return [t] if t.strip() else []

            sep = seps[0]
            remaining_seps = seps[1:]

            if sep:
                parts = t.split(sep)
            else:
                parts = list(t)

            chunks: List[str] = []
            current = ""

            for part in parts:
                candidate = f"{current}{sep}{part}" if current and sep else part
                if len(candidate) <= c_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    if len(part) > c_size:
                        sub_chunks = _split_recursive(part, remaining_seps)
                        chunks.extend(sub_chunks)
                        current = ""
                    else:
                        current = part

            if current.strip():
                chunks.append(current.strip())

            return [c for c in chunks if c]

        raw_chunks = _split_recursive(text, separators)

        # Apply overlap stitching if needed
        if c_overlap <= 0 or len(raw_chunks) <= 1:
            return raw_chunks

        overlapped: List[str] = []
        for i, chunk in enumerate(raw_chunks):
            if i > 0 and len(overlapped) > 0:
                prev = raw_chunks[i - 1]
                overlap_prefix = prev[-c_overlap:] if len(prev) > c_overlap else prev
                chunk_with_overlap = f"{overlap_prefix} {chunk}".strip()
                overlapped.append(chunk_with_overlap)
            else:
                overlapped.append(chunk)

        return overlapped

    def index_pdf(
        self,
        pdf_path: Union[str, Path],
        doc_id: Optional[str] = None,
    ) -> int:
        """
        Extract text from a PDF, recursively chunk, embed, and index in Qdrant.

        Args:
            pdf_path: File path to the PDF.
            doc_id: Optional document identifier. Defaults to filename stem.

        Returns:
            Total number of chunks indexed.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF text extraction.")

        document_id = doc_id or pdf_file.stem
        doc = fitz.open(str(pdf_file))

        extracted_chunks: List[Dict[str, Any]] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_text = page.get_text() or ""
            page_num = page_idx + 1
            page_id = f"{document_id}_p{page_num}"

            chunks = self.chunk_text(page_text)
            for c_idx, chunk_str in enumerate(chunks):
                extracted_chunks.append(
                    {
                        "chunk_id": f"{page_id}_c{c_idx}",
                        "page_id": page_id,
                        "doc_id": document_id,
                        "page_number": page_num,
                        "text": chunk_str,
                        "pdf_path": str(pdf_file.resolve()),
                    }
                )

        doc.close()
        return self._embed_and_upsert_chunks(extracted_chunks)

    def index_text_pages(
        self,
        pages: List[Dict[str, Any]],
    ) -> int:
        """
        Index pre-extracted page text representations (e.g. from ViDoRe datasets).

        Args:
            pages: List of dicts containing 'page_id', 'text', and optional metadata.

        Returns:
            Number of indexed chunks.
        """
        all_chunks: List[Dict[str, Any]] = []

        for p in pages:
            raw_text = p.get("text", "")
            page_id = str(p.get("page_id", uuid.uuid4().hex[:12]))
            page_num = p.get("page_number", 1)

            chunks = self.chunk_text(raw_text)
            if not chunks:
                chunks = [""]  # maintain page entry even if text was empty

            for c_idx, c_text in enumerate(chunks):
                chunk_meta = {k: v for k, v in p.items() if k != "text"}
                chunk_meta.update(
                    {
                        "chunk_id": f"{page_id}_c{c_idx}",
                        "page_id": page_id,
                        "page_number": page_num,
                        "text": c_text,
                    }
                )
                all_chunks.append(chunk_meta)

        return self._embed_and_upsert_chunks(all_chunks)

    def _embed_and_upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed list of chunk dicts and upsert to Qdrant."""
        if not chunks:
            return 0

        if self.client is None or models is None:
            raise RuntimeError("QdrantClient is not available.")

        texts = [c["text"] for c in chunks]

        if self.embedder is not None:
            dense_vectors = self.embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        else:
            # Fallback stub vectors if embedder unavailable (e.g. mock test)
            dense_vectors = np.zeros((len(texts), self.EMBEDDING_DIM), dtype=np.float32)

        points: List[models.PointStruct] = []
        for idx, chunk_meta in enumerate(chunks):
            chunk_id = chunk_meta["chunk_id"]
            point_id = _normalize_point_id(chunk_id)
            vector = dense_vectors[idx].tolist()

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk_meta,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info(
            "Upserted %d text chunks into '%s'.",
            len(points),
            self.collection_name,
        )
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_conditions: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search dense text baseline collection with a query string.

        Args:
            query: User question or search query.
            top_k: Number of chunks to retrieve.
            filter_conditions: Optional Qdrant filter object.

        Returns:
            List of matched chunk dicts:
            [{'chunk_id': str, 'page_id': str, 'page_number': int, 'score': float, 'text': str, 'metadata': dict}, ...]
        """
        if self.client is None or models is None:
            raise RuntimeError("QdrantClient is not available.")

        if self.embedder is not None:
            query_vector = self.embedder.encode(
                query,
                normalize_embeddings=True,
            ).tolist()
        else:
            query_vector = [0.0] * self.EMBEDDING_DIM

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=filter_conditions,
                with_payload=True,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=filter_conditions,
                with_payload=True,
            )

        results: List[Dict[str, Any]] = []
        for p in points:
            payload = p.payload or {}
            results.append(
                {
                    "id": str(p.id),
                    "chunk_id": str(payload.get("chunk_id", p.id)),
                    "page_id": str(payload.get("page_id", "")),
                    "page_number": payload.get("page_number", 1),
                    "score": float(p.score),
                    "text": payload.get("text", ""),
                    "metadata": payload,
                }
            )

        return results

    def search_pages(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search and aggregate chunk matches up to page level (taking max chunk score).

        Args:
            query: Search query text.
            top_k: Number of top pages to return.

        Returns:
            List of aggregated top pages: [{'page_id': str, 'score': float, 'best_chunk': str, ...}, ...]
        """
        # Retrieve extra candidate chunks to ensure top_k distinct pages
        chunk_results = self.search(query=query, top_k=top_k * 4)

        pages_map: Dict[str, Dict[str, Any]] = {}
        for r in chunk_results:
            page_id = r["page_id"]
            score = r["score"]

            if page_id not in pages_map or score > pages_map[page_id]["score"]:
                pages_map[page_id] = {
                    "page_id": page_id,
                    "page_number": r["page_number"],
                    "score": score,
                    "best_chunk": r["text"],
                    "metadata": r["metadata"],
                }

        ranked_pages = sorted(
            pages_map.values(), key=lambda x: x["score"], reverse=True
        )
        return ranked_pages[:top_k]

    def get_storage_info(self) -> Dict[str, Any]:
        """Inspect storage footprint of the baseline collection."""
        if self.client is None:
            return {"error": "QdrantClient is not available."}

        try:
            info = self.client.get_collection(self.collection_name)
        except Exception as exc:
            return {"error": str(exc), "collection_name": self.collection_name}

        disk_bytes = 0
        if self.qdrant_mode == "local" and Path(self.storage_path).exists():
            for root, _, files in os.walk(self.storage_path):
                for f in files:
                    try:
                        disk_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass

        points_count = getattr(info, "points_count", 0)
        return {
            "collection_name": self.collection_name,
            "mode": self.qdrant_mode,
            "embedding_dim": self.EMBEDDING_DIM,
            "points_count": points_count,
            "disk_size_bytes": disk_bytes,
            "disk_size_mb": round(disk_bytes / (1024 * 1024), 3),
        }
