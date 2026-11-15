"""Retrieval module: ColQwen2.5/ColPali visual encoder, Qdrant multi-vector store, and OCR baseline."""

from .ocr_baseline import OCRTextRetriever
from .qdrant_store import QdrantMultiVectorStore
from .visual_encoder import VisualEncoder

__all__ = [
    "VisualEncoder",
    "QdrantMultiVectorStore",
    "OCRTextRetriever",
]
