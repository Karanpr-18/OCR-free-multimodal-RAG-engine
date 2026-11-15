"""Ingestion module: PDF rendering and dataset loading for multimodal document RAG."""

from .pdf_renderer import (
    generate_page_id,
    render_pdf_bytes,
    render_pdf_pages,
    render_single_page,
)
from .dataset_loader import (
    build_unanswerable_split,
    get_evaluation_splits,
    load_vidore_dataset,
)

__all__ = [
    "render_pdf_pages",
    "render_single_page",
    "render_pdf_bytes",
    "generate_page_id",
    "load_vidore_dataset",
    "build_unanswerable_split",
    "get_evaluation_splits",
]
