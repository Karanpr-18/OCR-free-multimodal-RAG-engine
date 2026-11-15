"""
PDF rendering utilities using PyMuPDF (fitz) to convert documents into PIL RGB images.

Provides deterministic page identification via SHA-256 hashing and high-quality
rasterization tailored for visual document retrieval models (e.g. ColPali / ColQwen).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def generate_page_id(
    pdf_path: Union[str, Path], page_num: int, dpi: int = 150
) -> str:
    """
    Generate a deterministic SHA-256 identifier for a rendered PDF page.

    Args:
        pdf_path: Path or filename of the PDF document.
        page_num: Zero-indexed page number.
        dpi: Dots-per-inch rendering resolution.

    Returns:
        Hexadecimal SHA-256 digest uniquely identifying this page rendering.
    """
    p = Path(pdf_path)
    norm_path = str(p.resolve()) if p.exists() else str(pdf_path)
    key = f"{norm_path}:{page_num}:{dpi}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def _pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    """Convert a PyMuPDF Pixmap to a PIL RGB Image."""
    if pix.alpha:
        # Flatten alpha onto white/RGB colorspace
        pix = fitz.Pixmap(fitz.csRGB, pix)
    elif pix.colorspace and pix.colorspace.name != fitz.csRGB.name:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_single_page(
    pdf_path: Union[str, Path],
    page_num: int,
    dpi: int = 150,
) -> Tuple[str, Image.Image]:
    """
    Render a single page of a PDF document to a PIL RGB image.

    Args:
        pdf_path: Path to the target PDF file.
        page_num: Zero-indexed page number to render.
        dpi: Target resolution in dots per inch (default 150).

    Returns:
        A tuple of (page_id, pil_image) where page_id is a deterministic SHA-256 string.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        IndexError: If page_num is out of document bounds.
        RuntimeError: If PyMuPDF encounters rendering or parsing errors.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    if dpi <= 0:
        raise ValueError(f"DPI must be a positive integer, got {dpi}")

    page_id = generate_page_id(path, page_num, dpi)

    try:
        with fitz.open(str(path)) as doc:
            total_pages = len(doc)
            if page_num < 0 or page_num >= total_pages:
                raise IndexError(
                    f"Page index {page_num} out of bounds for document "
                    f"'{path.name}' with {total_pages} pages (valid range: 0..{total_pages - 1})"
                )

            page = doc.load_page(page_num)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            image = _pixmap_to_pil(pix)

            logger.debug(
                "Rendered page %d/%d from '%s' at %d DPI -> %s (dim=%dx%d)",
                page_num,
                total_pages,
                path.name,
                dpi,
                page_id[:8],
                image.width,
                image.height,
            )
            return page_id, image

    except (fitz.FileDataError, fitz.EmptyFileError) as e:
        logger.error("Failed to parse PDF '%s': %s", path, e)
        raise RuntimeError(f"Corrupt or unreadable PDF document '{path}': {e}") from e
    except Exception as e:
        if isinstance(e, (FileNotFoundError, IndexError, ValueError)):
            raise
        logger.error("Unexpected error rendering page %d of '%s': %s", page_num, path, e)
        raise RuntimeError(f"Error rendering page {page_num} of '{path}': {e}") from e


def render_pdf_pages(
    pdf_path: Union[str, Path],
    dpi: int = 150,
) -> List[Tuple[str, Image.Image]]:
    """
    Render all pages of a PDF document to PIL RGB images at the given DPI.

    Args:
        pdf_path: Path to the target PDF file.
        dpi: Target resolution in dots per inch (default 150).

    Returns:
        List of (page_id, pil_image) tuples in sequential document order.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        RuntimeError: If PyMuPDF encounters rendering or parsing errors.
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    if dpi <= 0:
        raise ValueError(f"DPI must be a positive integer, got {dpi}")

    try:
        with fitz.open(str(path)) as doc:
            total_pages = len(doc)
            logger.info(
                "Rendering %d pages from '%s' at %d DPI",
                total_pages,
                path.name,
                dpi,
            )
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            rendered: List[Tuple[str, Image.Image]] = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = _pixmap_to_pil(pix)
                page_id = generate_page_id(path, page_num, dpi)
                rendered.append((page_id, img))

            logger.info("Successfully rendered %d pages for '%s'", len(rendered), path.name)
            return rendered

    except (fitz.FileDataError, fitz.EmptyFileError) as e:
        logger.error("Failed to parse PDF '%s': %s", path, e)
        raise RuntimeError(f"Corrupt or unreadable PDF document '{path}': {e}") from e
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        logger.error("Unexpected error rendering '%s': %s", path, e)
        raise RuntimeError(f"Error rendering PDF '{path}': {e}") from e


def render_pdf_bytes(
    pdf_bytes: bytes,
    dpi: int = 150,
    doc_name: str = "in_memory.pdf",
) -> List[Tuple[str, Image.Image]]:
    """
    Render all pages of an in-memory PDF (bytes) to PIL RGB images.

    Useful for FastAPI file upload endpoints where files may not be saved to disk.

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        dpi: Target resolution in dots per inch (default 150).
        doc_name: Logical document name for logging and identifier generation.

    Returns:
        List of (page_id, pil_image) tuples.
    """
    if not pdf_bytes:
        raise ValueError("Received empty PDF byte buffer")

    if dpi <= 0:
        raise ValueError(f"DPI must be a positive integer, got {dpi}")

    content_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            total_pages = len(doc)
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            rendered: List[Tuple[str, Image.Image]] = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = _pixmap_to_pil(pix)

                key = f"{doc_name}:{content_hash}:{page_num}:{dpi}".encode("utf-8")
                page_id = hashlib.sha256(key).hexdigest()
                rendered.append((page_id, img))

            logger.info("Rendered %d pages from byte stream (%s)", len(rendered), doc_name)
            return rendered

    except Exception as e:
        logger.error("Failed to render in-memory PDF '%s': %s", doc_name, e)
        raise RuntimeError(f"Error rendering PDF byte stream '{doc_name}': {e}") from e
