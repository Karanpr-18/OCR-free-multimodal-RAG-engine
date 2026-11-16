"""
FastAPI application for the OCR-Free Multimodal RAG Engine.

Provides high-performance asynchronous REST endpoints for:
- Document ingestion and multi-vector late-interaction indexing (/v1/ingest)
- Visual QA querying with confidence-calibrated abstention (/v1/query)
- Rolling latency and abstention telemetry metrics (/v1/metrics)
- System health and component status checks (/health)
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LatencyBreakdown,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from src.config import settings

logger = logging.getLogger("src.api.main")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Service Container & In-Memory State
# ---------------------------------------------------------------------------

class ServiceContainer:
    """Dependency container for retrieval, generation, and storage services."""

    def __init__(self) -> None:
        self.visual_encoder: Any = None
        self.qdrant_store: Any = None
        self.vlm_reader: Any = None
        self.abstention_calibrator: Any = None
        # In-memory page image cache mapping page_id -> PIL Image
        self.page_cache: Dict[str, Image.Image] = {}
        # Metadata cache mapping page_id -> dict(pdf_path=..., page_number=...)
        self.page_metadata: Dict[str, Dict[str, Any]] = {}
        self.is_initialized: bool = False

    def get_page_image(self, page_id: str) -> Image.Image:
        """Retrieve PIL image for a given page_id from cache or fallback."""
        if page_id in self.page_cache:
            return self.page_cache[page_id]

        meta = self.page_metadata.get(page_id)
        if meta and "pdf_path" in meta and "page_number" in meta:
            try:
                from src.ingestion.pdf_renderer import render_single_page
                _, img = render_single_page(
                    meta["pdf_path"],
                    meta["page_number"] - 1,
                    dpi=meta.get("dpi", 150),
                )
                self.page_cache[page_id] = img
                return img
            except Exception as err:
                logger.warning("Failed to render page %s on-demand: %s", page_id, err)

        # Fallback dummy RGB image if not found (e.g. during mock/synthetic testing)
        return Image.new("RGB", (224, 224), color=(240, 240, 240))


class MetricsTracker:
    """Thread-safe telemetry tracker for query latencies and abstention rates."""

    def __init__(self, max_history: int = 1000) -> None:
        self.start_time: float = time.time()
        self.total_queries: int = 0
        self.total_abstained: int = 0
        self.latencies: Deque[float] = collections.deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def record_query(self, total_latency_ms: float, abstained: bool) -> None:
        async with self._lock:
            self.total_queries += 1
            if abstained:
                self.total_abstained += 1
            self.latencies.append(total_latency_ms)

    def get_metrics(self) -> MetricsResponse:
        uptime = time.time() - self.start_time
        if self.latencies:
            lat_array = np.array(list(self.latencies), dtype=np.float64)
            p50 = float(np.percentile(lat_array, 50))
            p95 = float(np.percentile(lat_array, 95))
        else:
            p50 = 0.0
            p95 = 0.0

        abstention_rate = (
            float(self.total_abstained / self.total_queries)
            if self.total_queries > 0
            else 0.0
        )

        return MetricsResponse(
            total_queries=self.total_queries,
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            abstention_rate=round(abstention_rate, 4),
            uptime_seconds=round(uptime, 2),
        )


# Global metrics tracker
metrics_tracker = MetricsTracker()


# ---------------------------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup initialization and graceful shutdown."""
    logger.info("Initializing OCR-Free Multimodal RAG Engine services...")
    services = ServiceContainer()

    # 1. Initialize Abstention Calibrator
    try:
        from src.generation.abstention import AbstentionCalibrator
        services.abstention_calibrator = AbstentionCalibrator(
            default_threshold=settings.confidence_threshold
        )
        logger.info("AbstentionCalibrator initialized (threshold=%.2f)", settings.confidence_threshold)
    except Exception as exc:
        logger.warning("Could not initialize AbstentionCalibrator: %s", exc)

    # 2. Initialize Qdrant Multi-Vector Store
    try:
        from src.retrieval.qdrant_store import QdrantMultiVectorStore
        services.qdrant_store = QdrantMultiVectorStore(
            collection_name=settings.retrieval.collection_name,
            qdrant_mode=settings.qdrant.mode,
            qdrant_url=settings.qdrant.url,
            storage_path=settings.qdrant.local_path,
        )
        logger.info(
            "QdrantMultiVectorStore initialized (mode=%s, collection=%s)",
            settings.qdrant.mode,
            settings.retrieval.collection_name,
        )
    except Exception as exc:
        logger.warning("Could not initialize QdrantMultiVectorStore: %s", exc)

    # 3. Initialize Visual Encoder (ColQwen2.5 / ColPali)
    try:
        from src.retrieval.visual_encoder import VisualEncoder
        services.visual_encoder = VisualEncoder(
            model_name=settings.model.colqwen_model,
            device=settings.model.device,
            dtype=settings.model.dtype,
        )
        logger.info("VisualEncoder initialized (%s)", settings.model.colqwen_model)
    except Exception as exc:
        logger.warning("Could not initialize VisualEncoder: %s", exc)

    # 4. Initialize VLM Reader
    try:
        from src.generation.vlm_reader import VLMReader
        services.vlm_reader = VLMReader(
            model_name=settings.vlm_model or settings.model.vlm_model,
            device=settings.model.device,
            dtype=settings.model.dtype,
            api_endpoint=settings.vlm_api_endpoint or getattr(settings.model, "vlm_api_endpoint", None),
            api_key=settings.vlm_api_key or getattr(settings.model, "vlm_api_key", None),
            lazy_load=True,
        )
        logger.info("VLMReader initialized (%s, lazy_load=True)", settings.model.vlm_model)
    except Exception as exc:
        logger.warning("Could not initialize VLMReader: %s", exc)

    services.is_initialized = True
    app.state.services = services

    yield

    logger.info("Shutting down OCR-Free Multimodal RAG Engine services...")
    # Optional cleanup of client connections or caches
    services.page_cache.clear()
    services.page_metadata.clear()


# ---------------------------------------------------------------------------
# FastAPI App Construction
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="OCR-Free Multimodal RAG Engine API",
        description=(
            "High-performance document retrieval and visual QA service using "
            "ColQwen2.5 multi-vector late interaction, Qdrant binary quantization, "
            "confidence-calibrated abstention, and Qwen2.5-VL grounded generation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request latency tracking middleware
    @application.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_t = time.perf_counter()
        response = await call_next(request)
        process_ms = (time.perf_counter() - start_t) * 1000.0
        response.headers["X-Process-Time-Ms"] = f"{process_ms:.2f}"
        return response

    # Global Exception Handlers
    @application.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Internal Server Error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": f"Internal server error: {str(exc)}",
                "status_code": 500,
            },
        )

    return application


app = create_app()


def get_services(request: Request) -> ServiceContainer:
    """FastAPI dependency to retrieve the active service container."""
    if hasattr(request.app.state, "services"):
        return request.app.state.services
    # Fallback container if app.state.services is missing (e.g. in standalone test fixtures)
    return ServiceContainer()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(services: ServiceContainer = Depends(get_services)) -> HealthResponse:
    """
    Check connection status with Qdrant and model loading state.
    """
    qdrant_connected = False
    if services.qdrant_store is not None:
        try:
            client = getattr(services.qdrant_store, "client", None)
            if client is not None:
                # If Qdrant client exists, verify connectivity
                if hasattr(client, "get_collections"):
                    client.get_collections()
                qdrant_connected = True
            else:
                qdrant_connected = True
        except Exception as err:
            logger.warning("Health check Qdrant ping failed: %s", err)
            qdrant_connected = False

    models_loaded = (
        services.visual_encoder is not None
        and services.vlm_reader is not None
    )

    overall_status = "healthy" if (qdrant_connected or models_loaded) else "degraded"

    return HealthResponse(
        status=overall_status,
        qdrant_connected=qdrant_connected,
        model_loaded=models_loaded,
        version="0.1.0",
    )


@app.get("/v1/metrics", response_model=MetricsResponse, tags=["Telemetry"])
async def get_metrics() -> MetricsResponse:
    """
    Retrieve rolling operational telemetry including p50/p95 latency and abstention rates.
    """
    return metrics_tracker.get_metrics()


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(
    request: Request,
    payload: Optional[IngestRequest] = None,
    file: Optional[UploadFile] = File(None),
    services: ServiceContainer = Depends(get_services),
) -> IngestResponse:
    """
    Ingest a PDF document via file upload or local file path.

    Renders pages to visual images, encodes them into multi-vector patch embeddings,
    and indexes them into Qdrant using binary quantization late-interaction.
    """
    start_time = time.perf_counter()
    temp_file_path: Optional[str] = None

    content_type = request.headers.get("content-type", "")

    # Resolve input source: either multipart file upload or JSON payload
    pdf_path: Optional[str] = None
    collection_name = settings.retrieval.collection_name
    quantization_mode = "binary_quantized"

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded = form.get("file")
        if isinstance(uploaded, UploadFile):
            file = uploaded

        if file is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Multipart request missing 'file' upload.",
            )

        collection_name = str(form.get("collection_name", collection_name))
        quantization_mode = str(form.get("quantization_mode", quantization_mode))

        # Save uploaded bytes to temporary file
        suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            temp_file_path = tmp.name
        pdf_path = temp_file_path

    else:
        # JSON Body
        if payload is None:
            try:
                body = await request.json()
                payload = IngestRequest.model_validate(body)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid JSON request body: {e}",
                )

        pdf_path = payload.pdf_path
        collection_name = payload.collection_name or collection_name
        quantization_mode = payload.quantization_mode or quantization_mode

        if not pdf_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'pdf_path' is required when not uploading a file.",
            )

    target_path = Path(pdf_path)
    if not target_path.exists() or not target_path.is_file():
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Specified PDF file not found at: {pdf_path}",
        )

    try:
        # 1. Render PDF pages to PIL RGB Images
        from src.ingestion.pdf_renderer import render_pdf_pages

        pages_data = await asyncio.to_thread(render_pdf_pages, target_path, 150)
        if not pages_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF file contains no renderable pages: {pdf_path}",
            )

        page_ids = [p[0] for p in pages_data]
        page_images = [p[1] for p in pages_data]

        # Cache images and metadata for fast retrieval and generation
        for idx, (pid, img) in enumerate(pages_data):
            services.page_cache[pid] = img
            services.page_metadata[pid] = {
                "pdf_path": str(target_path.resolve()),
                "page_number": idx + 1,
                "dpi": 150,
            }

        # 2. Encode page images into multi-vector representations
        if services.visual_encoder is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Visual encoder service is not loaded.",
            )

        embeddings = await asyncio.to_thread(
            services.visual_encoder.encode_images,
            page_images,
        )

        # 3. Upsert into Qdrant Multi-Vector Store
        if services.qdrant_store is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant vector store service is not available.",
            )

        # Ensure collection exists
        if hasattr(services.qdrant_store, "create_collection"):
            try:
                services.qdrant_store.create_collection(quantization_mode=quantization_mode)
            except Exception as err:
                logger.debug("Collection create call returned: %s", err)

        metadata_list = [
            {
                "page_id": pid,
                "page_number": idx + 1,
                "pdf_path": str(target_path.resolve()),
                "total_pages": len(pages_data),
            }
            for idx, pid in enumerate(page_ids)
        ]

        await asyncio.to_thread(
            services.qdrant_store.upsert_pages,
            page_ids=page_ids,
            embeddings=embeddings,
            metadata_list=metadata_list,
        )

        index_time_ms = (time.perf_counter() - start_time) * 1000.0

        return IngestResponse(
            status="success",
            pages_indexed=len(pages_data),
            collection_name=collection_name,
            index_time_ms=round(index_time_ms, 2),
        )

    finally:
        # Clean up temporary uploaded file if created
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                logger.warning("Could not clean up temp file %s: %s", temp_file_path, e)


@app.post("/v1/query", response_model=QueryResponse, tags=["Retrieval & QA"])
async def query_pipeline(
    request: QueryRequest,
    services: ServiceContainer = Depends(get_services),
) -> QueryResponse:
    """
    Execute end-to-end OCR-free multimodal retrieval and grounded QA:
    1. Multi-vector query encoding (ColQwen2.5)
    2. MaxSim multi-vector search in Qdrant with binary quantization
    3. Confidence calibration & abstention gate evaluation
    4. Grounded VLM visual answer generation with citations (Qwen2.5-VL)
    """
    t_start = time.perf_counter()
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    top_k = request.top_k

    # 1. Encode query
    if services.visual_encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Visual encoder service is not initialized.",
        )

    t_enc_start = time.perf_counter()
    query_embeddings = await asyncio.to_thread(
        services.visual_encoder.encode_queries,
        [query],
    )
    encode_ms = (time.perf_counter() - t_enc_start) * 1000.0

    # 2. Multi-Vector Search via MaxSim
    if services.qdrant_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Qdrant store service is not initialized.",
        )

    t_search_start = time.perf_counter()
    query_emb = query_embeddings[0] if isinstance(query_embeddings, list) else query_embeddings
    search_hits = await asyncio.to_thread(
        services.qdrant_store.search,
        query_embedding=query_emb,
        top_k=top_k,
    )
    search_ms = (time.perf_counter() - t_search_start) * 1000.0

    # Parse search hits
    retrieved_scores: List[float] = []
    retrieved_page_ids: List[str] = []

    if search_hits:
        for hit in search_hits:
            score = getattr(hit, "score", None) if not isinstance(hit, dict) else hit.get("score")
            if score is not None:
                retrieved_scores.append(float(score))

            payload = getattr(hit, "payload", {}) if not isinstance(hit, dict) else hit.get("payload", {})
            if payload is None:
                payload = {}
            pid = payload.get("page_id") or getattr(hit, "id", None) or (hit.get("id") if isinstance(hit, dict) else None)
            if pid is not None:
                retrieved_page_ids.append(str(pid))

    # 3. Confidence Calibration & Abstention Decision
    confidence: float = 0.0
    abstained: bool = False

    if not search_hits or not retrieved_scores:
        confidence = 0.0
        abstained = True
    elif services.abstention_calibrator is not None:
        try:
            confidence = float(
                services.abstention_calibrator.compute_confidence(retrieved_scores)
            )
        except Exception as err:
            logger.warning("Error computing calibrated confidence: %s", err)
            confidence = float(retrieved_scores[0]) if retrieved_scores else 0.0

        try:
            abstained = bool(services.abstention_calibrator.should_abstain(confidence))
        except TypeError:
            abstained = bool(
                services.abstention_calibrator.should_abstain(
                    confidence, settings.confidence_threshold
                )
            )
    else:
        # Fallback simple thresholding if calibrator is not present
        confidence = float(retrieved_scores[0]) if retrieved_scores else 0.0
        abstained = confidence < settings.confidence_threshold

    # 4. If Abstained, bypass expensive VLM generation
    if abstained:
        vlm_ms = 0.0
        total_ms = (time.perf_counter() - t_start) * 1000.0
        await metrics_tracker.record_query(total_latency_ms=total_ms, abstained=True)

        return QueryResponse(
            answer="I cannot answer this question based on the retrieved document pages with sufficient confidence.",
            citations=[],
            page_references=retrieved_page_ids,
            confidence=round(confidence, 4),
            abstained=True,
            latency=LatencyBreakdown(
                encode_ms=round(encode_ms, 2),
                search_ms=round(search_ms, 2),
                vlm_ms=round(vlm_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            images=None,
        )

    # 5. VLM Generation with Retrieved Pages
    if services.vlm_reader is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VLM reader service is not initialized.",
        )

    t_vlm_start = time.perf_counter()
    retrieved_images = [services.get_page_image(pid) for pid in retrieved_page_ids]

    # Dynamically use selected reader model if specified
    vlm_reader = services.vlm_reader
    if request.model and hasattr(vlm_reader, "model_name") and vlm_reader.model_name != request.model:
        # Check if model has changed and update reader dynamically
        vlm_reader.model_name = request.model

    vlm_result = await asyncio.to_thread(
        vlm_reader.generate_answer,
        query=query,
        page_images=retrieved_images,
        page_ids=retrieved_page_ids,
    )
    vlm_ms = (time.perf_counter() - t_vlm_start) * 1000.0
    total_ms = (time.perf_counter() - t_start) * 1000.0

    # Parse VLM output
    if isinstance(vlm_result, dict):
        answer = str(vlm_result.get("answer", ""))
        citations = list(vlm_result.get("citations", []))
        page_references = list(vlm_result.get("page_references", retrieved_page_ids))
    else:
        answer = str(vlm_result)
        citations = []
        page_references = retrieved_page_ids

    # Optional image base64 inclusion
    encoded_images: Optional[List[str]] = None
    if request.include_images and retrieved_images:
        encoded_images = [_pil_to_base64(img) for img in retrieved_images]

    await metrics_tracker.record_query(total_latency_ms=total_ms, abstained=False)

    return QueryResponse(
        answer=answer,
        citations=citations,
        page_references=page_references,
        confidence=round(confidence, 4),
        abstained=False,
        latency=LatencyBreakdown(
            encode_ms=round(encode_ms, 2),
            search_ms=round(search_ms, 2),
            vlm_ms=round(vlm_ms, 2),
            total_ms=round(total_ms, 2),
        ),
        images=encoded_images,
    )


# ---------------------------------------------------------------------------
# Static Files & Web UI Mount
# ---------------------------------------------------------------------------

_webapp_dir = Path(__file__).resolve().parent.parent.parent / "webapp"
if _webapp_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_webapp_dir), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def redirect_root_to_ui():
        """Redirect root URL to web application UI."""
        return RedirectResponse(url="/ui/")

