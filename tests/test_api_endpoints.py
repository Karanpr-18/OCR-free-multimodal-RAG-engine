"""
Integration tests for FastAPI REST service endpoints.

Uses httpx.AsyncClient to test all 4 application endpoints:
  - GET /health (service diagnostics and component connectivity)
  - GET /v1/metrics (telemetry tracking, percentiles, abstention rate)
  - POST /v1/ingest (PDF ingestion via JSON payload and multipart upload)
  - POST /v1/query (late-interaction retrieval, abstention gating, VLM QA)

Mocks heavy ML models and external dependencies (ColQwen2.5, Qwen2.5-VL, Qdrant).
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

try:
    from httpx import ASGITransport, AsyncClient
except ImportError:
    ASGITransport = None
    AsyncClient = None

from src.api.main import app, metrics_tracker, ServiceContainer


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_services():
    """Build a mock ServiceContainer with mocked ML models and vector store."""
    services = ServiceContainer()

    # 1. Mock Visual Encoder (ColQwen2.5)
    mock_encoder = MagicMock()
    # (N_tokens, 128)
    mock_encoder.encode_queries.return_value = [[[0.01] * 128 for _ in range(4)]]
    # (N_patches, 128)
    mock_encoder.encode_images.return_value = [[[0.02] * 128 for _ in range(10)]]
    services.visual_encoder = mock_encoder

    # 2. Mock Qdrant Multi-Vector Store
    mock_store = MagicMock()
    mock_store.client = MagicMock()
    mock_store.client.get_collections.return_value = ["multimodal_docs"]
    mock_store.create_collection.return_value = None
    mock_store.upsert_pages.return_value = 1
    mock_store.search.return_value = [
        {
            "id": "mock_page_001",
            "page_id": "doc1_page_1",
            "score": 28.5,
            "payload": {"page_id": "doc1_page_1", "page_number": 1},
        },
        {
            "id": "mock_page_002",
            "page_id": "doc1_page_2",
            "score": 14.2,
            "payload": {"page_id": "doc1_page_2", "page_number": 2},
        },
    ]
    services.qdrant_store = mock_store

    # 3. Mock Abstention Calibrator
    mock_calibrator = MagicMock()
    mock_calibrator.compute_confidence.return_value = 0.885
    mock_calibrator.should_abstain.return_value = False
    services.abstention_calibrator = mock_calibrator

    # 4. Mock VLM Reader (Qwen2.5-VL)
    mock_vlm = MagicMock()
    mock_vlm.generate_answer.return_value = {
        "answer": "Operating margin in Q3 was 21.8% according to Figure 4.",
        "citations": ["Figure 4, Page 1: Operating Margin Growth"],
        "page_references": ["doc1_page_1"],
    }
    services.vlm_reader = mock_vlm

    # 5. Populate mock page cache
    services.page_cache["doc1_page_1"] = Image.new("RGB", (100, 100), color="white")
    services.page_cache["doc1_page_2"] = Image.new("RGB", (100, 100), color="blue")
    services.is_initialized = True

    return services


async def create_test_client(mock_services_instance):
    """Create configured AsyncClient with injected mock services."""
    app.state.services = mock_services_instance
    if ASGITransport is not None:
        transport = ASGITransport(app=app)
        return AsyncClient(transport=transport, base_url="http://testserver")
    elif AsyncClient is not None:
        return AsyncClient(app=app, base_url="http://testserver")
    else:
        raise RuntimeError("httpx is required for API integration tests.")


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestApiEndpoints:

    def test_health_endpoint_healthy(self, mock_services):
        """GET /health should return 200 and healthy status when dependencies are connected."""
        async def _run():
            async with await create_test_client(mock_services) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["qdrant_connected"] is True
                assert data["model_loaded"] is True
                assert data["version"] == "0.1.0"

        asyncio.run(_run())

    def test_health_endpoint_degraded(self):
        """GET /health should report degraded status if services are uninitialized."""
        degraded_services = ServiceContainer()
        degraded_services.qdrant_store = None
        degraded_services.visual_encoder = None
        degraded_services.vlm_reader = None

        async def _run():
            async with await create_test_client(degraded_services) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "degraded"
                assert data["qdrant_connected"] is False
                assert data["model_loaded"] is False

        asyncio.run(_run())

    def test_metrics_endpoint(self, mock_services):
        """GET /v1/metrics returns query latency percentiles and abstention telemetry."""
        async def _run():
            # Reset tracker
            metrics_tracker.total_queries = 5
            metrics_tracker.total_abstained = 1
            metrics_tracker.latencies.clear()
            metrics_tracker.latencies.extend([100.0, 150.0, 200.0, 250.0, 300.0])

            async with await create_test_client(mock_services) as client:
                response = await client.get("/v1/metrics")
                assert response.status_code == 200
                data = response.json()
                assert data["total_queries"] == 5
                assert data["abstention_rate"] == 0.2
                assert data["p50_latency_ms"] == 200.0
                assert data["p95_latency_ms"] > 0
                assert data["uptime_seconds"] >= 0.0

        asyncio.run(_run())

    def test_query_endpoint_answered_successfully(self, mock_services):
        """POST /v1/query executes late interaction, passes abstention gate, and generates grounded answer."""
        async def _run():
            mock_services.abstention_calibrator.should_abstain.return_value = False
            mock_services.abstention_calibrator.compute_confidence.return_value = 0.912

            payload = {
                "query": "What is the operating margin reported in Figure 4?",
                "top_k": 2,
                "collection_name": "multimodal_docs",
                "include_images": False,
            }

            async with await create_test_client(mock_services) as client:
                response = await client.post("/v1/query", json=payload)
                assert response.status_code == 200
                data = response.json()

                assert data["abstained"] is False
                assert "21.8%" in data["answer"]
                assert len(data["citations"]) > 0
                assert "Figure 4" in data["citations"][0]
                assert data["page_references"] == ["doc1_page_1"]
                assert data["confidence"] == 0.912

                # Latency breakdown validation
                latency = data["latency"]
                assert latency["encode_ms"] >= 0.0
                assert latency["search_ms"] >= 0.0
                assert latency["vlm_ms"] >= 0.0
                assert latency["total_ms"] > 0.0
                assert data["images"] is None

        asyncio.run(_run())

    def test_query_endpoint_abstained_low_confidence(self, mock_services):
        """POST /v1/query abstains and bypasses VLM when confidence falls below threshold."""
        async def _run():
            mock_services.abstention_calibrator.compute_confidence.return_value = 0.220
            mock_services.abstention_calibrator.should_abstain.return_value = True

            payload = {
                "query": "What was the CEO compensation in 1985?",
                "top_k": 2,
            }

            async with await create_test_client(mock_services) as client:
                response = await client.post("/v1/query", json=payload)
                assert response.status_code == 200
                data = response.json()

                assert data["abstained"] is True
                assert "cannot answer" in data["answer"].lower()
                assert data["citations"] == []
                assert data["confidence"] == 0.220
                assert data["latency"]["vlm_ms"] == 0.0

                # VLM generation should NOT have been invoked
                mock_services.vlm_reader.generate_answer.assert_not_called()

        asyncio.run(_run())

    def test_query_endpoint_with_image_inclusion(self, mock_services):
        """POST /v1/query returns base64 images when include_images is True."""
        async def _run():
            mock_services.abstention_calibrator.should_abstain.return_value = False

            payload = {
                "query": "Show me the chart from page 1",
                "top_k": 1,
                "include_images": True,
            }

            async with await create_test_client(mock_services) as client:
                response = await client.post("/v1/query", json=payload)
                assert response.status_code == 200
                data = response.json()
                assert data["abstained"] is False
                assert data["images"] is not None
                assert len(data["images"]) >= 1
                assert data["images"][0].startswith("data:image/jpeg;base64,")

        asyncio.run(_run())

    def test_query_endpoint_empty_query_rejected(self, mock_services):
        """POST /v1/query rejects empty queries with 400 Bad Request."""
        async def _run():
            async with await create_test_client(mock_services) as client:
                response = await client.post("/v1/query", json={"query": "   "})
                assert response.status_code == 400
                assert "empty" in response.json()["detail"].lower()

        asyncio.run(_run())

    def test_ingest_json_valid_file(self, mock_services):
        """POST /v1/ingest via JSON body with valid PDF file path."""
        async def _run():
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_pdf:
                # Patch PDF page renderer to avoid needing true PyMuPDF PDF binary
                mock_page_image = Image.new("RGB", (200, 200), color="white")
                with patch(
                    "src.ingestion.pdf_renderer.render_pdf_pages",
                    return_value=[("page_hash_1", mock_page_image)],
                ):
                    payload = {
                        "pdf_path": tmp_pdf.name,
                        "collection_name": "multimodal_docs",
                        "quantization_mode": "binary_quantized",
                    }

                    async with await create_test_client(mock_services) as client:
                        response = await client.post("/v1/ingest", json=payload)
                        assert response.status_code == 200
                        data = response.json()
                        assert data["status"] == "success"
                        assert data["pages_indexed"] == 1
                        assert data["collection_name"] == "multimodal_docs"
                        assert data["index_time_ms"] > 0

                        # Verify encoder and store were invoked
                        mock_services.visual_encoder.encode_images.assert_called_once()
                        mock_services.qdrant_store.upsert_pages.assert_called_once()

        asyncio.run(_run())

    def test_ingest_nonexistent_file_returns_404(self, mock_services):
        """POST /v1/ingest returns 404 when target file does not exist."""
        async def _run():
            payload = {"pdf_path": "/path/to/nonexistent/document_12345.pdf"}
            async with await create_test_client(mock_services) as client:
                response = await client.post("/v1/ingest", json=payload)
                assert response.status_code == 404
                assert "not found" in response.json()["detail"].lower()

        asyncio.run(_run())

    def test_ingest_multipart_file_upload(self, mock_services):
        """POST /v1/ingest via multipart file upload."""
        async def _run():
            dummy_pdf_bytes = b"%PDF-1.5 \n%mock content\n%%EOF"
            mock_page_image = Image.new("RGB", (200, 200), color="yellow")

            with patch(
                "src.ingestion.pdf_renderer.render_pdf_pages",
                return_value=[("page_hash_upload", mock_page_image)],
            ):
                files = {
                    "file": ("quarterly_report.pdf", dummy_pdf_bytes, "application/pdf")
                }
                data = {
                    "collection_name": "multimodal_docs",
                    "quantization_mode": "binary_quantized",
                }

                async with await create_test_client(mock_services) as client:
                    response = await client.post("/v1/ingest", files=files, data=data)
                    assert response.status_code == 200
                    resp_data = response.json()
                    assert resp_data["status"] == "success"
                    assert resp_data["pages_indexed"] == 1

        asyncio.run(_run())
