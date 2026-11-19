"""
Pydantic v2 schemas for the OCR-Free Multimodal RAG Engine API.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class IngestRequest(BaseModel):
    """Request payload for document ingestion."""

    model_config = ConfigDict(extra="ignore")

    pdf_path: Optional[str] = Field(
        default=None,
        description="Path to local PDF file to ingest",
        examples=["/path/to/document.pdf"],
    )
    collection_name: str = Field(
        default="multimodal_docs",
        description="Target Qdrant collection name",
        examples=["multimodal_docs"],
    )
    quantization_mode: Literal["binary_quantized", "fp32_exact"] = Field(
        default="binary_quantized",
        description="Vector quantization mode: 'binary_quantized' or 'fp32_exact'",
        examples=["binary_quantized"],
    )


class IngestResponse(BaseModel):
    """Response payload for document ingestion."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="success",
        description="Ingestion execution status",
        examples=["success"],
    )
    pages_indexed: int = Field(
        ...,
        description="Number of document pages successfully rendered and indexed",
        examples=[12],
    )
    collection_name: str = Field(
        ...,
        description="Qdrant collection name containing indexed vectors",
        examples=["multimodal_docs"],
    )
    index_time_ms: float = Field(
        ...,
        description="Total time in milliseconds taken to render, encode, and index the document",
        examples=[1452.8],
    )


class QueryRequest(BaseModel):
    """Request payload for multi-vector late-interaction visual document QA query."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(
        ...,
        description="Natural language question about the ingested documents",
        examples=["What is the total revenue reported in table 3?"],
    )
    collection_name: str = Field(
        default="multimodal_docs",
        description="Qdrant collection name to retrieve pages from",
        examples=["multimodal_docs"],
    )
    top_k: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of top visual pages to retrieve via MaxSim",
        examples=[2],
    )
    model: Optional[str] = Field(
        default=None,
        description="Override reader model ID (e.g. opencode/deepseek-v4.1-flash-free, Qwen/Qwen2.5-VL-7B-Instruct)",
        examples=["opencode/deepseek-v4.1-flash-free"],
    )
    vlm_endpoint: Optional[str] = Field(
        default=None,
        description="Optional OpenAI-compatible VLM bridge or local endpoint URL",
        examples=["http://localhost:18787/v1"],
    )
    vlm_api_key: Optional[str] = Field(
        default=None,
        description="Optional API key for OpenCode or Google Gemini",
        examples=["AIzaSy..."],
    )
    include_images: bool = Field(
        default=False,
        description="Whether to return retrieved page images (base64 or references)",
        examples=[False],
    )


class LatencyBreakdown(BaseModel):
    """Breakdown of query pipeline latency in milliseconds."""

    model_config = ConfigDict(extra="ignore")

    encode_ms: float = Field(
        default=0.0,
        description="Latency for query multi-vector token encoding (ColQwen2.5)",
        examples=[24.5],
    )
    search_ms: float = Field(
        default=0.0,
        description="Latency for Qdrant multi-vector MaxSim search",
        examples=[18.2],
    )
    vlm_ms: float = Field(
        default=0.0,
        description="Latency for VLM generation (Qwen2.5-VL)",
        examples=[350.0],
    )
    total_ms: float = Field(
        default=0.0,
        description="Total roundtrip pipeline latency",
        examples=[392.7],
    )


class QueryResponse(BaseModel):
    """Response payload for visual QA query."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(
        ...,
        description="Generated answer grounded on retrieved visual pages, or abstention refusal message",
        examples=["The total revenue reported in table 3 is $4.2B."],
    )
    citations: List[str] = Field(
        default_factory=list,
        description="Specific grounded citations extracted from page content (e.g., table cells, chart axes)",
        examples=["Table 3, Row 4: Total Revenue"],
    )
    page_references: List[str] = Field(
        default_factory=list,
        description="Page identifiers or numbers used to answer the question",
        examples=["doc_p1", "doc_p3"],
    )
    confidence: float = Field(
        ...,
        description="Calibrated confidence score C(q) in range [0, 1]",
        examples=[0.87],
    )
    abstained: bool = Field(
        ...,
        description="True if query was rejected by confidence abstention gate",
        examples=[False],
    )
    latency: LatencyBreakdown = Field(
        ...,
        description="Detailed latency breakdown for pipeline stages",
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="Optional base64-encoded visual page images if include_images was requested",
    )


class MetricsResponse(BaseModel):
    """System operational metrics and latency quantiles."""

    model_config = ConfigDict(extra="ignore")

    total_queries: int = Field(
        ...,
        description="Total number of queries handled since server start",
        examples=[120],
    )
    p50_latency_ms: float = Field(
        ...,
        description="50th percentile (median) query total latency in milliseconds",
        examples=[380.5],
    )
    p95_latency_ms: float = Field(
        ...,
        description="95th percentile query total latency in milliseconds",
        examples=[520.1],
    )
    abstention_rate: float = Field(
        ...,
        description="Proportion of total queries that were abstained",
        examples=[0.15],
    )
    uptime_seconds: float = Field(
        ...,
        description="Server uptime in seconds",
        examples=[3600.0],
    )


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(extra="ignore")

    status: str = Field(
        default="healthy",
        description="Service health status ('healthy' or 'degraded')",
        examples=["healthy"],
    )
    qdrant_connected: bool = Field(
        ...,
        description="Indicates whether Qdrant vector database is reachable and operational",
        examples=[True],
    )
    model_loaded: bool = Field(
        ...,
        description="Indicates whether visual encoder and VLM models are initialized",
        examples=[True],
    )
    version: str = Field(
        default="0.1.0",
        description="API service version",
        examples=["0.1.0"],
    )
