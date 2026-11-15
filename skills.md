# Engineering Specification: OCR-Free Multimodal Document & Diagram RAG Engine

## 1. Execution Directive for Autonomous Agent
You are tasked with building, testing, and benchmarking a production-grade **Multimodal Document & Diagram Retrieval-Augmented Generation (RAG) Engine** from scratch. 

Execute this specification systematically in phases. Do not leave placeholder `TODO` functions, mock metrics, or untested endpoints. Every component must run end-to-end, pass automated unit/integration tests, and generate verified benchmark artifacts (`results/benchmark_summary.json` and `results/ablation_table.md`) comparing an OCR-free Vision-Language Retriever (**ColQwen2.5 / ColPali** + **Qdrant Multi-Vector Binary Quantization**) against a traditional **OCR + Dense Text RAG** baseline, complete with **confidence-calibrated abstention**.

---

## 2. System Architecture & Core Design Decisions

1. **OCR-Free Visual Indexing (`ColQwen2.5` / `ColPali`):**
   - Instead of extracting text via brittle OCR, render PDF pages as RGB images and encode them into multi-vector patch representations ($N \times 128$, where $N \approx 768\text{--}1030$ visual patch tokens per page).
   - Primary model: `vidore/colqwen2.5-v0.2` via `colpali-engine` (with configurable fallback to `vidore/colpali-v1.3` for lower-VRAM environments).
2. **Multi-Vector Late-Interaction Storage (`Qdrant`):**
   - Store page patch embeddings in Qdrant using `MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)`.
   - Enable **Binary Quantization** (`BinaryQuantization(binary=BinaryQuantizationConfig(always_ram=True))`) with two-stage search (`oversampling=2.0`, `rescore=True`) to benchmark memory compression vs. retrieval recall retention.
   - Support both Qdrant local persistent disk storage (`./qdrant_storage`) and Dockerized Qdrant server via environment configuration (`QDRANT_MODE=local|server`).
3. **Traditional Baseline for Quantitative Comparison:**
   - Implement a modular baseline pipeline (`src/baselines/ocr_text_rag.py`) using `PyMuPDF` (`fitz`) text extraction + recursive token chunking + `BAAI/bge-small-en-v1.5` dense embeddings in Qdrant.
   - Both pipelines must be evaluated on the exact same benchmark splits to quantify gains on tables, charts, and multi-column layouts.
4. **Grounded VLM Answer Generation & Confidence-Calibrated Abstention:**
   - Pass top-$k$ retrieved page images directly into a Vision-Language Model (`Qwen/Qwen2.5-VL-3B-Instruct` or `Qwen/Qwen2.5-VL-7B-Instruct`, plus optional OpenAI-compatible VLM endpoint support).
   - Compute a calibrated **Retrieval-Generation Confidence Score** $C(q, \mathcal{D}_k)$ combining normalized `MaxSim` retrieval margin, top-1 vs. top-2 score gap, and VLM token-level / self-consistency grounding score.
   - Trigger deterministic **Abstention** (`"INSUFFICIENT_VISUAL_EVIDENCE"`) when $C(q, \mathcal{D}_k) < \tau$, where threshold $\tau$ is tuned on a validation split containing unanswerable/out-of-corpus queries.
5. **Production Serving & Telemetry:**
   - Expose ingestion, querying, health, and telemetry endpoints via **FastAPI** with asynchronous request handling, structured latency instrumentation (`encode_ms`, `search_ms`, `vlm_ms`, `total_ms`), and a `Dockerfile` + `docker-compose.yml`.

---

## 3. Repository Structure

Create the exact directory layout below:

```text
multimodal-doc-rag/
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── README.md
├── configs/
│   └── default.yaml
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_renderer.py         # PDF-to-PIL conversion & page metadata hashing
│   │   └── dataset_loader.py       # Hugging Face ViDoRe loader + unanswerable split builder
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── visual_encoder.py       # ColQwen2.5 / ColPali batch image & query encoder
│   │   ├── qdrant_store.py         # MultiVector MaxSim + Binary Quantization manager
│   │   └── ocr_baseline.py         # PyMuPDF + BGE dense text baseline for comparison
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── vlm_reader.py           # Qwen2.5-VL grounded QA with bounding/region citation
│   │   └── abstention.py           # Calibrated confidence scoring & threshold abstention
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Recall@1, Recall@5, MRR, nDCG@5, Abstention AUROC, ECE
│   │   └── run_benchmark.py        # End-to-end comparative benchmark & ablation script
│   └── api/
│       ├── __init__.py
│       ├── schemas.py              # Pydantic v2 request/response models
│       └── main.py                 # FastAPI app with p50/p95 latency middleware
├── scripts/
│   └── generate_readme_charts.py   # Generates latency & recall comparison plots
├── tests/
│   ├── test_qdrant_multivector.py
│   ├── test_abstention_calibration.py
│   └── test_api_endpoints.py
└── results/
    └── .gitkeep
```

---

## 4. Environment & Dependencies

Pin clean, compatible dependencies in `requirements.txt`:

```text
torch>=2.4.0
torchvision>=0.19.0
transformers>=4.49.0
colpali-engine>=0.3.8
qdrant-client>=1.13.0
qwen-vl-utils>=0.0.10
accelerate>=1.3.0
sentence-transformers>=3.4.0
datasets>=3.2.0
pymupdf>=1.25.0
pillow>=11.0.0
numpy>=1.26.0
scipy>=1.14.0
scikit-learn>=1.6.0
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
pydantic>=2.10.0
pydantic-settings>=2.7.0
pyyaml>=6.0.2
matplotlib>=3.10.0
pytest>=8.3.0
httpx>=0.28.0
```

---

## 5. Mathematical & Algorithmic Specifications

### 5.1 Late-Interaction Scoring (`MaxSim`)
Given a query tokenized into $N_q$ text embeddings $E_q \in \mathbb{R}^{N_q \times d}$ ($d = 128$) and a document page encoded into $N_p$ visual patch embeddings $E_p \in \mathbb{R}^{N_p \times d}$, compute late-interaction similarity as:

$$S(q, p) = \sum_{i=1}^{N_q} \max_{j \in \{1, \dots, N_p\}} \langle E_q^{(i)}, E_p^{(j)} \rangle$$

Configure Qdrant's collection vector parameters explicitly with:
- `size=128`
- `distance=models.Distance.COSINE`
- `multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)`

### 5.2 Binary Quantization & Rescoring
In `src/retrieval/qdrant_store.py`, support two index modes for ablation:
1. `fp32_exact`: Full-precision multi-vector storage without quantization.
2. `binary_quantized`: `models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True))` queried with `models.QuantizationSearchParams(rescore=True, oversampling=2.0)`.
Measure and log exact RAM/storage footprint per 100 pages and `nDCG@5` delta between `fp32_exact` and `binary_quantized`.

### 5.3 Confidence-Calibrated Abstention
For a query $q$ with retrieved top-$k$ pages $\{p_1, \dots, p_k\}$ and raw `MaxSim` scores $\{s_1, \dots, s_k\}$:
1. Normalize retrieval score by query token length: $\tilde{s}_1 = \frac{s_1}{N_q}$.
2. Compute retrieval margin: $\Delta s = \frac{s_1 - s_2}{\max(s_1, \epsilon)}$.
3. Combine with VLM answer log-probability / self-verification confidence $p_{\text{vlm}} \in [0, 1]$ via a logistic or weighted calibration model:

$$C(q) = \sigma\left(w_1 \tilde{s}_1 + w_2 \Delta s + w_3 \log\frac{p_{\text{vlm}}}{1 - p_{\text{vlm}}} + b\right)$$

If $C(q) < \tau$, the system must abstain from answering and return `abstained=True` alongside the top candidate page references.

---

## 6. Step-by-Step Implementation Plan

### Phase 1: Dataset & Ingestion Pipeline
1. Implement `src/ingestion/dataset_loader.py` to pull evaluation subsets from Hugging Face (`vidore/tabfquad_test_subsampled` and `vidore/docvqa_test_subsampled`).
2. Construct a deterministic **Unanswerable / Negative Query Split** (25% of evaluation queries) by pairing out-of-domain or counterfactual questions against the indexed corpus so abstention calibration can be rigorously measured.
3. Implement `src/ingestion/pdf_renderer.py` using `PyMuPDF` to convert arbitrary multi-page PDFs into 150-DPI PIL images with deterministic SHA-256 page IDs.

### Phase 2: Retrieval Engines & Qdrant Multi-Vector Store
1. Implement `src/retrieval/visual_encoder.py` wrapping `ColQwen2_5` / `ColPali` in `torch.bfloat16` with automatic CUDA/MPS/CPU device fallback and batched inference (`batch_size=4`).
2. Implement `src/retrieval/qdrant_store.py` supporting collection creation, batch upsert of multi-vectors + metadata payloads, `MaxSim` search, and storage footprint inspection.
3. Implement `src/retrieval/ocr_baseline.py` using `PyMuPDF` text extraction + `BAAI/bge-small-en-v1.5` single-vector Qdrant indexing as the comparative baseline.

### Phase 3: VLM Reader & Abstention Module
1. Implement `src/generation/vlm_reader.py` using `Qwen/Qwen2.5-VL-3B-Instruct` (configurable to `7B` or API mode) taking top-$k$ page images ($k=2$ default) and instructing the model to cite exact page numbers and visual elements (table row/column, chart axis).
2. Implement `src/generation/abstention.py` to compute $C(q)$, calibrate $\tau$ on a validation fold targeting $\ge 90\%$ precision on answered queries, and compute **Abstention AUROC** and **Risk-Coverage curves**.

### Phase 4: FastAPI Service & Containerization
1. Build `src/api/main.py` with endpoints:
   - `POST /v1/ingest`: Accepts PDF upload or directory path, renders pages, indexes multi-vectors.
   - `POST /v1/query`: Runs retrieval, abstention gate, and VLM generation; returns answer, citations, confidence score, `abstained` flag, and per-stage latency breakdown (`encode_ms`, `search_ms`, `vlm_ms`).
   - `GET /v1/metrics`: Exposes rolling `p50`/`p95` latencies, query counts, and abstention rate.
2. Create `Dockerfile` and `docker-compose.yml` orchestrating the FastAPI app and Qdrant service.

### Phase 5: Automated Benchmarking & Reporting
1. Implement `src/evaluation/run_benchmark.py` to evaluate:
   - **System 1:** OCR (`PyMuPDF`) + Dense Text (`BGE-small`)
   - **System 2:** ColQwen2.5 / ColPali (`fp32_exact`)
   - **System 3:** ColQwen2.5 / ColPali (`binary_quantized` + `rescore=True`)
   - **System 4:** Full Multimodal RAG + Calibrated Abstention
2. Output `results/benchmark_summary.json` and `results/ablation_table.md` containing `Recall@1`, `Recall@5`, `nDCG@5`, `Index Size (MB)`, `p95 Search Latency (ms)`, `Abstention AUROC`, and `Answered Accuracy`.
3. Run `scripts/generate_readme_charts.py` to save high-resolution comparison plots into `results/` and update `README.md` with actual measured numbers.

---

## 7. Verification & Acceptance Criteria

Before concluding execution, verify all of the following:
1. `pytest tests/ -v` passes with zero errors (using lightweight mock/tiny tensors for fast CI unit tests plus an integration test).
2. `python -m src.evaluation.run_benchmark --max-samples 100` completes end-to-end without OOM errors and populates `results/benchmark_summary.json` and `results/ablation_table.md`.
3. `uvicorn src.api.main:app` boots cleanly and passes `httpx` smoke tests against `/v1/ingest`, `/v1/query`, and `/v1/metrics`.
4. `README.md` contains zero placeholder numbers—all metrics reflect actual experimental outputs from `results/`.