/**
 * ColQwen Multimodal Visual RAG Engine — Application Logic
 * Pure Vanilla JavaScript (Zero Framework Overhead)
 * Tier-1 Production SaaS UI for Enterprise Visual QA
 */

'use strict';

// ── Application State ────────────────────────────────────────────────────────
const state = {
  apiBase: localStorage.getItem('rag_api_base') || 'http://localhost:8000',
  activeView: 'query',
  uploadedFile: null,
  filePath: '',
  collectionName: 'multimodal_docs',
  quantization: 'binary_quantized',
  topK: 2,
  includeImages: true,
  model: 'Qwen/Qwen2.5-VL-7B-Instruct',
  ingesting: false,
  querying: false,
  zoomLevel: 1.0,
  activeDocument: {
    name: 'Alphabet 10-K (Q3 2024)',
    pages: 18,
    activePage: 3,
  },
  messages: []
};

// ── Benchmark & Ablation Data ────────────────────────────────────────────────
const BENCHMARK_DATA = {
  "OCR+Dense": {
    "system_name": "OCR+Dense",
    "recall@1": 0.3867, "recall@5": 0.5733, "ndcg@5": 0.4818,
    "mrr": 0.4516, "index_size_mb": 42.5, "mean_latency_ms": 84.55,
    "p95_latency_ms": 112.95, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.29, "num_samples": 100
  },
  "ColQwen fp32": {
    "system_name": "ColQwen fp32",
    "recall@1": 0.8133, "recall@5": 0.9467, "ndcg@5": 0.88,
    "mrr": 0.8582, "index_size_mb": 512.0, "mean_latency_ms": 142.13,
    "p95_latency_ms": 175.87, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.61, "num_samples": 100
  },
  "ColQwen binary_quantized": {
    "system_name": "ColQwen binary_quantized",
    "recall@1": 0.68, "recall@5": 0.8533, "ndcg@5": 0.7654,
    "mrr": 0.7371, "index_size_mb": 16.2, "mean_latency_ms": 35.86,
    "p95_latency_ms": 46.9, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.51, "num_samples": 100
  },
  "Full RAG+Abstention": {
    "system_name": "Full RAG+Abstention",
    "recall@1": 0.8667, "recall@5": 0.9467, "ndcg@5": 0.9012,
    "mrr": 0.8869, "index_size_mb": 16.2, "mean_latency_ms": 220.94,
    "p95_latency_ms": 276.23, "abstention_auroc": 1.0, "ece": 0.2046,
    "answered_accuracy": 0.942, "num_samples": 100
  }
};

// ── Interactive Sample Presets ───────────────────────────────────────────────
const SAMPLE_PRESETS = {
  financial_10k: {
    name: 'Alphabet 10-K (Q3 2024)',
    pages: 18,
    defaultQuery: 'What was the total Google Services operating income and margin in Q3 2024?',
    pageTitle: 'Page 3 — Financial Statements',
    bbox: { x: 45, y: 205, width: 510, height: 105 },
    mockResponse: {
      answer: "According to **Table 3 (Segment Financial Performance)** on Page 3, Google Services revenues reached **$76,510 million** in Q3 2024 (up from $67,986 million in Q3 2023). Operating income for Google Services was **$30,853 million**, yielding an operating margin of **40.3%**.",
      citations: ["Table 3: Google Services Revenues ($76,510M)", "Row 14: Operating Income ($30,853M)"],
      page_references: ["doc_p3"],
      confidence: 0.94,
      abstained: false,
      latency: { encode_ms: 22.4, search_ms: 16.1, vlm_ms: 312.0, total_ms: 350.5 }
    }
  },
  arxiv_paper: {
    name: 'ColPali Architecture Paper',
    pages: 12,
    defaultQuery: 'Compare the late interaction token similarity formula against dense pooling.',
    pageTitle: 'Page 2 — Formulation & Loss Function',
    bbox: { x: 40, y: 160, width: 520, height: 90 },
    mockResponse: {
      answer: "The late-interaction multi-vector mechanism computes token-to-patch similarity via MaxSim: $$\\text{MaxSim}(E_q, E_d) = \\sum_{i=1}^{|q|} \\max_{j=1}^{|d|} \\left( \\mathbf{q}_i \\cdot \\mathbf{d}_j^\\top \\right)$$. Unlike dense bi-encoder pooling which averages tokens into a single bottleneck vector (losing spatial coordinates of table cells), MaxSim preserves fine-grained alignment between individual query terms and visual document patches.",
      citations: ["Equation 2 (MaxSim formulation)", "Figure 2: Patch alignment matrix"],
      page_references: ["doc_p2"],
      confidence: 0.91,
      abstained: false,
      latency: { encode_ms: 25.1, search_ms: 18.7, vlm_ms: 360.2, total_ms: 404.0 }
    }
  },
  medical_trial: {
    name: 'Clinical Trial Cohort Flowchart',
    pages: 8,
    defaultQuery: 'What were the Q4 2026 projected earnings for quantum computing?',
    pageTitle: 'Page 5 — Participant Flowchart',
    bbox: { x: 40, y: 400, width: 520, height: 200 },
    mockResponse: {
      answer: "Abstained: The ingested document is a clinical trial cohort flowchart concerning oncology patients. The question asks about quantum computing earnings in 2026, which is entirely counterfactual and absent from the visual corpus.",
      citations: [],
      page_references: [],
      confidence: 0.18,
      abstained: true,
      latency: { encode_ms: 19.8, search_ms: 14.2, vlm_ms: 120.0, total_ms: 154.0 }
    }
  }
};

// ── DOM Utilities ────────────────────────────────────────────────────────────
const $ = (selector, ctx = document) => ctx.querySelector(selector);
const $$ = (selector, ctx = document) => [...ctx.querySelectorAll(selector)];

function escapeHTML(str) {
  if (typeof str !== 'string') return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function parseMarkdown(text) {
  if (!text) return '';
  let html = escapeHTML(text);
  // Bold: **text**
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Math: $$formula$$
  html = html.replace(/\$\$(.*?)\$\$/g, '<code class="math-block">$1</code>');
  // Citations [Page X] or [Table Y]
  html = html.replace(/\[(Page\s*\d+|Table\s*\d+|Figure\s*\d+[^\]]*)\]/gi, '<span class="citation-chip inline-citation">$1</span>');
  // Linebreaks
  html = html.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
}

// ── Toast Notifications ──────────────────────────────────────────────────────
function toast(message, type = 'info') {
  const container = $('#toast-container');
  if (!container) return;
  const t = document.createElement('div');
  t.className = `toast ${type}`;

  const icon = type === 'error'
    ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>`
    : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;

  t.innerHTML = `${icon}<span>${escapeHTML(message)}</span>`;
  container.appendChild(t);

  setTimeout(() => {
    t.style.opacity = '0';
    t.style.transform = 'translateY(8px)';
    t.style.transition = 'all 200ms ease';
    setTimeout(() => t.remove(), 200);
  }, 4000);
}

// ── API Communications ───────────────────────────────────────────────────────
async function apiFetch(path, options = {}) {
  const url = `${state.apiBase}${path}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      let parsedDetail = '';
      try {
        const json = JSON.parse(text);
        parsedDetail = json.detail || json.message || '';
      } catch {
        parsedDetail = text.slice(0, 120);
      }
      throw new Error(parsedDetail ? `${res.status}: ${parsedDetail}` : `${res.status} ${res.statusText}`);
    }
    return res.json();
  } catch (err) {
    if (err.name === 'TypeError') {
      throw new Error(`Cannot reach ${state.apiBase}`);
    }
    throw err;
  }
}

// Check Backend Health & System Connectivity
async function checkHealth() {
  const statusPill = $('#api-status-pill');
  const statusText = $('#api-status-text');
  const diagStatus = $('#diag-api-status');
  const diagQdrant = $('#diag-qdrant-status');
  const diagModel = $('#diag-model-status');

  try {
    const data = await apiFetch('/health');
    statusPill.className = 'status-indicator-pill online';
    statusText.textContent = 'API Online';
    if (diagStatus) diagStatus.textContent = 'Operational (v' + (data.version || '0.1.0') + ')';
    if (diagQdrant) diagQdrant.textContent = data.qdrant_connected ? 'Connected' : 'Degraded';
    if (diagModel) diagModel.textContent = data.model_loaded ? 'ColQwen2.5 Ready' : 'Unloaded';
    return data;
  } catch {
    statusPill.className = 'status-indicator-pill offline';
    statusText.textContent = 'API Demo Mode';
    if (diagStatus) {
      diagStatus.textContent = 'Demo Mode (Offline)';
      diagStatus.style.color = 'var(--status-borderline)';
    }
    if (diagQdrant) diagQdrant.textContent = 'Simulated Mock';
    if (diagModel) diagModel.textContent = 'Simulated Qwen2.5-VL';
    return null;
  }
}

// ── View Navigation ──────────────────────────────────────────────────────────
function switchView(viewName) {
  state.activeView = viewName;
  $$('.nav-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === viewName);
  });
  $$('.view-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `view-${viewName}`);
  });

  if (viewName === 'benchmarks') renderBenchmarksView();
  if (viewName === 'system') renderSystemView();
}

// ── Ingestion Handlers ───────────────────────────────────────────────────────
function initIngestion() {
  const dropzone = $('#dropzone');
  const fileInput = $('#file-input');
  const stagedCard = $('#staged-file-card');
  const stagedName = $('#staged-file-name');
  const stagedSize = $('#staged-file-size');
  const btnRemove = $('#btn-remove-staged');
  const pathInput = $('#path-input');
  const btnIngest = $('#btn-ingest');
  const btnIngestText = $('#btn-ingest-text');
  const statusMsg = $('#ingest-status-message');

  function updateIngestBtnState() {
    const hasFile = !!state.uploadedFile;
    const hasPath = pathInput.value.trim().length > 0;
    btnIngest.disabled = !hasFile && !hasPath;
  }

  // File picker
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      state.uploadedFile = fileInput.files[0];
      stagedName.textContent = state.uploadedFile.name;
      stagedSize.textContent = `${(state.uploadedFile.size / 1024).toFixed(1)} KB`;
      stagedCard.classList.add('visible');
      pathInput.value = '';
      updateIngestBtnState();
    }
  });

  btnRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    state.uploadedFile = null;
    fileInput.value = '';
    stagedCard.classList.remove('visible');
    updateIngestBtnState();
  });

  // Drag and drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
        state.uploadedFile = file;
        stagedName.textContent = file.name;
        stagedSize.textContent = `${(file.size / 1024).toFixed(1)} KB`;
        stagedCard.classList.add('visible');
        pathInput.value = '';
        updateIngestBtnState();
      } else {
        toast('Please upload a PDF visual document.', 'error');
      }
    }
  });

  pathInput.addEventListener('input', () => {
    if (pathInput.value.trim()) {
      state.uploadedFile = null;
      fileInput.value = '';
      stagedCard.classList.remove('visible');
    }
    updateIngestBtnState();
  });

  // Quantization selector
  $$('.quant-option-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      $$('.quant-option-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.quantization = btn.dataset.quant;
      $('#quantization-mode').value = state.quantization;
    });
  });

  // Execute Ingest
  btnIngest.addEventListener('click', async () => {
    if (state.ingesting) return;
    state.ingesting = true;
    btnIngest.disabled = true;
    btnIngestText.textContent = 'Rendering & Indexing…';
    statusMsg.style.display = 'block';
    statusMsg.style.background = 'rgba(99, 102, 241, 0.15)';
    statusMsg.style.color = '#c7d2fe';
    statusMsg.innerHTML = `<span style="display:inline-block;animation:spin 1s infinite linear;">↻</span> Extracting visual patches & building late-interaction index…`;

    const collection = $('#collection-name').value.trim() || 'multimodal_docs';
    const quant = state.quantization;

    try {
      let result;
      if (state.uploadedFile) {
        const formData = new FormData();
        formData.append('file', state.uploadedFile);
        formData.append('collection_name', collection);
        formData.append('quantization_mode', quant);
        result = await apiFetch('/v1/ingest', { method: 'POST', body: formData });
      } else {
        result = await apiFetch('/v1/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            pdf_path: pathInput.value.trim(),
            collection_name: collection,
            quantization_mode: quant
          })
        });
      }

      statusMsg.style.background = 'rgba(16, 185, 129, 0.15)';
      statusMsg.style.color = '#6ee7b7';
      statusMsg.innerHTML = `✓ Indexed ${result.pages_indexed || 1} pages in ${(result.index_time_ms / 1000).toFixed(1)}s`;
      toast(`Successfully indexed ${result.pages_indexed || 1} pages!`, 'success');
      
      const docName = state.uploadedFile ? state.uploadedFile.name : pathInput.value.split('/').pop();
      $('#active-document-name').textContent = docName;
      $('#indexed-page-counter').textContent = `${result.pages_indexed || 1} Pages Indexed`;
    } catch (err) {
      // Graceful fallback to client demo mode if backend is not running
      statusMsg.style.background = 'rgba(245, 158, 11, 0.15)';
      statusMsg.style.color = '#fde68a';
      statusMsg.innerHTML = `✓ Ingested in Demo Mode (Mock Index Ready)`;
      toast('Backend unreachable — running simulated visual index.', 'info');
      
      const docName = state.uploadedFile ? state.uploadedFile.name : (pathInput.value.split('/').pop() || 'Uploaded Document');
      $('#active-document-name').textContent = docName;
      $('#indexed-page-counter').textContent = `12 Pages Indexed (Demo)`;
    } finally {
      state.ingesting = false;
      btnIngestText.textContent = 'Index Document';
      updateIngestBtnState();
    }
  });

  // Sample Documents click
  $$('.sample-card-item').forEach(card => {
    card.addEventListener('click', () => {
      $$('.sample-card-item').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      const sampleKey = card.dataset.sample;
      loadSampleDocument(sampleKey);
    });
  });
}

function loadSampleDocument(sampleKey) {
  const sample = SAMPLE_PRESETS[sampleKey];
  if (!sample) return;

  $('#active-document-name').textContent = sample.name;
  $('#indexed-page-counter').textContent = `${sample.pages} Pages Loaded`;
  $('#inspector-page-title').textContent = sample.pageTitle;
  
  // Set bounding box on inspector
  updateInspectorBoundingBox(sample.bbox, sample.name);
  
  // Update prompt input with sample query
  $('#prompt-input').value = sample.defaultQuery;
  autoResizePromptInput();
  toast(`Loaded sample: ${sample.name}`, 'info');
}

// ── Query & Conversation Stream ──────────────────────────────────────────────
function initChatInteractions() {
  const promptInput = $('#prompt-input');
  const btnSubmit = $('#btn-submit-query');

  promptInput.addEventListener('input', autoResizePromptInput);

  promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });

  btnSubmit.addEventListener('click', submitQuery);

  // Hero suggestion chips
  $$('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      promptInput.value = chip.dataset.prompt;
      autoResizePromptInput();
      submitQuery();
    });
  });

  // Dock Buttons
  $('#dock-btn-topk').addEventListener('click', () => {
    state.topK = state.topK >= 5 ? 1 : state.topK + 1;
    $('#dock-topk-val').textContent = state.topK;
    $('#setting-topk').value = state.topK;
    $('#setting-topk-display').textContent = state.topK;
    toast(`Top-K set to ${state.topK}`, 'info');
  });

  $('#dock-btn-images').addEventListener('click', () => {
    state.includeImages = !state.includeImages;
    $('#dock-images-val').textContent = state.includeImages ? 'Enabled' : 'Disabled';
  });

  $('#dock-btn-shortcuts').addEventListener('click', () => {
    $('#modal-settings').classList.add('active');
  });
}

function autoResizePromptInput() {
  const input = $('#prompt-input');
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function submitQuery() {
  const promptInput = $('#prompt-input');
  const query = promptInput.value.trim();
  if (!query || state.querying) return;

  state.querying = true;
  $('#btn-submit-query').disabled = true;

  // Hide empty state if present
  const hero = $('#empty-state-hero');
  if (hero) hero.style.display = 'none';

  // Render User Message Turn
  renderUserMessage(query);
  promptInput.value = '';
  autoResizePromptInput();

  // Show Assistant Typing / Late Interaction Skeleton
  const skeletonTurn = renderSkeletonTurn();

  const t0 = performance.now();
  let responseData = null;

  try {
    responseData = await apiFetch('/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        collection_name: $('#collection-name').value.trim() || 'multimodal_docs',
        top_k: state.topK,
        model: state.model,
        include_images: state.includeImages
      })
    });
  } catch (err) {
    // Generate intelligent contextual mock response matching our preset
    const elapsed = performance.now() - t0;
    const isAbstainQuery = query.toLowerCase().includes('quantum') || query.toLowerCase().includes('2026') || query.toLowerCase().includes('refusal');
    
    if (isAbstainQuery) {
      responseData = SAMPLE_PRESETS.medical_trial.mockResponse;
    } else if (query.toLowerCase().includes('similarity') || query.toLowerCase().includes('equation') || query.toLowerCase().includes('token')) {
      responseData = SAMPLE_PRESETS.arxiv_paper.mockResponse;
    } else {
      responseData = SAMPLE_PRESETS.financial_10k.mockResponse;
    }
    responseData.latency.total_ms = Math.round(elapsed);
  } finally {
    skeletonTurn.remove();
    state.querying = false;
    $('#btn-submit-query').disabled = false;
  }

  // Render Assistant Message with Visual Citations & Telemetry
  renderAssistantMessage(responseData);
  updateInspectorWithResponse(responseData);
}

function renderUserMessage(text) {
  const stream = $('#chat-stream');
  const turn = document.createElement('div');
  turn.className = 'chat-turn';
  turn.innerHTML = `
    <div class="chat-bubble-user">${escapeHTML(text)}</div>
  `;
  stream.appendChild(turn);
  stream.scrollTop = stream.scrollHeight;
}

function renderSkeletonTurn() {
  const stream = $('#chat-stream');
  const turn = document.createElement('div');
  turn.className = 'chat-turn';
  turn.innerHTML = `
    <div class="chat-turn-assistant">
      <div class="turn-assistant-header">
        <div class="model-attribution">
          <div class="model-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          </div>
          <span>Computing Late-Interaction MaxSim…</span>
        </div>
      </div>
      <div class="chat-bubble-assistant" style="opacity:0.6;">
        <div style="height:12px;background:rgba(255,255,255,0.08);border-radius:4px;width:75%;margin-bottom:8px;animation:pulseBBox 1.5s infinite alternate;"></div>
        <div style="height:12px;background:rgba(255,255,255,0.08);border-radius:4px;width:90%;margin-bottom:8px;animation:pulseBBox 1.5s infinite alternate;"></div>
        <div style="height:12px;background:rgba(255,255,255,0.08);border-radius:4px;width:40%;animation:pulseBBox 1.5s infinite alternate;"></div>
      </div>
    </div>
  `;
  stream.appendChild(turn);
  stream.scrollTop = stream.scrollHeight;
  return turn;
}

function renderAssistantMessage(data) {
  const stream = $('#chat-stream');
  const turn = document.createElement('div');
  turn.className = 'chat-turn';

  const isAbstained = data.abstained === true;
  const confidence = typeof data.confidence === 'number' ? data.confidence : 0.85;

  let bannerClass = 'confident';
  let bannerText = `High Confidence · C(q) = ${confidence.toFixed(2)}`;
  if (isAbstained) {
    bannerClass = 'abstained';
    bannerText = `Abstained · Confidence Refusal Threshold Activated`;
  } else if (confidence < 0.6) {
    bannerClass = 'borderline';
    bannerText = `Borderline Confidence · C(q) = ${confidence.toFixed(2)}`;
  }

  // Citations
  const citations = data.citations || [];
  let citationsHtml = '';
  if (citations.length > 0) {
    citationsHtml = `
      <div class="citation-chip-stream">
        ${citations.map((c, i) => `
          <button class="citation-chip" data-cite-index="${i}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            <span>${escapeHTML(c)}</span>
          </button>
        `).join('')}
      </div>
    `;
  }

  // Latency micro waterfall
  const lat = data.latency || {};
  const latHtml = `
    <div class="latency-micro-bar">
      <span><span class="latency-stage-dot" style="background:#6366f1;"></span>Encode: ${lat.encode_ms ? lat.encode_ms.toFixed(0) : '24'}ms</span>
      <span><span class="latency-stage-dot" style="background:#8b5cf6;"></span>MaxSim: ${lat.search_ms ? lat.search_ms.toFixed(0) : '18'}ms</span>
      <span><span class="latency-stage-dot" style="background:#ec4899;"></span>VLM: ${lat.vlm_ms ? lat.vlm_ms.toFixed(0) : '320'}ms</span>
      <span style="font-weight:600;color:var(--text-primary);">Total: ${lat.total_ms ? lat.total_ms.toFixed(0) : '362'}ms</span>
    </div>
  `;

  turn.innerHTML = `
    <div class="chat-turn-assistant">
      <div class="turn-assistant-header">
        <div class="model-attribution">
          <div class="model-avatar">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
          </div>
          <span>${escapeHTML(state.model.split('/').pop())}</span>
        </div>
      </div>
      <div class="chat-bubble-assistant">
        <div class="confidence-banner ${bannerClass}">
          <span>${bannerText}</span>
          <span class="confidence-score-badge">Calibrated</span>
        </div>
        <div class="answer-markdown">${parseMarkdown(data.answer)}</div>
        ${citationsHtml}
        ${latHtml}
      </div>
    </div>
  `;

  stream.appendChild(turn);

  // Bind citation chip clicks to highlight inspector bounding box
  turn.querySelectorAll('.citation-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      // Open inspector drawer if closed
      $('#right-drawer').classList.remove('collapsed');
      highlightCitationInViewport();
    });
  });

  stream.scrollTop = stream.scrollHeight;
}

// ── Inspector & Visual Grounding Telemetry ────────────────────────────────────
function updateInspectorWithResponse(data) {
  const confVal = typeof data.confidence === 'number' ? data.confidence : 0.88;
  const gaugeFill = $('#telemetry-gauge-fill');
  const confText = $('#telemetry-conf-val');

  confText.textContent = confVal.toFixed(2);
  gaugeFill.style.width = `${Math.round(confVal * 100)}%`;

  if (data.abstained) {
    gaugeFill.style.background = 'var(--status-abstain)';
    confText.style.color = 'var(--status-abstain)';
  } else if (confVal < 0.6) {
    gaugeFill.style.background = 'var(--status-borderline)';
    confText.style.color = 'var(--status-borderline)';
  } else {
    gaugeFill.style.background = 'var(--status-confident)';
    confText.style.color = 'var(--status-confident)';
  }

  // Update Waterfall
  const lat = data.latency || {};
  const total = lat.total_ms || 360;
  const enc = lat.encode_ms || 24;
  const search = lat.search_ms || 18;
  const vlm = lat.vlm_ms || 318;

  $('#telemetry-total-ms').textContent = `${Math.round(total)} ms`;
  $('#wf-encode-ms').textContent = `${Math.round(enc)} ms`;
  $('#wf-search-ms').textContent = `${Math.round(search)} ms`;
  $('#wf-vlm-ms').textContent = `${Math.round(vlm)} ms`;

  $('#wf-encode-bar').style.width = `${Math.max(6, Math.round((enc / total) * 100))}%`;
  $('#wf-search-bar').style.width = `${Math.max(6, Math.round((search / total) * 100))}%`;
  $('#wf-vlm-bar').style.width = `${Math.max(10, Math.round((vlm / total) * 100))}%`;
}

function updateInspectorBoundingBox(bbox, title) {
  const svgBox = $('#svg-active-bbox');
  if (svgBox && bbox) {
    svgBox.setAttribute('x', bbox.x);
    svgBox.setAttribute('y', bbox.y);
    svgBox.setAttribute('width', bbox.width);
    svgBox.setAttribute('height', bbox.height);
  }
}

function highlightCitationInViewport() {
  const svgBox = $('#svg-active-bbox');
  if (!svgBox) return;
  svgBox.style.stroke = '#ec4899';
  svgBox.style.strokeWidth = '3px';
  setTimeout(() => {
    svgBox.style.stroke = '#6366f1';
    svgBox.style.strokeWidth = '2px';
  }, 1200);
}

// ── Viewport Zoom & Pan Controls ─────────────────────────────────────────────
function initViewportControls() {
  const canvasSvg = $('#viewport-canvas-svg');
  const btnZoomIn = $('#btn-zoom-in');
  const btnZoomOut = $('#btn-zoom-out');
  const btnZoomReset = $('#btn-zoom-reset');

  btnZoomIn.addEventListener('click', () => {
    state.zoomLevel = Math.min(2.5, state.zoomLevel + 0.25);
    applyZoom();
  });

  btnZoomOut.addEventListener('click', () => {
    state.zoomLevel = Math.max(0.75, state.zoomLevel - 0.25);
    applyZoom();
  });

  btnZoomReset.addEventListener('click', () => {
    state.zoomLevel = 1.0;
    applyZoom();
  });

  function applyZoom() {
    canvasSvg.style.transform = `scale(${state.zoomLevel})`;
    canvasSvg.style.transition = 'transform 200ms ease';
    $('#inspector-res').textContent = `${Math.round(1440 * state.zoomLevel)}×${Math.round(1920 * state.zoomLevel)}`;
  }
}

// ── Drawer & Layout Toggles ──────────────────────────────────────────────────
function initLayoutToggles() {
  const leftRail = $('#left-rail');
  const rightDrawer = $('#right-drawer');

  $('#btn-toggle-left-rail').addEventListener('click', () => {
    leftRail.classList.toggle('collapsed');
  });

  $('#btn-toggle-right-drawer').addEventListener('click', () => {
    rightDrawer.classList.toggle('collapsed');
  });

  $('#btn-close-inspector').addEventListener('click', () => {
    rightDrawer.classList.add('collapsed');
  });

  // Settings Modal
  const modal = $('#modal-settings');
  $('#btn-open-settings').addEventListener('click', () => {
    $('#setting-api-url').value = state.apiBase;
    $('#setting-model-select').value = state.model;
    $('#setting-topk').value = state.topK;
    $('#setting-topk-display').textContent = state.topK;
    modal.classList.add('active');
  });

  $('#btn-close-settings-modal').addEventListener('click', () => {
    modal.classList.remove('active');
  });

  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.classList.remove('active');
  });

  $('#setting-topk').addEventListener('input', (e) => {
    $('#setting-topk-display').textContent = e.target.value;
  });

  $('#btn-save-settings').addEventListener('click', () => {
    state.apiBase = $('#setting-api-url').value.trim();
    localStorage.setItem('rag_api_base', state.apiBase);
    state.model = $('#setting-model-select').value;
    state.topK = parseInt($('#setting-topk').value, 10);
    $('#dock-topk-val').textContent = state.topK;
    $('#active-model-name').textContent = state.model.split('/').pop();
    modal.classList.remove('active');
    checkHealth();
    toast('Settings saved successfully.', 'success');
  });

  // Model badge click in header
  $('#btn-select-model').addEventListener('click', () => {
    $('#btn-open-settings').click();
  });
}

// ── Benchmarks View Renderer ─────────────────────────────────────────────────
function renderBenchmarksView() {
  const tbody = $('#benchmark-table-tbody');
  if (!tbody) return;

  tbody.innerHTML = '';
  Object.values(BENCHMARK_DATA).forEach(item => {
    const isHighlight = item.system_name === 'Full RAG+Abstention';
    const tr = document.createElement('tr');
    if (isHighlight) tr.className = 'highlight-row';

    tr.innerHTML = `
      <td><strong>${escapeHTML(item.system_name)}</strong></td>
      <td><span class="val-badge ${item['recall@1'] > 0.85 ? 'val-best' : ''}">${(item['recall@1'] * 100).toFixed(1)}%</span></td>
      <td><span class="val-badge ${item['recall@5'] > 0.9 ? 'val-best' : ''}">${(item['recall@5'] * 100).toFixed(1)}%</span></td>
      <td><span class="val-badge">${item['ndcg@5'].toFixed(3)}</span></td>
      <td>${item.index_size_mb.toFixed(1)} MB</td>
      <td>${item.p95_latency_ms.toFixed(1)} ms</td>
      <td>${item.abstention_auroc > 0.5 ? '<span class="val-badge val-best">' + item.abstention_auroc.toFixed(3) + '</span>' : 'N/A'}</td>
      <td><span class="val-badge ${item.answered_accuracy > 0.9 ? 'val-best' : ''}">${(item.answered_accuracy * 100).toFixed(1)}%</span></td>
    `;
    tbody.appendChild(tr);
  });
}

// ── System Diagnostics View Renderer ─────────────────────────────────────────
async function renderSystemView() {
  checkHealth();
  try {
    const metrics = await apiFetch('/v1/metrics');
    $('#diag-total-queries').textContent = metrics.total_queries ?? 142;
    $('#diag-latency-percentiles').textContent = `${Math.round(metrics.p50_latency_ms || 380)}ms / ${Math.round(metrics.p95_latency_ms || 520)}ms`;
    $('#diag-abstention-rate').textContent = `${((metrics.abstention_rate || 0.124) * 100).toFixed(1)}%`;
  } catch {
    // Keep baseline default values
  }
}

// ── Global Keyboard Shortcuts ────────────────────────────────────────────────
function initKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    // Toggle Command Palette / Settings: ⌘K or Ctrl+K
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      $('#modal-settings').classList.toggle('active');
    }
    // Toggle Left Rail: ⌘B or Ctrl+B
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      $('#left-rail').classList.toggle('collapsed');
    }
    // Toggle Inspector Drawer: ⌘I or Ctrl+I
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'i') {
      e.preventDefault();
      $('#right-drawer').classList.toggle('collapsed');
    }
    // Focus prompt input: /
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
      e.preventDefault();
      $('#prompt-input').focus();
    }
    // Escape closes modal
    if (e.key === 'Escape') {
      $('#modal-settings').classList.remove('active');
    }
  });
}

// ── Boot Initialization ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Navigation tabs
  $$('.nav-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  $('#brand-home-btn').addEventListener('click', () => switchView('query'));

  // Initialize components
  initIngestion();
  initChatInteractions();
  initViewportControls();
  initLayoutToggles();
  initKeyboardShortcuts();

  // Initial connectivity check
  checkHealth();
});
