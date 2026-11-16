// RAG Engine — Application Logic
// Vanilla JS, no frameworks, no jQuery

'use strict';

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  apiBase: localStorage.getItem('rag_api_base') || 'http://localhost:8000',
  activeView: 'query',
  uploadedFile: null,
  filePath: '',
  collectionName: 'multimodal_docs',
  quantization: 'binary_quantized',
  topK: 5,
  includeImages: false,
  ingesting: false,
  querying: false,
};

// ── Benchmark data (from results/benchmark_summary.json) ──────────────────
const BENCHMARK_DATA = {
  "OCR+Dense": {
    "recall@1": 0.3867, "recall@5": 0.5733, "ndcg@5": 0.4818,
    "mrr": 0.4516, "index_size_mb": 42.5, "mean_latency_ms": 84.55,
    "p95_latency_ms": 112.95, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.29, "num_samples": 100
  },
  "ColQwen fp32": {
    "recall@1": 0.8133, "recall@5": 0.9467, "ndcg@5": 0.88,
    "mrr": 0.8582, "index_size_mb": 512.0, "mean_latency_ms": 142.13,
    "p95_latency_ms": 175.87, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.61, "num_samples": 100
  },
  "ColQwen binary_quantized": {
    "recall@1": 0.68, "recall@5": 0.8533, "ndcg@5": 0.7654,
    "mrr": 0.7371, "index_size_mb": 16.2, "mean_latency_ms": 35.86,
    "p95_latency_ms": 46.9, "abstention_auroc": 0.5, "ece": 0.0,
    "answered_accuracy": 0.51, "num_samples": 100
  },
  "Full RAG+Abstention": {
    "recall@1": 0.8667, "recall@5": 0.9467, "ndcg@5": 0.9012,
    "mrr": 0.8869, "index_size_mb": 16.2, "mean_latency_ms": 220.94,
    "p95_latency_ms": 276.23, "abstention_auroc": 1.0, "ece": 0.2046,
    "answered_accuracy": 0.942, "num_samples": 100
  }
};

const SYSTEM_NAMES = Object.keys(BENCHMARK_DATA);

// ── DOM helpers ───────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'className') e.className = v;
    else if (k === 'textContent') e.textContent = v;
    else if (k === 'innerHTML') e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child == null) continue;
    e.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return e;
}

// ── Toast notifications ───────────────────────────────────────────────────
function toast(msg, type = 'error') {
  const container = $('#toast-container');
  if (!container) return;
  const t = el('div', { className: `toast ${type}` });

  const icon = type === 'error'
    ? `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 4v3M7 9.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`
    : `<svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M4.5 7l2 2 3-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  t.innerHTML = `${icon}<span>${escapeHTML(String(msg))}</span>`;
  container.appendChild(t);

  setTimeout(() => {
    t.classList.add('dismissing');
    setTimeout(() => t.remove(), 200);
  }, 4000);
}

// ── API helpers ───────────────────────────────────────────────────────────
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
      throw new Error(`Cannot reach ${state.apiBase} — is the server running?`);
    }
    throw err;
  }
}

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  const pill = $('#api-status');
  if (!pill) return null;
  try {
    const data = await apiFetch('/health');
    pill.className = 'status-pill online';
    pill.innerHTML = `<span class="status-dot"></span><span>API Online</span>`;
    return data;
  } catch {
    pill.className = 'status-pill offline';
    pill.innerHTML = `<span class="status-dot"></span><span>API Offline</span>`;
    return null;
  }
}

// ── Tab navigation ────────────────────────────────────────────────────────
function switchView(viewName) {
  state.activeView = viewName;

  $$('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.view === viewName);
  });
  $$('.view').forEach(v => {
    v.classList.toggle('active', v.id === `view-${viewName}`);
  });

  if (viewName === 'benchmarks') renderBenchmarks();
  if (viewName === 'system') initSystemView();
}

// ── Upload zone ───────────────────────────────────────────────────────────
function initUploadZone() {
  const zone = $('#upload-zone');
  const fileInput = $('#file-input');
  const pathInput = $('#path-input');
  const ingestBtn = $('#ingest-btn');

  function updateZoneState() {
    const hasFile = !!state.uploadedFile;
    const hasPath = pathInput.value.trim().length > 0;
    zone.classList.toggle('has-file', hasFile);
    ingestBtn.disabled = !hasFile && !hasPath;

    if (hasFile) {
      zone.querySelector('.upload-label').innerHTML = `<strong>${escapeHTML(state.uploadedFile.name)}</strong>`;
      zone.querySelector('.upload-hint').textContent = `${(state.uploadedFile.size / 1024).toFixed(0)} KB · Ready to index`;
    } else {
      zone.querySelector('.upload-label').innerHTML = `Drop PDF or <strong>click to browse</strong>`;
      zone.querySelector('.upload-hint').textContent = 'PDF files only';
    }
  }

  fileInput.addEventListener('change', () => {
    state.uploadedFile = fileInput.files[0] || null;
    state.filePath = '';
    pathInput.value = '';
    updateZoneState();
  });

  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && (f.type === 'application/pdf' || f.name.toLowerCase().endsWith('.pdf'))) {
      state.uploadedFile = f;
      state.filePath = '';
      pathInput.value = '';
      updateZoneState();
    } else if (f) {
      toast('Only PDF files are supported.');
    }
  });

  pathInput.addEventListener('input', () => {
    state.filePath = pathInput.value.trim();
    if (state.filePath) {
      state.uploadedFile = null;
      fileInput.value = '';
      zone.classList.remove('has-file');
      zone.querySelector('.upload-label').innerHTML = `Drop PDF or <strong>click to browse</strong>`;
      zone.querySelector('.upload-hint').textContent = 'PDF files only';
    }
    ingestBtn.disabled = !state.filePath && !state.uploadedFile;
  });

  updateZoneState();
}

// ── Collapsible ───────────────────────────────────────────────────────────
function initCollapsibles() {
  $$('.collapsible-trigger').forEach(trigger => {
    const bodyId = trigger.dataset.target;
    const body = document.getElementById(bodyId);
    if (!body) return;

    body.style.maxHeight = '0px';

    trigger.addEventListener('click', () => {
      const isOpen = trigger.classList.contains('open');
      trigger.classList.toggle('open', !isOpen);
      body.style.maxHeight = isOpen ? '0px' : body.scrollHeight + 'px';
    });
  });
}

// ── Slider ───────────────────────────────────────────────────────────────
function initSlider() {
  const slider = $('#topk-slider');
  const display = $('#topk-value');
  slider.value = state.topK;
  display.textContent = state.topK;
  slider.addEventListener('input', () => {
    state.topK = parseInt(slider.value, 10);
    display.textContent = state.topK;
  });
}

// ── Toggle ────────────────────────────────────────────────────────────────
function initToggle() {
  const toggle = $('#include-images-toggle');
  toggle.addEventListener('change', () => {
    state.includeImages = toggle.checked;
  });
}

// ── Ingest ────────────────────────────────────────────────────────────────
async function ingest() {
  if (state.ingesting) return;
  const statusStrip = $('#ingest-status');
  const ingestBtn = $('#ingest-btn');

  const collectionName = $('#collection-name').value.trim() || 'multimodal_docs';
  const quantization = $('#quantization-mode').value;

  state.ingesting = true;
  ingestBtn.disabled = true;
  statusStrip.className = 'ingest-status visible';
  statusStrip.innerHTML = `<span class="spinner"></span><span>Indexing document pages…</span>`;

  const t0 = Date.now();

  try {
    let body, headers;

    if (state.uploadedFile) {
      const fd = new FormData();
      fd.append('file', state.uploadedFile);
      fd.append('collection_name', collectionName);
      fd.append('quantization_mode', quantization);
      fd.append('quantization', quantization);
      body = fd;
    } else {
      body = JSON.stringify({
        pdf_path: state.filePath,
        file_path: state.filePath,
        collection_name: collectionName,
        quantization_mode: quantization,
      });
      headers = { 'Content-Type': 'application/json' };
    }

    const res = await apiFetch('/v1/ingest', { method: 'POST', body, headers });
    const elapsed = res.index_time_ms ? (res.index_time_ms / 1000).toFixed(1) : ((Date.now() - t0) / 1000).toFixed(1);
    const pages = res.pages_indexed != null ? res.pages_indexed : (res.pages || res.num_pages || '?');

    statusStrip.className = 'ingest-status visible success';
    statusStrip.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6l3 3 5-5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg><span>✓ ${pages} pages indexed in ${elapsed}s</span>`;
  } catch (err) {
    statusStrip.className = 'ingest-status visible error';
    statusStrip.innerHTML = `<svg width="12" height="12" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="5" stroke="currentColor" stroke-width="1.5"/><path d="M6 3.5v3M6 8.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg><span>${escapeHTML(err.message)}</span>`;
    toast(err.message);
  } finally {
    state.ingesting = false;
    ingestBtn.disabled = false;
  }
}

// ── Query ─────────────────────────────────────────────────────────────────
function showSkeleton() {
  const area = $('#result-area');
  area.innerHTML = `
    <div class="skeleton-block">
      <div class="skeleton skeleton-title"></div>
      <div class="skeleton skeleton-line"></div>
      <div class="skeleton skeleton-line w80"></div>
      <div class="skeleton skeleton-line w60"></div>
    </div>
    <div class="skeleton" style="height:28px;border-radius:var(--radius);"></div>
  `;
}

function showResult(data) {
  const area = $('#result-area');
  area.innerHTML = '';

  const abstained = data.abstained === true;
  const confidence = typeof data.confidence === 'number' ? data.confidence : null;

  if (abstained) {
    // Abstained state
    const abstainCard = el('div', { className: 'abstain-card' });
    let confBadge = '';
    if (confidence !== null) {
      confBadge = `<span class="confidence-pill confidence-low" style="margin-left:auto;">${confidence.toFixed(2)} confidence</span>`;
    }

    abstainCard.innerHTML = `
      <div class="abstain-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.5"/>
          <path d="M12 7v5M12 15v2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
        </svg>
      </div>
      <div style="flex:1;">
        <div style="display:flex;align-items:center;margin-bottom:3px;">
          <span style="font-size:13px;font-weight:600;color:var(--text);">Abstained</span>
          ${confBadge}
        </div>
        <div class="abstain-text">Insufficient visual evidence to answer confidently.</div>
      </div>
    `;
    area.appendChild(abstainCard);
  } else {
    // Answer card
    const card = el('div', { className: 'answer-card' });

    let confidenceHTML = '';
    if (confidence !== null) {
      const cls = confidence >= 0.6 ? 'confidence-high' : confidence >= 0.3 ? 'confidence-mid' : 'confidence-low';
      confidenceHTML = `<span class="confidence-pill ${cls}">${confidence.toFixed(2)} confidence</span>`;
    }

    const rawCitations = data.citations || [];
    const pageRefs = data.page_references || [];
    const allCitations = [...rawCitations, ...pageRefs.filter(p => !rawCitations.includes(p))];

    const citationsHTML = allCitations.length > 0
      ? `<div class="citations">${allCitations.map(c => `<span class="citation-tag">${escapeHTML(typeof c === 'object' ? (c.page || c.id || JSON.stringify(c)) : String(c))}</span>`).join('')}</div>`
      : '';

    card.innerHTML = `
      <div class="answer-header">
        <span class="answer-label">Answer</span>
        ${confidenceHTML}
      </div>
      <div class="answer-body">
        <div class="answer-text">${escapeHTML(data.answer || '')}</div>
        ${citationsHTML}
      </div>
    `;
    area.appendChild(card);
  }

  // Latency strip
  const lat = data.latency || data.latency_ms || {};
  if (lat && Object.keys(lat).length > 0) {
    const strip = el('div', { className: 'latency-strip' });
    const encodeMs = lat.encode_ms ?? lat.encode;
    const searchMs = lat.search_ms ?? lat.search;
    const vlmMs = lat.vlm_ms ?? lat.vlm;
    const totalMs = lat.total_ms ?? lat.total;

    const parts = [
      encodeMs != null ? { label: 'encode', val: encodeMs } : null,
      searchMs != null ? { label: 'search', val: searchMs } : null,
      vlmMs != null ? { label: 'vlm', val: vlmMs } : null,
      totalMs != null ? { label: 'total', val: totalMs } : null,
    ].filter(Boolean);

    if (parts.length === 0 && typeof lat === 'number') {
      strip.innerHTML = `<span class="latency-label">total</span>&nbsp;<span class="latency-value">${lat}ms</span>`;
    } else {
      strip.innerHTML = parts.map((p, i) =>
        `<span class="latency-item"><span class="latency-label">${p.label}</span>&nbsp;<span class="latency-value">${typeof p.val === 'number' ? p.val.toFixed(0) + 'ms' : p.val}</span></span>${i < parts.length - 1 ? '<span class="latency-sep">·</span>' : ''}`
      ).join('');
    }
    area.appendChild(strip);
  }

  // Retrieved page images strip
  if (state.includeImages && data.images && data.images.length > 0) {
    const strip = el('div', { className: 'image-strip' });
    data.images.forEach((imgData, idx) => {
      const img = el('img', {});
      let src = '';
      if (typeof imgData === 'string') {
        src = imgData.startsWith('data:') ? imgData : `data:image/png;base64,${imgData}`;
      } else if (imgData && imgData.url) {
        src = imgData.url;
      } else if (imgData && imgData.base64) {
        src = imgData.base64.startsWith('data:') ? imgData.base64 : `data:image/png;base64,${imgData.base64}`;
      }
      img.src = src;
      img.alt = `Retrieved page ${idx + 1}`;
      img.title = `Retrieved page ${idx + 1}`;
      strip.appendChild(img);
    });
    area.appendChild(strip);
  }
}

function escapeHTML(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function query() {
  if (state.querying) return;
  const q = $('#query-textarea').value.trim();
  if (!q) { toast('Please enter a question.', 'error'); return; }

  const collectionName = $('#collection-name').value.trim() || 'multimodal_docs';

  state.querying = true;
  const askBtn = $('#ask-btn');
  askBtn.disabled = true;
  askBtn.innerHTML = `<span class="spinner" style="border-color:#fff;border-top-color:transparent;"></span> Asking…`;

  showSkeleton();

  try {
    const payload = {
      query: q,
      collection_name: collectionName,
      top_k: state.topK,
      include_images: state.includeImages,
    };
    const data = await apiFetch('/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    showResult(data);
  } catch (err) {
    $('#result-area').innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--danger); border: 1px dashed rgba(239,68,68,0.3); border-radius: var(--radius-lg); background: var(--danger-dim);">
        <div style="font-size: 13px; font-weight: 500;">Query Failed</div>
        <div style="font-size: 12px; margin-top: 4px; color: var(--text-muted);">${escapeHTML(err.message)}</div>
      </div>
    `;
    toast(err.message);
  } finally {
    state.querying = false;
    askBtn.disabled = false;
    askBtn.innerHTML = 'Ask';
  }
}

// ── Auto-resize textarea ──────────────────────────────────────────────────
function initTextarea() {
  const ta = $('#query-textarea');
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.max(100, ta.scrollHeight) + 'px';
  });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      query();
    }
  });
}

// ── Benchmarks ────────────────────────────────────────────────────────────
function renderBenchmarks() {
  renderMetricCard('chart-recall1', 'Recall@1', 'recall@1', false);
  renderMetricCard('chart-recall5', 'Recall@5', 'recall@5', false);
  renderMetricCard('chart-ndcg5', 'nDCG@5', 'ndcg@5', false);
  renderMetricCard('chart-latency', 'p95 Latency (ms)', 'p95_latency_ms', true);
  renderComparisonTable();
}

function renderMetricCard(containerId, title, metricKey, isLatency) {
  const container = $(`#${containerId}`);
  if (!container) return;

  const values = SYSTEM_NAMES.map(name => ({
    name,
    val: BENCHMARK_DATA[name][metricKey]
  }));

  const vals = values.map(v => v.val);
  const maxVal = Math.max(...vals);
  const bestVal = isLatency ? Math.min(...vals) : maxVal;

  container.innerHTML = '';
  const header = el('div', { className: 'metric-card-header' });
  header.innerHTML = `<div class="metric-card-title">${title}</div>`;
  container.appendChild(header);

  const body = el('div', { className: 'metric-card-body' });

  values.forEach(({ name, val }) => {
    const isBest = val === bestVal;
    const pct = isLatency
      ? (val / maxVal) * 100
      : (val / maxVal) * 100;

    const fillClass = isLatency ? 'bar-fill bar-latency' : (isBest ? 'bar-fill bar-top' : 'bar-fill');

    const row = el('div', { className: 'bar-row' });
    const displayVal = isLatency ? `${val.toFixed(0)} ms` : val.toFixed(4);

    row.innerHTML = `
      <div class="bar-label-row">
        <span class="bar-system-name">${name}</span>
        <span class="bar-value">${displayVal}</span>
      </div>
      <div class="bar-track">
        <div class="${fillClass}" style="width:0%" data-pct="${pct.toFixed(1)}"></div>
      </div>
    `;
    body.appendChild(row);
  });

  container.appendChild(body);

  // Animate bars in smoothly
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      $$('.bar-fill[data-pct]', container).forEach(bar => {
        bar.style.width = bar.dataset.pct + '%';
      });
    });
  });
}

function renderComparisonTable() {
  const table = $('#comparison-table');
  if (!table) return;

  const metrics = [
    { key: 'recall@1', label: 'Recall@1', fmt: v => v.toFixed(4), lowerBetter: false },
    { key: 'recall@5', label: 'Recall@5', fmt: v => v.toFixed(4), lowerBetter: false },
    { key: 'ndcg@5', label: 'nDCG@5', fmt: v => v.toFixed(4), lowerBetter: false },
    { key: 'mrr', label: 'MRR', fmt: v => v.toFixed(4), lowerBetter: false },
    { key: 'p95_latency_ms', label: 'p95 Latency', fmt: v => v.toFixed(1) + ' ms', lowerBetter: true },
    { key: 'index_size_mb', label: 'Index Size', fmt: v => v.toFixed(1) + ' MB', lowerBetter: true },
    { key: 'answered_accuracy', label: 'Accuracy', fmt: v => (v * 100).toFixed(1) + '%', lowerBetter: false },
  ];

  const thead = el('thead');
  const headerRow = el('tr');
  headerRow.innerHTML = `<th>System</th>${metrics.map(m => `<th>${m.label}</th>`).join('')}`;
  thead.appendChild(headerRow);

  const tbody = el('tbody');
  SYSTEM_NAMES.forEach(name => {
    const row = el('tr');
    let cells = `<td class="system-name-cell">${name}</td>`;
    metrics.forEach(m => {
      const val = BENCHMARK_DATA[name][m.key];
      const allVals = SYSTEM_NAMES.map(n => BENCHMARK_DATA[n][m.key]);
      const best = m.lowerBetter ? Math.min(...allVals) : Math.max(...allVals);
      const isBest = val === best;
      cells += `<td class="${isBest ? 'best-value' : ''}">${m.fmt(val)}</td>`;
    });
    row.innerHTML = cells;
    tbody.appendChild(row);
  });

  table.innerHTML = '';
  table.appendChild(thead);
  table.appendChild(tbody);
}

// ── System View ───────────────────────────────────────────────────────────
async function initSystemView() {
  loadHealthStatus();
  loadMetrics();
}

async function loadHealthStatus() {
  const container = $('#health-rows');
  if (!container) return;

  const services = [
    { key: 'api', label: 'API Server' },
    { key: 'qdrant', label: 'Qdrant Multi-Vector DB' },
    { key: 'visual_encoder', label: 'Visual Encoder (ColQwen2.5)' },
    { key: 'vlm', label: 'VLM Reader (Qwen2.5-VL)' },
  ];

  // Skeleton loading state
  container.innerHTML = services.map(s => `
    <div class="health-row">
      <span class="health-row-name">${s.label}</span>
      <div class="health-status">
        <span class="health-dot loading"></span>
        <span class="text-dim text-xs">Connecting…</span>
      </div>
    </div>
  `).join('');

  try {
    const data = await apiFetch('/health');
    const isApiHealthy = data.status === 'healthy';
    const isQdrantHealthy = data.qdrant_connected === true;
    const isModelLoaded = data.model_loaded === true;

    const statuses = {
      api: isApiHealthy,
      qdrant: isQdrantHealthy,
      visual_encoder: isModelLoaded,
      vlm: isModelLoaded,
    };

    container.innerHTML = services.map(s => {
      const ok = statuses[s.key];
      const dotClass = ok ? 'healthy' : 'unhealthy';
      const label = ok ? 'Operational' : 'Degraded';
      return `
        <div class="health-row">
          <span class="health-row-name">${s.label}</span>
          <div class="health-status">
            <span class="health-dot ${dotClass}"></span>
            <span style="color:${ok ? 'var(--success)' : 'var(--danger)'};">${label}</span>
          </div>
        </div>
      `;
    }).join('');
  } catch {
    container.innerHTML = services.map(s => `
      <div class="health-row">
        <span class="health-row-name">${s.label}</span>
        <div class="health-status">
          <span class="health-dot unhealthy"></span>
          <span style="color:var(--text-dim)">Offline</span>
        </div>
      </div>
    `).join('');
  }
}

async function loadMetrics() {
  const container = $('#metrics-rows');
  if (!container) return;

  const metricDefs = [
    { key: 'total_queries', label: 'Total Queries', fmt: v => v?.toLocaleString() ?? '0' },
    { key: 'p50_latency_ms', label: 'p50 Latency', fmt: v => v != null ? v.toFixed(1) + ' ms' : '—' },
    { key: 'p95_latency_ms', label: 'p95 Latency', fmt: v => v != null ? v.toFixed(1) + ' ms' : '—' },
    { key: 'abstention_rate', label: 'Abstention Rate', fmt: v => v != null ? (v * 100).toFixed(1) + '%' : '—' },
    { key: 'uptime_seconds', label: 'Uptime', fmt: v => v != null ? formatUptime(v) : '—' },
  ];

  container.innerHTML = metricDefs.map(m => `
    <div class="metric-row">
      <span class="metric-row-label">${m.label}</span>
      <span class="metric-row-value skeleton" style="width:60px;height:14px;"></span>
    </div>
  `).join('');

  try {
    const data = await apiFetch('/v1/metrics');
    container.innerHTML = metricDefs.map(m => `
      <div class="metric-row">
        <span class="metric-row-label">${m.label}</span>
        <span class="metric-row-value">${m.fmt(data[m.key])}</span>
      </div>
    `).join('');
  } catch {
    container.innerHTML = metricDefs.map(m => `
      <div class="metric-row">
        <span class="metric-row-label">${m.label}</span>
        <span class="metric-row-value" style="color:var(--text-dim)">—</span>
      </div>
    `).join('');
  }
}

function formatUptime(seconds) {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

// ── Settings popover ──────────────────────────────────────────────────────
function initSettings() {
  const btn = $('#settings-btn');
  const popover = $('#settings-popover');
  const input = $('#api-url-input');

  input.value = state.apiBase;

  btn.addEventListener('click', e => {
    e.stopPropagation();
    popover.classList.toggle('open');
  });

  document.addEventListener('click', e => {
    if (!popover.contains(e.target) && e.target !== btn) {
      popover.classList.remove('open');
    }
  });

  input.addEventListener('change', () => {
    const val = input.value.trim().replace(/\/$/, '');
    if (val) {
      state.apiBase = val;
      localStorage.setItem('rag_api_base', val);
      checkHealth();
      toast('API URL updated', 'success');
    }
  });

  $('#settings-save-btn').addEventListener('click', () => {
    const val = input.value.trim().replace(/\/$/, '');
    if (val) {
      state.apiBase = val;
      localStorage.setItem('rag_api_base', val);
      checkHealth();
      popover.classList.remove('open');
      toast('API URL updated', 'success');
    }
  });
}

// ── Init ──────────────────────────────────────────────────────────────────
function init() {
  // Tab navigation
  $$('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => switchView(tab.dataset.view));
  });

  // Default view
  switchView('query');

  // Query panel components
  initUploadZone();
  initCollapsibles();
  initSlider();
  initToggle();
  initTextarea();

  // Action buttons
  $('#ingest-btn').addEventListener('click', ingest);
  $('#ask-btn').addEventListener('click', query);

  // API settings
  initSettings();

  // Health poll
  checkHealth();
  setInterval(checkHealth, 30000);
}

document.addEventListener('DOMContentLoaded', init);
