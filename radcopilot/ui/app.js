/*
 * radcopilot/ui/app.js
 *
 * Browser application layer for the modular RadCopilot refactor.
 *
 * Design goals:
 * - plain JavaScript, no framework required
 * - works with the current modular Python server + Ollama proxy
 * - gracefully handles missing DOM elements while the UI is still being built
 * - centralizes settings, history, report generation, logs, RAG, and benchmark actions
 * - remains usable from both browser globals and CommonJS tooling
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(root.RadCopilotTemplates || null);
  } else {
    root.RadCopilotApp = factory(root.RadCopilotTemplates || null);
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function (TemplatesApi) {
  'use strict';

  const APP_NAME = 'RadCopilot Local';
  const STORAGE_KEYS = Object.freeze({
    settings: 'rcSettings',
    history: 'rcHistory',
    lastMode: 'rcMode'
  });

  const DEFAULTS = Object.freeze({
    mode: 'report',
    model: 'llama3.1:8b',
    useRag: true,
    maxRagExamples: 3,
    maxHistoryItems: 20,
    logTail: 25,
    benchmarkSampleLimit: 25
  });

  const Templates = TemplatesApi || createFallbackTemplatesApi();

  const state = {
    initialized: false,
    mode: DEFAULTS.mode,
    config: null,
    health: null,
    report: null,
    differential: null,
    guideline: null,
    history: loadHistory(),
    logs: [],
    ragStatus: null,
    benchmark: null,
    settings: loadSettings(),
    abortController: null,
    root: null
  };

  // ---------------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------------
  function $(id, rootNode) {
    const scope = rootNode || state.root || document;
    return scope && typeof scope.getElementById === 'function' ? scope.getElementById(id) : null;
  }

  function qs(selector, rootNode) {
    const scope = rootNode || state.root || document;
    return scope && typeof scope.querySelector === 'function' ? scope.querySelector(selector) : null;
  }

  function qsa(selector, rootNode) {
    const scope = rootNode || state.root || document;
    return scope && typeof scope.querySelectorAll === 'function' ? Array.from(scope.querySelectorAll(selector)) : [];
  }

  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value == null ? '' : String(value);
  }

  function setHTML(id, value) {
    const node = $(id);
    if (node) node.innerHTML = value == null ? '' : String(value);
  }

  function setValue(id, value) {
    const node = $(id);
    if (node && 'value' in node) node.value = value == null ? '' : String(value);
  }

  function getValue(id, fallback) {
    const node = $(id);
    if (!node || !('value' in node)) return fallback == null ? '' : fallback;
    return String(node.value || fallback || '');
  }

  function setChecked(id, checked) {
    const node = $(id);
    if (node && 'checked' in node) node.checked = Boolean(checked);
  }

  function getChecked(id, fallback) {
    const node = $(id);
    if (!node || !('checked' in node)) return Boolean(fallback);
    return Boolean(node.checked);
  }

  function setDisabled(id, disabled) {
    const node = $(id);
    if (node) node.disabled = Boolean(disabled);
  }

  function show(id, visible) {
    const node = $(id);
    if (!node) return;
    node.hidden = !visible;
    if (node.style) node.style.display = visible ? '' : 'none';
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function humanTime(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      if (Number.isNaN(date.getTime())) return String(isoString);
      return date.toLocaleString();
    } catch (_err) {
      return String(isoString);
    }
  }

  // ---------------------------------------------------------------------------
  // Settings / local storage
  // ---------------------------------------------------------------------------
  function loadJson(key, fallback) {
    try {
      const raw = root.localStorage ? root.localStorage.getItem(key) : null;
      if (!raw) return clone(fallback);
      return JSON.parse(raw);
    } catch (_err) {
      return clone(fallback);
    }
  }

  function saveJson(key, value) {
    try {
      if (root.localStorage) {
        root.localStorage.setItem(key, JSON.stringify(value));
      }
    } catch (_err) {
      // ignore storage write failures in local UI
    }
  }

  function loadSettings() {
    return Object.assign({
      model: DEFAULTS.model,
      templateId: Templates.getDefaultTemplateId(),
      useRag: DEFAULTS.useRag,
      maxRagExamples: DEFAULTS.maxRagExamples
    }, loadJson(STORAGE_KEYS.settings, {}));
  }

  function saveSettings(settings) {
    state.settings = Object.assign({}, state.settings, settings || {});
    saveJson(STORAGE_KEYS.settings, state.settings);
    saveJson(STORAGE_KEYS.lastMode, state.mode);
    return clone(state.settings);
  }

  function loadHistory() {
    const items = loadJson(STORAGE_KEYS.history, []);
    return Array.isArray(items) ? items : [];
  }

  function saveHistory(items) {
    state.history = Array.isArray(items) ? items.slice(0, DEFAULTS.maxHistoryItems) : [];
    saveJson(STORAGE_KEYS.history, state.history);
  }

  function addHistoryEntry(entry) {
    const next = [Object.assign({
      id: cryptoId(),
      createdAt: new Date().toISOString()
    }, entry || {}), ...state.history];
    saveHistory(next.slice(0, DEFAULTS.maxHistoryItems));
    renderHistory();
  }

  // ---------------------------------------------------------------------------
  // Network layer
  // ---------------------------------------------------------------------------
  async function request(method, path, options) {
    const opts = Object.assign({ json: undefined, body: undefined, headers: {}, signal: undefined }, options || {});
    const headers = Object.assign({}, opts.headers || {});
    let body = opts.body;

    if (opts.json !== undefined) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(opts.json);
    }

    const response = await fetch(path, {
      method,
      headers,
      body,
      signal: opts.signal
    });

    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');
    const payload = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const message = isJson && payload && payload.error ? payload.error : `${method} ${path} failed with ${response.status}`;
      const error = new Error(String(message));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    return payload;
  }

  const api = Object.freeze({
    getHealth() {
      return request('GET', '/health');
    },
    getConfig() {
      return request('GET', '/config');
    },
    getRecentLogs(limit) {
      const qs = new URLSearchParams({ limit: String(limit || DEFAULTS.logTail) });
      return request('GET', `/logs/recent?${qs.toString()}`);
    },
    getRagStatus() {
      return request('GET', '/rag/status');
    },
    queryRag(findings, modality, k) {
      const qs = new URLSearchParams({ findings: String(findings || ''), k: String(k || DEFAULTS.maxRagExamples) });
      if (modality) qs.set('modality', modality);
      return request('GET', `/rag/query?${qs.toString()}`);
    },
    getRagExamples(findings, modality, k) {
      const qs = new URLSearchParams({ findings: String(findings || ''), k: String(k || DEFAULTS.maxRagExamples) });
      if (modality) qs.set('modality', modality);
      return request('GET', `/rag/examples?${qs.toString()}`);
    },
    trainRag(sourcePath) {
      return request('POST', '/rag/train', { json: { path: sourcePath } });
    },
    rateLine(payload) {
      return request('POST', '/rag/rate', { json: payload || {} });
    },
    listBenchmarkDatasets() {
      return request('GET', '/benchmark/datasets');
    },
    loadBenchmarkPath(sourcePath) {
      return request('POST', '/benchmark/load-path', { json: { path: sourcePath } });
    },
    ollamaTags(signal) {
      return request('GET', '/api/tags', { signal });
    },
    ollamaChat(payload, signal) {
      return request('POST', '/api/chat', { json: payload, signal });
    },
    async tryServerReport(payload, signal) {
      try {
        return await request('POST', '/report/generate', { json: payload, signal });
      } catch (error) {
        if (error && (error.status === 404 || error.status === 405)) {
          return null;
        }
        throw error;
      }
    }
  });

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------
  async function initialize(rootNode) {
    state.root = rootNode || document;
    if (state.initialized) return getPublicState();

    state.mode = loadJson(STORAGE_KEYS.lastMode, DEFAULTS.mode) || DEFAULTS.mode;

    populateTemplateSelect();
    restoreSettingsToUi();
    applyMode(state.mode);
    renderTemplateSkeleton();
    renderChecklist();
    renderHistory();
    bindUiEvents();

    await Promise.allSettled([
      refreshConfig(),
      refreshHealth(),
      refreshRagStatus(),
      refreshLogs(),
      refreshModelList(),
      refreshBenchmarkDatasets()
    ]);

    state.initialized = true;
    return getPublicState();
  }

  async function refreshConfig() {
    try {
      state.config = await api.getConfig();
      setText('appName', state.config.app_name || APP_NAME);
      setText('serverStatus', state.config.base_url || '');
      setText('ollamaUrlStatus', state.config.ollama_url || '');
      return state.config;
    } catch (error) {
      setStatus(`Config load failed: ${error.message}`, 'warn');
      return null;
    }
  }

  async function refreshHealth() {
    try {
      state.health = await api.getHealth();
      setText('healthStatus', state.health.ok ? 'Healthy' : 'Degraded');
      setText('ollamaStatus', state.health.ollama_up ? 'Connected' : 'Unavailable');
      return state.health;
    } catch (error) {
      setText('healthStatus', 'Unavailable');
      setText('ollamaStatus', 'Unavailable');
      setStatus(`Health check failed: ${error.message}`, 'warn');
      return null;
    }
  }

  async function refreshRagStatus() {
    try {
      const payload = await api.getRagStatus();
      state.ragStatus = payload;
      renderRagStatus();
      return payload;
    } catch (error) {
      renderRagStatus({ ok: false, error: error.message });
      return null;
    }
  }

  async function refreshLogs() {
    try {
      const payload = await api.getRecentLogs(DEFAULTS.logTail);
      state.logs = Array.isArray(payload.items) ? payload.items : [];
      renderLogs();
      return state.logs;
    } catch (error) {
      setHTML('logsOutput', `<div class="muted">Unable to load logs: ${escapeHtml(error.message)}</div>`);
      return [];
    }
  }

  async function refreshModelList() {
    const modelSelect = $('modelSelect') || $('modelInput');
    if (!modelSelect || !('innerHTML' in modelSelect)) return [];
    try {
      const payload = await api.ollamaTags();
      const models = Array.isArray(payload.models) ? payload.models : [];
      if (!models.length) return [];
      const selected = getValue('modelSelect', getValue('modelInput', state.settings.model || DEFAULTS.model));
      modelSelect.innerHTML = models
        .map((item) => `<option value="${escapeHtml(item.name || '')}">${escapeHtml(item.name || '')}</option>`)
        .join('');
      modelSelect.value = selected || state.settings.model || DEFAULTS.model;
      return models;
    } catch (_err) {
      return [];
    }
  }

  async function refreshBenchmarkDatasets() {
    const output = $('benchmarkDatasets');
    try {
      const payload = await api.listBenchmarkDatasets();
      const items = Array.isArray(payload.items) ? payload.items : [];
      if (!output) return items;
      if (!items.length) {
        output.innerHTML = '<div class="muted">No benchmark datasets discovered.</div>';
        return items;
      }
      output.innerHTML = items.map((item) => {
        return `<div class="dataset-item"><strong>${escapeHtml(item.name || item.path || 'dataset')}</strong><br><span class="muted">${escapeHtml(item.path || '')}</span></div>`;
      }).join('');
      return items;
    } catch (error) {
      if (output) output.innerHTML = `<div class="muted">Unable to list benchmark datasets: ${escapeHtml(error.message)}</div>`;
      return [];
    }
  }

  // ---------------------------------------------------------------------------
  // UI binding / rendering
  // ---------------------------------------------------------------------------
  function bindUiEvents() {
    const seen = new WeakSet();
    const bindings = [
      ['generateReportBtn', 'click', onGenerateReportClick],
      ['cancelBtn', 'click', onCancelClick],
      ['refreshLogsBtn', 'click', () => refreshLogs()],
      ['refreshHealthBtn', 'click', () => Promise.allSettled([refreshHealth(), refreshConfig()])],
      ['trainRagBtn', 'click', onTrainRagClick],
      ['loadBenchmarkBtn', 'click', onLoadBenchmarkClick],
      ['saveSettingsBtn', 'click', onSaveSettingsClick],
      ['templateSelect', 'change', onTemplateChange],
      ['modeSelect', 'change', onModeSelectChange],
      ['runDifferentialBtn', 'click', onRunDifferentialClick],
      ['runGuidelineBtn', 'click', onRunGuidelineClick]
    ];

    bindings.forEach(([id, evt, handler]) => {
      const node = $(id);
      if (node && !seen.has(node)) {
        node.addEventListener(evt, handler);
        seen.add(node);
      }
    });

    qsa('[data-mode]').forEach((node) => {
      if (seen.has(node)) return;
      node.addEventListener('click', function () {
        const nextMode = node.getAttribute('data-mode') || DEFAULTS.mode;
        applyMode(nextMode);
      });
      seen.add(node);
    });
  }

  function populateTemplateSelect() {
    const select = $('templateSelect');
    if (!select) return;
    const options = Templates.getTemplateOptions();
    select.innerHTML = options
      .map((item) => `<option value="${escapeHtml(item.value)}">${escapeHtml(item.label)}</option>`)
      .join('');
  }

  function restoreSettingsToUi() {
    const settings = state.settings || loadSettings();
    setValue('modelInput', settings.model || DEFAULTS.model);
    setValue('modelSelect', settings.model || DEFAULTS.model);
    setValue('templateSelect', settings.templateId || Templates.getDefaultTemplateId());
    setChecked('useRagInput', settings.useRag !== false);
    setValue('maxRagExamplesInput', settings.maxRagExamples || DEFAULTS.maxRagExamples);
  }

  function applyMode(mode) {
    state.mode = normalizeMode(mode);
    saveJson(STORAGE_KEYS.lastMode, state.mode);
    qsa('[data-mode-panel]').forEach((node) => {
      const panelMode = node.getAttribute('data-mode-panel');
      const active = panelMode === state.mode;
      node.hidden = !active;
      if (node.style) node.style.display = active ? '' : 'none';
    });
    qsa('[data-mode]').forEach((node) => {
      const active = node.getAttribute('data-mode') === state.mode;
      node.setAttribute('aria-pressed', active ? 'true' : 'false');
      node.classList.toggle('is-active', active);
    });
    const modeSelect = $('modeSelect');
    if (modeSelect) modeSelect.value = state.mode;
    setStatus(`Mode: ${titleCase(state.mode)}`, 'info');
    return state.mode;
  }

  function renderTemplateSkeleton() {
    const templateId = getSelectedTemplateId();
    const preview = Templates.renderTemplateSkeleton(templateId, { includeDefaults: true });
    setValue('reportOutput', preview);
    setText('templateLabel', Templates.getTemplate(templateId).label || templateId);
    const previewNode = $('templatePreview');
    if (previewNode) {
      if ('value' in previewNode) previewNode.value = preview;
      else previewNode.textContent = preview;
    }
  }

  function renderChecklist() {
    const listNode = $('checklistList');
    if (!listNode) return;
    const checklist = Templates.getChecklist(getSelectedTemplateId());
    if (!checklist.length) {
      listNode.innerHTML = '<li class="muted">No checklist defined.</li>';
      return;
    }
    listNode.innerHTML = checklist.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
  }

  function renderHistory() {
    const container = $('historyList');
    if (!container) return;
    if (!state.history.length) {
      container.innerHTML = '<div class="muted">No saved reports yet.</div>';
      return;
    }
    container.innerHTML = state.history.map((item, index) => {
      const title = item.templateLabel || item.templateId || 'Report';
      const subtitle = item.indication || item.findings || 'No description';
      return (
        `<button class="history-item" data-history-index="${index}" type="button">` +
        `<strong>${escapeHtml(title)}</strong><br>` +
        `<span class="muted">${escapeHtml(truncate(subtitle, 96))}</span><br>` +
        `<span class="muted">${escapeHtml(humanTime(item.createdAt))}</span>` +
        `</button>`
      );
    }).join('');

    qsa('[data-history-index]', container).forEach((node) => {
      node.addEventListener('click', function () {
        const index = parseInt(node.getAttribute('data-history-index') || '-1', 10);
        loadHistoryEntry(index);
      });
    });
  }

  function renderLogs(payload) {
    const output = $('logsOutput');
    if (!output) return;
    const items = payload && Array.isArray(payload.items) ? payload.items : state.logs;
    if (!items || !items.length) {
      output.innerHTML = '<div class="muted">No recent logs.</div>';
      return;
    }
    output.innerHTML = items.map((entry) => {
      return (
        `<div class="log-line">` +
        `<strong>${escapeHtml(entry.type || 'LOG')}</strong> ` +
        `<span class="muted">${escapeHtml(humanTime(entry.ts))}</span><br>` +
        `<code>${escapeHtml(typeof entry.detail === 'string' ? entry.detail : JSON.stringify(entry.detail))}</code>` +
        `</div>`
      );
    }).join('');
  }

  function renderRagStatus(payload) {
    const target = $('ragStatus');
    const info = payload || state.ragStatus;
    if (!target) return;
    if (!info) {
      target.innerHTML = '<div class="muted">RAG status unavailable.</div>';
      return;
    }
    if (info.ok === false && info.error) {
      target.innerHTML = `<div class="muted">${escapeHtml(info.error)}</div>`;
      return;
    }
    const parts = [];
    if (typeof info.record_count !== 'undefined') parts.push(`Records: ${info.record_count}`);
    if (typeof info.ratings_count !== 'undefined') parts.push(`Ratings: ${info.ratings_count}`);
    if (typeof info.index_ready !== 'undefined') parts.push(`Index: ${info.index_ready ? 'Ready' : 'Not Ready'}`);
    target.innerHTML = `<div>${escapeHtml(parts.join(' · ') || 'RAG status loaded')}</div>`;
  }

  function renderBenchmark(payload) {
    const output = $('benchmarkOutput');
    if (!output) return;
    if (!payload || !Array.isArray(payload.items)) {
      output.innerHTML = '<div class="muted">Benchmark data unavailable.</div>';
      return;
    }
    if (!payload.items.length) {
      output.innerHTML = '<div class="muted">No benchmark records loaded.</div>';
      return;
    }
    const lines = payload.items.slice(0, DEFAULTS.benchmarkSampleLimit).map((item, index) => {
      return `${index + 1}. ${truncate(item.findings || '', 160)}\n   Ref: ${truncate(item.impression || '', 140)}`;
    });
    output.textContent = lines.join('\n\n');
  }

  function renderReportResult(result) {
    state.report = result;
    if (!result) return;

    const finalReport = result.final_report || result.finalReport || result.report || '';
    const finalImpression = result.final_impression || result.finalImpression || '';

    setValue('reportOutput', finalReport);
    setValue('impressionOutput', finalImpression);
    setText('reportStatus', result.ok === false ? 'ERROR' : 'COMPLETE');
    setText('templateLabel', result.template_label || result.templateLabel || getSelectedTemplateId());

    const traceOutput = $('traceOutput');
    if (traceOutput) {
      const trace = Array.isArray(result.trace) ? result.trace : [];
      traceOutput.textContent = trace.join('\n');
    }

    const issuesOutput = $('validationOutput');
    if (issuesOutput) {
      const issues = result.validation && Array.isArray(result.validation.issues)
        ? result.validation.issues
        : [];
      if (!issues.length) {
        issuesOutput.textContent = 'No validation issues.';
      } else {
        issuesOutput.textContent = issues.map((item) => `- ${item.code || 'issue'}: ${item.message || ''}`).join('\n');
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Primary workflows
  // ---------------------------------------------------------------------------
  async function generateReport(partialRequest) {
    const requestPayload = buildRequestPayload(partialRequest);
    if (!requestPayload.findings.trim()) {
      throw new Error('Findings are required before generating a report.');
    }

    if (state.abortController) {
      state.abortController.abort();
    }
    state.abortController = new AbortController();

    show('cancelBtn', true);
    setDisabled('generateReportBtn', true);
    setText('reportStatus', 'RUNNING');
    setStatus('Generating report...', 'info');

    try {
      const serverResult = await api.tryServerReport(requestPayload, state.abortController.signal);
      if (serverResult && serverResult.ok !== undefined) {
        renderReportResult(serverResult);
        persistCompletedReport(serverResult, requestPayload);
        setStatus('Report generated using server pipeline.', 'success');
        return serverResult;
      }

      const fallbackResult = await generateReportClientSide(requestPayload, state.abortController.signal);
      renderReportResult(fallbackResult);
      persistCompletedReport(fallbackResult, requestPayload);
      setStatus('Report generated using local browser workflow.', 'success');
      return fallbackResult;
    } catch (error) {
      if (error && error.name === 'AbortError') {
        setText('reportStatus', 'CANCELLED');
        setStatus('Generation cancelled.', 'warn');
        throw error;
      }
      setText('reportStatus', 'ERROR');
      setStatus(`Report generation failed: ${error.message}`, 'error');
      throw error;
    } finally {
      setDisabled('generateReportBtn', false);
      show('cancelBtn', false);
      state.abortController = null;
    }
  }

  async function generateReportClientSide(requestPayload, signal) {
    const template = Templates.getTemplate(requestPayload.template_id);
    const ragExamples = requestPayload.use_rag
      ? await fetchRagExamples(requestPayload.findings, template.modality, requestPayload.max_rag_examples)
      : [];

    const mappingPrompt = buildFindingsMappingPrompt(template, requestPayload.findings);
    const mappingReply = await chatText(mappingPrompt, requestPayload.model, signal);
    const sectionMap = parseMappings(mappingReply, template, requestPayload.findings);
    const renderedSections = buildRenderedSections(template, sectionMap);

    const impressionPrompt = buildImpressionPrompt({
      request: requestPayload,
      template,
      renderedSections,
      ragExamples
    });
    const rawImpression = await chatText(impressionPrompt, requestPayload.model, signal);
    const fixedImpression = normalizeImpression(rawImpression, template);
    const finalReport = buildFinalReport(template, requestPayload, renderedSections, fixedImpression);

    return {
      ok: true,
      template_id: template.id,
      template_label: template.label,
      findings: requestPayload.findings,
      indication: requestPayload.indication,
      section_map: sectionMap,
      rendered_sections: renderedSections,
      rag_examples: ragExamples,
      raw_impression: rawImpression,
      final_impression: fixedImpression,
      validation: validateClientImpression(fixedImpression, template),
      final_report: finalReport,
      trace: [
        'Browser fallback pipeline used.',
        `Template: ${template.id}`,
        `RAG examples: ${ragExamples.length}`
      ],
      model_used: requestPayload.model
    };
  }

  async function runDifferential(partialRequest) {
    const requestPayload = buildRequestPayload(partialRequest);
    if (!requestPayload.findings.trim()) {
      throw new Error('Findings are required before generating a differential.');
    }
    setStatus('Generating differential...', 'info');
    const prompt = buildDifferentialPrompt(requestPayload);
    const text = await chatText(prompt, requestPayload.model);
    state.differential = { request: requestPayload, text };
    setValue('differentialOutput', text);
    setStatus('Differential generated.', 'success');
    return state.differential;
  }

  async function runGuidelineLookup(partialRequest) {
    const requestPayload = buildRequestPayload(partialRequest);
    if (!requestPayload.findings.trim() && !requestPayload.indication.trim()) {
      throw new Error('Findings or indication are required before guideline lookup.');
    }
    setStatus('Looking up guideline recommendations...', 'info');
    const prompt = buildGuidelinePrompt(requestPayload);
    const text = await chatText(prompt, requestPayload.model);
    state.guideline = { request: requestPayload, text };
    setValue('guidelineOutput', text);
    setStatus('Guideline recommendations generated.', 'success');
    return state.guideline;
  }

  async function fetchRagExamples(findings, modality, k) {
    try {
      const [queryPayload, examplePayload] = await Promise.allSettled([
        api.queryRag(findings, modality, k),
        api.getRagExamples(findings, modality, k)
      ]);

      const collected = [];
      [queryPayload, examplePayload].forEach((result) => {
        if (result.status !== 'fulfilled') return;
        const items = Array.isArray(result.value.items) ? result.value.items : [];
        items.forEach((item) => {
          if (!item || !item.findings || !item.impression) return;
          collected.push({
            findings: String(item.findings),
            impression: String(item.impression),
            score: typeof item.score === 'number' ? item.score : null,
            modality: item.modality || modality || null
          });
        });
      });

      return dedupeByKey(collected, function (item) {
        return `${item.findings}||${item.impression}`;
      }).slice(0, k || DEFAULTS.maxRagExamples);
    } catch (_err) {
      return [];
    }
  }

  async function chatText(prompt, model, signal) {
    const payload = {
      model: model || state.settings.model || DEFAULTS.model,
      stream: false,
      messages: [
        {
          role: 'user',
          content: String(prompt || '')
        }
      ]
    };
    const result = await api.ollamaChat(payload, signal);
    return extractChatContent(result);
  }

  // ---------------------------------------------------------------------------
  // Prompt builders / report helpers
  // ---------------------------------------------------------------------------
  function buildRequestPayload(partialRequest) {
    const payload = Object.assign({
      indication: getValue('indicationInput', ''),
      findings: getValue('findingsInput', ''),
      age: getValue('ageInput', ''),
      sex: getValue('sexInput', ''),
      template_id: getSelectedTemplateId(),
      model: getSelectedModel(),
      use_rag: getChecked('useRagInput', state.settings.useRag !== false),
      max_rag_examples: parseInt(getValue('maxRagExamplesInput', state.settings.maxRagExamples || DEFAULTS.maxRagExamples), 10) || DEFAULTS.maxRagExamples
    }, partialRequest || {});

    return {
      indication: String(payload.indication || '').trim(),
      findings: String(payload.findings || '').trim(),
      age: String(payload.age || '').trim(),
      sex: String(payload.sex || '').trim(),
      template_id: Templates.normalizeTemplateId(payload.template_id),
      model: String(payload.model || DEFAULTS.model).trim(),
      use_rag: Boolean(payload.use_rag),
      max_rag_examples: Math.max(0, Math.min(8, Number(payload.max_rag_examples || DEFAULTS.maxRagExamples)))
    };
  }

  function buildFindingsMappingPrompt(template, findings) {
    const labels = template.sectionOrder
      .map((key) => `${key} => ${template.sectionLabels[key] || titleCase(key)}`)
      .join('\n');
    return [
      'You are mapping radiology findings into report sections.',
      'Return one line per relevant finding in the exact format: section_key | sentence',
      'Only use these section keys:',
      labels,
      'Do not invent findings. Preserve measurements and laterality.',
      'If a finding does not fit well, choose the closest section.',
      '',
      'Findings:',
      findings
    ].join('\n');
  }

  function buildImpressionPrompt(context) {
    const requestPayload = context.request;
    const template = context.template;
    const sectionText = template.sectionOrder
      .map((key) => `${template.sectionLabels[key] || titleCase(key)}: ${(context.renderedSections[key] || '').trim()}`)
      .join('\n');
    const ragText = (context.ragExamples || []).length
      ? context.ragExamples.map((item, index) => {
          return `Example ${index + 1}\nFindings: ${item.findings}\nImpression: ${item.impression}`;
        }).join('\n\n')
      : 'No retrieval examples available.';

    return [
      'You are drafting the IMPRESSION section of a radiology report.',
      'Write only concise clinically significant impression lines.',
      'Prefer numbered impression lines.',
      template.allowNegativesInImpression
        ? 'Negatives may be included only when clinically important.'
        : 'Do not include purely negative normal statements in the impression.',
      template.guidelineHint ? `Guideline hint: ${template.guidelineHint}` : '',
      requestPayload.indication ? `Indication: ${requestPayload.indication}` : '',
      requestPayload.age ? `Age: ${requestPayload.age}` : '',
      requestPayload.sex ? `Sex: ${requestPayload.sex}` : '',
      '',
      'Rendered findings by section:',
      sectionText,
      '',
      'Similar examples:',
      ragText,
      '',
      'Return only the final impression text.'
    ].filter(Boolean).join('\n');
  }

  function buildDifferentialPrompt(requestPayload) {
    const template = Templates.getTemplate(requestPayload.template_id);
    return [
      'You are generating a differential diagnosis from radiology findings.',
      'Return a ranked list with the most likely possibilities first.',
      'Be concise and explain the reasoning briefly.',
      `Template: ${template.label}`,
      requestPayload.indication ? `Indication: ${requestPayload.indication}` : '',
      '',
      'Findings:',
      requestPayload.findings
    ].filter(Boolean).join('\n');
  }

  function buildGuidelinePrompt(requestPayload) {
    const template = Templates.getTemplate(requestPayload.template_id);
    return [
      'You are summarizing guideline-oriented follow-up or recommendation logic for a radiology case.',
      'Do not invent society names or thresholds that are not supported by the findings.',
      template.guidelineHint ? `Template guidance: ${template.guidelineHint}` : '',
      requestPayload.indication ? `Indication: ${requestPayload.indication}` : '',
      requestPayload.findings ? `Findings: ${requestPayload.findings}` : '',
      '',
      'Return concise recommendation language suitable for the radiologist to review.'
    ].filter(Boolean).join('\n');
  }

  function parseMappings(rawText, template, originalFindings) {
    const mapped = {};
    template.sectionOrder.forEach((key) => {
      mapped[key] = [];
    });

    splitLines(rawText).forEach((line) => {
      const match = line.match(/^([a-zA-Z0-9_-]+)\s*\|\s*(.+)$/);
      if (!match) return;
      const sectionKey = normalizeSectionKey(match[1], template);
      const text = cleanSentence(match[2]);
      if (!sectionKey || !text) return;
      mapped[sectionKey].push(text);
    });

    const hasAny = Object.values(mapped).some((items) => items.length);
    if (!hasAny) {
      fallbackMapFromFindings(originalFindings, template).forEach((item) => {
        mapped[item.section].push(item.text);
      });
    }

    Object.keys(mapped).forEach((key) => {
      mapped[key] = dedupeByKey(mapped[key], function (line) { return line.toLowerCase(); });
    });
    return mapped;
  }

  function buildRenderedSections(template, sectionMap) {
    const rendered = {};
    template.sectionOrder.forEach((key) => {
      const abnormal = Array.isArray(sectionMap[key]) ? sectionMap[key] : [];
      rendered[key] = abnormal.length ? abnormal.join(' ') : String(template.sectionDefaults[key] || '');
    });
    return rendered;
  }

  function normalizeImpression(rawText, template) {
    let lines = splitLines(String(rawText || ''))
      .map((line) => line.replace(/^impression\s*:?/i, '').trim())
      .map((line) => line.replace(/^[-*•]\s*/, '').trim())
      .filter(Boolean);

    lines = lines.map((line) => line.replace(/^\d+[.)]\s*/, '').trim()).filter(Boolean);
    lines = dedupeByKey(lines, function (line) { return line.toLowerCase(); });

    if (!template.allowNegativesInImpression) {
      lines = lines.filter((line) => !/^no\b/i.test(line));
    }

    if (!lines.length) {
      lines = ['No acute abnormality identified on this exam.'];
    }

    return lines.map((line, index) => `${index + 1}. ${cleanSentence(line)}`).join('\n');
  }

  function validateClientImpression(impression, template) {
    const issues = [];
    const lines = splitLines(impression).filter(Boolean);
    if (!lines.length) {
      issues.push({ code: 'empty_impression', message: 'No impression text returned.' });
    }
    if (!template.allowNegativesInImpression) {
      lines.forEach((line) => {
        if (/^\d+\.\s*no\b/i.test(line)) {
          issues.push({ code: 'negative_impression', message: `Negative line detected: ${line}` });
        }
      });
    }
    return {
      ok: issues.length === 0,
      issue_count: issues.length,
      issues: issues
    };
  }

  function buildFinalReport(template, requestPayload, renderedSections, impressionText) {
    const parts = [];
    if (requestPayload.indication) {
      parts.push(`INDICATION: ${requestPayload.indication}`);
      parts.push('');
    }
    parts.push('FINDINGS:');
    template.sectionOrder.forEach((key) => {
      const label = template.sectionLabels[key] || titleCase(key);
      parts.push(`${label}: ${(renderedSections[key] || '').trim()}`);
    });
    parts.push('');
    parts.push('IMPRESSION:');
    parts.push(impressionText.trim());
    return parts.join('\n').trim();
  }

  function persistCompletedReport(result, requestPayload) {
    const historyEntry = {
      templateId: result.template_id || requestPayload.template_id,
      templateLabel: result.template_label || Templates.getTemplate(requestPayload.template_id).label,
      indication: requestPayload.indication,
      findings: requestPayload.findings,
      age: requestPayload.age,
      sex: requestPayload.sex,
      report: result.final_report || '',
      impression: result.final_impression || '',
      model: requestPayload.model,
      metadata: {
        ragExampleCount: Array.isArray(result.rag_examples) ? result.rag_examples.length : 0
      }
    };
    addHistoryEntry(historyEntry);
    saveSettings({
      model: requestPayload.model,
      templateId: requestPayload.template_id,
      useRag: requestPayload.use_rag,
      maxRagExamples: requestPayload.max_rag_examples
    });
  }

  // ---------------------------------------------------------------------------
  // Event handlers
  // ---------------------------------------------------------------------------
  async function onGenerateReportClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    try {
      await generateReport();
    } catch (error) {
      if (error && error.name === 'AbortError') return;
      console.error(error);
    }
  }

  function onCancelClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    if (state.abortController) {
      state.abortController.abort();
      state.abortController = null;
    }
  }

  function onTemplateChange(event) {
    const templateId = event && event.target ? event.target.value : getSelectedTemplateId();
    saveSettings({ templateId: Templates.normalizeTemplateId(templateId) });
    renderTemplateSkeleton();
    renderChecklist();
  }

  function onModeSelectChange(event) {
    applyMode(event && event.target ? event.target.value : DEFAULTS.mode);
  }

  async function onRunDifferentialClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    try {
      await runDifferential();
    } catch (error) {
      setStatus(`Differential failed: ${error.message}`, 'error');
    }
  }

  async function onRunGuidelineClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    try {
      await runGuidelineLookup();
    } catch (error) {
      setStatus(`Guideline lookup failed: ${error.message}`, 'error');
    }
  }

  async function onTrainRagClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    const path = getValue('trainPathInput', '').trim();
    if (!path) {
      setStatus('Enter a file or directory path before training RAG.', 'warn');
      return;
    }
    try {
      setStatus('Training RAG library...', 'info');
      const payload = await api.trainRag(path);
      const target = $('trainRagOutput');
      if (target) {
        target.textContent = JSON.stringify(payload, null, 2);
      }
      await refreshRagStatus();
      setStatus('RAG training completed.', 'success');
    } catch (error) {
      setStatus(`RAG training failed: ${error.message}`, 'error');
    }
  }

  async function onLoadBenchmarkClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    const path = getValue('benchmarkPathInput', '').trim();
    if (!path) {
      setStatus('Enter a benchmark dataset path first.', 'warn');
      return;
    }
    try {
      const payload = await api.loadBenchmarkPath(path);
      state.benchmark = payload;
      renderBenchmark(payload);
      setStatus('Benchmark data loaded.', 'success');
    } catch (error) {
      setStatus(`Benchmark load failed: ${error.message}`, 'error');
    }
  }

  function onSaveSettingsClick(event) {
    if (event && typeof event.preventDefault === 'function') event.preventDefault();
    saveSettings({
      model: getSelectedModel(),
      templateId: getSelectedTemplateId(),
      useRag: getChecked('useRagInput', DEFAULTS.useRag),
      maxRagExamples: parseInt(getValue('maxRagExamplesInput', DEFAULTS.maxRagExamples), 10) || DEFAULTS.maxRagExamples
    });
    setStatus('Settings saved locally.', 'success');
  }

  // ---------------------------------------------------------------------------
  // Utility helpers
  // ---------------------------------------------------------------------------
  function getSelectedTemplateId() {
    return Templates.normalizeTemplateId(getValue('templateSelect', state.settings.templateId || Templates.getDefaultTemplateId()));
  }

  function getSelectedModel() {
    return String(getValue('modelSelect', getValue('modelInput', state.settings.model || DEFAULTS.model)) || DEFAULTS.model).trim();
  }

  function loadHistoryEntry(index) {
    const item = state.history[index];
    if (!item) return null;
    setValue('indicationInput', item.indication || '');
    setValue('findingsInput', item.findings || '');
    setValue('ageInput', item.age || '');
    setValue('sexInput', item.sex || '');
    setValue('templateSelect', item.templateId || Templates.getDefaultTemplateId());
    renderTemplateSkeleton();
    renderChecklist();
    setValue('reportOutput', item.report || '');
    setValue('impressionOutput', item.impression || '');
    setStatus(`Loaded history item: ${item.templateLabel || item.templateId || 'Report'}`, 'info');
    return item;
  }

  function setStatus(message, level) {
    const node = $('appStatus');
    if (!node) return;
    node.textContent = String(message || '');
    node.setAttribute('data-level', level || 'info');
  }

  function splitLines(text) {
    return String(text || '')
      .split(/\r?\n+/)
      .map((line) => line.trim())
      .filter(Boolean);
  }

  function cleanSentence(value) {
    return String(value || '')
      .replace(/^[-*•\s]+/, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function normalizeSectionKey(rawKey, template) {
    const normalized = String(rawKey || '').trim().toLowerCase().replace(/\s+/g, '_');
    if (template.sectionOrder.includes(normalized)) return normalized;

    const labelMatch = template.sectionOrder.find((key) => {
      const label = String(template.sectionLabels[key] || '').trim().toLowerCase().replace(/[\s/]+/g, '_');
      return label === normalized || label.includes(normalized) || normalized.includes(label);
    });
    return labelMatch || template.sectionOrder[0] || 'general';
  }

  function fallbackMapFromFindings(findings, template) {
    const lines = splitLines(findings);
    if (!lines.length) return [];
    return lines.map((line) => ({
      section: guessSection(line, template),
      text: cleanSentence(line)
    }));
  }

  function guessSection(line, template) {
    const text = String(line || '').toLowerCase();
    const choices = template.sectionOrder || [];
    const hints = {
      lungs: ['lung', 'pulmonary', 'airspace', 'nodule', 'opacity', 'atelect', 'consolid'],
      pleura: ['pleura', 'pleural', 'effusion', 'pneumothorax'],
      mediastinum: ['mediast', 'hilar', 'lymph node'],
      heart: ['heart', 'cardiac', 'pericard'],
      vasculature: ['vascular', 'aorta', 'artery', 'vein', 'embol'],
      chest_wall: ['rib', 'osseous', 'bone', 'fracture', 'chest wall'],
      liver: ['liver', 'hepatic'],
      gallbladder: ['gallbladder', 'biliary', 'duct'],
      spleen: ['spleen', 'splenic'],
      pancreas: ['pancrea'],
      adrenals: ['adrenal'],
      kidneys: ['kidney', 'renal', 'ureter', 'urinary', 'bladder'],
      bowel: ['bowel', 'colon', 'small bowel', 'append'],
      peritoneum: ['ascites', 'mesent', 'peritone'],
      pelvis: ['uterus', 'ovar', 'adnex', 'pelvis', 'prostate'],
      brain: ['brain', 'intracranial', 'hemorrhage', 'infarct', 'ventric'],
      sinuses: ['sinus', 'mastoid'],
      general: []
    };
    let best = choices[0] || 'general';
    let bestScore = -1;
    choices.forEach((key) => {
      const score = (hints[key] || []).reduce((sum, hint) => sum + (text.includes(hint) ? 1 : 0), 0);
      if (score > bestScore) {
        best = key;
        bestScore = score;
      }
    });
    return best;
  }

  function extractChatContent(payload) {
    if (!payload) return '';
    if (typeof payload.content === 'string') return payload.content;
    if (payload.message && typeof payload.message.content === 'string') return payload.message.content;
    if (typeof payload.response === 'string') return payload.response;
    return '';
  }

  function normalizeMode(mode) {
    const value = String(mode || '').trim().toLowerCase();
    if (['report', 'differential', 'guideline'].includes(value)) return value;
    return DEFAULTS.mode;
  }

  function truncate(value, length) {
    const text = String(value || '');
    if (text.length <= length) return text;
    return text.slice(0, Math.max(0, length - 1)) + '…';
  }

  function titleCase(value) {
    return String(value || '')
      .replace(/[_-]+/g, ' ')
      .split(' ')
      .filter(Boolean)
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }

  function dedupeByKey(items, keyFn) {
    const seen = new Set();
    const output = [];
    (Array.isArray(items) ? items : []).forEach((item) => {
      const key = String(keyFn(item));
      if (seen.has(key)) return;
      seen.add(key);
      output.push(item);
    });
    return output;
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value == null ? null : value));
  }

  function cryptoId() {
    try {
      if (root.crypto && typeof root.crypto.randomUUID === 'function') {
        return root.crypto.randomUUID();
      }
    } catch (_err) {
      // fall through
    }
    return `rc-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
  }

  function getPublicState() {
    return {
      initialized: state.initialized,
      mode: state.mode,
      config: clone(state.config),
      health: clone(state.health),
      history: clone(state.history),
      settings: clone(state.settings),
      ragStatus: clone(state.ragStatus)
    };
  }

  function createFallbackTemplatesApi() {
    return {
      DEFAULT_TEMPLATE_ID: 'generic',
      getDefaultTemplateId: function () { return 'generic'; },
      normalizeTemplateId: function (id) { return String(id || 'generic').trim().toLowerCase() || 'generic'; },
      getTemplate: function () {
        return {
          id: 'generic',
          label: 'Generic Report',
          modality: 'generic',
          sectionOrder: ['general'],
          sectionLabels: { general: 'Findings' },
          sectionDefaults: { general: 'No acute abnormality is identified on this exam.' },
          allowNegativesInImpression: false,
          guidelineHint: '',
          checklist: ['Findings reviewed']
        };
      },
      getTemplateOptions: function () {
        return [{ value: 'generic', label: 'Generic Report', modality: 'generic' }];
      },
      getChecklist: function () { return ['Findings reviewed']; },
      renderTemplateSkeleton: function () {
        return 'FINDINGS:\nFindings: No acute abnormality is identified on this exam.\n\nIMPRESSION:';
      }
    };
  }

  const publicApi = {
    APP_NAME,
    DEFAULTS: clone(DEFAULTS),
    initialize,
    refreshConfig,
    refreshHealth,
    refreshRagStatus,
    refreshLogs,
    refreshBenchmarkDatasets,
    generateReport,
    runDifferential,
    runGuidelineLookup,
    applyMode,
    buildRequestPayload,
    buildFindingsMappingPrompt,
    buildImpressionPrompt,
    buildDifferentialPrompt,
    buildGuidelinePrompt,
    buildFinalReport,
    loadHistoryEntry,
    getState: getPublicState,
    saveSettings,
    api,
    Templates
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', function () {
        initialize(document).catch(function (error) {
          console.error('RadCopilot initialization failed:', error);
        });
      });
    } else {
      initialize(document).catch(function (error) {
        console.error('RadCopilot initialization failed:', error);
      });
    }
  }

  return Object.freeze(publicApi);
});
