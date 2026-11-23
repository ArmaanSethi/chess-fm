/**
 * ChessFM Public Project & Benchmarks Dashboard.
 *
 * Zero-dependency browser module that loads ./data/dashboard.json, validates
 * its structure, renders architecture, milestones, P1/P2 benchmarks, the
 * engineering quirks ledger, and quick links with strict XSS escaping and
 * URL sanitization.
 */

const VALID_MILESTONE_STATUSES = new Set(['DONE', 'IN_PROGRESS', 'PLANNED']);
const VALID_QUIRK_TRACKS = new Set(['P1', 'P2', 'Infra']);

/**
 * Escape HTML special characters so untrusted or dynamic JSON strings can
 * never inject markup or event handlers.
 *
 * @param {unknown} value
 * @returns {string}
 */
export function escapeHtml(value) {
  if (value === null || value === undefined) return '';
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

/**
 * Validate and sanitize a URL before placing it in an `href` attribute.
 * Permits only `https://`, `http://`, `./`, or `#` targets; blocks
 * `javascript:`, `data:`, `vbscript:`, protocol-relative `//`, etc.
 *
 * @param {unknown} url
 * @returns {string}
 */
export function sanitizeUrl(url) {
  if (typeof url !== 'string') return '#';
  const trimmed = url.trim();
  if (!trimmed) return '#';
  // Reject control characters or whitespace inside scheme
  if (/[\u0000-\u001f\u007f]/.test(trimmed)) return '#';

  if (trimmed.startsWith('./') || trimmed.startsWith('#')) {
    return trimmed;
  }
  if (/^https?:\/\//i.test(trimmed)) {
    try {
      const parsed = new URL(trimmed);
      if (parsed.protocol === 'https:' || parsed.protocol === 'http:') {
        return parsed.href;
      }
    } catch {
      return '#';
    }
  }
  return '#';
}

/**
 * Validate the loaded dashboard.json payload.
 *
 * @param {unknown} data
 * @returns {{ ok: boolean, errors: string[] }}
 */
export function validateDashboardData(data) {
  const errors = [];
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return { ok: false, errors: ['Dashboard payload must be a JSON object'] };
  }

  if (typeof data.schema_version !== 'string' || !data.schema_version.trim()) {
    errors.push('Missing schema_version');
  }
  if (typeof data.updated_utc !== 'string' || Number.isNaN(Date.parse(data.updated_utc))) {
    errors.push('Invalid updated_utc ISO-8601 timestamp');
  }
  if (
    typeof data.overall_progress_pct !== 'number' ||
    !Number.isFinite(data.overall_progress_pct) ||
    data.overall_progress_pct < 0 ||
    data.overall_progress_pct > 100
  ) {
    errors.push('overall_progress_pct must be a finite number between 0 and 100');
  }

  if (!data.budget || typeof data.budget !== 'object') {
    errors.push('Missing budget object');
  } else {
    const { cap_usd: cap, spent_usd_est: spent } = data.budget;
    if (typeof cap !== 'number' || !Number.isFinite(cap) || cap <= 0) {
      errors.push('budget.cap_usd must be a positive finite number');
    }
    if (typeof spent !== 'number' || !Number.isFinite(spent) || spent < 0) {
      errors.push('budget.spent_usd_est must be a non-negative finite number');
    }
  }

  if (!Array.isArray(data.architecture) || data.architecture.length === 0) {
    errors.push('architecture must be a non-empty array');
  }

  if (!Array.isArray(data.milestones) || data.milestones.length === 0) {
    errors.push('milestones must be a non-empty array');
  } else {
    for (const m of data.milestones) {
      if (!m || typeof m !== 'object' || typeof m.id !== 'string') {
        errors.push('Invalid milestone entry');
        continue;
      }
      if (!VALID_MILESTONE_STATUSES.has(m.status)) {
        errors.push(`Invalid status "${String(m.status)}" on milestone ${m.id}`);
      }
    }
  }

  if (
    !data.benchmarks ||
    typeof data.benchmarks !== 'object' ||
    !Array.isArray(data.benchmarks.p1_models) ||
    !Array.isArray(data.benchmarks.p2_layer_a) ||
    !Array.isArray(data.benchmarks.p2_layer_b)
  ) {
    errors.push('benchmarks must include p1_models, p2_layer_a, and p2_layer_b arrays');
  }

  if (!Array.isArray(data.quirks_ledger)) {
    errors.push('quirks_ledger must be an array');
  } else {
    for (const q of data.quirks_ledger) {
      if (!q || typeof q !== 'object' || typeof q.id !== 'string') {
        errors.push('Invalid quirk entry');
        continue;
      }
      if (!VALID_QUIRK_TRACKS.has(q.track)) {
        errors.push(`Invalid track "${String(q.track)}" on quirk ${q.id}`);
      }
    }
  }

  if (!data.links || typeof data.links !== 'object') {
    errors.push('Missing links object');
  }

  return { ok: errors.length === 0, errors };
}

/**
 * Compute aggregate milestone completion counts.
 *
 * @param {Array<{ status: string }>} milestones
 * @returns {{ total: number, done: number, inProgress: number, planned: number, pct: number }}
 */
export function computeMilestoneCounts(milestones) {
  if (!Array.isArray(milestones) || milestones.length === 0) {
    return { total: 0, done: 0, inProgress: 0, planned: 0, pct: 0 };
  }
  let done = 0;
  let inProgress = 0;
  let planned = 0;
  for (const m of milestones) {
    if (m?.status === 'DONE') done += 1;
    else if (m?.status === 'IN_PROGRESS') inProgress += 1;
    else if (m?.status === 'PLANNED') planned += 1;
  }
  const total = milestones.length;
  return {
    total,
    done,
    inProgress,
    planned,
    pct: Math.round((done / total) * 100),
  };
}

/**
 * Filter the engineering quirks ledger by track ('All', 'P1', 'P2', 'Infra').
 *
 * @param {Array<{ track: string }>} quirks
 * @param {string} track
 * @returns {Array<object>}
 */
export function filterQuirks(quirks, track = 'All') {
  if (!Array.isArray(quirks)) return [];
  if (!track || track === 'All') return [...quirks];
  return quirks.filter((q) => q && q.track === track);
}

/**
 * Format a numeric or nullable benchmark cell cleanly.
 *
 * @param {number|string|null|undefined} value
 * @param {string} suffix
 * @returns {string}
 */
export function formatMetricCell(value, suffix = '') {
  if (value === null || value === undefined || value === '') {
    return '<span class="metric-pending">&mdash;</span>';
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return '<span class="metric-pending">&mdash;</span>';
    const formatted = Number.isInteger(value) ? String(value) : value.toFixed(1);
    return `${escapeHtml(formatted)}${escapeHtml(suffix)}`;
  }
  return `${escapeHtml(value)}${escapeHtml(suffix)}`;
}

function statusPillClass(status) {
  if (status === 'DONE' || status === 'measured' || status === 'Fixed') return 'pill-ok';
  if (status === 'IN_PROGRESS' || status === 'In Progress' || status === 'Mitigated') {
    return 'pill-warn';
  }
  return 'pill-info';
}

function formatStatusLabel(status) {
  if (status === 'IN_PROGRESS') return 'IN PROGRESS';
  if (status === 'measured') return 'Measured';
  if (status === 'pending') return 'Pending';
  return String(status || '');
}

function formatUtcTimestamp(isoString) {
  const parsed = Date.parse(isoString);
  if (Number.isNaN(parsed)) return escapeHtml(isoString || 'Unknown');
  const d = new Date(parsed);
  return `${d.toISOString().slice(0, 16).replace('T', ' ')} UTC`;
}

/**
 * Render an architecture overview card.
 *
 * @param {object} card
 * @returns {string}
 */
export function renderArchitectureCard(card) {
  const highlights = Array.isArray(card.highlights)
    ? `<ul class="arch-list">${card.highlights
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join('')}</ul>`
    : '';
  return `
    <article class="arch-card">
      <div class="arch-card-head">
        <span class="pill pill-info">${escapeHtml(card.track)}</span>
        <span class="muted-text arch-badge">${escapeHtml(card.badge)}</span>
      </div>
      <h4 class="arch-title">${escapeHtml(card.title)}</h4>
      <p class="muted-text">${escapeHtml(card.summary)}</p>
      ${highlights}
    </article>
  `;
}

/**
 * Render a single milestone timeline table row.
 *
 * @param {object} milestone
 * @param {string} githubRepoUrl
 * @returns {string}
 */
export function renderMilestoneRow(milestone, githubRepoUrl = 'https://github.com/ArmaanSethi/chess-fm') {
  const pillCls = statusPillClass(milestone.status);
  const statusLabel = formatStatusLabel(milestone.status);
  const safeRepo = sanitizeUrl(githubRepoUrl).replace(/\/+$/, '');
  const shaHtml = milestone.commit_sha
    ? `<a class="sha-link" href="${escapeHtml(sanitizeUrl(`${safeRepo}/commit/${encodeURIComponent(milestone.commit_sha)}`))}" target="_blank" rel="noopener"><code>${escapeHtml(milestone.commit_sha)}</code></a>`
    : '<span class="metric-pending">&mdash;</span>';
  const metrics = Array.isArray(milestone.verified_metrics) && milestone.verified_metrics.length > 0
    ? `<div class="chip-row">${milestone.verified_metrics
        .map((metric) => `<span class="metric-chip">${escapeHtml(metric)}</span>`)
        .join('')}</div>`
    : '<span class="metric-pending">Pending verification</span>';

  return `
    <tr>
      <td><strong>${escapeHtml(milestone.id)}</strong></td>
      <td><span class="pill pill-muted">${escapeHtml(milestone.track)}</span></td>
      <td>
        <div class="ms-name">${escapeHtml(milestone.name)}</div>
        <div class="ms-summary">${escapeHtml(milestone.summary)}</div>
      </td>
      <td><span class="pill ${pillCls}">${escapeHtml(statusLabel)}</span></td>
      <td>${shaHtml}</td>
      <td>${metrics}</td>
    </tr>
  `;
}

/**
 * Render a P1 RL LLM benchmark progression table row.
 *
 * @param {object} row
 * @returns {string}
 */
export function renderP1BenchmarkRow(row) {
  const pillCls = statusPillClass(row.status);
  const statusBadge = row.status === 'measured'
    ? '<span class="pill pill-ok">Measured</span>'
    : `<span class="pill ${pillCls}">${escapeHtml(row.notes || 'Pending (M3)')}</span>`;

  return `
    <tr>
      <td><strong>${escapeHtml(row.stage)}</strong></td>
      <td><code>${escapeHtml(row.model)}</code></td>
      <td>${statusBadge}</td>
      <td>${formatMetricCell(row.strict_legal_pct, '%')}</td>
      <td>${formatMetricCell(row.constrained_legal_pct, '%')}</td>
      <td>${formatMetricCell(row.acpl)}</td>
      <td>${formatMetricCell(row.stockfish_score)}</td>
      <td>${formatMetricCell(row.maia_score)}</td>
    </tr>
  `;
}

/**
 * Render a P2 Layer A measured capability row.
 *
 * @param {object} row
 * @returns {string}
 */
export function renderP2LayerARow(row) {
  const pillCls = statusPillClass(row.status);
  return `
    <tr>
      <td><strong>${escapeHtml(row.capability)}</strong></td>
      <td><span class="pill ${pillCls}">${escapeHtml(formatStatusLabel(row.status))}</span></td>
      <td>${escapeHtml(row.value)}</td>
      <td class="muted-cell">${escapeHtml(row.notes)}</td>
    </tr>
  `;
}

/**
 * Render a P2 Layer B distillation benchmark table row.
 *
 * @param {object} row
 * @returns {string}
 */
export function renderP2LayerBRow(row) {
  const pillCls = statusPillClass(row.status);
  const statusBadge = row.status === 'measured'
    ? '<span class="pill pill-ok">Measured</span>'
    : `<span class="pill ${pillCls}">${escapeHtml(row.notes || 'Pending (M4)')}</span>`;

  return `
    <tr>
      <td><strong>${escapeHtml(row.stage)}</strong></td>
      <td><code>${escapeHtml(row.model)}</code></td>
      <td>${statusBadge}</td>
      <td>${formatMetricCell(row.act_eval_faithfulness_pct, '%')}</td>
      <td>${formatMetricCell(row.spoiler_rate_pct, '%')}</td>
      <td>${formatMetricCell(row.webgpu_tok_s, ' tok/s')}</td>
    </tr>
  `;
}

/**
 * Render an engineering quirk card for the Quirks & Failure Ledger.
 *
 * @param {object} quirk
 * @returns {string}
 */
export function renderQuirkCard(quirk) {
  const statusCls = statusPillClass(quirk.status);
  return `
    <article class="quirk-card" data-track="${escapeHtml(quirk.track)}">
      <div class="quirk-head">
        <div class="quirk-meta">
          <code class="quirk-id">${escapeHtml(quirk.id)}</code>
          <span class="pill pill-muted">${escapeHtml(quirk.track)}</span>
          <span class="quirk-sub">${escapeHtml(quirk.subsystem)}</span>
        </div>
        <span class="pill ${statusCls}">${escapeHtml(quirk.status)}</span>
      </div>
      <h3 class="quirk-title">${escapeHtml(quirk.title)}</h3>
      <p class="quirk-detail"><strong>Impact:</strong> ${escapeHtml(quirk.impact)}</p>
      <p class="quirk-detail"><strong>Resolution:</strong> ${escapeHtml(quirk.resolution)}</p>
    </article>
  `;
}

function renderLinksGrid(links) {
  const primaryCards = [
    {
      label: 'Live Coach Studio',
      desc: 'Client-side Stockfish-19 WASM + Maia-2 Coach',
      url: links.live_coach,
      external: false,
    },
    {
      label: 'GitHub Repository',
      desc: 'Source code, CI workflows, and issues',
      url: links.github_repo,
      external: true,
    },
    {
      label: 'Weights & Biases',
      desc: 'Live SFT & GRPO training curves and telemetry',
      url: links.wandb_project,
      external: true,
    },
    {
      label: 'Hugging Face Hub',
      desc: 'Published checkpoints, manifests, and ONNX models',
      url: links.huggingface,
      external: true,
    },
  ];

  const primaryHtml = primaryCards
    .map((item) => {
      const safeHref = escapeHtml(sanitizeUrl(item.url));
      const attrs = item.external ? ' target="_blank" rel="noopener"' : '';
      return `
        <a class="link-card" href="${safeHref}"${attrs}>
          <span class="link-card-title">${escapeHtml(item.label)}</span>
          <span class="link-card-desc">${escapeHtml(item.desc)}</span>
        </a>
      `;
    })
    .join('');

  const docsHtml = Array.isArray(links.docs)
    ? links.docs
        .map((doc) => {
          const safeHref = escapeHtml(sanitizeUrl(doc.url));
          return `<a class="btn btn-sm btn-secondary" href="${safeHref}" target="_blank" rel="noopener">${escapeHtml(doc.label)}</a>`;
        })
        .join('')
    : '';

  return { primaryHtml, docsHtml };
}

function showError(doc, message) {
  const banner = doc.getElementById('dashboard-error');
  if (!banner) return;
  banner.textContent = message;
  banner.classList.remove('hidden');
}

/**
 * Initialize and render the dashboard in the browser DOM.
 *
 * @param {Document} doc
 * @param {string} dataUrl
 */
export async function initDashboard(doc = document, dataUrl = './data/dashboard.json') {
  try {
    const response = await fetch(dataUrl, { cache: 'no-store' });
    if (!response.ok) {
      showError(doc, `Unable to load dashboard data (HTTP ${response.status}).`);
      return false;
    }

    let data;
    try {
      data = await response.json();
    } catch {
      showError(doc, 'Dashboard data file returned invalid JSON.');
      return false;
    }

    const validation = validateDashboardData(data);
    if (!validation.ok) {
      showError(doc, `Dashboard data validation failed: ${validation.errors.join('; ')}`);
      return false;
    }

    const counts = computeMilestoneCounts(data.milestones);
    const pct = Math.max(0, Math.min(100, Math.round(data.overall_progress_pct)));

    // Hero KPI Strip
    const progressEl = doc.getElementById('kpi-progress-val');
    if (progressEl) progressEl.textContent = `${pct}%`;
    const progressFill = doc.getElementById('kpi-progress-fill');
    if (progressFill) progressFill.style.width = `${pct}%`;

    const milestonesEl = doc.getElementById('kpi-milestones-val');
    if (milestonesEl) milestonesEl.textContent = `${counts.done} / ${counts.total}`;
    const milestonesSub = doc.getElementById('kpi-milestones-sub');
    if (milestonesSub) {
      milestonesSub.textContent = `${counts.inProgress} in progress · ${counts.planned} planned`;
    }

    const budgetEl = doc.getElementById('kpi-budget-val');
    if (budgetEl) {
      budgetEl.textContent = `$${data.budget.spent_usd_est.toFixed(2)} / $${data.budget.cap_usd}`;
    }
    const budgetSub = doc.getElementById('kpi-budget-sub');
    if (budgetSub && data.budget.notes) {
      budgetSub.textContent = data.budget.notes;
    }

    const testsEl = doc.getElementById('kpi-tests-val');
    if (testsEl && data.test_summary) {
      testsEl.textContent = `${data.test_summary.js_tests_passed} JS passing`;
    }
    const testsSub = doc.getElementById('kpi-tests-sub');
    if (testsSub && data.test_summary) {
      testsSub.textContent = `${data.test_summary.js_tests_files} JS files · ${data.test_summary.python_tests_note}`;
    }

    const updatedEl = doc.getElementById('kpi-updated-val');
    if (updatedEl) updatedEl.textContent = formatUtcTimestamp(data.updated_utc);

    // Architecture Cards
    const archGrid = doc.getElementById('arch-grid');
    if (archGrid) {
      archGrid.innerHTML = data.architecture.map(renderArchitectureCard).join('');
    }

    // Milestones Table
    const msTbody = doc.getElementById('milestones-tbody');
    if (msTbody) {
      msTbody.innerHTML = data.milestones
        .map((m) => renderMilestoneRow(m, data.links.github_repo))
        .join('');
    }

    // Benchmarks Tables
    const p1Tbody = doc.getElementById('p1-benchmarks-tbody');
    if (p1Tbody) {
      p1Tbody.innerHTML = data.benchmarks.p1_models.map(renderP1BenchmarkRow).join('');
    }

    const p2aTbody = doc.getElementById('p2-layer-a-tbody');
    if (p2aTbody) {
      p2aTbody.innerHTML = data.benchmarks.p2_layer_a.map(renderP2LayerARow).join('');
    }

    const p2bTbody = doc.getElementById('p2-layer-b-tbody');
    if (p2bTbody) {
      p2bTbody.innerHTML = data.benchmarks.p2_layer_b.map(renderP2LayerBRow).join('');
    }

    // Quirks Ledger & Track Filter
    const quirksGrid = doc.getElementById('quirks-grid');
    const quirksCount = doc.getElementById('quirks-count-badge');
    const filterButtons = Array.from(doc.querySelectorAll('.quirk-filter-btn'));

    const renderQuirksForTrack = (track) => {
      const filtered = filterQuirks(data.quirks_ledger, track);
      if (quirksCount) {
        quirksCount.textContent = `${filtered.length} ${filtered.length === 1 ? 'quirk' : 'quirks'}`;
      }
      if (quirksGrid) {
        quirksGrid.innerHTML = filtered.length > 0
          ? filtered.map(renderQuirkCard).join('')
          : '<div class="empty-state">No engineering quirks logged for this track.</div>';
      }
      for (const btn of filterButtons) {
        const isActive = btn.dataset.track === track;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-pressed', isActive ? 'true' : 'false');
      }
    };

    for (const btn of filterButtons) {
      btn.addEventListener('click', () => {
        renderQuirksForTrack(btn.dataset.track || 'All');
      });
    }
    renderQuirksForTrack('All');

    // Quick Links
    const { primaryHtml, docsHtml } = renderLinksGrid(data.links);
    const linksGrid = doc.getElementById('links-grid');
    if (linksGrid) linksGrid.innerHTML = primaryHtml;
    const docsLinks = doc.getElementById('docs-links');
    if (docsLinks) docsLinks.innerHTML = docsHtml;

    return true;
  } catch (err) {
    showError(doc, `Failed to initialize dashboard: ${err?.message || 'Network error'}`);
    return false;
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    initDashboard(document);
  });
}
