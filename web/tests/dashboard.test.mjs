import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  escapeHtml,
  sanitizeUrl,
  validateDashboardData,
  computeMilestoneCounts,
  filterQuirks,
  formatMetricCell,
  renderMilestoneRow,
  renderP1BenchmarkRow,
  renderP2LayerBRow,
  renderQuirkCard,
} from '../js/dashboard.js';

const dashboardData = JSON.parse(
  fs.readFileSync(new URL('../data/dashboard.json', import.meta.url), 'utf8'),
);

test('dashboard.json passes schema validation and top-level invariants', () => {
  const res = validateDashboardData(dashboardData);
  assert.equal(res.ok, true, `Validation errors: ${res.errors.join(', ')}`);
  assert.equal(dashboardData.schema_version, '1.0');
  assert.ok(!Number.isNaN(Date.parse(dashboardData.updated_utc)));
  assert.ok(dashboardData.overall_progress_pct >= 0 && dashboardData.overall_progress_pct <= 100);
  assert.equal(dashboardData.budget.cap_usd, 30);
  assert.ok(dashboardData.budget.spent_usd_est >= 0);
  assert.ok(dashboardData.budget.spent_usd_est <= dashboardData.budget.cap_usd);
});

test('dashboard.json contains all 8 milestones with valid statuses and consistent progress', () => {
  const expectedIds = ['M1', 'M2', 'M2B', 'M3', 'M4', 'M5', 'M6', 'E2E'];
  const actualIds = dashboardData.milestones.map((m) => m.id);
  assert.deepEqual(actualIds, expectedIds);

  const validStatuses = new Set(['DONE', 'IN_PROGRESS', 'PLANNED']);
  for (const m of dashboardData.milestones) {
    assert.ok(validStatuses.has(m.status), `Invalid status on ${m.id}: ${m.status}`);
    assert.ok(Array.isArray(m.verified_metrics));
  }

  const counts = computeMilestoneCounts(dashboardData.milestones);
  assert.equal(counts.total, 8);
  assert.equal(dashboardData.overall_progress_pct, counts.pct);

  const m2 = dashboardData.milestones.find((m) => m.id === 'M2');
  assert.equal(m2.status, 'DONE');
  assert.equal(m2.commit_sha, '28c0590');
});

test('dashboard.json has zero fabricated numbers on pending benchmark rows', () => {
  assert.equal(dashboardData.benchmarks.p1_models.length, 6);
  for (const row of dashboardData.benchmarks.p1_models) {
    if (row.status === 'pending') {
      assert.equal(row.strict_legal_pct, null);
      assert.equal(row.constrained_legal_pct, null);
      assert.equal(row.acpl, null);
      assert.equal(row.stockfish_score, null);
      assert.equal(row.maia_score, null);
    }
  }

  assert.equal(dashboardData.benchmarks.p2_layer_b.length, 3);
  for (const row of dashboardData.benchmarks.p2_layer_b) {
    if (row.status === 'pending') {
      assert.equal(row.act_eval_faithfulness_pct, null);
      assert.equal(row.spoiler_rate_pct, null);
      assert.equal(row.webgpu_tok_s, null);
    }
  }

  assert.ok(dashboardData.benchmarks.p2_layer_a.length >= 4);
  for (const row of dashboardData.benchmarks.p2_layer_a) {
    assert.equal(row.status, 'measured');
  }
});

test('dashboard.json contains all 8 verified engineering quirks with valid tracks', () => {
  const expectedIds = [
    'QUIRK-001',
    'QUIRK-002',
    'QUIRK-003',
    'QUIRK-004',
    'QUIRK-005',
    'QUIRK-006',
    'QUIRK-007',
    'QUIRK-008',
  ];
  assert.deepEqual(
    dashboardData.quirks_ledger.map((q) => q.id),
    expectedIds,
  );
  const validTracks = new Set(['P1', 'P2', 'Infra']);
  for (const q of dashboardData.quirks_ledger) {
    assert.ok(validTracks.has(q.track), `Invalid track on ${q.id}`);
    assert.ok(q.title && q.impact && q.resolution && q.status);
  }
});

test('dashboard.json links all pass strict URL sanitization', () => {
  const { links } = dashboardData;
  for (const key of ['github_repo', 'live_coach', 'wandb_project', 'huggingface']) {
    assert.notEqual(sanitizeUrl(links[key]), '#', `Unsafe link for ${key}`);
  }
  assert.ok(Array.isArray(links.docs) && links.docs.length >= 5);
  for (const doc of links.docs) {
    assert.ok(doc.label);
    assert.notEqual(sanitizeUrl(doc.url), '#', `Unsafe doc link for ${doc.label}`);
  }
});

test('escapeHtml neutralizes XSS vectors across all special characters', () => {
  const payload = `<img src=x onerror="alert('xss')">&`;
  const escaped = escapeHtml(payload);
  assert.equal(escaped, '&lt;img src=x onerror=&quot;alert(&#39;xss&#39;)&quot;&gt;&amp;');
  assert.equal(escapeHtml(null), '');
  assert.equal(escapeHtml(undefined), '');
});

test('sanitizeUrl allows safe protocols and blocks dangerous schemes', () => {
  assert.equal(sanitizeUrl('https://github.com/ArmaanSethi/chess-fm'), 'https://github.com/ArmaanSethi/chess-fm');
  assert.equal(sanitizeUrl('http://127.0.0.1:8000/'), 'http://127.0.0.1:8000/');
  assert.equal(sanitizeUrl('./index.html'), './index.html');
  assert.equal(sanitizeUrl('#benchmarks'), '#benchmarks');

  assert.equal(sanitizeUrl('javascript:alert(1)'), '#');
  assert.equal(sanitizeUrl('JAVASCRIPT:alert(1)'), '#');
  assert.equal(sanitizeUrl('data:text/html,<script>alert(1)</script>'), '#');
  assert.equal(sanitizeUrl('vbscript:msgbox(1)'), '#');
  assert.equal(sanitizeUrl('//evil.example/payload'), '#');
  assert.equal(sanitizeUrl('https://ok.com/\u0000bad'), '#');
  assert.equal(sanitizeUrl(null), '#');
});

test('computeMilestoneCounts and filterQuirks return accurate aggregates', () => {
  const counts = computeMilestoneCounts([
    { status: 'DONE' },
    { status: 'DONE' },
    { status: 'IN_PROGRESS' },
    { status: 'PLANNED' },
  ]);
  assert.deepEqual(counts, { total: 4, done: 2, inProgress: 1, planned: 1, pct: 50 });
  assert.deepEqual(computeMilestoneCounts([]), { total: 0, done: 0, inProgress: 0, planned: 0, pct: 0 });

  assert.equal(filterQuirks(dashboardData.quirks_ledger, 'All').length, 8);
  assert.equal(filterQuirks(dashboardData.quirks_ledger, 'P1').length, 3);
  assert.equal(filterQuirks(dashboardData.quirks_ledger, 'P2').length, 2);
  assert.equal(filterQuirks(dashboardData.quirks_ledger, 'Infra').length, 3);
});

test('formatMetricCell handles null, finite numbers, and non-finite edge cases', () => {
  assert.ok(formatMetricCell(null).includes('&mdash;'));
  assert.ok(formatMetricCell(undefined).includes('&mdash;'));
  assert.ok(formatMetricCell(Number.NaN).includes('&mdash;'));
  assert.equal(formatMetricCell(98.45, '%'), '98.5%');
  assert.equal(formatMetricCell(42, ' cp'), '42 cp');
});

test('row and card renderers escape untrusted dynamic strings safely', () => {
  const xss = '<script>alert(1)</script>';
  const msRow = renderMilestoneRow(
    {
      id: 'M9',
      name: xss,
      track: 'P1',
      status: 'PLANNED',
      commit_sha: null,
      summary: xss,
      verified_metrics: [xss],
    },
    'javascript:alert(1)',
  );
  assert.ok(!msRow.includes('<script>'));
  assert.ok(msRow.includes('&lt;script&gt;'));

  const p1Row = renderP1BenchmarkRow({
    stage: xss,
    model: 'Qwen/Qwen3-4B',
    status: 'pending',
    strict_legal_pct: null,
    constrained_legal_pct: null,
    acpl: null,
    stockfish_score: null,
    maia_score: null,
    notes: 'Pending (M3)',
  });
  assert.ok(!p1Row.includes('<script>'));

  const p2bRow = renderP2LayerBRow({
    stage: xss,
    model: 'Qwen/Qwen3.5-0.8B',
    status: 'pending',
    act_eval_faithfulness_pct: null,
    spoiler_rate_pct: null,
    webgpu_tok_s: null,
    notes: 'Pending (M4)',
  });
  assert.ok(!p2bRow.includes('<script>'));

  const quirkHtml = renderQuirkCard({
    id: 'QUIRK-999',
    track: 'Infra',
    subsystem: xss,
    title: xss,
    impact: xss,
    resolution: xss,
    status: 'Fixed',
  });
  assert.ok(!quirkHtml.includes('<script>'));
});

test('validateDashboardData rejects malformed payloads cleanly', () => {
  assert.equal(validateDashboardData(null).ok, false);
  assert.equal(validateDashboardData([]).ok, false);
  const bad = structuredClone(dashboardData);
  bad.overall_progress_pct = 150;
  bad.updated_utc = 'not-a-date';
  bad.milestones[0].status = 'INVALID';
  bad.quirks_ledger[0].track = 'UNKNOWN';
  const res = validateDashboardData(bad);
  assert.equal(res.ok, false);
  assert.ok(res.errors.length >= 4);
});
