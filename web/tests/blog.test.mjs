import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  escapeHtml,
  parseFenToSquares,
  renderChessboardSvg,
  highlightSlopTerms,
  buildPassAtKSvg,
  buildProbeTrajectorySvg,
  buildHeadMatrixSvg,
  buildSvdSpectrumSvg,
  buildLogitLensSvg,
} from '../js/blog.js';

const blogData = JSON.parse(
  fs.readFileSync(new URL('../data/blog_data.json', import.meta.url), 'utf8'),
);
const blogHtml = fs.readFileSync(new URL('../blog.html', import.meta.url), 'utf8');

test('blog_data.json has schema 1.0 and verified non-NaN headline metrics', () => {
  assert.equal(blogData.schema_version, '1.0');
  assert.equal(blogData.hero_kpis.coach_all_clean_pct, 92.0);
  assert.equal(blogData.hero_kpis.apawn_before_pct, 60.3);
  assert.equal(blogData.hero_kpis.apawn_after_pct, 1.5);
  assert.equal(blogData.hero_kpis.sabotage_flip_rate_pct, 100.0);
  assert.equal(blogData.rlvr_apawn_hack.board_examples.length, 5);
  assert.equal(blogData.e5_coach.rows.length, 7);
});

test('blog.html is zero-CDN, semantic, and includes collapsible deep dives', () => {
  assert.ok(!blogHtml.includes('fonts.googleapis.com'));
  assert.ok(!blogHtml.includes('cdn.jsdelivr.net'));
  assert.ok(!blogHtml.includes('unpkg.com'));
  assert.ok(blogHtml.includes('<details class="deep-dive">'));
  assert.ok(blogHtml.includes('./data/blog_data.json') || blogHtml.includes('./js/blog.js'));
  assert.equal((blogHtml.match(/<h1[\s>]/g) || []).length, 1);
});

test('parseFenToSquares and renderChessboardSvg produce valid SVG with arrows', () => {
  const ex = blogData.rlvr_apawn_hack.board_examples[0];
  const sq = parseFenToSquares(ex.fen);
  assert.equal(sq.e1, 'K');
  assert.equal(sq.a2, 'P');
  assert.equal(sq.c1, 'B');
  const svg = renderChessboardSvg(ex.fen, {
    highlights: { a2: '#C83E2B', c1: '#147A53' },
    arrows: [
      { from: 'a2', to: 'a3', color: '#C83E2B' },
      { from: 'c1', to: 'd2', color: '#147A53' },
    ],
  });
  assert.ok(svg.startsWith('<svg '));
  assert.ok(svg.includes('class="anim-arrow"'));
  assert.ok(svg.endsWith('</svg>'));
});

test('escapeHtml and highlightSlopTerms sanitize XSS and mark boilerplate phrases', () => {
  assert.equal(escapeHtml('<script>x</script>'), '&lt;script&gt;x&lt;/script&gt;');
  const out = highlightSlopTerms('Consider the tension in the center <img onerror=1>');
  assert.ok(out.includes('<span class="hl-slop">tension in the center</span>'));
  assert.ok(!out.includes('<img'));
});

test('all inline SVG chart builders render valid, NaN-free SVG from blog_data.json', () => {
  const svgs = [
    buildPassAtKSvg(blogData.pass_at_k),
    buildProbeTrajectorySvg(blogData.interp.interp1_probes),
    buildHeadMatrixSvg(blogData.interp.interp4_piece_tracker, 'top5'),
    buildHeadMatrixSvg(blogData.interp.interp4_piece_tracker, 'random5'),
    buildSvdSpectrumSvg(blogData.interp.interp2_svd_slop, 'play'),
    buildSvdSpectrumSvg(blogData.interp.interp2_svd_slop, 'coach'),
    buildLogitLensSvg(blogData.interp.interp3_cot_sabotage),
  ];
  for (const svg of svgs) {
    assert.ok(svg.startsWith('<svg ') && svg.endsWith('</svg>'));
    assert.ok(!svg.includes('NaN'), `Found NaN in SVG output: ${svg.slice(0, 200)}`);
    assert.ok(!svg.includes('undefined'), `Found undefined in SVG output: ${svg.slice(0, 200)}`);
  }
});
