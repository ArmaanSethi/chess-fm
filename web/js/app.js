/**
 * ChessFM Coach Layer A — Standalone Static Browser Application Controller.
 *
 * Wires Stockfish 19 WASM, Maia-2 (instant heuristic + opt-in ONNX), Lichess/Chess.com
 * direct browser importers, 2 Critical Moments review, 43-line ECO Opening Coach,
 * SM-2 Blunder Drills, 3-level 0%-spoiler Socratic hints, ACT-Eval claim verifier,
 * and optional Layer B WebGPU dynamic import.
 */

import { Chess } from '../vendor/chess.mjs';
import {
  StockfishWorker,
  winProbFromCp,
  formatEvalScore,
  uciToMoveObj,
} from './engine.js';
import { MaiaClient } from './maia.js';
import {
  fetchLichessGames,
  fetchChessComGames,
  parsePgnGames,
} from './importers.js';
import {
  reviewGame,
  WEAKNESS_METADATA,
} from './review.js';
import { OpeningBook } from './openings.js';
import { DrillDeck } from './drills.js';
import {
  generateSocraticHint,
  containsMoveSpoiler,
} from './socratic.js';
import { verifyClaims } from './verifier.js';

const UNICODE_PIECES = {
  wp: '\u2659', wn: '\u2658', wb: '\u2657', wr: '\u2656', wq: '\u2655', wk: '\u2654',
  bp: '\u265F', bn: '\u265E', bb: '\u265D', br: '\u265C', bq: '\u265B', bk: '\u265A',
};

const SAMPLE_GAME_PGN = `[Event "ChessFM Sample Italian Game"]
[Site "https://lichess.org"]
[Date "2026.11.20"]
[White "Student"]
[Black "Opponent"]
[Result "0-1"]
[WhiteElo "1400"]
[BlackElo "1420"]
[ECO "C55"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. d3 h6 7. Nxf7 Kxf7 8. Qf3+ Ke6 9. Nc3 Nb4 10. Bb3 c6 11. O-O Qf6 12. Qe4 Qf5 13. Qc4 b5 14. Nxb5 cxb5 15. Qxb5 Bd7 16. Qa5 Bc6 0-1`;

const state = {
  elo: 1400,
  board: new Chess(),
  flipped: false,
  selectedSq: null,
  lastMove: null,
  inspectedMistakeUci: null,
  latestAnalysis: null,
  latestMaia: null,
  importedGames: [],
  selectedGameIdx: 0,
  lastReview: null,
  reviewAbort: null,
  activeDrill: null,
  drillBoard: new Chess(),
  drillSelectedSq: null,
  drillLastMove: null,
  webgpuCoach: null,
  webgpuEnabled: false,
};

const stockfish = new StockfishWorker();
const maia = new MaiaClient();
const deck = new DrillDeck();
let openingBook = new OpeningBook([]);

const $ = (id) => document.getElementById(id);

function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Chessboard & SVG Arrow Renderer
// ---------------------------------------------------------------------------

function squareCoords(sq, flipped) {
  const file = sq.charCodeAt(0) - 97;
  const rank = Number.parseInt(sq[1], 10) - 1;
  const col = flipped ? 7 - file : file;
  const row = flipped ? rank : 7 - rank;
  return { x: (col + 0.5) * 100, y: (row + 0.5) * 100 };
}

function renderArrows(svgEl, arrows, flipped) {
  if (!svgEl) return;
  const defs = arrows
    .map(
      (a, idx) =>
        `<marker id="ah-${svgEl.id}-${idx}" markerWidth="4" markerHeight="4" refX="2.5" refY="2" orient="auto">` +
        `<polygon points="0 0, 4 2, 0 4" fill="${a.color}" /></marker>`,
    )
    .join('');

  const lines = arrows
    .map((a, idx) => {
      if (!a.uci || a.uci.length < 4) return '';
      const from = squareCoords(a.uci.slice(0, 2), flipped);
      const to = squareCoords(a.uci.slice(2, 4), flipped);
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const len = Math.hypot(dx, dy) || 1;
      const shrink = 22;
      const x2 = to.x - (dx / len) * shrink;
      const y2 = to.y - (dy / len) * shrink;
      return `<line x1="${from.x}" y1="${from.y}" x2="${x2}" y2="${y2}" stroke="${a.color}" stroke-width="${a.width || 14}" stroke-opacity="${a.opacity || 0.82}" stroke-linecap="round" marker-end="url(#ah-${svgEl.id}-${idx})" />`;
    })
    .join('');

  svgEl.innerHTML = `<defs>${defs}</defs>${lines}`;
}

function renderBoardGrid(containerEl, chessInstance, options) {
  const { flipped, selectedSq, lastMove, onSquareClick } = options;
  const legalTargets = new Map();
  if (selectedSq) {
    for (const m of chessInstance.moves({ square: selectedSq, verbose: true })) {
      legalTargets.set(m.to, Boolean(m.captured));
    }
  }

  containerEl.innerHTML = '';
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

  for (let r = 0; r < 8; r += 1) {
    for (let c = 0; c < 8; c += 1) {
      const fileIdx = flipped ? 7 - c : c;
      const rankIdx = flipped ? r : 7 - r;
      const sq = `${files[fileIdx]}${rankIdx + 1}`;
      const isLight = (fileIdx + rankIdx) % 2 === 1;

      const cell = document.createElement('div');
      cell.className = `sq ${isLight ? 'sq-light' : 'sq-dark'}`;
      cell.dataset.square = sq;

      if (sq === selectedSq) cell.classList.add('selected');
      if (lastMove && (sq === lastMove.from || sq === lastMove.to)) {
        cell.classList.add('last-move');
      }
      if (legalTargets.has(sq)) {
        cell.classList.add(legalTargets.get(sq) ? 'legal-capture' : 'legal-target');
      }

      if (c === 0) {
        const rankSpan = document.createElement('span');
        rankSpan.className = 'coord-rank';
        rankSpan.textContent = String(rankIdx + 1);
        cell.appendChild(rankSpan);
      }
      if (r === 7) {
        const fileSpan = document.createElement('span');
        fileSpan.className = 'coord-file';
        fileSpan.textContent = files[fileIdx];
        cell.appendChild(fileSpan);
      }

      const piece = chessInstance.get(sq);
      if (piece) {
        const span = document.createElement('span');
        span.className = `piece piece-${piece.color}`;
        span.textContent = UNICODE_PIECES[`${piece.color}${piece.type}`] || '';
        cell.appendChild(span);
      }

      cell.addEventListener('click', () => onSquareClick(sq));
      containerEl.appendChild(cell);
    }
  }
}

// ---------------------------------------------------------------------------
// Tab Navigation
// ---------------------------------------------------------------------------

function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    const active = btn.dataset.tab === tabName;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `tab-${tabName}`);
  });
}

// ---------------------------------------------------------------------------
// Coach Board Controller
// ---------------------------------------------------------------------------

let boardAnalysisToken = 0;

function updateCoachBoardUI() {
  renderBoardGrid($('coach-board'), state.board, {
    flipped: state.flipped,
    selectedSq: state.selectedSq,
    lastMove: state.lastMove,
    onSquareClick: handleCoachSquareClick,
  });

  $('fen-input').value = state.board.fen();
  const turnText = state.board.isCheckmate()
    ? 'Checkmate'
    : state.board.isDraw()
      ? 'Draw'
      : state.board.turn() === 'w'
        ? 'White to move'
        : 'Black to move';
  $('board-turn-badge').textContent = turnText;

  updateLiveOpeningCard();
  refreshBoardArrows();
}

function refreshBoardArrows() {
  const arrows = [];
  if (state.inspectedMistakeUci) {
    arrows.push({ uci: state.inspectedMistakeUci, color: '#f85149', width: 14, opacity: 0.78 });
  }
  if (state.latestMaia?.top_moves?.[0]) {
    arrows.push({ uci: state.latestMaia.top_moves[0].move, color: '#a371f7', width: 12, opacity: 0.72 });
  }
  if (state.latestAnalysis?.bestMove) {
    arrows.push({ uci: state.latestAnalysis.bestMove, color: '#3fb950', width: 15, opacity: 0.85 });
  }
  renderArrows($('coach-arrows'), arrows, state.flipped);
}

function handleCoachSquareClick(sq) {
  const piece = state.board.get(sq);
  if (state.selectedSq) {
    if (state.selectedSq === sq) {
      state.selectedSq = null;
      updateCoachBoardUI();
      return;
    }
    try {
      const moved = state.board.move({ from: state.selectedSq, to: sq, promotion: 'q' });
      if (moved) {
        state.lastMove = { from: moved.from, to: moved.to };
        state.selectedSq = null;
        state.inspectedMistakeUci = null;
        updateCoachBoardUI();
        analyzeCurrentBoard();
        return;
      }
    } catch {
      // fall through to reselect
    }
  }

  if (piece && piece.color === state.board.turn()) {
    state.selectedSq = sq;
  } else {
    state.selectedSq = null;
  }
  updateCoachBoardUI();
}

async function analyzeCurrentBoard() {
  const token = ++boardAnalysisToken;
  const fen = state.board.fen();
  state.latestAnalysis = null;
  state.latestMaia = null;
  refreshBoardArrows();
  $('maia-elo-label').textContent = String(state.elo);

  // Instant Maia-2 prediction (heuristic or ONNX)
  const maiaPred = await maia.predict(fen, state.elo, { topK: 5 });
  if (token !== boardAnalysisToken) return;
  state.latestMaia = { ...maiaPred, fen };
  renderMaiaList(maiaPred);
  refreshBoardArrows();

  // Stockfish MultiPV 3 analysis
  if (!stockfish.ready) {
    $('multipv-list').innerHTML = '<p class="muted-text">Waiting for Stockfish 19 WASM&hellip;</p>';
    return;
  }

  $('multipv-list').innerHTML = '<p class="muted-text">Analyzing MultiPV 3&hellip;</p>';
  try {
    const analysis = await stockfish.analyseMultiPV(fen, { depth: 12, multipv: 3, pvPlies: 6 });
    if (token !== boardAnalysisToken) return;
    state.latestAnalysis = { ...analysis, fen };
    renderMultiPV(analysis);
    refreshBoardArrows();
  } catch (err) {
    if (token !== boardAnalysisToken) return;
    $('multipv-list').innerHTML = `<p class="muted-text">Engine unavailable: ${escapeHtml(err.message)}</p>`;
  }
}

function renderMultiPV(analysis) {
  const container = $('multipv-list');
  const isBlack = analysis.sideToMove === 'b';

  if (!analysis.topLines || analysis.topLines.length === 0) {
    container.innerHTML = '<p class="muted-text">No legal moves in this position.</p>';
    $('sf-depth-badge').textContent = 'Terminal';
    return;
  }

  const top = analysis.topLines[0];
  $('sf-depth-badge').textContent = `Depth ${top.depth}`;

  const whiteCp = isBlack ? -top.evalCp : top.evalCp;
  const whiteMate = top.mate !== null ? (isBlack ? -top.mate : top.mate) : null;
  const whiteWinPct = Math.max(5, Math.min(95, winProbFromCp(whiteCp)));
  $('eval-fill').style.height = `${whiteWinPct}%`;
  $('eval-label').textContent = formatEvalScore(whiteCp, whiteMate);

  container.innerHTML = '';
  for (const line of analysis.topLines) {
    const lineWhiteCp = isBlack ? -line.evalCp : line.evalCp;
    const lineWhiteMate = line.mate !== null ? (isBlack ? -line.mate : line.mate) : null;
    const row = document.createElement('div');
    row.className = 'pv-row';
    row.innerHTML = `<span class="pv-score">${escapeHtml(formatEvalScore(lineWhiteCp, lineWhiteMate))}</span><span>${escapeHtml(line.pvSan.join(' '))}</span>`;
    row.addEventListener('click', () => {
      const obj = uciToMoveObj(line.move);
      if (!obj) return;
      const moved = state.board.move(obj);
      if (moved) {
        state.lastMove = { from: moved.from, to: moved.to };
        state.inspectedMistakeUci = null;
        updateCoachBoardUI();
        analyzeCurrentBoard();
      }
    });
    container.appendChild(row);
  }
}

function renderMaiaList(pred) {
  $('maia-backend-label').textContent =
    pred.backend === 'maia2_onnx' ? 'Maia-2 ONNX Neural' : 'Calibrated Heuristic';
  const container = $('maia-list');
  if (!pred.top_moves || pred.top_moves.length === 0) {
    container.innerHTML = '<p class="muted-text">No legal moves.</p>';
    return;
  }
  container.innerHTML = pred.top_moves
    .map((m) => {
      const pct = Math.round(m.prob * 100);
      return `<div class="maia-row">
        <strong>${escapeHtml(m.san)}</strong>
        <div class="bar-track"><div class="bar-fill-purple" style="width:${pct}%"></div></div>
        <span>${pct}%</span>
      </div>`;
    })
    .join('');
}

async function handleSocraticHint(level) {
  const fen = state.board.fen();
  if (state.board.moves().length === 0) {
    $('socratic-output').textContent = 'No legal moves to hint in this position.';
    return;
  }

  const matchingAnalysis = state.latestAnalysis?.fen === fen ? state.latestAnalysis : null;
  const matchingMaia = state.latestMaia?.fen === fen ? state.latestMaia : null;

  if (!matchingAnalysis && !matchingMaia) {
    $('socratic-output').textContent = 'Analyzing position — please try again in a moment.';
    return;
  }

  const bestUci =
    matchingAnalysis?.bestMove ||
    matchingMaia?.top_moves?.[0]?.move;
  if (!bestUci) {
    $('socratic-output').textContent = 'No legal moves to hint in this position.';
    return;
  }

  const pvUci = matchingAnalysis?.topLines?.[0]?.pvUci || [bestUci];
  let hintRes;
  try {
    hintRes = generateSocraticHint(fen, bestUci, level, { pvUci });
  } catch {
    $('socratic-output').textContent = 'Analyzing position — please try again in a moment.';
    return;
  }
  let text = hintRes.hint;
  let sourceLabel = 'Layer A Verified Coach';

  // Optional Layer B WebGPU pass with mandatory ACT-Eval + spoiler verification
  if (state.webgpuEnabled && state.webgpuCoach) {
    try {
      const gen = await state.webgpuCoach.generate({
        fen,
        taskType: `socratic_l${level}`,
        context: { bestUci, pvUci, elo: state.elo },
      });
      if (gen && gen.text && !gen.fallback) {
        const vCheck = verifyClaims(fen, gen.text);
        const leaked = level < 3 && containsMoveSpoiler(gen.text, bestUci, hintRes.move_san);
        if (vCheck.passed && !leaked) {
          text = gen.text;
          sourceLabel = `Layer B WebGPU (${gen.modelId || 'local'})`;
        }
      }
    } catch {
      // Cleanly fall back to Layer A verified hint if Layer B errors
    }
  }

  const verification = verifyClaims(fen, text);
  $('socratic-output').innerHTML = `<div><strong>Level ${level} (${escapeHtml(sourceLabel)}):</strong> ${escapeHtml(text)}</div>`;

  const actBadge = $('act-eval-badge');
  actBadge.textContent = verification.badgeText;
  actBadge.className = `pill ${verification.passed ? 'pill-ok' : 'pill-danger'}`;

  const spoilerBadge = $('spoiler-badge');
  if (level < 3) {
    const clean = !containsMoveSpoiler(text, bestUci, hintRes.move_san);
    spoilerBadge.classList.remove('hidden');
    spoilerBadge.textContent = clean ? '0% Spoiler Guard' : 'Spoiler Flagged';
    spoilerBadge.className = `pill ${clean ? 'pill-ok' : 'pill-danger'}`;
  } else {
    spoilerBadge.classList.add('hidden');
  }
}

// ---------------------------------------------------------------------------
// Game Import & Review Controller
// ---------------------------------------------------------------------------

function setImportStatus(msg, isError = false) {
  const el = $('import-status');
  el.textContent = msg;
  el.classList.toggle('error', isError);
}

function loadGamesIntoUI(games) {
  state.importedGames = games;
  state.selectedGameIdx = 0;
  if (games.length === 0) {
    $('game-list-wrap').classList.add('hidden');
    setImportStatus('No valid PGN games found.', true);
    return;
  }

  $('game-list-wrap').classList.remove('hidden');
  const first = games[0];
  $('student-color-select').value = first.studentColor || 'w';
  renderGameList();
  updateDeviationCard(first.pgn, null);
}

function renderGameList() {
  const ul = $('game-list');
  ul.innerHTML = '';
  state.importedGames.forEach((g, idx) => {
    const li = document.createElement('li');
    li.className = `game-item ${idx === state.selectedGameIdx ? 'selected' : ''}`;
    li.innerHTML = `<span><strong>${escapeHtml(g.white)}</strong> vs <strong>${escapeHtml(g.black)}</strong> (${escapeHtml(g.result)})</span><span class="muted-text">${g.plyCount} plies</span>`;
    li.addEventListener('click', () => {
      state.selectedGameIdx = idx;
      if (g.studentColor) $('student-color-select').value = g.studentColor;
      renderGameList();
      updateDeviationCard(g.pgn, null);
    });
    ul.appendChild(li);
  });
}

async function runGameReview() {
  const game = state.importedGames[state.selectedGameIdx];
  if (!game) return;

  const studentColor = $('student-color-select').value === 'b' ? 'b' : 'w';
  state.reviewAbort = new AbortController();
  $('analyze-game-btn').disabled = true;
  $('cancel-analyze-btn').classList.remove('hidden');
  $('review-progress-wrap').classList.remove('hidden');
  $('review-progress-fill').style.width = '2%';
  $('review-progress-text').textContent = 'Starting analysis...';

  try {
    const result = await reviewGame(game.pgn, {
      studentColor,
      studentElo: state.elo,
      engine: stockfish.ready ? stockfish : null,
      maia,
      depth: 10,
      signal: state.reviewAbort.signal,
      onProgress: ({ current, total, pct }) => {
        $('review-progress-fill').style.width = `${pct}%`;
        $('review-progress-text').textContent = `Evaluating position ${current}/${total} (${pct}%)`;
      },
    });
    state.lastReview = result;
    renderReviewResults(result);
    updateDeviationCard(game.pgn, result.plyAnalyses);
  } catch (err) {
    setImportStatus(err.message || 'Analysis failed.', true);
  } finally {
    $('analyze-game-btn').disabled = false;
    $('cancel-analyze-btn').classList.add('hidden');
    $('review-progress-wrap').classList.add('hidden');
    state.reviewAbort = null;
  }
}

function renderReviewResults(res) {
  $('review-empty').classList.add('hidden');
  $('review-results').classList.remove('hidden');
  $('acc-white').textContent = `${res.whiteAccuracy}%`;
  $('acc-black').textContent = `${res.blackAccuracy}%`;
  $('acc-student').textContent = `${res.studentAccuracy}%`;

  const momentsEl = $('critical-moments-list');
  if (res.criticalMoments.length === 0) {
    momentsEl.innerHTML = '<p class="muted-text">Clean game! No major mistakes (&ge;80 cp loss) detected for your side.</p>';
  } else {
    momentsEl.innerHTML = '';
    for (const m of res.criticalMoments) {
      const card = document.createElement('div');
      card.className = 'moment-card';
      const meta = WEAKNESS_METADATA[m.weaknessCategory] || { label: m.weaknessCategory };
      card.innerHTML = `
        <div class="card-head-row">
          <strong>Move ${m.moveNumber} (${m.mover === 'w' ? 'White' : 'Black'}): ${escapeHtml(m.playedSan)} &rarr; ${escapeHtml(m.bestSan)}</strong>
          ${m.isHumanTrap ? '<span class="pill pill-warn">Human Trap</span>' : `<span class="pill pill-danger">${escapeHtml(m.classification)}</span>`}
        </div>
        <div class="muted-text">
          WDL Swing: <strong>${m.wdlSwing}%</strong> &middot;
          Maia Findability: <strong>${Math.round(m.maiaBestProb * 100)}%</strong> &middot;
          Tag: <strong>${escapeHtml(meta.label)}</strong>
        </div>
        <div class="btn-row">
          <button class="btn btn-sm btn-primary inspect-btn" type="button">Inspect on Board</button>
          <button class="btn btn-sm btn-accent add-drill-btn" type="button">Add to Blunder Drills</button>
        </div>
      `;
      card.querySelector('.inspect-btn').addEventListener('click', () => {
        inspectPositionOnBoard(m.fen, m.playedUci);
      });
      card.querySelector('.add-drill-btn').addEventListener('click', (e) => {
        const { added } = deck.addFromCriticalMoment(m);
        e.target.textContent = added ? 'Added to Drills' : 'Already in Deck';
        e.target.disabled = true;
        refreshDrillsUI();
      });
      momentsEl.appendChild(card);
    }
  }

  // 6-category Weakness Profile
  const weakEl = $('weakness-profile-list');
  const maxSev = Math.max(1, ...res.weaknessProfile.orderedCategories.map((c) => c.severityScore));
  weakEl.innerHTML = res.weaknessProfile.orderedCategories
    .map((c) => {
      const pct = Math.round((c.severityScore / maxSev) * 100);
      return `<div class="weakness-row" title="${escapeHtml(c.recommendation)}">
        <span>${escapeHtml(c.label)}</span>
        <div class="bar-track"><div class="bar-fill-red" style="width:${c.count > 0 ? Math.max(8, pct) : 0}%"></div></div>
        <span>${c.count}x</span>
      </div>`;
    })
    .join('');

  // Move-by-move table
  const tbody = $('review-moves-tbody');
  tbody.innerHTML = '';
  for (const p of res.plyAnalyses) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${p.moveNumber}${p.mover === 'w' ? '.' : '...'}</td>
      <td><strong>${escapeHtml(p.playedSan)}</strong></td>
      <td>${escapeHtml(p.bestSan)}</td>
      <td>${escapeHtml(formatEvalScore(p.evalWhiteCp))}</td>
      <td>${p.cpLoss} cp</td>
      <td class="cls-${escapeHtml(p.classification)}">${escapeHtml(p.classification)}</td>
    `;
    tr.addEventListener('click', () => inspectPositionOnBoard(p.fen, p.playedUci));
    tbody.appendChild(tr);
  }
}

function inspectPositionOnBoard(fen, mistakeUci = null) {
  state.board = new Chess(fen);
  state.selectedSq = null;
  state.lastMove = null;
  state.inspectedMistakeUci = mistakeUci;
  switchTab('board');
  updateCoachBoardUI();
  analyzeCurrentBoard();
}

// ---------------------------------------------------------------------------
// Opening Coach Controller
// ---------------------------------------------------------------------------

function updateLiveOpeningCard() {
  const hit = openingBook.identifyOpening(state.board.fen(), state.board.history());
  if (!hit) {
    $('live-opening-eco').textContent = 'ECO';
    $('live-opening-name').textContent =
      state.board.history().length === 0 ? 'Starting Position' : 'Out of Book / Custom Position';
    $('live-opening-moves').textContent =
      state.board.history().length === 0
        ? 'Play moves on the Coach Board or load any opening below.'
        : state.board.history().join(' ');
    $('live-opening-plans').classList.add('hidden');
    return;
  }

  $('live-opening-eco').textContent = hit.eco;
  $('live-opening-name').textContent = hit.name;
  $('live-opening-moves').textContent = hit.moves.join(' ');
  $('live-white-plan').textContent = hit.white_plan;
  $('live-black-plan').textContent = hit.black_plan;
  $('live-common-trap').textContent = hit.common_trap;
  $('live-opening-plans').classList.remove('hidden');
}

function updateDeviationCard(pgn, evals) {
  if (!openingBook || openingBook.size === 0) return;
  const dev = openingBook.detectDeviation(pgn, evals);
  const card = $('deviation-card');
  if (!dev.opening) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  const dropStr = dev.evalDropCp !== null ? ` (${dev.evalDropCp} cp eval drop)` : '';
  $('deviation-summary').textContent = dev.deviationPly
    ? `${dev.eco} ${dev.name}: Stayed in book through ply ${dev.lastBookPly}; first deviation at ply ${dev.deviationPly} (${dev.deviationMoveSan})${dropStr}.`
    : `${dev.eco} ${dev.name}: Entire game stayed within book (${dev.lastBookPly} plies).`;
}

function renderOpeningsCatalog(query = '') {
  const list = openingBook.search(query);
  const container = $('openings-list');
  container.innerHTML = '';
  for (const entry of list) {
    const div = document.createElement('div');
    div.className = 'opening-item';
    div.innerHTML = `
      <div>
        <span class="eco-tag">${escapeHtml(entry.eco)}</span>
        <h4>${escapeHtml(entry.name)}</h4>
        <p class="muted-text">${escapeHtml(entry.moves.join(' '))}</p>
      </div>
      <button class="btn btn-sm btn-secondary load-opening-btn" type="button">Load Opening on Board</button>
    `;
    div.querySelector('.load-opening-btn').addEventListener('click', () => {
      const b = new Chess();
      for (const san of entry.moves) {
        b.move(san);
      }
      state.board = b;
      state.selectedSq = null;
      state.lastMove = null;
      state.inspectedMistakeUci = null;
      switchTab('board');
      updateCoachBoardUI();
      analyzeCurrentBoard();
    });
    container.appendChild(div);
  }
}

// ---------------------------------------------------------------------------
// Blunder Drills (SM-2) Controller
// ---------------------------------------------------------------------------

function refreshDrillsUI() {
  const stats = deck.getStats();
  $('stat-due').textContent = String(stats.due_cards);
  $('stat-total').textContent = String(stats.total_cards);
  $('stat-mastered').textContent = String(stats.mastered_cards);
  $('stat-rate').textContent = `${Math.round(stats.success_rate * 100)}%`;
  $('drills-due-badge').textContent = String(stats.due_cards);

  const ul = $('drill-card-list');
  ul.innerHTML = '';
  for (const card of deck.cards) {
    const li = document.createElement('li');
    li.className = `drill-item ${state.activeDrill?.card_id === card.card_id ? 'selected' : ''}`;
    const meta = WEAKNESS_METADATA[card.category] || { label: card.category };
    li.innerHTML = `
      <span><strong>${escapeHtml(meta.label)}</strong> &middot; Due ${escapeHtml(card.due_date)} (EF ${card.ease_factor.toFixed(2)})</span>
      <button class="btn btn-sm btn-ghost del-drill-btn" type="button" title="Remove card">&times;</button>
    `;
    li.addEventListener('click', (e) => {
      if (e.target.classList.contains('del-drill-btn')) return;
      loadDrillCard(card);
    });
    li.querySelector('.del-drill-btn').addEventListener('click', () => {
      deck.removeCard(card.card_id);
      if (state.activeDrill?.card_id === card.card_id) {
        state.activeDrill = null;
      }
      refreshDrillsUI();
      loadNextDueDrill();
    });
    ul.appendChild(li);
  }
}

function loadDrillCard(card) {
  state.activeDrill = card;
  state.drillBoard = new Chess(card.fen);
  state.drillSelectedSq = null;
  state.drillLastMove = null;
  const meta = WEAKNESS_METADATA[card.category] || { label: card.category };
  const side = state.drillBoard.turn() === 'w' ? 'White' : 'Black';
  $('drill-prompt-title').textContent = `Find the Best Move for ${side}`;
  $('drill-category-badge').textContent = meta.label;
  $('drill-feedback').textContent = card.note || `Category: ${meta.label}. Play the best move on the board.`;
  renderArrows($('drill-arrows'), [], state.drillBoard.turn() === 'b');
  renderDrillBoard();
  refreshDrillsUI();
}

function loadNextDueDrill() {
  const due = deck.getDueCards();
  if (due.length > 0) {
    loadDrillCard(due[0]);
  } else if (deck.cards.length > 0) {
    loadDrillCard(deck.cards[0]);
  } else {
    state.activeDrill = null;
    state.drillBoard = new Chess();
    renderDrillBoard();
    $('drill-feedback').textContent = 'No drill cards yet — click "Seed Sample Drills" or add critical moments from a reviewed game.';
  }
}

function renderDrillBoard() {
  const flipped = state.activeDrill ? new Chess(state.activeDrill.fen).turn() === 'b' : false;
  renderBoardGrid($('drill-board'), state.drillBoard, {
    flipped,
    selectedSq: state.drillSelectedSq,
    lastMove: state.drillLastMove,
    onSquareClick: handleDrillSquareClick,
  });
}

async function handleDrillSquareClick(sq) {
  if (!state.activeDrill) return;
  const piece = state.drillBoard.get(sq);
  if (state.drillSelectedSq) {
    if (state.drillSelectedSq === sq) {
      state.drillSelectedSq = null;
      renderDrillBoard();
      return;
    }
    try {
      const probe = new Chess(state.activeDrill.fen);
      const moved = probe.move({ from: state.drillSelectedSq, to: sq, promotion: 'q' });
      if (moved) {
        const playedUci = `${moved.from}${moved.to}${moved.promotion || ''}`;
        state.drillSelectedSq = null;
        await gradeDrillMove(playedUci, moved.san);
        return;
      }
    } catch {
      // fall through
    }
  }
  if (piece && piece.color === state.drillBoard.turn()) {
    state.drillSelectedSq = sq;
  } else {
    state.drillSelectedSq = null;
  }
  renderDrillBoard();
}

async function gradeDrillMove(playedUci, playedSan) {
  const card = state.activeDrill;
  if (!card) return;

  let cpLoss = null;
  const isBest = playedUci.toLowerCase() === card.best_move.toLowerCase();
  if (!isBest && stockfish.ready) {
    try {
      const [bestEval, playedEval] = await Promise.all([
        stockfish.evalMove(card.fen, card.best_move, 10),
        stockfish.evalMove(card.fen, playedUci, 10),
      ]);
      cpLoss = Math.max(0, bestEval.evalCp - playedEval.evalCp);
    } catch {
      cpLoss = null;
    }
  }

  const res = deck.recordAttempt(card.card_id, playedUci, cpLoss, isBest);
  state.activeDrill = res.card;
  const flipped = new Chess(card.fen).turn() === 'b';

  if (res.passed) {
    state.drillBoard = new Chess(card.fen);
    const mv = state.drillBoard.move(uciToMoveObj(playedUci));
    state.drillLastMove = mv ? { from: mv.from, to: mv.to } : null;
    renderArrows($('drill-arrows'), [{ uci: playedUci, color: '#3fb950', width: 15, opacity: 0.85 }], flipped);
    $('drill-feedback').innerHTML = `<strong>${escapeHtml(res.label)} (${escapeHtml(playedSan)})!</strong> Next review in ${res.card.interval_days} day(s) (EF ${res.card.ease_factor.toFixed(2)}).`;
  } else {
    renderArrows(
      $('drill-arrows'),
      [
        { uci: playedUci, color: '#f85149', width: 14, opacity: 0.8 },
        { uci: card.best_move, color: '#3fb950', width: 14, opacity: 0.85 },
      ],
      flipped,
    );
    const lossText = res.cp_loss !== null ? ` (~${res.cp_loss} cp loss)` : '';
    $('drill-feedback').innerHTML = `<strong>${escapeHtml(res.label)} (${escapeHtml(playedSan)})${lossText}.</strong> Best move was <code>${escapeHtml(card.best_move)}</code>. Card reset to 1 day.`;
  }

  renderDrillBoard();
  refreshDrillsUI();
}

// ---------------------------------------------------------------------------
// Bootstrap & Event Wiring
// ---------------------------------------------------------------------------

async function initApp() {
  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });

  // Elo selector
  $('elo-select').addEventListener('change', (e) => {
    state.elo = Number.parseInt(e.target.value, 10) || 1400;
    analyzeCurrentBoard();
  });

  // Maia-2 opt-in ONNX upgrade
  $('maia-upgrade-btn').addEventListener('click', async () => {
    const btn = $('maia-upgrade-btn');
    const pill = $('maia-pill');
    btn.disabled = true;
    pill.textContent = 'Downloading ONNX...';
    pill.className = 'pill pill-warn';

    const ok = await maia.loadOnnx({
      onProgress: ({ phase, pct }) => {
        pill.textContent =
          phase === 'download'
            ? `Downloading ONNX... ${pct}%`
            : phase === 'ready'
              ? 'Maia-2: ONNX Ready'
              : `Maia-2: ${phase}...`;
      },
    });

    if (ok) {
      pill.textContent = 'Maia-2: ONNX Ready';
      pill.className = 'pill pill-ok';
      btn.classList.add('hidden');
      analyzeCurrentBoard();
    } else {
      pill.textContent = 'Maia-2: Calibrated Heuristic';
      pill.className = 'pill pill-info';
      btn.disabled = false;
      const banner = $('webgpu-banner');
      banner.textContent = `Maia-2 ONNX upgrade unavailable (${maia.error || 'offline'}) — continuing with instant calibrated heuristic.`;
      banner.classList.remove('hidden');
    }
  });

  // Optional Layer B WebGPU dynamic import
  $('webgpu-toggle').addEventListener('change', async (e) => {
    const banner = $('webgpu-banner');
    if (!e.target.checked) {
      state.webgpuEnabled = false;
      banner.classList.add('hidden');
      return;
    }
    try {
      const mod = await import('./webgpu_coach.js');
      if (!state.webgpuCoach) {
        state.webgpuCoach = new mod.WebGPUCoach();
      }
      const res = await state.webgpuCoach.init();
      state.webgpuEnabled = res.ready;
      banner.textContent = res.message;
      banner.classList.remove('hidden');
    } catch (err) {
      state.webgpuEnabled = false;
      banner.textContent = `WebGPU coach unavailable (${err.message}) — using verified Layer A deterministic coach.`;
      banner.classList.remove('hidden');
    }
  });

  // Importers & Sample Game
  $('import-lichess-btn').addEventListener('click', async () => {
    const user = $('lichess-user').value.trim();
    setImportStatus(`Fetching recent Lichess games for ${user}...`);
    try {
      const games = await fetchLichessGames(user, { max: 10 });
      setImportStatus(`Loaded ${games.length} Lichess game(s) for ${user}.`);
      loadGamesIntoUI(games);
    } catch (err) {
      setImportStatus(err.message, true);
    }
  });

  $('import-chesscom-btn').addEventListener('click', async () => {
    const user = $('chesscom-user').value.trim();
    setImportStatus(`Fetching recent Chess.com games for ${user}...`);
    try {
      const games = await fetchChessComGames(user, { max: 10 });
      setImportStatus(`Loaded ${games.length} Chess.com game(s) for ${user}.`);
      loadGamesIntoUI(games);
    } catch (err) {
      setImportStatus(err.message, true);
    }
  });

  $('load-pgn-btn').addEventListener('click', () => {
    const text = $('pgn-input').value.trim();
    if (!text) {
      setImportStatus('Paste a PGN first or click Load Sample Game.', true);
      return;
    }
    const games = parsePgnGames(text).filter((g) => g.valid && g.plyCount > 0);
    if (games.length === 0) {
      loadGamesIntoUI([]);
      return;
    }
    setImportStatus(`Parsed ${games.length} PGN game(s).`);
    loadGamesIntoUI(games);
  });

  $('load-sample-btn').addEventListener('click', () => {
    $('pgn-input').value = SAMPLE_GAME_PGN;
    const games = parsePgnGames(SAMPLE_GAME_PGN);
    games[0].studentColor = 'w';
    setImportStatus('Loaded sample Italian Game (Student vs Opponent). Click "Analyze Game"!');
    loadGamesIntoUI(games);
  });

  $('analyze-game-btn').addEventListener('click', runGameReview);
  $('cancel-analyze-btn').addEventListener('click', () => {
    if (state.reviewAbort) state.reviewAbort.abort();
  });

  // Coach Board Controls
  $('board-flip-btn').addEventListener('click', () => {
    state.flipped = !state.flipped;
    updateCoachBoardUI();
  });
  $('board-undo-btn').addEventListener('click', () => {
    state.board.undo();
    state.selectedSq = null;
    state.lastMove = null;
    state.inspectedMistakeUci = null;
    updateCoachBoardUI();
    analyzeCurrentBoard();
  });
  $('board-reset-btn').addEventListener('click', () => {
    state.board = new Chess();
    state.selectedSq = null;
    state.lastMove = null;
    state.inspectedMistakeUci = null;
    updateCoachBoardUI();
    analyzeCurrentBoard();
  });
  $('fen-load-btn').addEventListener('click', () => {
    try {
      state.board = new Chess($('fen-input').value.trim());
      state.selectedSq = null;
      state.lastMove = null;
      state.inspectedMistakeUci = null;
      updateCoachBoardUI();
      analyzeCurrentBoard();
    } catch {
      $('socratic-output').textContent = 'Invalid FEN string.';
    }
  });
  $('fen-copy-btn').addEventListener('click', () => {
    navigator.clipboard?.writeText(state.board.fen()).catch(() => {});
  });

  // Socratic Hint Buttons
  $('hint-l1-btn').addEventListener('click', () => handleSocraticHint(1));
  $('hint-l2-btn').addEventListener('click', () => handleSocraticHint(2));
  $('hint-l3-btn').addEventListener('click', () => handleSocraticHint(3));

  // Opening search
  $('opening-search').addEventListener('input', (e) => renderOpeningsCatalog(e.target.value));

  // Drills controls
  $('seed-drills-btn').addEventListener('click', () => {
    const added = deck.seedSampleDrills();
    refreshDrillsUI();
    if (!state.activeDrill || added > 0) loadNextDueDrill();
  });
  $('drill-next-btn').addEventListener('click', loadNextDueDrill);
  $('drill-hint-l1-btn').addEventListener('click', () => {
    if (!state.activeDrill) return;
    const h = generateSocraticHint(state.activeDrill.fen, state.activeDrill.best_move, 1);
    $('drill-feedback').textContent = `L1 Nudge (0% Spoiler): ${h.hint}`;
  });
  $('drill-hint-l2-btn').addEventListener('click', () => {
    if (!state.activeDrill) return;
    const h = generateSocraticHint(state.activeDrill.fen, state.activeDrill.best_move, 2);
    $('drill-feedback').textContent = `L2 Clue (0% Spoiler): ${h.hint}`;
  });
  $('drill-solution-btn').addEventListener('click', () => {
    if (!state.activeDrill) return;
    const h = generateSocraticHint(state.activeDrill.fen, state.activeDrill.best_move, 3);
    const flipped = new Chess(state.activeDrill.fen).turn() === 'b';
    renderArrows($('drill-arrows'), [{ uci: state.activeDrill.best_move, color: '#3fb950', width: 15, opacity: 0.85 }], flipped);
    $('drill-feedback').textContent = `Solution: ${h.hint}`;
  });

  // Load 43-line ECO Opening Book
  try {
    openingBook = await OpeningBook.fromUrl('./data/openings.json');
    $('openings-count-badge').textContent = String(openingBook.size);
    renderOpeningsCatalog('');
  } catch (err) {
    console.error('Failed to load openings.json:', err);
  }

  // Initial board & drills render
  updateCoachBoardUI();
  refreshDrillsUI();
  loadNextDueDrill();

  // Boot Stockfish 19 WASM Worker
  try {
    await stockfish.init();
    $('sf-pill').textContent = 'Stockfish 19 WASM: Ready';
    $('sf-pill').className = 'pill pill-ok';
    analyzeCurrentBoard();
  } catch (err) {
    $('sf-pill').textContent = 'Stockfish WASM: Fallback';
    $('sf-pill').className = 'pill pill-warn';
    console.warn('Stockfish init failed:', err);
    analyzeCurrentBoard();
  }
}

initApp();
