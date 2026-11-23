/**
 * Maia-2 Human-Elo Move Calibration & Trap Detection (Layer A).
 *
 * Boots immediately in deterministic ``calibratedFallback`` mode (exact port of
 * ``src/chessfm/coach/maia_sidecar.py``) and supports opt-in progressive upgrade
 * to the MIT-licensed Maia-2 ONNX model (``maia_rapid.onnx``) via
 * ``onnxruntime-web@1.20.1`` with ``numThreads = 1`` (zero COOP/COEP headers)
 * and IndexedDB caching.
 */

import { Chess } from '../vendor/chess.mjs';

export const ELO_BRACKETS = [600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600];

export const MAIA2_ONNX_URL =
  'https://raw.githubusercontent.com/CSSLab/maia-platform-frontend/c2afeeaa96c7664dc9bb81954960b75ede7e4c20/public/maia2/maia_rapid.onnx';
export const ORT_CDN_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs';
export const ORT_WASM_PATHS = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const PIECE_VALUES = { p: 1.0, n: 3.0, b: 3.2, r: 5.0, q: 9.0, k: 0.0 };
const CENTER_SQUARES = new Set(['d4', 'e4', 'd5', 'e5']);
const EXTENDED_CENTER = new Set([
  'c3', 'd3', 'e3', 'f3',
  'c4', 'f4',
  'c5', 'f5',
  'c6', 'd6', 'e6', 'f6',
]);
const PIECE_PLANES = {
  P: 0, N: 1, B: 2, R: 3, Q: 4, K: 5,
  p: 6, n: 7, b: 8, r: 9, q: 10, k: 11,
};

/**
 * Snap an arbitrary Elo rating to the nearest supported bracket (ties go lower).
 *
 * @param {number} elo
 * @returns {number}
 */
export function snapEloBracket(elo) {
  const num = Number(elo) || 1400;
  let best = ELO_BRACKETS[0];
  let bestDist = Math.abs(num - best);
  for (let i = 1; i < ELO_BRACKETS.length; i += 1) {
    const dist = Math.abs(num - ELO_BRACKETS[i]);
    if (dist < bestDist) {
      best = ELO_BRACKETS[i];
      bestDist = dist;
    }
  }
  return best;
}

/**
 * Map a student Elo to a normalized skill factor in [0, 1].
 *
 * @param {number} studentElo
 * @returns {number}
 */
export function eloSkillFactor(studentElo) {
  const clamped = Math.max(600, Math.min(2600, Number(studentElo) || 1400));
  return (clamped - 600) / 2000.0;
}

/**
 * Map Elo to Maia-2's 11 discrete rating buckets (0: <1100, 1..9: 1100..1999, 10: >=2000).
 *
 * @param {number} elo
 * @returns {number}
 */
export function eloToBucket(elo) {
  const n = Number(elo);
  if (Number.isNaN(n) || n < 1100) return 0;
  if (n >= 2000) return 10;
  return Math.floor((n - 1100) / 100) + 1;
}

/**
 * Vertically mirror a single algebraic square (e.g. ``e2`` <-> ``e7``).
 *
 * @param {string} sq
 * @returns {string}
 */
export function mirrorSquare(sq) {
  const file = sq[0];
  const rank = 9 - Number.parseInt(sq[1], 10);
  return `${file}${rank}`;
}

/**
 * Vertically mirror a UCI move string (e.g. ``e2e4`` <-> ``e7e5``, ``a7a8q`` <-> ``a2a1q``).
 *
 * @param {string} uci
 * @returns {string}
 */
export function mirrorUci(uci) {
  if (typeof uci !== 'string' || uci.length < 4) return uci;
  const from = mirrorSquare(uci.slice(0, 2));
  const to = mirrorSquare(uci.slice(2, 4));
  const promo = uci.length > 4 ? uci.slice(4) : '';
  return `${from}${to}${promo}`;
}

/**
 * Mirror a FEN vertically and swap piece colors, turn, castling rights, and en passant.
 * Used to canonicalize Black-to-move positions into White-to-move perspective for Maia-2.
 *
 * @param {string} fen
 * @returns {string}
 */
export function mirrorFen(fen) {
  const parts = fen.trim().split(/\s+/);
  const ranks = parts[0].split('/').reverse();
  const swappedRanks = ranks.map((r) =>
    r.replace(/[a-zA-Z]/g, (ch) => (ch === ch.toUpperCase() ? ch.toLowerCase() : ch.toUpperCase())),
  );

  const turn = parts[1] === 'w' ? 'b' : 'w';

  let castling = '-';
  if (parts[2] && parts[2] !== '-') {
    const swapped = parts[2].replace(/[KQkq]/g, (ch) =>
      ch === ch.toUpperCase() ? ch.toLowerCase() : ch.toUpperCase(),
    );
    const ordered = ['K', 'Q', 'k', 'q'].filter((c) => swapped.includes(c)).join('');
    castling = ordered || '-';
  }

  let ep = '-';
  if (parts[3] && parts[3] !== '-' && /^[a-h][1-8]$/.test(parts[3])) {
    ep = mirrorSquare(parts[3]);
  }

  const halfmove = parts[4] ?? '0';
  const fullmove = parts[5] ?? '1';
  return `${swappedRanks.join('/')} ${turn} ${castling} ${ep} ${halfmove} ${fullmove}`;
}

/**
 * Encode a FEN into Maia-2's 18x8x8 float32 board tensor (length 1,152).
 * Automatically mirrors Black-to-move positions into White-to-move perspective.
 *
 * @param {string} fen
 * @returns {Float32Array}
 */
export function encodeBoardPlanes(fen) {
  const rawParts = fen.trim().split(/\s+/);
  const canonicalFen = rawParts[1] === 'b' ? mirrorFen(fen) : fen;
  const parts = canonicalFen.trim().split(/\s+/);

  const tensor = new Float32Array(18 * 64);
  const ranks = parts[0].split('/');

  for (let rankIdx = 0; rankIdx < 8; rankIdx += 1) {
    const row = 7 - rankIdx;
    let file = 0;
    for (const ch of ranks[rankIdx]) {
      if (ch >= '1' && ch <= '8') {
        file += Number.parseInt(ch, 10);
      } else {
        const plane = PIECE_PLANES[ch];
        if (plane !== undefined) {
          tensor[plane * 64 + row * 8 + file] = 1.0;
        }
        file += 1;
      }
    }
  }

  // Plane 12: side to move (1.0 for White; always 1.0 after Black mirroring)
  if (parts[1] === 'w') {
    tensor.fill(1.0, 12 * 64, 13 * 64);
  }

  // Planes 13..16: castling rights K, Q, k, q
  const castling = parts[2] || '-';
  if (castling.includes('K')) tensor.fill(1.0, 13 * 64, 14 * 64);
  if (castling.includes('Q')) tensor.fill(1.0, 14 * 64, 15 * 64);
  if (castling.includes('k')) tensor.fill(1.0, 15 * 64, 16 * 64);
  if (castling.includes('q')) tensor.fill(1.0, 16 * 64, 17 * 64);

  // Plane 17: en passant square
  const ep = parts[3] || '-';
  if (ep !== '-' && /^[a-h][1-8]$/.test(ep)) {
    const epFile = ep.charCodeAt(0) - 97;
    const epRow = Number.parseInt(ep[1], 10) - 1;
    tensor[17 * 64 + epRow * 8 + epFile] = 1.0;
  }

  return tensor;
}

/**
 * Decode Maia-2's 1,880 policy logits into a legal move probability distribution.
 *
 * @param {Float32Array | number[]} logits Length-1880 array of raw logits.
 * @param {string} fen Original (unmirrored) FEN.
 * @param {Record<string, number>} movesDict Mapping from UCI string to logit index.
 * @returns {{ moves: Array<{ move: string, san: string, prob: number }>, policy: Record<string, number> }}
 */
export function decodePolicyLogits(logits, fen, movesDict) {
  const board = new Chess(fen);
  const isBlack = board.turn() === 'b';
  const legal = board.moves({ verbose: true });
  if (legal.length === 0) {
    return { moves: [], policy: {} };
  }

  const scored = [];
  let maxLogit = -Infinity;

  for (const m of legal) {
    const uci = `${m.from}${m.to}${m.promotion || ''}`;
    const lookupUci = isBlack ? mirrorUci(uci) : uci;
    const idx = movesDict[lookupUci];
    const val = idx !== undefined && idx < logits.length ? Number(logits[idx]) : -20.0;
    if (val > maxLogit) maxLogit = val;
    scored.push({ move: uci, san: m.san, logit: val });
  }

  let sumExp = 0;
  const exps = scored.map((item) => {
    const e = Math.exp(item.logit - maxLogit);
    sumExp += e;
    return e;
  });

  const policy = {};
  const moves = scored.map((item, i) => {
    const prob = sumExp > 0 ? exps[i] / sumExp : 1 / scored.length;
    policy[item.move] = prob;
    return { move: item.move, san: item.san, prob };
  });

  moves.sort((a, b) => (b.prob !== a.prob ? b.prob - a.prob : a.move.localeCompare(b.move)));
  return { moves, policy };
}

/**
 * Numerically stable temperature-scaled softmax over a UCI->score dictionary.
 *
 * @param {Record<string, number>} scores
 * @param {number} temperature
 * @returns {Record<string, number>}
 */
export function softmax(scores, temperature) {
  const entries = Object.entries(scores);
  if (entries.length === 0) return {};
  const temp = Math.max(1e-6, temperature);
  let maxScaled = -Infinity;
  for (const [, v] of entries) {
    const s = v / temp;
    if (s > maxScaled) maxScaled = s;
  }
  let sum = 0;
  const exps = entries.map(([k, v]) => {
    const e = Math.exp(v / temp - maxScaled);
    sum += e;
    return [k, e];
  });
  const out = {};
  for (const [k, e] of exps) {
    out[k] = sum > 0 ? e / sum : 1 / entries.length;
  }
  return out;
}

/**
 * Heuristic move score mirroring ``src/chessfm/coach/maia_sidecar.py::_heuristic_move_score``.
 *
 * @param {Chess} board
 * @param {import('../vendor/chess.mjs').Move} move
 * @param {number} skill Skill factor in [0, 1].
 * @returns {number}
 */
function heuristicMoveScore(board, move, skill) {
  let score = 0.0;
  const mover = board.turn();
  const opponent = mover === 'w' ? 'b' : 'w';
  const movingVal = PIECE_VALUES[move.piece] ?? 1.0;

  // Capture (MVV-LVA)
  if (move.captured) {
    const isEp = move.flags.includes('e');
    const victimVal = isEp ? 1.0 : (PIECE_VALUES[move.captured] ?? 1.0);
    score += 0.9 + 0.35 * victimVal - 0.08 * movingVal;
  }

  // Promotion
  if (move.promotion) {
    score += 0.5 + 0.25 * (PIECE_VALUES[move.promotion] ?? 1.0);
  }

  // Castling
  if (move.flags.includes('k') || move.flags.includes('q')) {
    score += 1.0 + 0.4 * skill;
  }

  // Center control
  if (CENTER_SQUARES.has(move.to)) {
    score += 0.55 + 0.25 * skill;
  } else if (EXTENDED_CENTER.has(move.to)) {
    score += 0.25 + 0.1 * skill;
  }

  // Minor piece development off back rank
  if (move.piece === 'n' || move.piece === 'b') {
    const backRank = mover === 'w' ? '1' : '8';
    if (move.from[1] === backRank) {
      score += 0.45 + 0.25 * skill;
    }
  }

  // Post-move tactical features (push then undo)
  board.move(move);
  try {
    if (board.isCheckmate()) {
      score += 4.0 + 2.0 * skill;
    } else if (board.inCheck()) {
      score += 0.85 - 0.25 * skill;
    }

    const oppAttackers = board.attackers(move.to, opponent);
    const defenders = board.attackers(move.to, mover);
    if (oppAttackers.length > 0 && defenders.length === 0) {
      score -= (0.25 + 0.95 * skill) * Math.min(movingVal, 5.0);
    } else if (oppAttackers.length > 0) {
      let minAttackerVal = Infinity;
      for (const sq of oppAttackers) {
        const p = board.get(sq);
        const v = p ? (PIECE_VALUES[p.type] ?? 1.0) : 1.0;
        if (v < minAttackerVal) minAttackerVal = v;
      }
      if (minAttackerVal < movingVal) {
        score -= (0.2 + 0.75 * skill) * (movingVal - minAttackerVal);
      }
    }
  } finally {
    board.undo();
  }

  return score;
}

/**
 * Instant, deterministic human-Elo move distribution without requiring ONNX weights.
 * Mirrors ``src/chessfm/coach/maia_sidecar.py::_predict_fallback``.
 *
 * @param {string} fen
 * @param {number} [studentElo=1400]
 * @param {{ engineCpMap?: Record<string, number> | null, topK?: number }} [options]
 */
export function calibratedFallback(fen, studentElo = 1400, options = {}) {
  const board = new Chess(fen);
  const legal = board.moves({ verbose: true });
  const skill = eloSkillFactor(studentElo);
  const topK = options.topK ?? 5;

  if (legal.length === 0) {
    return {
      fen,
      student_elo: Number(studentElo) || 1400,
      elo_bracket: snapEloBracket(studentElo),
      backend: 'calibrated_fallback',
      policy: {},
      top_moves: [],
      win_prob: 0.5,
    };
  }

  const scores = {};
  const sanByUci = {};
  for (const m of legal) {
    const uci = `${m.from}${m.to}${m.promotion || ''}`;
    sanByUci[uci] = m.san;
    scores[uci] = heuristicMoveScore(board, m, skill);
  }

  // Optional Stockfish centipawn blend
  const cpMap = options.engineCpMap || null;
  if (cpMap && Object.keys(cpMap).length > 0) {
    const bestCp = Math.max(...Object.values(cpMap));
    const engineWeight = 0.4 + 1.8 * skill;
    for (const [uci, cp] of Object.entries(cpMap)) {
      if (scores[uci] !== undefined) {
        const lossPawns = Math.max(0, (bestCp - cp) / 100.0);
        scores[uci] += engineWeight * Math.max(-3.0, 1.5 - lossPawns);
      }
    }
  }

  const temperature = 1.8 - 1.15 * skill;
  const policy = softmax(scores, temperature);

  const sorted = Object.entries(policy)
    .map(([move, prob]) => ({ move, san: sanByUci[move] || move, prob }))
    .sort((a, b) => (b.prob !== a.prob ? b.prob - a.prob : a.move.localeCompare(b.move)));

  return {
    fen,
    student_elo: Number(studentElo) || 1400,
    elo_bracket: snapEloBracket(studentElo),
    backend: 'calibrated_fallback',
    policy,
    top_moves: sorted.slice(0, topK),
    win_prob: null,
  };
}

/**
 * Classify whether a student's mistake is a common human trap at their Elo.
 * Mirrors ``src/chessfm/coach/maia_sidecar.py::detect_human_trap``.
 *
 * @param {string} fen
 * @param {string} playedMove
 * @param {string} bestMove
 * @param {number} cpLoss
 * @param {number} [studentElo=1200]
 * @param {{ policy: Record<string, number>, backend?: string } | null} [prediction=null]
 */
export function detectHumanTrap(fen, playedMove, bestMove, cpLoss, studentElo = 1200, prediction = null) {
  const pred = prediction || calibratedFallback(fen, studentElo);
  const playedUci = String(playedMove || '').toLowerCase();
  const bestUci = String(bestMove || '').toLowerCase();
  const playedProb = pred.policy[playedUci] ?? 0;
  const bestProb = pred.policy[bestUci] ?? 0;
  const loss = Math.max(0, Number(cpLoss) || 0);

  let category;
  let isTrap = false;
  let coachingContext;

  if (loss >= 100.0 && playedProb >= 0.15) {
    category = 'common_human_trap';
    isTrap = true;
    coachingContext = `Common human trap at ~${studentElo} Elo (${Math.round(playedProb * 100)}% of peers play this move).`;
  } else if (loss >= 100.0) {
    category = 'unusual_oversight';
    coachingContext = `Unusual tactical oversight at ~${studentElo} Elo; double-check checks, captures, and threats.`;
  } else {
    category = 'sound_move';
    coachingContext = 'Reasonable human choice within acceptable accuracy bounds.';
  }

  return {
    fen,
    played_move: playedUci,
    best_move: bestUci,
    cp_loss: Math.round(loss * 100) / 100,
    student_elo: Number(studentElo) || 1200,
    backend: pred.backend || 'calibrated_fallback',
    played_move_prob: Math.round(playedProb * 1e6) / 1e6,
    best_move_prob: Math.round(bestProb * 1e6) / 1e6,
    is_common_human_trap: isTrap,
    category,
    coaching_context: coachingContext,
  };
}

// ---------------------------------------------------------------------------
// IndexedDB cache helpers (browser-only, safe no-op in Node)
// ---------------------------------------------------------------------------

const IDB_NAME = 'MaiaModels';
const IDB_STORE = 'models';
const IDB_KEY = 'maia2-rapid-v1';

function openMaiaDb() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') {
      resolve(null);
      return;
    }
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function readCachedOnnxBuffer() {
  try {
    const db = await openMaiaDb();
    if (!db) return null;
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const req = tx.objectStore(IDB_STORE).get(IDB_KEY);
      req.onsuccess = async () => {
        const val = req.result;
        if (!val) return resolve(null);
        if (val instanceof ArrayBuffer) return resolve(val);
        if (typeof Blob !== 'undefined' && val instanceof Blob) {
          return resolve(await val.arrayBuffer());
        }
        return resolve(null);
      };
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;
  }
}

async function writeCachedOnnxBuffer(buffer) {
  try {
    const db = await openMaiaDb();
    if (!db) return;
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(buffer, IDB_KEY);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    // Ignore private-browsing quota errors; model still works in-memory.
  }
}

/**
 * Progressive Maia-2 client: instant ``heuristic`` mode with opt-in ONNX upgrade.
 */
export class MaiaClient {
  /**
   * @param {{ movesDictUrl?: string, modelUrl?: string, ortCdnUrl?: string }} [options]
   */
  constructor(options = {}) {
    this.movesDictUrl = options.movesDictUrl || './data/maia2_moves.json';
    this.modelUrl = options.modelUrl || MAIA2_ONNX_URL;
    this.ortCdnUrl = options.ortCdnUrl || ORT_CDN_URL;
    this.status = 'heuristic'; // 'heuristic' | 'loading' | 'onnx' | 'error'
    this.error = null;
    this.ort = null;
    this.session = null;
    this.movesDict = null;
  }

  /**
   * Download (or read from IndexedDB) and initialize Maia-2 ONNX in single-threaded WASM.
   *
   * @param {{ onProgress?: (info: { phase: string, loaded?: number, total?: number, pct?: number }) => void }} [options]
   */
  async loadOnnx(options = {}) {
    if (this.status === 'onnx' && this.session) return true;
    const onProgress = options.onProgress || (() => {});
    this.status = 'loading';
    this.error = null;

    try {
      onProgress({ phase: 'dictionary', pct: 2 });
      if (!this.movesDict) {
        const resp = await fetch(this.movesDictUrl);
        if (!resp.ok) throw new Error(`Failed to load move dictionary (HTTP ${resp.status})`);
        this.movesDict = await resp.json();
      }

      onProgress({ phase: 'cache', pct: 5 });
      let buffer = await readCachedOnnxBuffer();

      if (!buffer) {
        onProgress({ phase: 'download', loaded: 0, total: 93246338, pct: 6 });
        const resp = await fetch(this.modelUrl);
        if (!resp.ok) throw new Error(`Failed to fetch Maia-2 ONNX (HTTP ${resp.status})`);
        const total = Number.parseInt(resp.headers.get('content-length') || '93246338', 10);

        if (resp.body && typeof resp.body.getReader === 'function') {
          const reader = resp.body.getReader();
          const chunks = [];
          let loaded = 0;
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            chunks.push(value);
            loaded += value.byteLength;
            const pct = Math.min(90, Math.round(6 + (loaded / Math.max(1, total)) * 84));
            onProgress({ phase: 'download', loaded, total, pct });
          }
          const combined = new Uint8Array(loaded);
          let offset = 0;
          for (const chunk of chunks) {
            combined.set(chunk, offset);
            offset += chunk.byteLength;
          }
          buffer = combined.buffer;
        } else {
          buffer = await resp.arrayBuffer();
        }
        await writeCachedOnnxBuffer(buffer);
      }

      onProgress({ phase: 'runtime', pct: 92 });
      const ort = await import(/* @vite-ignore */ this.ortCdnUrl);
      ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
      ort.env.wasm.numThreads = 1; // Critical: zero COOP/COEP required
      this.ort = ort;

      onProgress({ phase: 'session', pct: 96 });
      this.session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'basic',
      });

      this.status = 'onnx';
      onProgress({ phase: 'ready', pct: 100 });
      return true;
    } catch (err) {
      this.error = err instanceof Error ? err.message : String(err);
      this.status = 'heuristic';
      return false;
    }
  }

  /**
   * Predict human move probabilities for ``fen`` at ``studentElo``.
   * Uses Maia-2 ONNX when loaded, otherwise falls back instantly to ``calibratedFallback``.
   *
   * @param {string} fen
   * @param {number} [studentElo=1400]
   * @param {{ engineCpMap?: Record<string, number> | null, topK?: number }} [options]
   */
  async predict(fen, studentElo = 1400, options = {}) {
    const topK = options.topK ?? 5;
    if (this.status === 'onnx' && this.session && this.ort && this.movesDict) {
      try {
        const boardPlanes = encodeBoardPlanes(fen);
        const bucket = BigInt(eloToBucket(studentElo));
        const feeds = {
          boards: new this.ort.Tensor('float32', boardPlanes, [1, 18, 8, 8]),
          elo_self: new this.ort.Tensor('int64', BigInt64Array.from([bucket]), [1]),
          elo_oppo: new this.ort.Tensor('int64', BigInt64Array.from([bucket]), [1]),
        };
        const results = await this.session.run(feeds);
        const logits = results.logits_maia.data;
        const { moves, policy } = decodePolicyLogits(logits, fen, this.movesDict);

        let winProb = null;
        if (results.logits_value?.data?.length) {
          let wp = Math.min(Math.max(Number(results.logits_value.data[0]) / 2 + 0.5, 0), 1);
          const isBlack = fen.trim().split(/\s+/)[1] === 'b';
          if (isBlack) wp = 1 - wp;
          winProb = Math.round(wp * 10000) / 10000;
        }

        if (moves.length > 0) {
          return {
            fen,
            student_elo: Number(studentElo) || 1400,
            elo_bracket: snapEloBracket(studentElo),
            backend: 'maia2_onnx',
            policy,
            top_moves: moves.slice(0, topK),
            win_prob: winProb,
          };
        }
      } catch (err) {
        console.warn('Maia-2 ONNX inference failed, using calibratedFallback:', err);
      }
    }

    return calibratedFallback(fen, studentElo, options);
  }
}
