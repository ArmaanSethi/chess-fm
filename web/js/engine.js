/**
 * Browser Stockfish 19 WASM Web Worker wrapper, UCI parser, and WDL move classifier.
 *
 * Pure functions (parseUciInfoLine, scoreToCp, winProbFromCp, classifyMove, pvUciToSan)
 * run identically in Node 22 unit tests and browsers with zero DOM dependencies.
 * StockfishWorker uses the vendored single-threaded lite NNUE build requiring zero
 * COOP/COEP headers or SharedArrayBuffer.
 */

import { Chess } from '../vendor/chess.mjs';

export const LICHESS_CP_COEFF = 0.00368208;
export const MATE_SCORE_CP = 10000;

/**
 * Convert a centipawn evaluation (mover's perspective) to win probability in [0, 100].
 * Mirrors ``src/chessfm/coach/classifier.py::win_prob_from_cp``.
 *
 * @param {number} cp
 * @returns {number}
 */
export function winProbFromCp(cp) {
  const exponent = Math.max(-500, Math.min(500, -LICHESS_CP_COEFF * Number(cp)));
  const prob = 50 + 50 * (2 / (1 + Math.exp(exponent)) - 1);
  return Math.max(0, Math.min(100, prob));
}

/**
 * Convert a parsed UCI score object ``{ type: 'cp'|'mate', value: number }`` to centipawns.
 *
 * @param {{ type: string, value: number } | null | undefined} score
 * @returns {number}
 */
export function scoreToCp(score) {
  if (!score) return 0;
  if (score.type === 'mate') {
    if (score.value > 0) return MATE_SCORE_CP;
    if (score.value < 0) return -MATE_SCORE_CP;
    return 0;
  }
  return Number(score.value) || 0;
}

/**
 * Parse a UCI ``info`` line emitted by Stockfish into a structured object.
 * Returns ``null`` for non-scoring or malformed lines.
 *
 * @param {string} line
 * @returns {{
 *   depth: number,
 *   seldepth: number | null,
 *   multipv: number,
 *   score: { type: 'cp' | 'mate', value: number },
 *   cp: number,
 *   mate: number | null,
 *   bound: 'lowerbound' | 'upperbound' | null,
 *   nodes: number | null,
 *   nps: number | null,
 *   pv: string[],
 * } | null}
 */
export function parseUciInfoLine(line) {
  if (typeof line !== 'string' || !line.startsWith('info ')) {
    return null;
  }
  const tokens = line.trim().split(/\s+/);
  let depth = null;
  let seldepth = null;
  let multipv = 1;
  let score = null;
  let bound = null;
  let nodes = null;
  let nps = null;
  let pv = [];

  for (let i = 1; i < tokens.length; i += 1) {
    const tok = tokens[i];
    if (tok === 'depth' && i + 1 < tokens.length) {
      depth = Number.parseInt(tokens[++i], 10);
    } else if (tok === 'seldepth' && i + 1 < tokens.length) {
      seldepth = Number.parseInt(tokens[++i], 10);
    } else if (tok === 'multipv' && i + 1 < tokens.length) {
      multipv = Number.parseInt(tokens[++i], 10);
    } else if (tok === 'nodes' && i + 1 < tokens.length) {
      nodes = Number.parseInt(tokens[++i], 10);
    } else if (tok === 'nps' && i + 1 < tokens.length) {
      nps = Number.parseInt(tokens[++i], 10);
    } else if (tok === 'score' && i + 2 < tokens.length) {
      const type = tokens[++i];
      const val = Number.parseInt(tokens[++i], 10);
      if ((type === 'cp' || type === 'mate') && !Number.isNaN(val)) {
        score = { type, value: val };
      }
      if (i + 1 < tokens.length && (tokens[i + 1] === 'lowerbound' || tokens[i + 1] === 'upperbound')) {
        bound = tokens[++i];
      }
    } else if (tok === 'pv') {
      pv = tokens.slice(i + 1);
      break;
    } else if (tok === 'string') {
      break;
    }
  }

  if (depth === null || Number.isNaN(depth) || !score || pv.length === 0) {
    return null;
  }

  return {
    depth,
    seldepth: Number.isNaN(seldepth) ? null : seldepth,
    multipv: Number.isNaN(multipv) || multipv < 1 ? 1 : multipv,
    score,
    cp: scoreToCp(score),
    mate: score.type === 'mate' ? score.value : null,
    bound,
    nodes: Number.isNaN(nodes) ? null : nodes,
    nps: Number.isNaN(nps) ? null : nps,
    pv,
  };
}

/**
 * Classify a chess move using WDL win-probability drop and centipawn loss.
 * Mirrors ``src/chessfm/coach/classifier.py::classify_move``.
 *
 * Accepts either:
 *   classifyMove(cpBefore, cpAfter, bestMoveUci, playedMoveUci)
 * or:
 *   classifyMove(cpBefore, cpAfter, isBestMoveBoolean)
 *
 * @param {number} evalBeforeCp Mover-perspective eval of best move before playing.
 * @param {number} evalAfterCp Mover-perspective eval after playing the move.
 * @param {string | boolean} bestMoveOrFlag Best UCI move or boolean isBestMove.
 * @param {string} [playedMove] Played UCI move when 3rd arg is bestMove string.
 */
export function classifyMove(evalBeforeCp, evalAfterCp, bestMoveOrFlag = false, playedMove = undefined) {
  const isBestMove =
    typeof bestMoveOrFlag === 'boolean'
      ? bestMoveOrFlag
      : Boolean(bestMoveOrFlag && playedMove && bestMoveOrFlag.toLowerCase() === playedMove.toLowerCase());

  const cpLoss = Math.max(0, Number(evalBeforeCp) - Number(evalAfterCp));
  const winProbBefore = winProbFromCp(evalBeforeCp);
  const winProbAfter = winProbFromCp(evalAfterCp);
  const winProbDrop = Math.max(0, winProbBefore - winProbAfter);

  let label;
  if (isBestMove) {
    label = 'Best';
  } else if (
    winProbBefore >= 80.0 &&
    winProbAfter >= 35.0 &&
    winProbAfter <= 60.0 &&
    winProbDrop >= 20.0
  ) {
    label = 'Missed Win';
  } else if (winProbDrop >= 20.0 || cpLoss >= 250.0) {
    label = 'Blunder';
  } else if (winProbDrop >= 10.0 || cpLoss >= 100.0) {
    label = 'Mistake';
  } else if (winProbDrop >= 5.0 || cpLoss >= 50.0) {
    label = 'Inaccuracy';
  } else if (winProbDrop >= 2.0 || cpLoss >= 20.0) {
    label = 'Good';
  } else {
    label = 'Excellent';
  }

  return {
    label,
    cp_loss: Math.round(cpLoss * 100) / 100,
    win_prob_before: Math.round(winProbBefore * 100) / 100,
    win_prob_after: Math.round(winProbAfter * 100) / 100,
    win_prob_drop: Math.round(winProbDrop * 100) / 100,
    is_best_move: isBestMove,
  };
}

/**
 * Convert a UCI move string (e.g. ``e2e4`` or ``e7e8q``) into a chess.js move object.
 *
 * @param {string} uci
 * @returns {{ from: string, to: string, promotion?: string } | null}
 */
export function uciToMoveObj(uci) {
  if (typeof uci !== 'string') return null;
  const trimmed = uci.trim().toLowerCase();
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(trimmed)) return null;
  const obj = { from: trimmed.slice(0, 2), to: trimmed.slice(2, 4) };
  if (trimmed.length === 5) obj.promotion = trimmed[4];
  return obj;
}

/**
 * Convert a UCI PV array into SAN strings from ``fen`` up to ``maxPlies``.
 *
 * @param {string} fen
 * @param {string[]} pvUci
 * @param {number} [maxPlies=6]
 * @returns {string[]}
 */
export function pvUciToSan(fen, pvUci, maxPlies = 6) {
  const board = new Chess(fen);
  const out = [];
  const limit = Math.min(pvUci.length, maxPlies);
  for (let i = 0; i < limit; i += 1) {
    const moveObj = uciToMoveObj(pvUci[i]);
    if (!moveObj) break;
    try {
      const moved = board.move(moveObj);
      if (!moved) break;
      out.push(moved.san);
    } catch {
      break;
    }
  }
  return out;
}

/**
 * Format a centipawn or mate score from White's perspective for UI display.
 *
 * @param {number} cpWhite
 * @param {number | null} [mateWhite=null]
 * @returns {string}
 */
export function formatEvalScore(cpWhite, mateWhite = null) {
  if (mateWhite !== null && mateWhite !== undefined) {
    return `${mateWhite > 0 ? '+' : '-'}M${Math.abs(mateWhite)}`;
  }
  if (Math.abs(cpWhite) >= MATE_SCORE_CP - 100) {
    return cpWhite > 0 ? '+M' : '-M';
  }
  const pawns = cpWhite / 100;
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`;
}

/**
 * Serialized Stockfish 19 WASM Web Worker client.
 */
export class StockfishWorker {
  /**
   * @param {{ workerUrl?: string, workerFactory?: () => Worker }} [options]
   */
  constructor(options = {}) {
    this.workerUrl = options.workerUrl || './vendor/stockfish/stockfish-19-lite-single.js';
    this.workerFactory = options.workerFactory || null;
    this.worker = null;
    this.ready = false;
    this.status = 'idle';
    this._listeners = new Set();
    this._queue = Promise.resolve();
    this._lastMultiPV = 1;
  }

  _enqueue(task) {
    const run = this._queue.then(task, task);
    this._queue = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  _addListener(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _send(cmd) {
    if (!this.worker) {
      throw new Error('Stockfish worker is not initialized');
    }
    this.worker.postMessage(cmd);
  }

  _waitFor(predicate, timeoutMs = 30000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        off();
        reject(new Error(`Stockfish timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      const off = this._addListener((line) => {
        if (predicate(line)) {
          clearTimeout(timer);
          off();
          resolve(line);
        }
      });
    });
  }

  /**
   * Spawn the Stockfish Web Worker and complete the UCI handshake.
   */
  async init() {
    if (this.ready && this.worker) return;
    this.status = 'loading';
    this.worker = this.workerFactory ? this.workerFactory() : new Worker(this.workerUrl);
    this.worker.onmessage = (event) => {
      const raw = typeof event === 'string' ? event : event?.data;
      if (typeof raw !== 'string') return;
      for (const line of raw.split(/\r?\n/)) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        for (const fn of this._listeners) {
          fn(trimmed);
        }
      }
    };
    this.worker.onerror = (err) => {
      this.status = 'error';
      console.error('Stockfish worker error:', err);
    };

    const uciOk = this._waitFor((line) => line === 'uciok', 20000);
    this._send('uci');
    await uciOk;

    const readyOk = this._waitFor((line) => line === 'readyok', 15000);
    this._send('isready');
    await readyOk;

    this.ready = true;
    this.status = 'ready';
  }

  /**
   * Run a MultiPV analysis on ``fen``.
   *
   * @param {string} fen
   * @param {{ depth?: number, multipv?: number, pvPlies?: number, onInfo?: (info: any) => void }} [options]
   */
  analyseMultiPV(fen, options = {}) {
    const depth = options.depth ?? 12;
    const requestedMultiPV = options.multipv ?? 3;
    const pvPlies = options.pvPlies ?? 6;
    const onInfo = options.onInfo || null;

    return this._enqueue(async () => {
      if (!this.ready) await this.init();

      const board = new Chess(fen);
      const sideToMove = board.turn();
      const legalMoves = board.moves({ verbose: true });

      if (legalMoves.length === 0) {
        const mateEval = board.isCheckmate() ? -MATE_SCORE_CP : 0;
        return {
          fen,
          sideToMove,
          bestMove: null,
          bestEvalCp: mateEval,
          bestMate: board.isCheckmate() ? 0 : null,
          topLines: [],
        };
      }

      const actualMultiPV = Math.max(1, Math.min(requestedMultiPV, legalMoves.length));
      if (this._lastMultiPV !== actualMultiPV) {
        this._send(`setoption name MultiPV value ${actualMultiPV}`);
        const readyOk = this._waitFor((l) => l === 'readyok', 10000);
        this._send('isready');
        await readyOk;
        this._lastMultiPV = actualMultiPV;
      }

      const bestByPv = new Map();
      const off = this._addListener((line) => {
        const parsed = parseUciInfoLine(line);
        if (!parsed) return;
        const prev = bestByPv.get(parsed.multipv);
        // Prefer deeper search, and non-bound exact scores at equal depth
        if (!prev || parsed.depth > prev.depth || (parsed.depth === prev.depth && !parsed.bound)) {
          bestByPv.set(parsed.multipv, parsed);
          if (onInfo) onInfo(parsed);
        }
      });

      try {
        const bestMoveDone = this._waitFor((l) => l.startsWith('bestmove '), 45000);
        this._send(`position fen ${fen}`);
        this._send(`go depth ${depth}`);
        const bestMoveLine = await bestMoveDone;
        const fallbackMove = bestMoveLine.split(/\s+/)[1] || null;

        const sortedSlots = Array.from(bestByPv.keys()).sort((a, b) => a - b);
        const topLines = [];
        for (const slot of sortedSlots) {
          const info = bestByPv.get(slot);
          const uci = info.pv[0];
          const pvUci = info.pv.slice(0, pvPlies);
          const pvSan = pvUciToSan(fen, pvUci, pvPlies);
          topLines.push({
            multipv: slot,
            move: uci,
            san: pvSan[0] || uci,
            evalCp: info.cp,
            mate: info.mate,
            depth: info.depth,
            pvUci,
            pvSan,
          });
        }

        if (topLines.length === 0 && fallbackMove && fallbackMove !== '(none)') {
          const pvSan = pvUciToSan(fen, [fallbackMove], 1);
          topLines.push({
            multipv: 1,
            move: fallbackMove,
            san: pvSan[0] || fallbackMove,
            evalCp: 0,
            mate: null,
            depth,
            pvUci: [fallbackMove],
            pvSan,
          });
        }

        return {
          fen,
          sideToMove,
          bestMove: topLines[0]?.move || fallbackMove,
          bestEvalCp: topLines[0]?.evalCp ?? 0,
          bestMate: topLines[0]?.mate ?? null,
          topLines,
        };
      } finally {
        off();
      }
    });
  }

  /**
   * Evaluate a specific UCI move from ``fen`` returning mover-perspective centipawns.
   *
   * @param {string} fen
   * @param {string} uci
   * @param {number} [depth=12]
   */
  async evalMove(fen, uci, depth = 12) {
    const board = new Chess(fen);
    const moveObj = uciToMoveObj(uci);
    if (!moveObj) {
      throw new Error(`Invalid UCI move: ${uci}`);
    }
    const moved = board.move(moveObj);
    if (!moved) {
      throw new Error(`Illegal move ${uci} for FEN ${fen}`);
    }
    if (board.isCheckmate()) {
      return {
        move: uci.toLowerCase(),
        san: moved.san,
        evalCp: MATE_SCORE_CP,
        mate: 1,
        pvUci: [uci.toLowerCase()],
        pvSan: [moved.san],
      };
    }
    if (board.isDraw() || board.isStalemate() || board.isInsufficientMaterial()) {
      return {
        move: uci.toLowerCase(),
        san: moved.san,
        evalCp: 0,
        mate: null,
        pvUci: [uci.toLowerCase()],
        pvSan: [moved.san],
      };
    }

    const opp = await this.analyseMultiPV(board.fen(), { depth, multipv: 1, pvPlies: 5 });
    const top = opp.topLines[0];
    const evalCp = top ? -top.evalCp : 0;
    const mate = top && top.mate !== null ? -top.mate : null;
    const pvUci = [uci.toLowerCase(), ...(top?.pvUci || [])].slice(0, 6);
    const pvSan = [moved.san, ...(top?.pvSan || [])].slice(0, 6);

    return {
      move: uci.toLowerCase(),
      san: moved.san,
      evalCp,
      mate,
      pvUci,
      pvSan,
    };
  }

  stop() {
    if (this.worker) {
      this._send('stop');
    }
  }

  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
      this.ready = false;
      this.status = 'terminated';
    }
  }
}
