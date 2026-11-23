/**
 * Full-game PGN review, Lichess accuracy curve, 6-category Weakness Profile,
 * and "2 Critical Moments at Your Rating" ranking (``wdlSwing * maiaFindability``).
 *
 * Faithful JavaScript port of ``src/chessfm/coach/game_review.py`` with Layer A
 * Maia-2 findability weighting.
 */

import { Chess } from '../vendor/chess.mjs';
import { MATE_SCORE_CP, LICHESS_CP_COEFF, uciToMoveObj } from './engine.js';
import { calibratedFallback, detectHumanTrap } from './maia.js';

export const PIECE_VALUES_CP = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };

const CENTER_SQUARES = new Set(['d4', 'e4', 'd5', 'e5']);
const EXTENDED_CENTER = new Set([
  'c3', 'd3', 'e3', 'f3',
  'c4', 'f4',
  'c5', 'f5',
  'c6', 'd6', 'e6', 'f6',
]);

export const WEAKNESS_CATEGORIES = [
  'opening_inaccuracy',
  'hanging_piece',
  'missed_tactic',
  'missed_mate',
  'endgame_conversion',
  'positional_drift',
];

export const WEAKNESS_METADATA = {
  opening_inaccuracy: {
    label: 'Opening Inaccuracy',
    description: 'Early-game inaccuracies in development, center control, or king safety.',
    recommendation: 'Review opening principles: develop minor pieces, control the center, and castle early.',
  },
  hanging_piece: {
    label: 'Hanging Piece',
    description: 'Leaving pieces undefended or stepping onto squares where they can be won.',
    recommendation: 'Run a blunder check before every move: verify all captures, checks, and undefended pieces.',
  },
  missed_tactic: {
    label: 'Missed Tactic',
    description: 'Overlooking forcing tactical shots such as forks, pins, skewers, or winning captures.',
    recommendation: 'Practice daily tactical puzzles focused on checks, captures, and threats (CCT).',
  },
  missed_mate: {
    label: 'Missed Mate',
    description: "Missing a forced checkmate sequence when the opponent's king is exposed.",
    recommendation: 'Drill mate-in-1 through mate-in-3 patterns and always scan forcing checks first.',
  },
  endgame_conversion: {
    label: 'Endgame Conversion',
    description: 'Losing evaluation in simplified positions with few pieces remaining.',
    recommendation: 'Study king activity, passed pawns, and fundamental rook and pawn endgames.',
  },
  positional_drift: {
    label: 'Positional Drift',
    description: 'Gradual strategic concessions in the middlegame without an immediate tactical blow.',
    recommendation: 'Improve your worst-placed piece and formulate a concrete plan around pawn structure.',
  },
};

const SUBOPTIMAL_CLASSES = new Set(['Inaccuracy', 'Mistake', 'Blunder']);

/**
 * Convert a centipawn score (mover's perspective) to win probability in [0, 100].
 *
 * @param {number} cp
 * @returns {number}
 */
export function cpToWinProb(cp) {
  const clamped = Math.max(-MATE_SCORE_CP, Math.min(MATE_SCORE_CP, Number(cp) || 0));
  return 50.0 + 50.0 * (2.0 / (1.0 + Math.exp(-LICHESS_CP_COEFF * clamped)) - 1.0);
}

/**
 * Compute per-move Lichess accuracy percentage in [0, 100].
 * Accepts either ``moveAccuracy(winBefore, winAfter)`` or ``moveAccuracy(wdlSwing)``.
 *
 * @param {number} winBeforeOrSwing
 * @param {number} [winAfter]
 * @returns {number}
 */
export function moveAccuracy(winBeforeOrSwing, winAfter = undefined) {
  const rawSwing =
    winAfter !== undefined
      ? Number(winBeforeOrSwing) - Number(winAfter)
      : Number(winBeforeOrSwing);
  const swing = Math.max(0.0, rawSwing);
  if (swing === 0.0) return 100.0;
  const acc = 103.1668 * Math.exp(-0.04354 * swing) - 3.1669;
  return Math.max(0.0, Math.min(100.0, acc));
}

/**
 * Classify a game-review move into one of 5 labels:
 * ``Best``, ``Good``, ``Inaccuracy``, ``Mistake``, ``Blunder``.
 * Mirrors ``src/chessfm/coach/game_review.py::_classify_move``.
 *
 * @param {number} cpLoss
 * @param {boolean} [isBest=false]
 * @param {{ wdlSwing?: number, missedMate?: boolean, blunderedMate?: boolean }} [options]
 * @returns {'Best' | 'Good' | 'Inaccuracy' | 'Mistake' | 'Blunder'}
 */
export function classifyReviewMove(cpLoss, isBest = false, options = {}) {
  if (isBest) return 'Best';
  const loss = Math.max(0, Number(cpLoss) || 0);
  const wdlSwing = Math.max(0, Number(options.wdlSwing) || 0);
  if (options.missedMate || options.blunderedMate || loss > 200.0 || wdlSwing > 20.0) {
    return 'Blunder';
  }
  if (loss > 100.0 || wdlSwing > 10.0) {
    return 'Mistake';
  }
  if (loss > 50.0 || wdlSwing > 5.0) {
    return 'Inaccuracy';
  }
  if (loss > 15.0 || wdlSwing > 2.0) {
    return 'Good';
  }
  return 'Best';
}

/**
 * Determine whether ``board`` is in an endgame phase (exact ``game_review.py`` thresholds).
 *
 * @param {Chess} board
 * @returns {boolean}
 */
export function isEndgamePosition(board) {
  let nonPawnMaterial = 0;
  let totalQueens = 0;
  for (const row of board.board()) {
    for (const sq of row) {
      if (!sq) continue;
      if (sq.type === 'q') totalQueens += 1;
      if (sq.type !== 'p' && sq.type !== 'k') {
        nonPawnMaterial += PIECE_VALUES_CP[sq.type] || 0;
      }
    }
  }
  return nonPawnMaterial <= 2600 || (totalQueens === 0 && nonPawnMaterial <= 3400);
}

/**
 * Check if ``mover`` ('w' | 'b') has any hanging non-pawn, non-king piece on ``board``.
 *
 * @param {Chess} board
 * @param {'w' | 'b'} mover
 * @returns {boolean}
 */
function hasHangingPiece(board, mover) {
  const opponent = mover === 'w' ? 'b' : 'w';
  for (const row of board.board()) {
    for (const cell of row) {
      if (!cell || cell.color !== mover || cell.type === 'p' || cell.type === 'k') continue;
      const oppAttackers = board.attackers(cell.square, opponent);
      if (oppAttackers.length === 0) continue;
      const defenders = board.attackers(cell.square, mover);
      if (defenders.length === 0) return true;
      const pieceVal = PIECE_VALUES_CP[cell.type] || 0;
      let minAttackerVal = Infinity;
      for (const atkSq of oppAttackers) {
        const atkPiece = board.get(atkSq);
        const val = atkPiece ? (PIECE_VALUES_CP[atkPiece.type] ?? 100) : 100;
        if (val < minAttackerVal) minAttackerVal = val;
      }
      if (minAttackerVal < pieceVal) return true;
    }
  }
  return false;
}

/**
 * Tag a suboptimal move into one of the 6 canonical ``WEAKNESS_CATEGORIES``.
 * Mirrors ``src/chessfm/coach/game_review.py::_tag_weakness``.
 *
 * @param {{
 *   fen: string,
 *   ply: number,
 *   mover: 'w' | 'b',
 *   playedUci: string,
 *   bestUci: string | null,
 *   cpBefore: number,
 *   cpAfter: number,
 *   cpLoss: number,
 *   mateBefore?: number | null,
 *   classification: string,
 * }} info
 * @returns {string | null}
 */
export function tagWeakness(info) {
  if (!SUBOPTIMAL_CLASSES.has(info.classification)) return null;

  if ((info.mateBefore && info.mateBefore > 0) || (info.cpBefore >= 9000 && info.cpAfter < 9000)) {
    return 'missed_mate';
  }

  const boardBefore = new Chess(info.fen);
  const boardAfter = new Chess(info.fen);
  const playedObj = uciToMoveObj(info.playedUci);
  if (playedObj) {
    try {
      boardAfter.move(playedObj);
    } catch {
      // ignore invalid move
    }
  }

  if (info.cpLoss >= 150.0 && hasHangingPiece(boardAfter, info.mover)) {
    return 'hanging_piece';
  }

  if (info.bestUci && info.cpLoss >= 120.0) {
    const bestObj = uciToMoveObj(info.bestUci);
    if (bestObj) {
      const probe = new Chess(info.fen);
      try {
        const moved = probe.move(bestObj);
        if (moved && (moved.captured || moved.promotion || probe.inCheck())) {
          return 'missed_tactic';
        }
      } catch {
        // ignore
      }
    }
  }

  if (info.ply <= 16) {
    return 'opening_inaccuracy';
  }

  if (isEndgamePosition(boardBefore)) {
    return 'endgame_conversion';
  }

  return 'positional_drift';
}

/**
 * Fast static material + center + en-prise evaluation (mover's perspective, centipawns)
 * used as fallback when Stockfish WASM is not attached (e.g. unit tests or instant preview).
 *
 * @param {Chess} board
 * @returns {number}
 */
export function staticEvalCp(board) {
  if (board.isCheckmate()) return -MATE_SCORE_CP;
  if (board.isDraw() || board.isStalemate() || board.isInsufficientMaterial()) return 0;

  let whiteScore = 0;
  for (const row of board.board()) {
    for (const cell of row) {
      if (!cell) continue;
      const sign = cell.color === 'w' ? 1 : -1;
      let val = PIECE_VALUES_CP[cell.type] || 0;
      if (CENTER_SQUARES.has(cell.square)) {
        val += cell.type === 'p' || cell.type === 'n' || cell.type === 'b' ? 15 : 5;
      } else if (EXTENDED_CENTER.has(cell.square) && (cell.type === 'n' || cell.type === 'b')) {
        val += 8;
      }
      whiteScore += sign * val;
    }
  }

  const mover = board.turn();
  const opponent = mover === 'w' ? 'b' : 'w';
  let worstLoss = 0;
  for (const row of board.board()) {
    for (const cell of row) {
      if (!cell || cell.color !== mover || cell.type === 'k') continue;
      const oppAttackers = board.attackers(cell.square, opponent);
      if (oppAttackers.length === 0) continue;
      const defenders = board.attackers(cell.square, mover);
      const val = PIECE_VALUES_CP[cell.type] || 0;
      if (defenders.length === 0) {
        worstLoss = Math.max(worstLoss, 0.9 * val);
      } else {
        let minAtk = Infinity;
        for (const sq of oppAttackers) {
          const p = board.get(sq);
          const v = p ? (PIECE_VALUES_CP[p.type] ?? 100) : 100;
          if (v < minAtk) minAtk = v;
        }
        if (minAtk < val) {
          worstLoss = Math.max(worstLoss, 0.85 * (val - minAtk));
        }
      }
    }
  }

  const moverScore = mover === 'w' ? whiteScore : -whiteScore;
  return Math.round(moverScore - worstLoss);
}

/**
 * Rank the student's top teachable critical moments ("2 Critical Moments at Your Rating")
 * by ``wdlSwing * maiaFindability`` where ``maiaFindability = 0.35 + 0.65 * maiaBestProb``.
 *
 * @param {Array<{
 *   ply: number,
 *   moveNumber?: number,
 *   mover: 'w' | 'b',
 *   fen: string,
 *   playedUci: string,
 *   playedSan: string,
 *   bestUci: string,
 *   bestSan: string,
 *   cpLoss: number,
 *   wdlSwing: number,
 *   classification: string,
 *   weaknessCategory?: string | null,
 * }>} plyAnalyses
 * @param {{
 *   studentColor?: 'w' | 'b',
 *   studentElo?: number,
 *   maiaPredict?: (fen: string, elo: number) => Promise<{ policy: Record<string, number>, backend?: string }> | { policy: Record<string, number>, backend?: string },
 *   maxMoments?: number,
 * }} [options]
 */
export async function rankCriticalMoments(plyAnalyses, options = {}) {
  const studentColor = options.studentColor || 'w';
  const studentElo = Number(options.studentElo) || 1400;
  const maxMoments = options.maxMoments ?? 2;
  const maiaPredict =
    options.maiaPredict || ((fen, elo) => calibratedFallback(fen, elo));

  const candidates = plyAnalyses.filter(
    (p) => p.mover === studentColor && (p.cpLoss >= 80 || p.wdlSwing >= 8) && p.playedUci !== p.bestUci,
  );

  const ranked = [];
  for (const item of candidates) {
    const pred = await maiaPredict(item.fen, studentElo);
    const bestKey = String(item.bestUci || '').toLowerCase();
    const playedKey = String(item.playedUci || '').toLowerCase();
    const maiaBestProb = pred?.policy?.[bestKey] ?? 0;
    const maiaPlayedProb = pred?.policy?.[playedKey] ?? 0;
    const maiaFindability = 0.35 + 0.65 * maiaBestProb;
    const priorityScore = item.wdlSwing * maiaFindability;
    const trap = detectHumanTrap(
      item.fen,
      item.playedUci,
      item.bestUci,
      item.cpLoss,
      studentElo,
      pred,
    );

    ranked.push({
      ply: item.ply,
      moveNumber: item.moveNumber ?? Math.ceil(item.ply / 2),
      mover: item.mover,
      fen: item.fen,
      playedUci: item.playedUci,
      playedSan: item.playedSan,
      bestUci: item.bestUci,
      bestSan: item.bestSan,
      cpLoss: Math.round(item.cpLoss),
      wdlSwing: Math.round(item.wdlSwing * 10) / 10,
      classification: item.classification,
      maiaBestProb: Math.round(maiaBestProb * 10000) / 10000,
      maiaPlayedProb: Math.round(maiaPlayedProb * 10000) / 10000,
      maiaFindability: Math.round(maiaFindability * 10000) / 10000,
      priorityScore: Math.round(priorityScore * 100) / 100,
      isHumanTrap: trap.is_common_human_trap,
      trapContext: trap.coaching_context,
      weaknessCategory: item.weaknessCategory || 'positional_drift',
    });
  }

  ranked.sort((a, b) =>
    b.priorityScore !== a.priorityScore
      ? b.priorityScore - a.priorityScore
      : b.wdlSwing - a.wdlSwing,
  );
  return ranked.slice(0, maxMoments);
}

/**
 * Aggregate a student's suboptimal moves into the 6 canonical weakness categories.
 * Mirrors ``src/chessfm/coach/game_review.py::build_weakness_profile``.
 *
 * @param {Array<{
 *   mover: 'w' | 'b',
 *   fen: string,
 *   playedUci: string,
 *   bestUci: string,
 *   cpLoss: number,
 *   wdlSwing: number,
 *   classification: string,
 *   weaknessCategory?: string | null,
 * }>} plyAnalyses
 * @param {'w' | 'b'} [studentColor='w']
 */
export function buildWeaknessProfile(plyAnalyses, studentColor = 'w') {
  const buckets = {};
  for (const cat of WEAKNESS_CATEGORIES) {
    buckets[cat] = {
      category: cat,
      label: WEAKNESS_METADATA[cat].label,
      description: WEAKNESS_METADATA[cat].description,
      recommendation: WEAKNESS_METADATA[cat].recommendation,
      count: 0,
      totalCpLoss: 0,
      totalWdlSwing: 0,
      avgWdlSwing: 0,
      severityScore: 0,
    };
  }

  const studentSuboptimal = plyAnalyses.filter(
    (p) => p.mover === studentColor && SUBOPTIMAL_CLASSES.has(p.classification),
  );

  for (const item of studentSuboptimal) {
    const cat = WEAKNESS_CATEGORIES.includes(item.weaknessCategory)
      ? item.weaknessCategory
      : 'positional_drift';
    buckets[cat].count += 1;
    buckets[cat].totalCpLoss += Math.max(0, Number(item.cpLoss) || 0);
    buckets[cat].totalWdlSwing += Math.max(0, Number(item.wdlSwing) || 0);
  }

  for (const cat of WEAKNESS_CATEGORIES) {
    const b = buckets[cat];
    b.avgWdlSwing = b.count > 0 ? Math.round((b.totalWdlSwing / b.count) * 100) / 100 : 0;
    b.severityScore =
      b.count > 0
        ? Math.round((b.count * (1.0 + b.avgWdlSwing / 25.0) + b.totalCpLoss / 500.0) * 1000) / 1000
        : 0;
  }

  const sortedCategories = Object.values(buckets).sort((a, b) =>
    b.severityScore !== a.severityScore ? b.severityScore - a.severityScore : b.count - a.count,
  );
  const topPriorities = sortedCategories.filter((c) => c.count > 0).slice(0, 3);
  const topCats = new Set(topPriorities.map((c) => c.category));

  const seen = new Set();
  const drills = [];
  const sortedForDrills = [...studentSuboptimal].sort((a, b) => {
    const aTop = topCats.has(a.weaknessCategory) ? 1 : 0;
    const bTop = topCats.has(b.weaknessCategory) ? 1 : 0;
    if (bTop !== aTop) return bTop - aTop;
    if (b.wdlSwing !== a.wdlSwing) return b.wdlSwing - a.wdlSwing;
    return b.cpLoss - a.cpLoss;
  });

  for (const m of sortedForDrills) {
    const key = `${m.fen}|${m.bestUci}`;
    if (seen.has(key) || !m.bestUci) continue;
    seen.add(key);
    drills.push({
      fen: m.fen,
      best_move: m.bestUci,
      category: m.weaknessCategory || 'positional_drift',
      student_mistake: m.playedUci,
    });
  }

  return {
    studentColor,
    totalMistakes: studentSuboptimal.length,
    categories: buckets,
    orderedCategories: sortedCategories,
    topPriorities,
    drills,
  };
}

/**
 * Orchestrate a full-game review over ``pgn`` using Stockfish WASM (or static fallback)
 * and Maia-2 findability weighting.
 *
 * @param {string} pgn
 * @param {{
 *   studentColor?: 'w' | 'b',
 *   studentElo?: number,
 *   engine?: { analyseMultiPV: Function } | null,
 *   maia?: { predict: Function } | null,
 *   depth?: number,
 *   onProgress?: (info: { current: number, total: number, pct: number }) => void,
 *   signal?: AbortSignal | null,
 * }} [options]
 */
export async function reviewGame(pgn, options = {}) {
  const studentColor = options.studentColor || 'w';
  const studentElo = Number(options.studentElo) || 1400;
  const engine = options.engine || null;
  const maia = options.maia || null;
  const depth = options.depth ?? 10;
  const onProgress = options.onProgress || (() => {});
  const signal = options.signal || null;

  const loader = new Chess();
  loader.loadPgn(pgn);
  const verboseMoves = loader.history({ verbose: true });
  const totalPlies = verboseMoves.length;

  const startFen = verboseMoves.length > 0 ? verboseMoves[0].before : loader.fen();
  const replay = new Chess(startFen);
  const fens = [replay.fen()];
  for (const m of verboseMoves) {
    replay.move(m);
    fens.push(replay.fen());
  }

  // Evaluate each position once (N+1 evals instead of 2N!)
  const positionEvals = [];
  for (let i = 0; i <= totalPlies; i += 1) {
    if (signal?.aborted) {
      throw new Error('Game analysis cancelled');
    }
    const fen = fens[i];
    if (engine) {
      const res = await engine.analyseMultiPV(fen, { depth, multipv: 1, pvPlies: 4 });
      positionEvals.push({
        bestMove: res.bestMove,
        bestSan: res.topLines[0]?.san || res.bestMove || '',
        evalCp: res.bestEvalCp ?? 0,
        mate: res.bestMate ?? null,
      });
    } else {
      const b = new Chess(fen);
      const legal = b.moves({ verbose: true });
      let bestMove = null;
      let bestSan = '';
      let bestScore = -Infinity;
      for (const cand of legal) {
        b.move(cand);
        const s = -staticEvalCp(b);
        b.undo();
        if (s > bestScore) {
          bestScore = s;
          bestMove = `${cand.from}${cand.to}${cand.promotion || ''}`;
          bestSan = cand.san;
        }
      }
      positionEvals.push({
        bestMove,
        bestSan,
        evalCp: legal.length > 0 ? bestScore : staticEvalCp(b),
        mate: null,
      });
    }

    onProgress({
      current: i + 1,
      total: totalPlies + 1,
      pct: Math.round(((i + 1) / Math.max(1, totalPlies + 1)) * 100),
    });
  }

  const plyAnalyses = [];
  const accByColor = { w: [], b: [] };

  for (let i = 0; i < totalPlies; i += 1) {
    const move = verboseMoves[i];
    const ply = i + 1;
    const mover = move.color;
    const fenBefore = fens[i];
    const playedUci = `${move.from}${move.to}${move.promotion || ''}`;
    const evalBefore = positionEvals[i];
    const evalNext = positionEvals[i + 1];

    const cpBefore = evalBefore.evalCp;
    // evalNext is from opponent's POV -> negate for mover's POV
    const cpAfter = -evalNext.evalCp;
    const mateBefore = evalBefore.mate;
    const mateAfter = evalNext.mate !== null ? -evalNext.mate : null;

    const isBest =
      Boolean(evalBefore.bestMove) &&
      playedUci.toLowerCase() === evalBefore.bestMove.toLowerCase();

    const cpLoss = isBest ? 0 : Math.max(0, cpBefore - cpAfter);
    const winBefore = cpToWinProb(cpBefore);
    const winAfter = cpToWinProb(cpAfter);
    const wdlSwing = isBest ? 0 : Math.max(0, winBefore - winAfter);
    const accuracy = moveAccuracy(wdlSwing);
    accByColor[mover].push(accuracy);

    const missedMate = Boolean(
      ((mateBefore && mateBefore > 0) || cpBefore >= 9000) && cpAfter < 9000 && !isBest,
    );
    const blunderedMate = Boolean(
      (mateAfter && mateAfter < 0) || (cpBefore > -9000 && cpAfter <= -9000),
    );

    const classification = classifyReviewMove(cpLoss, isBest, {
      wdlSwing,
      missedMate,
      blunderedMate,
    });

    const weaknessCategory = tagWeakness({
      fen: fenBefore,
      ply,
      mover,
      playedUci,
      bestUci: evalBefore.bestMove,
      cpBefore,
      cpAfter,
      cpLoss,
      mateBefore,
      classification,
    });

    plyAnalyses.push({
      ply,
      moveNumber: Math.ceil(ply / 2),
      mover,
      fen: fenBefore,
      fenAfter: fens[i + 1],
      playedUci,
      playedSan: move.san,
      bestUci: evalBefore.bestMove || playedUci,
      bestSan: evalBefore.bestSan || move.san,
      cpBefore: Math.round(cpBefore),
      cpAfter: Math.round(cpAfter),
      evalWhiteCp: Math.round(mover === 'w' ? cpAfter : -cpAfter),
      cpLoss: Math.round(cpLoss),
      wdlSwing: Math.round(wdlSwing * 10) / 10,
      accuracy: Math.round(accuracy * 10) / 10,
      classification,
      weaknessCategory,
    });
  }

  const avg = (arr) => (arr.length ? Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) * 10) / 10 : 100);
  const whiteAccuracy = avg(accByColor.w);
  const blackAccuracy = avg(accByColor.b);

  const maiaPredict = maia
    ? (fen, elo) => maia.predict(fen, elo)
    : (fen, elo) => calibratedFallback(fen, elo);

  const criticalMoments = await rankCriticalMoments(plyAnalyses, {
    studentColor,
    studentElo,
    maiaPredict,
    maxMoments: 2,
  });

  const weaknessProfile = buildWeaknessProfile(plyAnalyses, studentColor);

  return {
    studentColor,
    studentElo,
    totalPlies,
    whiteAccuracy,
    blackAccuracy,
    studentAccuracy: studentColor === 'w' ? whiteAccuracy : blackAccuracy,
    plyAnalyses,
    criticalMoments,
    weaknessProfile,
  };
}
