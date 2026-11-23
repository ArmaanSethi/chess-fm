/**
 * 3-Level 0%-Spoiler Socratic Hint Generator and Spoiler Guard.
 *
 * Faithful JavaScript port of ``src/chessfm/coach/socratic.py``:
 * - Level 1: Strategic/tactical nudge with ZERO move spoilers and ZERO square names.
 * - Level 2: Piece/motif/zone clue with ZERO move spoilers (never bare destination square for pawns).
 * - Level 3: Full solution revealing SAN + UCI, concrete rationale, and verified PV continuation.
 */

import { Chess } from '../vendor/chess.mjs';
import { uciToMoveObj } from './engine.js';

const PIECE_NAMES = {
  p: 'pawn',
  n: 'knight',
  b: 'bishop',
  r: 'rook',
  q: 'queen',
  k: 'king',
};

const TACTIC_VALUES = {
  p: 1,
  n: 3,
  b: 3,
  r: 5,
  q: 9,
  k: 100,
};

const CENTER_SQUARES = new Set(['d4', 'd5', 'e4', 'e5']);

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function hasToken(text, token) {
  if (!token) return false;
  const re = new RegExp(`(?<![A-Za-z0-9])${escapeRegex(token)}(?![A-Za-z0-9])`, 'i');
  return re.test(text);
}

/**
 * Check whether ``text`` leaks ``bestMoveUci`` (or its SAN / from+to square pair).
 * Mirrors ``src/chessfm/coach/socratic.py::contains_move_spoiler``.
 *
 * The 3rd parameter accepts either a SAN string (e.g. ``"Nf3"``) OR a FEN string
 * (from which the SAN is computed automatically).
 *
 * @param {string} text
 * @param {string} bestMoveUci
 * @param {string | null} [bestMoveSanOrFen=null]
 * @returns {boolean}
 */
export function containsMoveSpoiler(text, bestMoveUci, bestMoveSanOrFen = null) {
  if (!text || !bestMoveUci) return false;
  const uciLower = String(bestMoveUci).trim().toLowerCase();
  if (uciLower.length < 4) return false;

  const fromSq = uciLower.slice(0, 2);
  const toSq = uciLower.slice(2, 4);
  const promoChar = uciLower.length >= 5 ? uciLower[4] : null;

  // 1. UCI & compact promotion variants (e.g. e7e8n, e8N, e8=N)
  const uciVariants = new Set([
    uciLower,
    `${fromSq}${toSq}`,
    `${fromSq}-${toSq}`,
    `${fromSq}->${toSq}`,
    `${fromSq}\u2192${toSq}`,
  ]);
  if (promoChar) {
    uciVariants.add(`${toSq}${promoChar}`);
    uciVariants.add(`${toSq}=${promoChar}`);
  }
  for (const variant of uciVariants) {
    if (hasToken(text, variant)) return true;
  }

  // 2. SAN variants & piece inference
  let san = null;
  let movedPieceName = null;
  if (typeof bestMoveSanOrFen === 'string' && bestMoveSanOrFen.trim()) {
    if (bestMoveSanOrFen.includes('/') && bestMoveSanOrFen.includes(' ')) {
      try {
        const b = new Chess(bestMoveSanOrFen);
        const obj = uciToMoveObj(uciLower);
        const moved = obj ? b.move(obj) : null;
        if (moved) {
          san = moved.san;
          movedPieceName = PIECE_NAMES[moved.piece] || null;
        }
      } catch {
        san = null;
      }
    } else {
      san = bestMoveSanOrFen.trim();
    }
  }

  if (san) {
    const stripped = san.replace(/[+#]+$/, '');
    const sanVariants = new Set([san, stripped]);
    if (stripped.includes('=')) {
      sanVariants.add(stripped.replace('=', ''));
    }
    if (stripped === 'O-O') {
      sanVariants.add('0-0');
      sanVariants.add('0-0+');
      sanVariants.add('0-0#');
    } else if (stripped === 'O-O-O') {
      sanVariants.add('0-0-0');
      sanVariants.add('0-0-0+');
      sanVariants.add('0-0-0#');
    }
    for (const variant of sanVariants) {
      if (hasToken(text, variant)) return true;
    }

    if (!movedPieceName) {
      const sanPieceMap = { N: 'knight', B: 'bishop', R: 'rook', Q: 'queen', K: 'king' };
      if (sanPieceMap[stripped[0]]) {
        movedPieceName = sanPieceMap[stripped[0]];
      } else if (/^[a-h]/.test(stripped)) {
        movedPieceName = 'pawn';
      }
    }
  }

  // 3. Both origin and destination squares mentioned anywhere in text
  if (hasToken(text, fromSq) && hasToken(text, toSq)) {
    return true;
  }

  // 4. Natural-language piece + destination square spoiler (e.g. "Move your knight to f3", "queen on f7")
  if (hasToken(text, toSq)) {
    if (movedPieceName && hasToken(text, movedPieceName)) {
      return true;
    }
    if (promoChar && PIECE_NAMES[promoChar] && hasToken(text, PIECE_NAMES[promoChar])) {
      return true;
    }
  }

  return false;
}

function containsAnyMoveSquare(text, bestMoveUci) {
  const uciLower = String(bestMoveUci || '').trim().toLowerCase();
  if (uciLower.length < 4) return false;
  return hasToken(text, uciLower.slice(0, 2)) || hasToken(text, uciLower.slice(2, 4));
}

function squareZone(sq) {
  if (CENTER_SQUARES.has(sq)) return 'the center';
  const file = sq[0];
  if (file === 'f' || file === 'g' || file === 'h') return 'the kingside';
  if (file === 'a' || file === 'b' || file === 'c') return 'the queenside';
  return 'the central files';
}

function analyzeBestMove(fen, bestMoveUci) {
  const board = new Chess(fen);
  const mover = board.turn();
  const opponent = mover === 'w' ? 'b' : 'w';
  const moveObj = uciToMoveObj(bestMoveUci);
  if (!moveObj) {
    throw new Error(`Invalid UCI move: ${bestMoveUci}`);
  }
  const moved = board.move(moveObj);
  if (!moved) {
    throw new Error(`Illegal move ${bestMoveUci} for FEN ${fen}`);
  }

  const pieceName = PIECE_NAMES[moved.piece] || 'piece';
  const isCastling = moved.flags.includes('k') || moved.flags.includes('q');
  const isKingsideCastle = moved.flags.includes('k');
  const isPromotion = Boolean(moved.promotion);
  const promoName = moved.promotion ? PIECE_NAMES[moved.promotion] : null;
  const isEnPassant = moved.flags.includes('e');
  const isCapture = Boolean(moved.captured);
  const capturedName = moved.captured ? PIECE_NAMES[moved.captured] : null;
  const givesCheck = board.inCheck();
  const isCheckmate = board.isCheckmate();

  // Discovered check: opponent king is in check, but moved piece at `to` is not attacking opponent king
  let isDiscoveredCheck = false;
  if (givesCheck) {
    let oppKingSq = null;
    for (const row of board.board()) {
      for (const cell of row) {
        if (cell && cell.color === opponent && cell.type === 'k') {
          oppKingSq = cell.square;
          break;
        }
      }
      if (oppKingSq) break;
    }
    if (oppKingSq) {
      const checkers = board.attackers(oppKingSq, mover);
      isDiscoveredCheck = checkers.length > 0 && !checkers.includes(moved.to);
    }
  }

  // Fork detection: moved piece at `to` attacks >= 2 valuable or undefended enemy targets
  let isFork = false;
  const moverVal = TACTIC_VALUES[moved.piece] || 1;
  let forkTargets = 0;
  for (const row of board.board()) {
    for (const cell of row) {
      if (!cell || cell.color !== opponent) continue;
      const attackers = board.attackers(cell.square, mover);
      if (!attackers.includes(moved.to)) continue;
      const targetVal = TACTIC_VALUES[cell.type] || 1;
      const oppDefenders = board.attackers(cell.square, opponent);
      if (
        cell.type === 'k' ||
        targetVal > moverVal ||
        (cell.type !== 'p' && oppDefenders.length === 0)
      ) {
        forkTargets += 1;
      }
    }
  }
  if (forkTargets >= 2) isFork = true;

  const backRank = mover === 'w' ? '1' : '8';
  const isDevelopment = (moved.piece === 'n' || moved.piece === 'b') && moved.from[1] === backRank;
  const seventhRank = mover === 'w' ? '7' : '2';
  const isSeventhRankRook = moved.piece === 'r' && moved.to[1] === seventhRank;
  const isBackRankMate = isCheckmate && (moved.to[1] === '8' || moved.to[1] === '1');

  const themes = [];
  if (isBackRankMate) themes.push('back-rank checkmate');
  else if (isCheckmate) themes.push('checkmate');
  if (isPromotion) themes.push('pawn promotion');
  if (isCastling) themes.push('king safety');
  if (isDiscoveredCheck) themes.push('discovered check');
  else if (givesCheck && !isCheckmate) themes.push('forcing check');
  if (isFork) themes.push('fork');
  if (isCapture) themes.push('winning material');
  if (isSeventhRankRook) themes.push('7th rank invasion');
  if (isDevelopment) themes.push('piece development');
  if (CENTER_SQUARES.has(moved.to)) themes.push('center control');
  if (themes.length === 0) themes.push('piece activity');

  return {
    san: moved.san,
    uci: bestMoveUci.toLowerCase(),
    pieceType: moved.piece,
    pieceName,
    from: moved.from,
    to: moved.to,
    toFile: moved.to[0],
    zone: squareZone(moved.to),
    isCastling,
    isKingsideCastle,
    isPromotion,
    promoName,
    isEnPassant,
    isCapture,
    capturedName,
    givesCheck,
    isCheckmate,
    isDiscoveredCheck,
    isFork,
    isDevelopment,
    isSeventhRankRook,
    themes,
  };
}

function verifyContinuation(fen, pvUci, maxPlies = 6) {
  if (!Array.isArray(pvUci) || pvUci.length === 0) return [];
  const board = new Chess(fen);
  const sanList = [];
  for (const uci of pvUci.slice(0, maxPlies)) {
    const obj = uciToMoveObj(uci);
    if (!obj) break;
    try {
      const moved = board.move(obj);
      if (!moved) break;
      sanList.push(moved.san);
    } catch {
      break;
    }
  }
  return sanList;
}

/**
 * Generate a deterministic 3-level Socratic hint with guaranteed 0% spoiler leakage on L1 & L2.
 *
 * @param {string} fen
 * @param {string} bestMoveUci
 * @param {1 | 2 | 3} [level=1]
 * @param {{ pvUci?: string[] }} [context={}]
 */
export function generateSocraticHint(fen, bestMoveUci, level = 1, context = {}) {
  const lvl = Number(level);
  if (lvl !== 1 && lvl !== 2 && lvl !== 3) {
    throw new RangeError(`Hint level must be 1, 2, or 3 (got ${level})`);
  }

  const info = analyzeBestMove(fen, bestMoveUci);
  const continuationSan = verifyContinuation(fen, context.pvUci || [info.uci], 6);

  let hint = '';

  if (lvl === 1) {
    if (info.isCheckmate) {
      hint = 'Look for a forcing check that immediately ends the game.';
    } else if (info.isPromotion) {
      hint = 'A passed pawn is ready to transform the position — look at your pawn advance.';
    } else if (info.isCastling) {
      hint = 'Prioritize king safety and connect your rooks before launching an attack.';
    } else if (info.isDiscoveredCheck) {
      hint = `Consider how moving your ${info.pieceName} can unleash a hidden attack on the enemy king.`;
    } else if (info.isFork) {
      hint = `Your ${info.pieceName} has a tactical double attack available — scan undefended targets.`;
    } else if (info.givesCheck) {
      hint = `Start with forcing checks: your ${info.pieceName} can seize the initiative.`;
    } else if (info.isCapture) {
      hint = `Scan all captures carefully — your ${info.pieceName} can win or trade key material.`;
    } else if (info.isDevelopment) {
      hint = `Focus on rapid piece development — activate your ${info.pieceName} from the back rank.`;
    } else {
      hint = `Look closely at how your ${info.pieceName} can improve its activity and coordination.`;
    }

    if (containsMoveSpoiler(hint, info.uci, info.san) || containsAnyMoveSquare(hint, info.uci)) {
      hint = `Look closely at what your ${info.pieceName} can accomplish.`;
    }
  } else if (lvl === 2) {
    if (info.isCastling) {
      hint = `Tuck your king away safely by castling on ${info.isKingsideCastle ? 'the kingside' : 'the queenside'}.`;
    } else if (info.pieceType === 'p') {
      // Never mention bare destination square for pawn pushes (SAN == square name)
      if (info.isPromotion) {
        hint = `Advance your pawn on the ${info.toFile}-file to promote to a ${info.promoName}.`;
      } else if (info.isCapture) {
        hint = `Use your pawn to capture the enemy ${info.capturedName} in ${info.zone}.`;
      } else {
        hint = `Push a pawn toward ${info.zone} along the ${info.toFile}-file to gain space.`;
      }
    } else if (info.isCheckmate) {
      hint = `Your ${info.pieceName} can deliver checkmate in ${info.zone}.`;
    } else if (info.isFork) {
      hint = `Your ${info.pieceName} has a decisive fork in ${info.zone} attacking multiple enemy targets.`;
    } else if (info.isCapture) {
      hint = `Use your ${info.pieceName} to capture the enemy ${info.capturedName} in ${info.zone}.`;
    } else if (info.givesCheck) {
      hint = `Bring your ${info.pieceName} into ${info.zone} with a forcing check.`;
    } else if (info.isDevelopment) {
      hint = `Develop your ${info.pieceName} toward ${info.zone} to strengthen your control.`;
    } else {
      hint = `Activate your ${info.pieceName} toward ${info.zone} to strengthen your control.`;
    }

    if (containsMoveSpoiler(hint, info.uci, info.san) || containsAnyMoveSquare(hint, info.uci)) {
      hint = `Look for a tactical opportunity with your ${info.pieceName} in ${info.zone}.`;
    }
  } else {
    // Level 3: Full solution + rationale + verified continuation
    const reasons = [];
    if (info.isCheckmate) reasons.push('delivers checkmate');
    else if (info.isDiscoveredCheck) reasons.push('unleashes a discovered check');
    else if (info.givesCheck) reasons.push('gives a forcing check');
    if (info.isFork) reasons.push('creates a decisive fork');
    if (info.isCapture) reasons.push(`captures the ${info.capturedName}`);
    if (info.isPromotion) reasons.push(`promotes to a ${info.promoName}`);
    if (info.isCastling) reasons.push('secures the king and connects the rooks');
    if (info.isSeventhRankRook) reasons.push('invades the 7th rank');
    if (info.isDevelopment) reasons.push('develops a minor piece into active play');
    if (reasons.length === 0) reasons.push(`improves your ${info.pieceName} in ${info.zone}`);

    hint = `Play ${info.san} (${info.uci}), which ${reasons.join(' and ')}.`;
    if (continuationSan.length > 1) {
      hint += ` Verified continuation: ${continuationSan.join(' ')}.`;
    }
  }

  const spoilerFree = lvl === 3 ? false : !containsMoveSpoiler(hint, info.uci, info.san);

  return {
    level: lvl,
    hint,
    spoiler_free: spoilerFree,
    themes: info.themes,
    move_san: info.san,
    move_uci: info.uci,
    continuation_san: continuationSan,
  };
}
