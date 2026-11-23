/**
 * ACT-Eval Deterministic Chess Claim Verifier.
 *
 * Faithful JavaScript port of ``src/chessfm/coach/verifier.py``:
 * extracts piece-on-square claims, UCI/SAN/contextual pawn moves, captures,
 * checks (+), checkmates (#), and variation resets, then verifies every claim
 * deterministically against ``chess.js`` without calling any LLM.
 */

import { Chess } from '../vendor/chess.mjs';
import { uciToMoveObj } from './engine.js';

const PIECE_NAME_TO_TYPE = {
  king: 'k',
  queen: 'q',
  rook: 'r',
  bishop: 'b',
  knight: 'n',
  pawn: 'p',
};

const PIECE_ON_SQUARE_RE =
  /(?<![A-Za-z0-9])(?:(?<color>white|black)(?:'s)?\s+)?(?<piece>king|queen|rook|bishop|knight|pawn)\s+(?:on|at|sitting\s+on|placed\s+on|stationed\s+on)\s+(?<square>[a-h][1-8])(?![A-Za-z0-9])/gi;

const UCI_MOVE_RE = /(?<![A-Za-z0-9])([a-h][1-8][a-h][1-8][qrbnQRBN]?)(?![A-Za-z0-9])/g;

const SAN_MOVE_RE =
  /(?<![A-Za-z0-9])((?:O-O-O|O-O|0-0-0|0-0)[+#]?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][18]=[QRBN][+#]?|[a-h][1-8][+#])(?![A-Za-z0-9])/g;

const CONTEXTUAL_PAWN_MOVE_RE =
  /(?:\b\d+\.(?:\.\.)?\s*|\b(?:play|plays|playing|played|push|pushes|pushing|move|moves|moving|try|tries|trying|consider|after|reply|replies|with|then|or)\s+|\bbest\s+move\s+(?:is|:)\s*|\bchoice\s+is\s+|\binstead\s+(?:of|,)\s*|\b(?:expected|key|main|best)?\s*(?:continuation|line|variation)\s*:\s*)(?<move>[a-h][2-7])(?![A-Za-z0-9=+#])/gi;

const FOLLOWUP_PAWN_MOVE_RE = /^\s+(?<move>[a-h][2-7])(?![A-Za-z0-9=+#])/;

const VARIATION_RESET_RE =
  /(?:(?<!\d)[.!?;]\s+|\b(?:continuation|line|variation|instead|alternatively)\s*[:,\s])/i;

const CHECKMATE_CLAIM_RE =
  /(?:delivers|gives|is|achieves|scores|forces)\s+(?:a\s+)?(?:checkmate|mate)(?:\s+on\s+(?<sq1>[a-h][1-8]))?|(?:checkmate|mate)\s+on\s+(?<sq2>[a-h][1-8])/gi;

const CHECK_CLAIM_RE =
  /(?:delivers|gives|is|comes\s+with|with)\s+(?:a\s+)?check(?!mate)(?:\s+on\s+(?<sq1>[a-h][1-8]))?|check(?!mate)\s+on\s+(?<sq2>[a-h][1-8])/gi;

const NEGATION_WINDOW_RE =
  /\b(?:not|no|never|without|n't|avoid|avoids|avoiding|prevent|prevents|preventing|stop|stops|stopping|defend|defends|defending|threaten|threatens|threatening)\s+(?:\w+\s+){0,2}$/i;

const AVOIDANCE_WINDOW_RE =
  /\b(?:avoid|avoids|avoiding|not|never|don't\s+play|instead\s+of|blunder|mistake|bad\s+move)\s+(?:\w+\s+){0,2}$/i;

function spansOverlap(a, b) {
  return a[0] < b[1] && b[0] < a[1];
}

function isNegated(text, startIdx) {
  const prefix = text.slice(Math.max(0, startIdx - 40), startIdx);
  return NEGATION_WINDOW_RE.test(prefix);
}

function isAvoided(text, startIdx) {
  const prefix = text.slice(Math.max(0, startIdx - 30), startIdx);
  return AVOIDANCE_WINDOW_RE.test(prefix);
}

function sameSentence(text, endA, startB) {
  if (endA > startB) return false;
  const between = text.slice(endA, startB).replace(/\b\d+\.(?:\.\.)?/g, '');
  return !/[.!?;]/.test(between);
}

/**
 * Extract structured verifiable chess claims from natural-language coaching text.
 *
 * @param {string} text
 * @param {string | null} [_fen=null]
 */
export function extractClaims(text, _fen = null) {
  if (typeof text !== 'string' || !text.trim()) return [];

  const claims = [];
  const pieceSpans = [];
  const moveSpans = [];

  // 1. Piece-on-square claims
  PIECE_ON_SQUARE_RE.lastIndex = 0;
  let m;
  while ((m = PIECE_ON_SQUARE_RE.exec(text)) !== null) {
    const span = [m.index, m.index + m[0].length];
    pieceSpans.push(span);
    if (isNegated(text, m.index)) continue;
    claims.push({
      type: 'piece_on_square',
      raw: m[0],
      span,
      color: m.groups.color ? (m.groups.color.toLowerCase() === 'white' ? 'w' : 'b') : null,
      piece: PIECE_NAME_TO_TYPE[m.groups.piece.toLowerCase()],
      pieceName: m.groups.piece.toLowerCase(),
      square: m.groups.square.toLowerCase(),
    });
  }

  // 2. UCI move claims
  UCI_MOVE_RE.lastIndex = 0;
  while ((m = UCI_MOVE_RE.exec(text)) !== null) {
    const span = [m.index, m.index + m[0].length];
    if (pieceSpans.some((ps) => spansOverlap(span, ps))) continue;
    moveSpans.push(span);
    claims.push({
      type: 'uci_move',
      raw: m[1],
      span,
      uci: m[1].toLowerCase(),
      avoided: isAvoided(text, m.index),
    });
  }

  // 3. Unambiguous SAN move claims
  SAN_MOVE_RE.lastIndex = 0;
  while ((m = SAN_MOVE_RE.exec(text)) !== null) {
    const span = [m.index, m.index + m[0].length];
    if (moveSpans.some((ms) => spansOverlap(span, ms))) continue;
    if (pieceSpans.some((ps) => spansOverlap(span, ps))) continue;
    moveSpans.push(span);
    const san = m[1].replace(/^0-0-0/, 'O-O-O').replace(/^0-0/, 'O-O');
    claims.push({
      type: 'san_move',
      raw: m[1],
      span,
      san,
      avoided: isAvoided(text, m.index),
    });
  }

  // 4. Contextual pawn moves + fixed-point follow-up pawn moves
  CONTEXTUAL_PAWN_MOVE_RE.lastIndex = 0;
  while ((m = CONTEXTUAL_PAWN_MOVE_RE.exec(text)) !== null) {
    const moveStr = m.groups.move;
    const moveStart = m.index + m[0].length - moveStr.length;
    const span = [moveStart, moveStart + moveStr.length];
    if (moveSpans.some((ms) => spansOverlap(span, ms))) continue;
    if (pieceSpans.some((ps) => spansOverlap(span, ps))) continue;
    moveSpans.push(span);
    claims.push({
      type: 'san_move',
      raw: moveStr,
      span,
      san: moveStr,
      avoided: isAvoided(text, moveStart),
    });
  }

  // Fixed-point follow-up pawn moves right after an existing move span (e.g. "e4 e5 Nf3")
  let changed = true;
  while (changed) {
    changed = false;
    moveSpans.sort((a, b) => a[0] - b[0]);
    for (const [, end] of [...moveSpans]) {
      const rest = text.slice(end);
      const fm = FOLLOWUP_PAWN_MOVE_RE.exec(rest);
      if (!fm) continue;
      const moveStr = fm.groups.move;
      const moveStart = end + fm[0].length - moveStr.length;
      const span = [moveStart, moveStart + moveStr.length];
      if (moveSpans.some((ms) => spansOverlap(span, ms))) continue;
      if (pieceSpans.some((ps) => spansOverlap(span, ps))) continue;
      moveSpans.push(span);
      claims.push({
        type: 'san_move',
        raw: moveStr,
        span,
        san: moveStr,
        avoided: isAvoided(text, moveStart),
      });
      changed = true;
    }
  }

  // 5. Checkmate claims then check claims
  const checkSpans = [];
  CHECKMATE_CLAIM_RE.lastIndex = 0;
  while ((m = CHECKMATE_CLAIM_RE.exec(text)) !== null) {
    const span = [m.index, m.index + m[0].length];
    if (isNegated(text, m.index)) continue;
    checkSpans.push(span);
    claims.push({
      type: 'checkmate_claim',
      raw: m[0],
      span,
      square: (m.groups.sq1 || m.groups.sq2 || '').toLowerCase() || null,
    });
  }

  CHECK_CLAIM_RE.lastIndex = 0;
  while ((m = CHECK_CLAIM_RE.exec(text)) !== null) {
    const span = [m.index, m.index + m[0].length];
    if (isNegated(text, m.index)) continue;
    if (checkSpans.some((cs) => spansOverlap(span, cs))) continue;
    checkSpans.push(span);
    claims.push({
      type: 'check_claim',
      raw: m[0],
      span,
      square: (m.groups.sq1 || m.groups.sq2 || '').toLowerCase() || null,
    });
  }

  claims.sort((a, b) => a.span[0] - b.span[0]);
  return claims;
}

function tryPlayUci(board, uci) {
  const obj = uciToMoveObj(uci);
  if (!obj) return null;
  try {
    return board.move(obj);
  } catch {
    return null;
  }
}

function tryPlaySan(board, san) {
  const requiresCheck = san.endsWith('+');
  const requiresMate = san.endsWith('#');
  const baseSan = san.replace(/[+#]+$/, '');

  let moved = null;
  try {
    moved = board.move(san);
  } catch {
    try {
      moved = board.move(baseSan);
    } catch {
      return { ok: false, reason: `Illegal SAN move '${san}'` };
    }
  }
  if (!moved) {
    return { ok: false, reason: `Illegal SAN move '${san}'` };
  }

  if (requiresMate && !board.isCheckmate()) {
    board.undo();
    return { ok: false, reason: `Move '${san}' is legal but does not deliver checkmate (#)` };
  }
  if (requiresCheck && !board.inCheck()) {
    board.undo();
    return { ok: false, reason: `Move '${san}' is legal but does not give check (+)` };
  }
  return { ok: true, moved };
}

/**
 * Verify all extracted chess claims in ``text`` against the position ``fen``.
 * Mirrors ``src/chessfm/coach/verifier.py::verify_claims``.
 *
 * @param {string} fen
 * @param {string} text
 */
export function verifyClaims(fen, text) {
  const rootBoard = new Chess(fen);
  let varBoard = new Chess(fen);
  const claims = extractClaims(text, fen);

  const verifiedList = [];
  const violations = [];
  let lastMoveEnd = 0;
  let lastMoveCtx = null;

  for (const claim of claims) {
    let ok = false;
    let reason = '';

    if (claim.type === 'piece_on_square') {
      const piece = rootBoard.get(claim.square);
      if (!piece) {
        reason = `No piece on ${claim.square} (claimed ${claim.raw})`;
      } else if (piece.type !== claim.piece) {
        reason = `Expected ${claim.pieceName} on ${claim.square}, found ${piece.type}`;
      } else if (claim.color && piece.color !== claim.color) {
        reason = `Wrong color on ${claim.square} (claimed ${claim.raw})`;
      } else {
        ok = true;
      }
    } else if (claim.type === 'uci_move' || claim.type === 'san_move') {
      const between = text.slice(lastMoveEnd, claim.span[0]);
      if (lastMoveEnd > 0 && VARIATION_RESET_RE.test(between)) {
        varBoard = new Chess(fen);
        lastMoveCtx = null;
      }

      if (claim.type === 'uci_move') {
        let moved = tryPlayUci(varBoard, claim.uci);
        if (!moved && varBoard.fen() !== rootBoard.fen()) {
          varBoard = new Chess(fen);
          moved = tryPlayUci(varBoard, claim.uci);
        }
        if (moved) {
          ok = true;
          lastMoveCtx = {
            moved,
            inCheck: varBoard.inCheck(),
            isCheckmate: varBoard.isCheckmate(),
            end: claim.span[1],
          };
        } else {
          reason = `Illegal UCI move '${claim.uci}'`;
        }
      } else {
        let res = tryPlaySan(varBoard, claim.san);
        if (!res.ok && varBoard.fen() !== rootBoard.fen()) {
          varBoard = new Chess(fen);
          res = tryPlaySan(varBoard, claim.san);
        }
        if (res.ok) {
          ok = true;
          lastMoveCtx = {
            moved: res.moved,
            inCheck: varBoard.inCheck(),
            isCheckmate: varBoard.isCheckmate(),
            end: claim.span[1],
          };
        } else {
          reason = res.reason;
        }
      }
      lastMoveEnd = claim.span[1];
    } else if (claim.type === 'checkmate_claim' || claim.type === 'check_claim') {
      const wantMate = claim.type === 'checkmate_claim';
      const targetSq = claim.square;

      if (lastMoveCtx && sameSentence(text, lastMoveCtx.end, claim.span[0])) {
        const sqMatches = !targetSq || lastMoveCtx.moved.to === targetSq;
        const condMatches = wantMate ? lastMoveCtx.isCheckmate : lastMoveCtx.inCheck;
        if (sqMatches && condMatches) {
          ok = true;
        } else {
          reason = `Move ${lastMoveCtx.moved.san} does not satisfy '${claim.raw}'`;
        }
      } else {
        // Check if current position or any legal move satisfies the claim
        if (!targetSq && (wantMate ? rootBoard.isCheckmate() : rootBoard.inCheck())) {
          ok = true;
        } else {
          const probe = new Chess(fen);
          const legal = probe.moves({ verbose: true });
          for (const cand of legal) {
            if (targetSq && cand.to !== targetSq) continue;
            probe.move(cand);
            const matched = wantMate ? probe.isCheckmate() : probe.inCheck();
            probe.undo();
            if (matched) {
              ok = true;
              break;
            }
          }
          if (!ok) {
            reason = `No legal move satisfies '${claim.raw}'`;
          }
        }
      }
    }

    const enriched = { ...claim, verified: ok, reason: ok ? null : reason };
    verifiedList.push(enriched);
    if (!ok) {
      violations.push({
        type: claim.type,
        claim: claim.raw,
        reason,
      });
    }
  }

  const totalClaims = verifiedList.length;
  const verifiedClaims = verifiedList.filter((c) => c.verified).length;
  const failedClaims = totalClaims - verifiedClaims;
  const passed = failedClaims === 0;
  const factualityScore = totalClaims > 0 ? Math.round((verifiedClaims / totalClaims) * 10000) / 10000 : 1.0;

  const badgeText =
    totalClaims === 0
      ? 'ACT-Eval: Verified (no board claims)'
      : passed
        ? `ACT-Eval: ${verifiedClaims}/${totalClaims} claims verified`
        : `ACT-Eval: ${verifiedClaims}/${totalClaims} verified (${failedClaims} flagged)`;

  return {
    passed,
    factuality_score: factualityScore,
    totalClaims,
    verifiedClaims,
    failedClaims,
    claims: verifiedList,
    violations,
    badgeText,
  };
}
