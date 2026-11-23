/**
 * ECO Opening Coach with transposition-aware 4-field FEN index and deviation detection.
 *
 * Loads the precomputed 43-line ``web/data/openings.json`` exported from
 * ``src/chessfm/coach/openings.py`` via ``scripts/export_openings_json.py``.
 */

import { Chess } from '../vendor/chess.mjs';

/**
 * Normalize a full FEN string to its 4-field board identity:
 * ``[piece_placement, active_color, castling_rights, en_passant]``.
 * Mirrors ``src/chessfm/data_generation/phase_sampler.py::normalize_board_fen``.
 *
 * @param {string} fen
 * @returns {string}
 */
export function normalizeBoardFen(fen) {
  if (typeof fen !== 'string') {
    throw new Error('FEN must be a string');
  }
  const parts = fen.trim().split(/\s+/);
  if (parts.length < 4) {
    throw new Error(`Invalid FEN (expected at least 4 fields): ${fen}`);
  }
  return parts.slice(0, 4).join(' ');
}

/**
 * Transposition-aware ECO Opening Book and repertoire deviation detector.
 */
export class OpeningBook {
  /**
   * @param {Array<{
   *   eco: string,
   *   name: string,
   *   moves: string[],
   *   moves_uci?: string[],
   *   white_plan: string,
   *   black_plan: string,
   *   common_trap: string,
   *   fens?: string[],
   *   final_fen?: string,
   * }>} entries
   */
  constructor(entries = []) {
    this.entries = entries;
    /** @type {Map<string, typeof entries[0]>} Exact final-position 4-field FEN -> entry */
    this.fenIndex = new Map();
    /** @type {Set<string>} All intermediate + final 4-field FENs in the book */
    this.bookPositions = new Set();

    for (const entry of entries) {
      let fens = entry.fens;
      if (!Array.isArray(fens) || fens.length === 0) {
        fens = [];
        const board = new Chess();
        for (const san of entry.moves || []) {
          const moved = board.move(san);
          if (!moved) break;
          fens.push(normalizeBoardFen(board.fen()));
        }
      }
      const finalFen = entry.final_fen || fens[fens.length - 1];
      if (finalFen) {
        // Prefer deeper/more specific line if two entries share a final FEN
        const existing = this.fenIndex.get(finalFen);
        if (!existing || (entry.moves?.length || 0) >= (existing.moves?.length || 0)) {
          this.fenIndex.set(finalFen, { ...entry, fens, final_fen: finalFen });
        }
      }
      for (const f of fens) {
        this.bookPositions.add(f);
      }
    }
  }

  /**
   * Fetch and construct an OpeningBook from ``web/data/openings.json``.
   *
   * @param {string} [url='./data/openings.json']
   * @returns {Promise<OpeningBook>}
   */
  static async fromUrl(url = './data/openings.json') {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to load openings book (HTTP ${resp.status})`);
    }
    const payload = await resp.json();
    return new OpeningBook(Array.isArray(payload) ? payload : payload.entries || []);
  }

  get size() {
    return this.entries.length;
  }

  /**
   * Identify the deepest matching ECO opening for a FEN and/or SAN move list.
   * Transposition-aware via 4-field canonical FEN lookup.
   *
   * @param {string | null} [fen=null]
   * @param {string[] | null} [movesSan=null]
   */
  identifyOpening(fen = null, movesSan = null) {
    let deepestMatch = null;

    if (Array.isArray(movesSan) && movesSan.length > 0) {
      const board = new Chess();
      for (const san of movesSan) {
        let moved = null;
        try {
          moved = board.move(san);
        } catch {
          break;
        }
        if (!moved) break;
        const key = normalizeBoardFen(board.fen());
        const hit = this.fenIndex.get(key);
        if (hit) {
          deepestMatch = hit;
        }
      }
      if (deepestMatch) return deepestMatch;
    }

    if (fen) {
      try {
        const key = normalizeBoardFen(fen);
        return this.fenIndex.get(key) || null;
      } catch {
        return null;
      }
    }

    return null;
  }

  /**
   * Search openings by ECO code, name, or plan keywords.
   *
   * @param {string} query
   */
  search(query = '') {
    const q = String(query || '').trim().toLowerCase();
    if (!q) return this.entries;
    return this.entries.filter(
      (e) =>
        e.eco.toLowerCase().includes(q) ||
        e.name.toLowerCase().includes(q) ||
        e.white_plan.toLowerCase().includes(q) ||
        e.black_plan.toLowerCase().includes(q) ||
        e.common_trap.toLowerCase().includes(q),
    );
  }

  /**
   * Detect the last book ply and first out-of-book deviation in a game.
   *
   * @param {string | string[]} pgnOrMoves PGN string or SAN move array.
   * @param {Array<{ ply: number, cpLoss?: number, cp_loss?: number }> | null} [evals=null]
   */
  detectDeviation(pgnOrMoves, evals = null) {
    let moves = [];
    if (Array.isArray(pgnOrMoves)) {
      moves = pgnOrMoves;
    } else if (typeof pgnOrMoves === 'string' && pgnOrMoves.trim()) {
      const loader = new Chess();
      try {
        loader.loadPgn(pgnOrMoves);
        moves = loader.history();
      } catch {
        moves = [];
      }
    }

    const board = new Chess();
    let matchedOpening = null;
    let lastBookPly = 0;

    for (let i = 0; i < moves.length; i += 1) {
      const ply = i + 1;
      let moved = null;
      try {
        moved = board.move(moves[i]);
      } catch {
        break;
      }
      if (!moved) break;
      const key = normalizeBoardFen(board.fen());
      const exact = this.fenIndex.get(key);
      if (exact) {
        matchedOpening = exact;
        lastBookPly = ply;
      } else if (this.bookPositions.has(key)) {
        lastBookPly = ply;
      }
    }

    const deviationPly = lastBookPly < moves.length ? lastBookPly + 1 : null;
    const deviationMoveSan = deviationPly ? moves[deviationPly - 1] : null;

    let evalDropCp = null;
    if (deviationPly && Array.isArray(evals)) {
      const hit = evals.find((e) => e.ply === deviationPly);
      if (hit) {
        evalDropCp = hit.cpLoss ?? hit.cp_loss ?? null;
      }
    }

    return {
      opening: matchedOpening,
      eco: matchedOpening?.eco || null,
      name: matchedOpening?.name || 'Unclassified Opening',
      lastBookPly,
      deviationPly,
      deviationMoveSan,
      evalDropCp,
      whitePlan: matchedOpening?.white_plan || null,
      blackPlan: matchedOpening?.black_plan || null,
      commonTrap: matchedOpening?.common_trap || null,
    };
  }
}
