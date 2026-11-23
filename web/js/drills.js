/**
 * SM-2 Spaced-Repetition Blunder Drills persisted in ``localStorage`` (``chessfm_drills_v1``).
 *
 * Faithful JavaScript port of ``src/chessfm/coach/drills.py`` including synchronous
 * SHA-256 ``makeDrillId(fen)`` parity and injectable storage for Node 22 unit tests.
 */

import { Chess } from '../vendor/chess.mjs';
import { normalizeBoardFen } from './openings.js';
import { uciToMoveObj } from './engine.js';

export const MIN_EF = 1.3;
export const DEFAULT_EF = 2.5;
export const ACCEPTABLE_CP_LOSS = 30.0;
export const INACCURATE_CP_LOSS = 100.0;
export const STORAGE_KEY = 'chessfm_drills_v1';

// ---------------------------------------------------------------------------
// Synchronous pure-JS SHA-256 (UTF-8) for 100% deterministic card_id parity
// ---------------------------------------------------------------------------

const SHA256_K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

function sha256Hex(str) {
  const bytes = new TextEncoder().encode(str);
  const bitLen = bytes.length * 8;
  const paddedLen = Math.ceil((bytes.length + 9) / 64) * 64;
  const buf = new Uint8Array(paddedLen);
  buf.set(bytes);
  buf[bytes.length] = 0x80;
  const view = new DataView(buf.buffer);
  view.setUint32(paddedLen - 8, Math.floor(bitLen / 0x100000000), false);
  view.setUint32(paddedLen - 4, bitLen >>> 0, false);

  let h0 = 0x6a09e667;
  let h1 = 0xbb67ae85;
  let h2 = 0x3c6ef372;
  let h3 = 0xa54ff53a;
  let h4 = 0x510e527f;
  let h5 = 0x9b05688c;
  let h6 = 0x1f83d9ab;
  let h7 = 0x5be0cd19;

  const w = new Uint32Array(64);
  for (let offset = 0; offset < paddedLen; offset += 64) {
    for (let i = 0; i < 16; i += 1) {
      w[i] = view.getUint32(offset + i * 4, false);
    }
    for (let i = 16; i < 64; i += 1) {
      const s0 = ((w[i - 15] >>> 7) | (w[i - 15] << 25)) ^ ((w[i - 15] >>> 18) | (w[i - 15] << 14)) ^ (w[i - 15] >>> 3);
      const s1 = ((w[i - 2] >>> 17) | (w[i - 2] << 15)) ^ ((w[i - 2] >>> 19) | (w[i - 2] << 13)) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }

    let a = h0;
    let b = h1;
    let c = h2;
    let d = h3;
    let e = h4;
    let f = h5;
    let g = h6;
    let h = h7;

    for (let i = 0; i < 64; i += 1) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const temp1 = (h + S1 + ch + SHA256_K[i] + w[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const temp2 = (S0 + maj) >>> 0;

      h = g;
      g = f;
      f = e;
      e = (d + temp1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (temp1 + temp2) >>> 0;
    }

    h0 = (h0 + a) >>> 0;
    h1 = (h1 + b) >>> 0;
    h2 = (h2 + c) >>> 0;
    h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0;
    h5 = (h5 + f) >>> 0;
    h6 = (h6 + g) >>> 0;
    h7 = (h7 + h) >>> 0;
  }

  return [h0, h1, h2, h3, h4, h5, h6, h7].map((v) => v.toString(16).padStart(8, '0')).join('');
}

/**
 * Format a Date or ISO date string as ``YYYY-MM-DD``.
 *
 * @param {Date | string} [date]
 * @returns {string}
 */
export function formatDateIso(date = new Date()) {
  if (typeof date === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return date;
  }
  const d = date instanceof Date ? date : new Date(date);
  return d.toISOString().slice(0, 10);
}

/**
 * Add ``days`` to a ``YYYY-MM-DD`` or Date value and return ``YYYY-MM-DD``.
 *
 * @param {Date | string} date
 * @param {number} days
 * @returns {string}
 */
export function addDaysIso(date, days) {
  const iso = formatDateIso(date);
  const d = new Date(`${iso}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() + Math.round(days));
  return d.toISOString().slice(0, 10);
}

/**
 * Compute deterministic drill card ID matching ``src/chessfm/coach/drills.py``.
 *
 * @param {string} fen
 * @returns {string}
 */
export function makeDrillId(fen) {
  const norm = normalizeBoardFen(fen);
  return `drill_${sha256Hex(norm).slice(0, 12)}`;
}

/**
 * Apply the SuperMemo-2 (SM-2) update rule to a drill card.
 * Mirrors ``src/chessfm/coach/drills.py::sm2_update``.
 *
 * @param {{
 *   ease_factor: number,
 *   interval_days: number,
 *   repetitions: number,
 *   attempts: number,
 *   successes: number,
 *   due_date: string,
 * }} card
 * @param {number} quality Integer in 0..5.
 * @param {Date | string} [now=new Date()]
 */
export function sm2Update(card, quality, now = new Date()) {
  if (!Number.isInteger(quality) || quality < 0 || quality > 5) {
    throw new RangeError(`SM-2 quality must be an integer in [0, 5], got ${quality}`);
  }

  let newReps;
  let newInterval;
  if (quality < 3) {
    newReps = 0;
    newInterval = 1;
  } else {
    if (card.repetitions === 0) {
      newInterval = 1;
    } else if (card.repetitions === 1) {
      newInterval = 6;
    } else {
      newInterval = Math.max(1, Math.round(card.interval_days * card.ease_factor));
    }
    newReps = card.repetitions + 1;
  }

  const diff = 5 - quality;
  const efDelta = 0.1 - diff * (0.08 + diff * 0.02);
  const newEf = Math.round(Math.max(MIN_EF, card.ease_factor + efDelta) * 10000) / 10000;
  const newDue = addDaysIso(now, newInterval);

  return {
    ...card,
    ease_factor: newEf,
    interval_days: newInterval,
    repetitions: newReps,
    due_date: newDue,
    attempts: (card.attempts || 0) + 1,
    successes: (card.successes || 0) + (quality >= 3 ? 1 : 0),
  };
}

/**
 * Grade a student's drill attempt against ``card``.
 * Mirrors ``src/chessfm/coach/drills.py::grade_drill_attempt``.
 *
 * @param {{ fen: string, best_move: string }} card
 * @param {string} playedUci
 * @param {number | null} [cpLoss=null]
 * @param {boolean} [isBest=false]
 */
export function gradeDrillAttempt(card, playedUci, cpLoss = null, isBest = false) {
  const played = String(playedUci || '').trim().toLowerCase();
  const best = String(card.best_move || '').trim().toLowerCase();

  // Verify move legality on card.fen
  const board = new Chess(card.fen);
  const moveObj = uciToMoveObj(played);
  if (!moveObj) {
    return { quality: 0, passed: false, cp_loss: null, status: 'illegal', label: 'Illegal Move' };
  }
  try {
    const moved = board.move(moveObj);
    if (!moved) {
      return { quality: 0, passed: false, cp_loss: null, status: 'illegal', label: 'Illegal Move' };
    }
  } catch {
    return { quality: 0, passed: false, cp_loss: null, status: 'illegal', label: 'Illegal Move' };
  }

  if (isBest || played === best) {
    return { quality: 5, passed: true, cp_loss: 0.0, status: 'exact', label: 'Exact Best' };
  }

  if (cpLoss !== null && cpLoss !== undefined && !Number.isNaN(Number(cpLoss))) {
    const loss = Math.max(0, Number(cpLoss));
    if (loss <= ACCEPTABLE_CP_LOSS) {
      return { quality: 4, passed: true, cp_loss: Math.round(loss), status: 'acceptable', label: 'Acceptable' };
    }
    if (loss <= INACCURATE_CP_LOSS) {
      return { quality: 2, passed: false, cp_loss: Math.round(loss), status: 'inaccurate', label: 'Inaccurate' };
    }
    return { quality: 1, passed: false, cp_loss: Math.round(loss), status: 'missed', label: 'Blunder' };
  }

  return { quality: 1, passed: false, cp_loss: null, status: 'missed', label: 'Missed' };
}

/**
 * Create a default in-memory storage adapter (for Node tests or SSR).
 */
export function createMemoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
  };
}

/**
 * Curated offline sample drills so new users can practice immediately without importing a game.
 */
export const SAMPLE_DRILLS = [
  {
    fen: 'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
    best_move: 'h5f7',
    category: 'missed_mate',
    student_mistake: 'g1f3',
    note: "Scholar's Mate pattern — spot the forced mate on f7.",
  },
  {
    fen: 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 b kq - 5 4',
    best_move: 'd7d6',
    category: 'opening_inaccuracy',
    student_mistake: 'f6e4',
    note: 'Italian Game — solidify e5 before tactics open the e-file.',
  },
  {
    fen: '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1',
    best_move: 'd1d8',
    category: 'missed_mate',
    student_mistake: 'g1f1',
    note: 'Back-rank checkmate — punish the boxed-in king.',
  },
  {
    fen: 'r2q1rk1/ppp2ppp/2n1bn2/3pp3/1bP5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8',
    best_move: 'c4d5',
    category: 'missed_tactic',
    student_mistake: 'c1g5',
    note: 'Central tension — win control of the center with a clean capture.',
  },
];

/**
 * Persistent SM-2 Blunder Drill Deck backed by ``localStorage`` or an injectable storage map.
 */
export class DrillDeck {
  /**
   * @param {{ storage?: { getItem: Function, setItem: Function, removeItem?: Function }, storageKey?: string }} [options]
   */
  constructor(options = {}) {
    this.storageKey = options.storageKey || STORAGE_KEY;
    this.storage =
      options.storage ||
      (typeof globalThis.localStorage !== 'undefined' ? globalThis.localStorage : createMemoryStorage());
    /** @type {Array<any>} */
    this.cards = [];
    this._load();
  }

  _load() {
    try {
      const raw = this.storage.getItem(this.storageKey);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed?.cards)) {
        this.cards = parsed.cards.filter(
          (c) =>
            c &&
            typeof c === 'object' &&
            typeof c.card_id === 'string' &&
            typeof c.fen === 'string' &&
            typeof c.due_date === 'string',
        );
      }
    } catch {
      this.cards = [];
    }
  }

  _save() {
    try {
      this.storage.setItem(
        this.storageKey,
        JSON.stringify({ version: 1, cards: this.cards }),
      );
    } catch {
      // ignore quota errors
    }
  }

  /**
   * Add a drill card, deduplicating by 4-field canonical FEN.
   *
   * @param {{ fen: string, best_move: string, category?: string, student_mistake?: string, note?: string }} data
   * @param {Date | string} [now=new Date()]
   * @returns {{ card: any, added: boolean }}
   */
  addCard(data, now = new Date()) {
    const cardId = makeDrillId(data.fen);
    const existing = this.cards.find((c) => c.card_id === cardId);
    if (existing) {
      return { card: existing, added: false };
    }
    const card = {
      card_id: cardId,
      fen: data.fen,
      best_move: String(data.best_move || '').toLowerCase(),
      category: data.category || 'positional_drift',
      student_mistake: String(data.student_mistake || '').toLowerCase(),
      note: data.note || '',
      ease_factor: DEFAULT_EF,
      interval_days: 0,
      repetitions: 0,
      due_date: formatDateIso(now),
      attempts: 0,
      successes: 0,
    };
    this.cards.push(card);
    this._save();
    return { card, added: true };
  }

  /**
   * Add a drill card directly from a Critical Moment object returned by ``rankCriticalMoments``.
   *
   * @param {{ fen: string, bestUci: string, playedUci?: string, weaknessCategory?: string, trapContext?: string }} moment
   * @param {{ note?: string, now?: Date | string }} [meta]
   */
  addFromCriticalMoment(moment, meta = {}) {
    return this.addCard(
      {
        fen: moment.fen,
        best_move: moment.bestUci || moment.best_move,
        category: moment.weaknessCategory || moment.category || 'positional_drift',
        student_mistake: moment.playedUci || moment.student_mistake || '',
        note: meta.note || moment.trapContext || '',
      },
      meta.now || new Date(),
    );
  }

  /**
   * Seed the curated sample drills into the deck (deduplicated).
   *
   * @param {Date | string} [now=new Date()]
   * @returns {number} Number of newly added cards.
   */
  seedSampleDrills(now = new Date()) {
    let addedCount = 0;
    for (const sample of SAMPLE_DRILLS) {
      const { added } = this.addCard(sample, now);
      if (added) addedCount += 1;
    }
    return addedCount;
  }

  /**
   * Return all cards due on or before ``now``, ordered by ``(due_date, ease_factor, card_id)``.
   *
   * @param {Date | string} [now=new Date()]
   * @param {number | null} [limit=null]
   */
  getDueCards(now = new Date(), limit = null) {
    const today = formatDateIso(now);
    const due = this.cards
      .filter((c) => c.due_date <= today)
      .sort((a, b) => {
        if (a.due_date !== b.due_date) return a.due_date.localeCompare(b.due_date);
        if (a.ease_factor !== b.ease_factor) return a.ease_factor - b.ease_factor;
        return a.card_id.localeCompare(b.card_id);
      });
    return limit !== null ? due.slice(0, limit) : due;
  }

  /**
   * Grade an attempt on ``cardId`` and persist the SM-2 update.
   *
   * @param {string} cardId
   * @param {string} playedUci
   * @param {number | null} [cpLoss=null]
   * @param {boolean} [isBest=false]
   * @param {Date | string} [now=new Date()]
   */
  recordAttempt(cardId, playedUci, cpLoss = null, isBest = false, now = new Date()) {
    const idx = this.cards.findIndex((c) => c.card_id === cardId);
    if (idx === -1) {
      throw new Error(`Drill card not found: ${cardId}`);
    }
    const card = this.cards[idx];
    const grade = gradeDrillAttempt(card, playedUci, cpLoss, isBest);
    const updated = sm2Update(card, grade.quality, now);
    this.cards[idx] = updated;
    this._save();
    return {
      card_id: cardId,
      ...grade,
      card: updated,
    };
  }

  removeCard(cardId) {
    const before = this.cards.length;
    this.cards = this.cards.filter((c) => c.card_id !== cardId);
    const removed = this.cards.length < before;
    if (removed) this._save();
    return removed;
  }

  getStats(now = new Date()) {
    const today = formatDateIso(now);
    const totalCards = this.cards.length;
    let dueCards = 0;
    let masteredCards = 0;
    let totalAttempts = 0;
    let totalSuccesses = 0;
    let efSum = 0;
    const byCategory = {};

    for (const c of this.cards) {
      if (c.due_date <= today) dueCards += 1;
      if (c.repetitions >= 3 && c.interval_days >= 14) masteredCards += 1;
      totalAttempts += c.attempts || 0;
      totalSuccesses += c.successes || 0;
      efSum += c.ease_factor || DEFAULT_EF;
      byCategory[c.category] = (byCategory[c.category] || 0) + 1;
    }

    return {
      total_cards: totalCards,
      due_cards: dueCards,
      mastered_cards: masteredCards,
      total_attempts: totalAttempts,
      total_successes: totalSuccesses,
      success_rate: totalAttempts > 0 ? Math.round((totalSuccesses / totalAttempts) * 10000) / 10000 : 0,
      average_ease_factor: totalCards > 0 ? Math.round((efSum / totalCards) * 10000) / 10000 : DEFAULT_EF,
      by_category: byCategory,
    };
  }

  exportJson() {
    return JSON.stringify({ version: 1, cards: this.cards }, null, 2);
  }

  importJson(jsonText) {
    const parsed = JSON.parse(jsonText);
    const incoming = Array.isArray(parsed) ? parsed : parsed?.cards;
    if (!Array.isArray(incoming)) {
      throw new Error('Invalid drill deck JSON format');
    }
    let added = 0;
    for (const item of incoming) {
      if (!item || typeof item !== 'object' || !item.fen || !item.best_move) continue;
      const cardId = item.card_id || makeDrillId(item.fen);
      if (this.cards.some((c) => c.card_id === cardId)) continue;
      const rawDue = item.due_date;
      const validDue =
        rawDue && !Number.isNaN(new Date(rawDue).getTime()) ? rawDue : new Date();
      this.cards.push({
        card_id: cardId,
        fen: item.fen,
        best_move: String(item.best_move).toLowerCase(),
        category: item.category || 'positional_drift',
        student_mistake: String(item.student_mistake || '').toLowerCase(),
        note: item.note || '',
        ease_factor: Math.max(MIN_EF, Number(item.ease_factor) || DEFAULT_EF),
        interval_days: Math.max(0, Number(item.interval_days) || 0),
        repetitions: Math.max(0, Number(item.repetitions) || 0),
        due_date: formatDateIso(validDue),
        attempts: Math.max(0, Number(item.attempts) || 0),
        successes: Math.max(0, Number(item.successes) || 0),
      });
      added += 1;
    }
    if (added > 0) this._save();
    return added;
  }
}
