import test from 'node:test';
import assert from 'node:assert/strict';
import {
  MIN_EF,
  DEFAULT_EF,
  ACCEPTABLE_CP_LOSS,
  INACCURATE_CP_LOSS,
  makeDrillId,
  sm2Update,
  gradeDrillAttempt,
  createMemoryStorage,
  DrillDeck,
} from '../js/drills.js';

const MATE_FEN = '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1';

test('makeDrillId is deterministic across halfmove/fullmove changes', () => {
  const id1 = makeDrillId('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1');
  const id2 = makeDrillId('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 14 42');
  assert.equal(id1, id2);
  assert.match(id1, /^drill_[0-9a-f]{12}$/);
});

test('sm2Update follows SM-2 intervals, EF bounds, and reset on failure', () => {
  assert.equal(MIN_EF, 1.3);
  assert.equal(DEFAULT_EF, 2.5);
  assert.equal(ACCEPTABLE_CP_LOSS, 30.0);
  assert.equal(INACCURATE_CP_LOSS, 100.0);

  let card = {
    ease_factor: DEFAULT_EF,
    interval_days: 0,
    repetitions: 0,
    due_date: '2026-11-23',
    attempts: 0,
    successes: 0,
  };

  card = sm2Update(card, 5, '2026-11-23');
  assert.equal(card.repetitions, 1);
  assert.equal(card.interval_days, 1);
  assert.equal(card.due_date, '2026-11-24');
  assert.ok(card.ease_factor > 2.5);

  card = sm2Update(card, 5, '2026-11-24');
  assert.equal(card.repetitions, 2);
  assert.equal(card.interval_days, 6);
  assert.equal(card.due_date, '2026-11-30');

  card = sm2Update(card, 5, '2026-11-30');
  assert.equal(card.repetitions, 3);
  assert.ok(card.interval_days >= 14);

  // Failure resets repetitions to 0 and interval to 1
  card = sm2Update(card, 1, '2026-12-16');
  assert.equal(card.repetitions, 0);
  assert.equal(card.interval_days, 1);

  // Repeated failures never drop ease_factor below MIN_EF (1.3)
  for (let i = 0; i < 10; i += 1) {
    card = sm2Update(card, 0, '2026-12-17');
  }
  assert.equal(card.ease_factor, MIN_EF);
});

test('gradeDrillAttempt enforces exact, acceptable, inaccurate, missed, and illegal thresholds', () => {
  const card = { fen: MATE_FEN, best_move: 'd1d8' };
  assert.equal(gradeDrillAttempt(card, 'd1d8').status, 'exact');
  assert.equal(gradeDrillAttempt(card, 'd1d8').quality, 5);
  assert.equal(gradeDrillAttempt(card, 'd1d7', 25).status, 'acceptable');
  assert.equal(gradeDrillAttempt(card, 'd1d7', 25).quality, 4);
  assert.equal(gradeDrillAttempt(card, 'd1d7', 80).status, 'inaccurate');
  assert.equal(gradeDrillAttempt(card, 'd1d7', 80).quality, 2);
  assert.equal(gradeDrillAttempt(card, 'd1d7', 250).status, 'missed');
  assert.equal(gradeDrillAttempt(card, 'd1d7', 250).quality, 1);
  assert.equal(gradeDrillAttempt(card, 'a1a8').status, 'illegal');
  assert.equal(gradeDrillAttempt(card, 'a1a8').quality, 0);
});

test('DrillDeck persists to storage, deduplicates, schedules due cards, and exports/imports JSON', () => {
  const storage = createMemoryStorage();
  const deck = new DrillDeck({ storage });

  const { card, added } = deck.addFromCriticalMoment(
    {
      fen: MATE_FEN,
      bestUci: 'd1d8',
      playedUci: 'g1f1',
      weaknessCategory: 'missed_mate',
    },
    { now: '2026-11-23' },
  );
  assert.equal(added, true);
  assert.equal(deck.getDueCards('2026-11-23').length, 1);

  // Duplicate position with different move counter is deduplicated
  const dup = deck.addCard(
    { fen: '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 5 20', best_move: 'd1d8' },
    '2026-11-23',
  );
  assert.equal(dup.added, false);

  const attempt = deck.recordAttempt(card.card_id, 'd1d8', 0, true, '2026-11-23');
  assert.equal(attempt.passed, true);
  assert.equal(deck.getDueCards('2026-11-23').length, 0);
  assert.equal(deck.getDueCards('2026-11-24').length, 1);

  // Reload from same storage verifies persistence
  const reloaded = new DrillDeck({ storage });
  assert.equal(reloaded.cards.length, 1);
  assert.equal(reloaded.getStats('2026-11-24').due_cards, 1);

  const exported = reloaded.exportJson();
  const freshDeck = new DrillDeck({ storage: createMemoryStorage() });
  assert.equal(freshDeck.importJson(exported), 1);
});

test('DrillDeck._load() survives corrupted storage entries and importJson() clamps EF & sanitizes bad dates', () => {
  const storage = createMemoryStorage();
  storage.setItem(
    'chessfm_drills_v1',
    JSON.stringify({
      version: 1,
      cards: [
        null,
        { bad: true },
        {
          card_id: 'drill_valid123456',
          fen: MATE_FEN,
          best_move: 'd1d8',
          category: 'missed_mate',
          due_date: '2026-11-22',
          ease_factor: 2.5,
          interval_days: 1,
          repetitions: 1,
          attempts: 1,
          successes: 1,
        },
      ],
    }),
  );

  const deck = new DrillDeck({ storage });
  assert.equal(deck.cards.length, 1);
  assert.equal(deck.getDueCards('2026-11-22').length, 1);
  assert.equal(deck.getStats('2026-11-22').total_cards, 1);

  const added = deck.importJson(
    JSON.stringify([
      {
        fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        best_move: 'e2e4',
        ease_factor: 0.2,
        due_date: 'not-a-date',
      },
    ]),
  );
  assert.equal(added, 1);
  const imported = deck.cards.find(
    (c) => c.fen === 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  );
  assert.ok(imported);
  assert.equal(imported.ease_factor, MIN_EF);
  assert.match(imported.due_date, /^\d{4}-\d{2}-\d{2}$/);
});
