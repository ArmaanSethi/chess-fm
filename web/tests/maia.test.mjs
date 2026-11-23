import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  ELO_BRACKETS,
  snapEloBracket,
  eloToBucket,
  mirrorUci,
  mirrorFen,
  encodeBoardPlanes,
  decodePolicyLogits,
  calibratedFallback,
  detectHumanTrap,
  MaiaClient,
} from '../js/maia.js';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const movesDict = JSON.parse(
  fs.readFileSync(new URL('../data/maia2_moves.json', import.meta.url), 'utf8'),
);

test('eloToBucket maps all 11 Maia-2 buckets accurately', () => {
  assert.equal(ELO_BRACKETS.length, 11);
  assert.equal(snapEloBracket(1350), 1400);
  assert.equal(eloToBucket(800), 0);
  assert.equal(eloToBucket(1099), 0);
  assert.equal(eloToBucket(1100), 1);
  assert.equal(eloToBucket(1250), 2);
  assert.equal(eloToBucket(1999), 9);
  assert.equal(eloToBucket(2000), 10);
  assert.equal(eloToBucket(2600), 10);
});

test('mirrorUci and mirrorFen preserve vertical symmetry and promotion', () => {
  assert.equal(mirrorUci('e2e4'), 'e7e5');
  assert.equal(mirrorUci('e7e5'), 'e2e4');
  assert.equal(mirrorUci('a7a8q'), 'a2a1q');

  const afterE4 = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
  const mirrored = mirrorFen(afterE4);
  assert.equal(
    mirrored,
    'rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1',
  );
});

test('encodeBoardPlanes produces 18x8x8 tensor with expected piece/turn/castling/ep planes', () => {
  const tensor = encodeBoardPlanes(START_FEN);
  assert.equal(tensor.length, 18 * 64);
  // White pawn on e2 => plane 0, row 1, file 4
  assert.equal(tensor[0 * 64 + 1 * 8 + 4], 1.0);
  // Black king on e8 => plane 11, row 7, file 4
  assert.equal(tensor[11 * 64 + 7 * 8 + 4], 1.0);
  // Plane 12 (turn) == 1.0
  assert.equal(tensor[12 * 64], 1.0);
  // Planes 13..16 (KQkq) == 1.0
  assert.equal(tensor[13 * 64], 1.0);
  assert.equal(tensor[16 * 64], 1.0);
  // Plane 17 (en passant) == 0.0
  assert.equal(tensor[17 * 64], 0.0);

  // Black-to-move with en passant e3 -> mirrored to White with ep e6 (row 5, file 4)
  const epFen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1';
  const epTensor = encodeBoardPlanes(epFen);
  assert.equal(epTensor[12 * 64], 1.0);
  assert.equal(epTensor[17 * 64 + 5 * 8 + 4], 1.0);
});

test('decodePolicyLogits filters strictly to legal moves and mirrors Black moves back', () => {
  assert.equal(Object.keys(movesDict).length, 1880);
  const logits = new Float32Array(1880).fill(-5.0);
  // Boost mirrored e7e5 -> e2e4
  logits[movesDict.e2e4] = 10.0;

  const blackTurnFen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1';
  const { moves, policy } = decodePolicyLogits(logits, blackTurnFen, movesDict);
  assert.equal(moves[0].move, 'e7e5');
  assert.equal(moves[0].san, 'e5');
  assert.ok(policy.e7e5 > 0.99);
  const sum = Object.values(policy).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sum - 1.0) < 1e-6);
});

test('calibratedFallback is Elo-monotonic (higher Elo sharper) and detectHumanTrap triggers at thresholds', () => {
  const mateInOneFen = '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1';
  const low = calibratedFallback(mateInOneFen, 600);
  const high = calibratedFallback(mateInOneFen, 2400);
  assert.equal(high.top_moves[0].move, 'd1d8');
  assert.ok(
    high.policy.d1d8 > low.policy.d1d8,
    `Expected higher Elo prob (${high.policy.d1d8}) > lower Elo prob (${low.policy.d1d8})`,
  );

  // Trap detection: cpLoss >= 100 and playedProb >= 0.15
  const trap = detectHumanTrap(START_FEN, 'e2e4', 'd2d4', 150, 1200, {
    policy: { e2e4: 0.35, d2d4: 0.25 },
    backend: 'calibrated_fallback',
  });
  assert.equal(trap.is_common_human_trap, true);
  assert.equal(trap.category, 'common_human_trap');

  const oversight = detectHumanTrap(START_FEN, 'a2a3', 'e2e4', 180, 1200, {
    policy: { a2a3: 0.02, e2e4: 0.45 },
    backend: 'calibrated_fallback',
  });
  assert.equal(oversight.is_common_human_trap, false);
  assert.equal(oversight.category, 'unusual_oversight');
});

test('MaiaClient boots in heuristic mode and predicts immediately offline', async () => {
  const client = new MaiaClient();
  assert.equal(client.status, 'heuristic');
  const pred = await client.predict(START_FEN, 1400);
  assert.equal(pred.backend, 'calibrated_fallback');
  assert.ok(pred.top_moves.length > 0);
});
