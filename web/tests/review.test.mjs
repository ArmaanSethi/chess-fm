import test from 'node:test';
import assert from 'node:assert/strict';
import {
  WEAKNESS_CATEGORIES,
  cpToWinProb,
  moveAccuracy,
  classifyReviewMove,
  rankCriticalMoments,
  buildWeaknessProfile,
  reviewGame,
} from '../js/review.js';

test('cpToWinProb and moveAccuracy match Lichess formulas', () => {
  assert.equal(cpToWinProb(0), 50);
  assert.equal(moveAccuracy(0), 100);
  assert.equal(moveAccuracy(60, 60), 100);
  const acc10 = moveAccuracy(60, 50); // 10 WDL drop
  assert.ok(acc10 > 60 && acc10 < 66, `Unexpected accuracy for 10% WDL drop: ${acc10}`);
  assert.equal(moveAccuracy(100, 0), Math.max(0, 103.1668 * Math.exp(-4.354) - 3.1669));
});

test('classifyReviewMove covers all 5 game_review labels', () => {
  assert.equal(classifyReviewMove(0, true), 'Best');
  assert.equal(classifyReviewMove(10, false), 'Best');
  assert.equal(classifyReviewMove(30, false), 'Good');
  assert.equal(classifyReviewMove(70, false), 'Inaccuracy');
  assert.equal(classifyReviewMove(150, false), 'Mistake');
  assert.equal(classifyReviewMove(250, false), 'Blunder');
  assert.equal(classifyReviewMove(50, false, { missedMate: true }), 'Blunder');
});

test('rankCriticalMoments ranks top 2 student mistakes by wdlSwing * maiaFindability', async () => {
  const fen1 = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const fen2 = 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3';
  const fen3 = '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1';

  const plies = [
    {
      ply: 1,
      mover: 'w',
      fen: fen1,
      playedUci: 'a2a3',
      playedSan: 'a3',
      bestUci: 'e2e4',
      bestSan: 'e4',
      cpLoss: 150,
      wdlSwing: 20, // Low Maia best prob (0.05) -> 20 * (0.35 + 0.0325) = 7.65
      classification: 'Mistake',
      weaknessCategory: 'opening_inaccuracy',
    },
    {
      ply: 3,
      mover: 'w',
      fen: fen3,
      playedUci: 'g1f1',
      playedSan: 'Kf1',
      bestUci: 'd1d8',
      bestSan: 'Rd8#',
      cpLoss: 500,
      wdlSwing: 18, // High Maia best prob (0.90) -> 18 * (0.35 + 0.585) = 16.83
      classification: 'Blunder',
      weaknessCategory: 'missed_mate',
    },
    {
      ply: 4,
      mover: 'b', // Opponent mistake -> excluded for studentColor 'w'
      fen: fen2,
      playedUci: 'a7a6',
      playedSan: 'a6',
      bestUci: 'g8f6',
      bestSan: 'Nf6',
      cpLoss: 300,
      wdlSwing: 30,
      classification: 'Blunder',
      weaknessCategory: 'opening_inaccuracy',
    },
  ];

  const fakeMaia = (fen) => {
    if (fen === fen3) return { policy: { d1d8: 0.9, g1f1: 0.05 }, backend: 'calibrated_fallback' };
    return { policy: { e2e4: 0.05, a2a3: 0.2 }, backend: 'calibrated_fallback' };
  };

  const moments = await rankCriticalMoments(plies, {
    studentColor: 'w',
    studentElo: 1400,
    maiaPredict: fakeMaia,
    maxMoments: 2,
  });

  assert.equal(moments.length, 2);
  // fen3 must outrank fen1 despite slightly smaller wdlSwing because maiaFindability is much higher!
  assert.equal(moments[0].ply, 3);
  assert.equal(moments[1].ply, 1);
  assert.equal(moments[1].isHumanTrap, true);
});

test('buildWeaknessProfile populates all 6 canonical weakness categories', () => {
  assert.equal(WEAKNESS_CATEGORIES.length, 6);
  const plies = [
    {
      mover: 'w',
      fen: '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1',
      playedUci: 'g1f1',
      bestUci: 'd1d8',
      cpLoss: 400,
      wdlSwing: 45,
      classification: 'Blunder',
      weaknessCategory: 'missed_mate',
    },
    {
      mover: 'w',
      fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
      playedUci: 'g1h3',
      bestUci: 'e2e4',
      cpLoss: 90,
      wdlSwing: 8,
      classification: 'Inaccuracy',
      weaknessCategory: 'opening_inaccuracy',
    },
  ];
  const profile = buildWeaknessProfile(plies, 'w');
  assert.equal(profile.totalMistakes, 2);
  assert.equal(Object.keys(profile.categories).length, 6);
  assert.equal(profile.topPriorities[0].category, 'missed_mate');
  assert.equal(profile.drills.length, 2);
});

test('reviewGame runs end-to-end offline with static fallback', async () => {
  const pgn = '1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0';
  const result = await reviewGame(pgn, { studentColor: 'b', studentElo: 1200 });
  assert.equal(result.totalPlies, 7);
  assert.ok(result.whiteAccuracy >= 0 && result.whiteAccuracy <= 100);
  assert.ok(result.blackAccuracy >= 0 && result.blackAccuracy <= 100);
  assert.ok(Array.isArray(result.criticalMoments));
  assert.ok(result.weaknessProfile);
});

test('reviewGame succeeds on custom [SetUp "1"] / [FEN "..."] PGNs', async () => {
  const customPgn = [
    '[Event "Puzzle Study"]',
    '[SetUp "1"]',
    '[FEN "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"]',
    '',
    '4. Qxf7# 1-0',
  ].join('\n');

  const result = await reviewGame(customPgn, { studentColor: 'w', studentElo: 1400 });
  assert.equal(result.totalPlies, 1);
  assert.equal(result.plyAnalyses.length, 1);
  assert.equal(result.plyAnalyses[0].playedSan, 'Qxf7#');
  assert.equal(result.plyAnalyses[0].playedUci, 'h5f7');
  assert.equal(
    result.plyAnalyses[0].fen,
    'r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4',
  );
});
