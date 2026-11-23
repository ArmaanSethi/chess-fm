import test from 'node:test';
import assert from 'node:assert/strict';
import { containsMoveSpoiler, generateSocraticHint } from '../js/socratic.js';
import { extractClaims, verifyClaims } from '../js/verifier.js';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const MATE_FEN = '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1';
const CASTLE_FEN = 'r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4';

test('containsMoveSpoiler detects UCI, SAN, castling aliases, and from+to square pairs', () => {
  assert.equal(containsMoveSpoiler('Play e2e4 now.', 'e2e4', 'e4'), true);
  assert.equal(containsMoveSpoiler('Try e2-e4 or e2->e4.', 'e2e4', 'e4'), true);
  assert.equal(containsMoveSpoiler('Consider playing Nf3.', 'g1f3', 'Nf3'), true);
  assert.equal(containsMoveSpoiler('Move the piece from g1 to f3.', 'g1f3', 'Nf3'), true);
  assert.equal(containsMoveSpoiler('Castle with 0-0!', 'e1g1', 'O-O'), true);
  assert.equal(containsMoveSpoiler('Develop your knight toward the center.', 'g1f3', 'Nf3'), false);
});

test('generateSocraticHint guarantees 0% spoiler leakage on L1 & L2 and reveals solution on L3', () => {
  const cases = [
    [START_FEN, 'e2e4'],
    [START_FEN, 'g1f3'],
    [MATE_FEN, 'd1d8'],
    [CASTLE_FEN, 'e1g1'],
  ];

  for (const [fen, uci] of cases) {
    const h1 = generateSocraticHint(fen, uci, 1);
    const h2 = generateSocraticHint(fen, uci, 2);
    const h3 = generateSocraticHint(fen, uci, 3, { pvUci: [uci] });

    assert.equal(h1.spoiler_free, true, `L1 leaked for ${uci}: ${h1.hint}`);
    assert.equal(containsMoveSpoiler(h1.hint, uci, h1.move_san), false);
    assert.equal(h2.spoiler_free, true, `L2 leaked for ${uci}: ${h2.hint}`);
    assert.equal(containsMoveSpoiler(h2.hint, uci, h2.move_san), false);
    assert.ok(h3.hint.includes(h3.move_san), `L3 should reveal SAN: ${h3.hint}`);
    assert.ok(h3.hint.includes(uci), `L3 should reveal UCI: ${h3.hint}`);
  }
});

test('L2 Socratic hints never leak destination square on checkmate, fork, capture, quiet, or promotion moves', () => {
  const cases = [
    // Scholar's Mate (checkmate)
    ['r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4', 'h5f7'],
    // Knight fork
    ['r3k3/ppp5/8/1N6/8/8/8/4K3 w q - 0 1', 'b5c7'],
    // Piece capture (Nxe5)
    ['r1bqkb1r/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3', 'f3e5'],
    // Quiet non-development piece move (Bb5)
    ['r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3', 'f1b5'],
    // Quiet development move (Nf3)
    [START_FEN, 'g1f3'],
    // Underpromotion (e8=N)
    ['7k/4P3/8/8/8/8/8/4K3 w - - 0 1', 'e7e8n'],
  ];

  for (const [fen, uci] of cases) {
    const toSq = uci.slice(2, 4);
    const fromSq = uci.slice(0, 2);
    const h2 = generateSocraticHint(fen, uci, 2);
    assert.equal(h2.spoiler_free, true, `L2 flagged spoiler for ${uci}: ${h2.hint}`);
    assert.equal(
      containsMoveSpoiler(h2.hint, uci, h2.move_san),
      false,
      `containsMoveSpoiler flagged L2 for ${uci}: ${h2.hint}`,
    );
    assert.equal(
      new RegExp(`(?<![A-Za-z0-9])${toSq}(?![A-Za-z0-9])`, 'i').test(h2.hint),
      false,
      `L2 leaked destination square ${toSq} for ${uci}: ${h2.hint}`,
    );
    assert.equal(
      new RegExp(`(?<![A-Za-z0-9])${fromSq}(?![A-Za-z0-9])`, 'i').test(h2.hint),
      false,
      `L2 leaked origin square ${fromSq} for ${uci}: ${h2.hint}`,
    );
  }
});

test('containsMoveSpoiler catches natural-language piece+square and compact promotion spoilers', () => {
  assert.equal(containsMoveSpoiler('Move your knight to f3 to develop.', 'g1f3', 'Nf3'), true);
  assert.equal(containsMoveSpoiler('Your queen on f7 ends the game.', 'h5f7', 'Qxf7#'), true);
  assert.equal(containsMoveSpoiler('Play bishop to b5 to pin the knight.', 'f1b5', 'Bb5'), true);
  assert.equal(containsMoveSpoiler('Play e8N to fork the king and queen!', 'e7e8n', 'e8=N+'), true);
  assert.equal(containsMoveSpoiler('Promote the pawn to a knight on e8.', 'e7e8n', 'e8=N+'), true);
  assert.equal(containsMoveSpoiler('Look for a forcing check in the kingside.', 'h5f7', 'Qxf7#'), false);
  assert.equal(containsMoveSpoiler('Your knight has a fork in the queenside.', 'b5c7', 'Nxc7+'), false);
});

test('ACT-Eval verifyClaims verifies true claims and flags false piece/move/checkmate claims', () => {
  // True claims on start position with sequential variation
  const good = verifyClaims(
    START_FEN,
    'The white knight on g1 is ready. Play 1. e4 e5 2. Nf3 Nc6 to fight for the center.',
  );
  assert.equal(good.passed, true);
  assert.ok(good.totalClaims >= 5);
  assert.equal(good.failedClaims, 0);
  assert.match(good.badgeText, /ACT-Eval: \d+\/\d+ claims verified/);

  // False piece-on-square claim
  const badPiece = verifyClaims(START_FEN, 'The white queen on e4 controls the center.');
  assert.equal(badPiece.passed, false);
  assert.equal(badPiece.failedClaims, 1);

  // True checkmate claim (#) vs false checkmate claim
  const goodMate = verifyClaims(MATE_FEN, 'Play Rd8# which delivers checkmate on d8.');
  assert.equal(goodMate.passed, true);
  assert.ok(goodMate.verifiedClaims >= 2);

  const badMate = verifyClaims(START_FEN, 'Play e4# which delivers checkmate.');
  assert.equal(badMate.passed, false);
  assert.ok(badMate.failedClaims >= 1);

  // Variation reset across sentence boundary
  const resetTest = verifyClaims(START_FEN, 'Play e4 to grab space. Alternatively, play d4 for a solid center.');
  assert.equal(resetTest.passed, true);
  assert.equal(extractClaims('Play e4. Alternatively, play d4.').length, 2);
});
