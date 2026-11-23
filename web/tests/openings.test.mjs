import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { normalizeBoardFen, OpeningBook } from '../js/openings.js';

const openingsPayload = JSON.parse(
  fs.readFileSync(new URL('../data/openings.json', import.meta.url), 'utf8'),
);

test('openings.json contains all 43 ECO entries with precomputed 4-field FENs', () => {
  assert.equal(openingsPayload.count, 43);
  assert.equal(openingsPayload.entries.length, 43);
  for (const entry of openingsPayload.entries) {
    assert.ok(entry.eco && entry.name && entry.white_plan && entry.black_plan && entry.common_trap);
    assert.ok(Array.isArray(entry.fens) && entry.fens.length === entry.moves.length);
    assert.equal(entry.final_fen.split(' ').length, 4);
  }
});

test('normalizeBoardFen strips halfmove and fullmove counters', () => {
  assert.equal(
    normalizeBoardFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'),
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -',
  );
  assert.throws(() => normalizeBoardFen('invalid'));
});

test('OpeningBook supports transposition-aware FEN identification and search', () => {
  const book = new OpeningBook(openingsPayload.entries);
  assert.equal(book.size, 43);

  // Standard Italian Game move order: 1. e4 e5 2. Nf3 Nc6 3. Bc4
  const italian = book.identifyOpening(null, ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4']);
  assert.ok(italian);
  assert.equal(italian.eco, 'C50');
  assert.equal(italian.name, 'Italian Game');

  // Transposed move order: 1. e4 Nc6 2. Nf3 e5 3. Bc4 reaches the exact same 4-field FEN!
  const transposed = book.identifyOpening(null, ['e4', 'Nc6', 'Nf3', 'e5', 'Bc4']);
  assert.ok(transposed);
  assert.equal(transposed.name, 'Italian Game');

  const sicilianResults = book.search('Sicilian');
  assert.ok(sicilianResults.length >= 3);
});

test('OpeningBook.detectDeviation finds last book ply and first out-of-book move', () => {
  const book = new OpeningBook(openingsPayload.entries);
  const pgn = '1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. a3 Nf6';
  const dev = book.detectDeviation(pgn, [{ ply: 7, cpLoss: 45 }]);
  assert.equal(dev.name, 'Giuoco Piano');
  assert.equal(dev.lastBookPly, 6);
  assert.equal(dev.deviationPly, 7);
  assert.equal(dev.deviationMoveSan, 'a3');
  assert.equal(dev.evalDropCp, 45);
});
