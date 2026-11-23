import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parsePgnHeaders,
  splitPgnGames,
  parsePgnGames,
  detectStudentColor,
  fetchLichessGames,
  fetchChessComGames,
} from '../js/importers.js';

const MULTI_PGN = `[Event "Rated Blitz"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[ECO "C50"]
[WhiteElo "1450"]
[BlackElo "1420"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 1-0

[Event "Rated Rapid"]
[White "Charlie"]
[Black "alice"]
[Result "0-1"]
[ECO "B20"]

1. e4 c5 2. Nf3 d6 0-1
`;

test('parsePgnGames splits multi-game PGNs and extracts headers and moves', () => {
  const raw = splitPgnGames(MULTI_PGN);
  assert.equal(raw.length, 2);

  const games = parsePgnGames(MULTI_PGN);
  assert.equal(games.length, 2);
  assert.equal(games[0].white, 'Alice');
  assert.equal(games[0].black, 'Bob');
  assert.equal(games[0].eco, 'C50');
  assert.equal(games[0].whiteElo, 1450);
  assert.deepEqual(games[0].moves, ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5']);
  assert.equal(games[1].moves.length, 4);
});

test('detectStudentColor matches case-insensitively and defaults to w', () => {
  const games = parsePgnGames(MULTI_PGN);
  assert.equal(detectStudentColor(games[0].headers, 'alice'), 'w');
  assert.equal(detectStudentColor(games[1].headers, 'ALICE'), 'b');
  assert.equal(detectStudentColor(games[0].pgn, 'bob'), 'b');
  assert.equal(detectStudentColor(games[0].headers, 'unknown'), 'w');
  assert.equal(parsePgnHeaders(games[0].pgn).Result, '1-0');
});

test('fetchLichessGames fetches PGN without User-Agent and handles 404/429', async () => {
  let seenHeaders = null;
  const mockFetch = async (url, init) => {
    seenHeaders = init?.headers;
    assert.ok(url.includes('lichess.org/api/games/user/Alice'));
    return { ok: true, status: 200, text: async () => MULTI_PGN };
  };

  const games = await fetchLichessGames('Alice', { max: 5, fetchImpl: mockFetch });
  assert.equal(games.length, 2);
  assert.equal(games[0].studentColor, 'w');
  assert.equal(games[1].studentColor, 'b');
  assert.equal(seenHeaders['User-Agent'], undefined);
  assert.equal(seenHeaders.Accept, 'application/x-chess-pgn');

  await assert.rejects(
    () =>
      fetchLichessGames('missing', {
        fetchImpl: async () => ({ ok: false, status: 404 }),
      }),
    /not found \(HTTP 404\)/,
  );
  await assert.rejects(
    () =>
      fetchLichessGames('busy', {
        fetchImpl: async () => ({ ok: false, status: 429 }),
      }),
    /rate limit exceeded \(HTTP 429\)|HTTP 429/,
  );
});

test('fetchChessComGames reads latest archive in reverse order and handles 404', async () => {
  const mockFetch = async (url) => {
    if (url.endsWith('/games/archives')) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          archives: ['https://api.chess.com/pub/player/alice/games/2026/10'],
        }),
      };
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        games: [
          { pgn: '[White "Bob"]\n[Black "Alice"]\n[Result "0-1"]\n\n1. d4 d5 0-1' },
        ],
      }),
    };
  };

  const games = await fetchChessComGames('Alice', { max: 3, fetchImpl: mockFetch });
  assert.equal(games.length, 1);
  assert.equal(games[0].platform, 'chesscom');
  assert.equal(games[0].studentColor, 'b');

  await assert.rejects(
    () =>
      fetchChessComGames('nobody', {
        fetchImpl: async () => ({ ok: false, status: 404 }),
      }),
    /HTTP 404/,
  );
});

test('splitPgnGames separates a header-only (0-move) game from a following valid game', () => {
  const bundle = [
    '[Event "Empty"]',
    '[White "A"]',
    '[Black "B"]',
    '[Result "*"]',
    '',
    '[Event "Real"]',
    '[White "C"]',
    '[Black "D"]',
    '[Result "1-0"]',
    '',
    '1. e4 e5 1-0',
  ].join('\n');

  const split = splitPgnGames(bundle);
  assert.equal(split.length, 2);

  const parsed = parsePgnGames(bundle);
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0].event, 'Empty');
  assert.equal(parsed[0].plyCount, 0);
  assert.equal(parsed[1].event, 'Real');
  assert.equal(parsed[1].white, 'C');
  assert.equal(parsed[1].plyCount, 2);
  assert.equal(parsed[1].valid, true);
});
