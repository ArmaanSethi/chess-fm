/**
 * Direct browser Lichess & Chess.com public game importers and multi-game PGN parser.
 *
 * Both Lichess and Chess.com public game endpoints allow direct browser CORS requests.
 * Per browser Fetch spec, ``User-Agent`` is omitted so requests never trigger preflight failures.
 */

import { Chess } from '../vendor/chess.mjs';

/**
 * Map HTTP error status codes from Lichess / Chess.com to clear user-facing messages.
 *
 * @param {string} platform
 * @param {string} username
 * @param {number} status
 * @returns {Error}
 */
function buildHttpError(platform, username, status) {
  let message;
  if (status === 404) {
    message = `${platform} user '${username}' not found (HTTP 404).`;
  } else if (status === 429) {
    message = `${platform} API rate limit exceeded for '${username}' (HTTP 429). Please wait a moment and retry.`;
  } else {
    message = `${platform} API returned HTTP ${status} for '${username}'.`;
  }
  const err = new Error(message);
  err.status = status;
  return err;
}

/**
 * Extract PGN tag-pair headers from a single PGN string.
 *
 * @param {string} pgn
 * @returns {Record<string, string>}
 */
export function parsePgnHeaders(pgn) {
  const headers = {};
  if (typeof pgn !== 'string') return headers;
  const re = /^\s*\[([A-Za-z0-9_]+)\s+"((?:\\.|[^"\\])*)"\]\s*$/gm;
  let match;
  while ((match = re.exec(pgn)) !== null) {
    headers[match[1]] = match[2].replace(/\\"/g, '"').replace(/\\\\/g, '\\');
  }
  return headers;
}

/**
 * Split a multi-game PGN text bundle into individual PGN strings.
 *
 * @param {string} pgnBundle
 * @param {number | null} [maxGames=null]
 * @returns {string[]}
 */
export function splitPgnGames(pgnBundle, maxGames = null) {
  if (typeof pgnBundle !== 'string' || !pgnBundle.trim()) return [];
  if (maxGames !== null && maxGames <= 0) return [];

  const normalized = pgnBundle.replace(/\r\n/g, '\n').trim();
  const lines = normalized.split('\n');
  const games = [];
  let current = [];
  let seenMoves = false;
  let seenBlankAfterTags = false;
  const seenTags = new Set();

  for (const line of lines) {
    const trimmed = line.trim();
    const tagMatch = /^\[([A-Za-z0-9_]+)\s+"/.exec(trimmed);
    const isTag = Boolean(tagMatch);

    if (isTag && current.length > 0) {
      const tagName = tagMatch[1];
      const startsNewHeaderBlock =
        seenMoves || (seenBlankAfterTags && (tagName === 'Event' || seenTags.has(tagName)));
      if (startsNewHeaderBlock) {
        const joined = current.join('\n').trim();
        if (joined) games.push(joined);
        if (maxGames !== null && games.length >= maxGames) return games;
        current = [line];
        seenMoves = false;
        seenBlankAfterTags = false;
        seenTags.clear();
        seenTags.add(tagName);
        continue;
      }
    }

    if (isTag) {
      seenTags.add(tagMatch[1]);
      seenBlankAfterTags = false;
    } else if (trimmed === '') {
      if (seenTags.size > 0 && !seenMoves) {
        seenBlankAfterTags = true;
      }
    } else {
      seenMoves = true;
    }
    current.push(line);
  }

  if (current.length > 0) {
    const joined = current.join('\n').trim();
    if (joined) games.push(joined);
  }

  return maxGames !== null ? games.slice(0, maxGames) : games;
}

/**
 * Parse a single or multi-game PGN bundle into structured game records with headers & moves.
 *
 * @param {string} text
 * @param {number | null} [maxGames=null]
 */
export function parsePgnGames(text, maxGames = null) {
  const rawGames = splitPgnGames(text, maxGames);
  const parsed = [];

  for (const pgn of rawGames) {
    const headers = parsePgnHeaders(pgn);
    const board = new Chess();
    let moves = [];
    let valid = true;
    try {
      board.loadPgn(pgn);
      moves = board.history();
    } catch {
      valid = false;
    }

    parsed.push({
      pgn,
      headers,
      white: headers.White || 'White',
      black: headers.Black || 'Black',
      result: headers.Result || '*',
      date: headers.Date || headers.UTCDate || '',
      event: headers.Event || 'Casual Game',
      opening: headers.Opening || '',
      eco: headers.ECO || '',
      whiteElo: headers.WhiteElo ? Number.parseInt(headers.WhiteElo, 10) || null : null,
      blackElo: headers.BlackElo ? Number.parseInt(headers.BlackElo, 10) || null : null,
      site: headers.Site || '',
      moves,
      plyCount: moves.length,
      valid,
    });
  }

  return parsed;
}

/**
 * Detect whether ``username`` played White ('w') or Black ('b') in a game.
 * Case-insensitive trim match against White/Black headers; defaults to 'w'.
 *
 * @param {string | Record<string, string>} headersOrPgn
 * @param {string} username
 * @returns {'w' | 'b'}
 */
export function detectStudentColor(headersOrPgn, username) {
  if (!username || typeof username !== 'string') return 'w';
  const target = username.trim().toLowerCase();
  if (!target) return 'w';

  const headers =
    typeof headersOrPgn === 'string'
      ? parsePgnHeaders(headersOrPgn)
      : (headersOrPgn?.headers || headersOrPgn || {});

  const white = String(headers.White || headers.white || '').trim().toLowerCase();
  const black = String(headers.Black || headers.black || '').trim().toLowerCase();

  if (white === target) return 'w';
  if (black === target) return 'b';
  return 'w';
}

/**
 * Fetch recent public games from Lichess for ``username`` directly in the browser.
 *
 * @param {string} username
 * @param {{ max?: number, fetchImpl?: typeof fetch }} [options]
 */
export async function fetchLichessGames(username, options = {}) {
  const cleanUser = String(username || '').trim();
  if (!cleanUser) {
    throw new Error('Please enter a Lichess username.');
  }
  const max = Math.max(1, Math.min(50, Number(options.max) || 10));
  const fetchFn = options.fetchImpl || globalThis.fetch;
  const url =
    `https://lichess.org/api/games/user/${encodeURIComponent(cleanUser)}` +
    `?max=${max}&pgnInJson=false&clocks=false&evals=false&opening=true`;

  const response = await fetchFn(url, {
    headers: { Accept: 'application/x-chess-pgn' },
  });

  if (!response.ok) {
    throw buildHttpError('Lichess', cleanUser, response.status);
  }

  const text = await response.text();
  return parsePgnGames(text, max).map((g) => ({
    ...g,
    platform: 'lichess',
    studentColor: detectStudentColor(g.headers, cleanUser),
  }));
}

/**
 * Fetch recent public games from Chess.com for ``username`` directly in the browser.
 *
 * @param {string} username
 * @param {{ max?: number, fetchImpl?: typeof fetch }} [options]
 */
export async function fetchChessComGames(username, options = {}) {
  const cleanUser = String(username || '').trim();
  if (!cleanUser) {
    throw new Error('Please enter a Chess.com username.');
  }
  const max = Math.max(1, Math.min(50, Number(options.max) || 10));
  const fetchFn = options.fetchImpl || globalThis.fetch;
  const archivesUrl = `https://api.chess.com/pub/player/${encodeURIComponent(cleanUser.toLowerCase())}/games/archives`;

  const archResp = await fetchFn(archivesUrl, {
    headers: { Accept: 'application/json' },
  });
  if (!archResp.ok) {
    throw buildHttpError('Chess.com', cleanUser, archResp.status);
  }

  const archData = await archResp.json();
  const archives = Array.isArray(archData?.archives) ? [...archData.archives].reverse() : [];
  const collectedPgns = [];

  for (const monthUrl of archives) {
    if (collectedPgns.length >= max) break;
    const monthResp = await fetchFn(monthUrl, {
      headers: { Accept: 'application/json' },
    });
    if (!monthResp.ok) {
      if (monthResp.status === 429) {
        throw buildHttpError('Chess.com', cleanUser, 429);
      }
      continue;
    }
    const monthData = await monthResp.json();
    const games = Array.isArray(monthData?.games) ? [...monthData.games].reverse() : [];
    for (const game of games) {
      if (typeof game?.pgn === 'string' && game.pgn.trim()) {
        collectedPgns.push(game.pgn.trim());
        if (collectedPgns.length >= max) break;
      }
    }
  }

  const parsed = [];
  for (const pgn of collectedPgns) {
    const [game] = parsePgnGames(pgn, 1);
    if (game) {
      parsed.push({
        ...game,
        platform: 'chesscom',
        studentColor: detectStudentColor(game.headers, cleanUser),
      });
    }
  }
  return parsed;
}
