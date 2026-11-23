import test from 'node:test';
import assert from 'node:assert/strict';
import {
  LICHESS_CP_COEFF,
  MATE_SCORE_CP,
  winProbFromCp,
  scoreToCp,
  parseUciInfoLine,
  classifyMove,
  pvUciToSan,
  formatEvalScore,
  StockfishWorker,
} from '../js/engine.js';

test('winProbFromCp matches Lichess logistic curve', () => {
  assert.equal(LICHESS_CP_COEFF, 0.00368208);
  assert.equal(winProbFromCp(0), 50);
  assert.ok(winProbFromCp(200) > 65 && winProbFromCp(200) < 70);
  assert.ok(winProbFromCp(-200) > 30 && winProbFromCp(-200) < 35);
  assert.ok(Math.abs(winProbFromCp(300) + winProbFromCp(-300) - 100) < 1e-9);
  assert.equal(winProbFromCp(100000), 100);
  assert.equal(winProbFromCp(-100000), 0);
});

test('scoreToCp handles cp and mate scores', () => {
  assert.equal(scoreToCp({ type: 'cp', value: 145 }), 145);
  assert.equal(scoreToCp({ type: 'cp', value: -80 }), -80);
  assert.equal(scoreToCp({ type: 'mate', value: 3 }), MATE_SCORE_CP);
  assert.equal(scoreToCp({ type: 'mate', value: -2 }), -MATE_SCORE_CP);
  assert.equal(scoreToCp(null), 0);
});

test('parseUciInfoLine parses multipv, cp, mate, and pv tokens', () => {
  const cpLine =
    'info depth 14 seldepth 21 multipv 2 score cp 42 nodes 12345 nps 500000 pv e2e4 e7e5 g1f3';
  const parsed = parseUciInfoLine(cpLine);
  assert.ok(parsed);
  assert.equal(parsed.depth, 14);
  assert.equal(parsed.seldepth, 21);
  assert.equal(parsed.multipv, 2);
  assert.equal(parsed.cp, 42);
  assert.equal(parsed.mate, null);
  assert.deepEqual(parsed.pv, ['e2e4', 'e7e5', 'g1f3']);

  const mateLine = 'info depth 10 multipv 1 score mate -3 pv h5f7';
  const parsedMate = parseUciInfoLine(mateLine);
  assert.ok(parsedMate);
  assert.equal(parsedMate.mate, -3);
  assert.equal(parsedMate.cp, -MATE_SCORE_CP);

  assert.equal(parseUciInfoLine('info string NNUE evaluation'), null);
  assert.equal(parseUciInfoLine('bestmove e2e4'), null);
});

test('classifyMove matches classifier.py thresholds and priority order', () => {
  assert.equal(classifyMove(50, 50, 'e2e4', 'e2e4').label, 'Best');
  assert.equal(classifyMove(50, 45, false).label, 'Excellent');
  assert.equal(classifyMove(50, 20, false).label, 'Good');
  assert.equal(classifyMove(50, -10, false).label, 'Inaccuracy');
  assert.equal(classifyMove(50, -70, false).label, 'Mistake');
  assert.equal(classifyMove(50, -300, false).label, 'Blunder');
  // Missed Win: winProbBefore >= 80, winProbAfter in [35, 60], drop >= 20
  assert.equal(classifyMove(450, 0, false).label, 'Missed Win');
});

test('pvUciToSan converts legal UCI line to SAN', () => {
  const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  assert.deepEqual(pvUciToSan(startFen, ['e2e4', 'e7e5', 'g1f3', 'b8c6']), [
    'e4',
    'e5',
    'Nf3',
    'Nc6',
  ]);
  assert.equal(formatEvalScore(125), '+1.25');
  assert.equal(formatEvalScore(-50), '-0.50');
  assert.equal(formatEvalScore(10000, 2), '+M2');
  assert.equal(formatEvalScore(-10000, -1), '-M1');
});

test('StockfishWorker serialized queue works with mock workerFactory', async () => {
  const sent = [];
  const fakeWorker = {
    onmessage: null,
    onerror: null,
    postMessage(cmd) {
      sent.push(cmd);
      queueMicrotask(() => {
        if (cmd === 'uci') this.onmessage({ data: 'uciok' });
        else if (cmd === 'isready') this.onmessage({ data: 'readyok' });
        else if (cmd.startsWith('go depth')) {
          this.onmessage({
            data: 'info depth 8 multipv 1 score cp 35 pv e2e4 e7e5\ninfo depth 8 multipv 2 score cp 28 pv d2d4 d7d5\nbestmove e2e4',
          });
        }
      });
    },
    terminate() {},
  };

  const sf = new StockfishWorker({ workerFactory: () => fakeWorker });
  const res = await sf.analyseMultiPV(
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    { depth: 8, multipv: 2 },
  );
  assert.equal(res.bestMove, 'e2e4');
  assert.equal(res.topLines.length, 2);
  assert.equal(res.topLines[0].san, 'e4');
  assert.equal(res.topLines[1].san, 'd4');
  assert.ok(sent.includes('setoption name MultiPV value 2'));
  sf.terminate();
});
