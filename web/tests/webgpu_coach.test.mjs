import test from 'node:test';
import assert from 'node:assert/strict';
import { isWebGpuSupported, WebGPUCoach } from '../js/webgpu_coach.js';

test('isWebGpuSupported returns false cleanly in Node environment', () => {
  assert.equal(isWebGpuSupported(), false);
});

test('WebGPUCoach adheres to Interface Contract #2 and falls back cleanly without throwing', async () => {
  const coach = new WebGPUCoach();
  const initRes = await coach.init();
  assert.equal(initRes.ready, false);
  assert.equal(initRes.status, 'unsupported');
  assert.ok(initRes.message.includes('Layer A'));

  const genRes = await coach.generate({
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    taskType: 'socratic_l1',
  });
  assert.equal(genRes.text, null);
  assert.equal(genRes.fallback, true);
  assert.equal(typeof genRes.latencyMs, 'number');
  assert.equal(genRes.modelId, null);
});
