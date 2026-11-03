import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BOS, EOS, SEP, allowedTokens, completeClimb, seededRandom, validateRequest } from '../src/decoding.mjs';

// Six holds on a 2x3 grid, ranked like kilter.data.Board: role order, then y, then x.
const ROLE_ORDER = { 2: 0, 0: 1, 3: 2, 1: 3 };
const holds = Array.from({ length: 6 }, (_, i) => ({ id: 100 + i, x: (i % 2) / 2, y: Math.floor(i / 2) / 3 }));
const order = holds.flatMap((hold, i) => [0, 1, 2, 3].map(role => ({ token: 4 + 4 * i + role, key: [ROLE_ORDER[role], hold.y, hold.x, i] })))
  .sort((a, b) => a.key.reduce((d, v, k) => d || v - b.key[k], 0));
const rank = Array(4 + 4 * holds.length).fill(-1);
order.forEach(({ token }, position) => { rank[token] = position; });
const manifest = { format: 'kilter-v2', holds, rank, max_holds: 8, max_starts: 2, max_finishes: 2 };
const request = { grade: 5, angle: 45, temperature: 0.9, seed: 42, selected: [{ id: 101, role: 2 }, { id: 100, role: 1 }] };
const token = (index, role) => 4 + 4 * index + role;

test('picks are the given block, and generation never repeats or recolors them', async () => {
  let first, conditions;
  const predict = async (tokens, c) => {
    first ??= [...tokens];
    conditions = c;
    const logits = new Float32Array(rank.length).fill(-10);
    logits[EOS] = 100; // Masked until there is a finish.
    logits[token(1, 0)] = 90; // A picked hold again, as a hand: masked.
    logits[token(0, 3)] = 80; // A picked hold again, as a finish: masked.
    return logits;
  };
  const result = await completeClimb({ request, manifest, predict });
  assert.deepEqual(first, [BOS, token(1, 2), token(0, 1), SEP]);
  assert.deepEqual([...conditions], [5, 0.5]);
  assert.equal(result.endedWithEos, true);
  assert.deepEqual(result.holds.slice(0, 2), [{ id: 101, role: 2, locked: true }, { id: 100, role: 1, locked: true }]);
  assert.equal(new Set(result.holds.map(h => h.id)).size, result.holds.length);
  assert.ok(result.holds.some(h => h.role === 3 && !h.locked));
});

test('from nothing, a climb starts with a start hold and must reach a finish before feet or EOS', () => {
  const empty = allowedTokens({ manifest, used: new Set(), lastRank: -1, starts: 0, finishes: 0, count: 0 });
  assert.deepEqual(empty.sort((a, b) => a - b), holds.map((_, i) => token(i, 2)));
  const afterStart = allowedTokens({ manifest, used: new Set([1]), lastRank: rank[token(1, 2)], starts: 1, finishes: 0, count: 1 });
  assert.ok(!afterStart.includes(EOS));
  assert.ok(afterStart.every(t => (t - 4) % 4 !== 1));
  assert.ok(!afterStart.includes(token(0, 2))); // Same row but further left comes earlier in order.
  assert.ok(afterStart.includes(token(2, 2)));
  const full = allowedTokens({ manifest, used: new Set([1, 5]), lastRank: 0, starts: 1, finishes: 1, count: 8 });
  assert.deepEqual(full, [EOS]);
  const reserve = allowedTokens({ manifest, used: new Set([1]), lastRank: rank[token(1, 2)], starts: 1, finishes: 0, count: 7 });
  assert.ok(reserve.length && reserve.every(t => (t - 4) % 4 === 3));
});

test('the browser rules match kilter exactly on exported generation states', () => {
  const real = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
  const fixtures = JSON.parse(readFileSync(new URL('../public/model/decoding-fixtures.json', import.meta.url)));
  assert.ok(fixtures.length > 50);
  for (const state of fixtures) {
    const allowed = allowedTokens({ manifest: real, used: new Set(state.used), lastRank: state.last_rank,
      starts: state.starts, finishes: state.finishes, count: state.count });
    assert.deepEqual(allowed.sort((a, b) => a - b), state.allowed);
  }
});

test('stops at the hold limit with EOS and fits the model context', async () => {
  let longest = 0;
  const result = await completeClimb({ request: { ...request, selected: [] }, manifest,
    predict: async tokens => {
      longest = Math.max(longest, tokens.length);
      const logits = new Float32Array(rank.length);
      logits[EOS] = -100;
      return logits;
    } });
  assert.equal(result.endedWithEos, true);
  assert.ok(result.holds.length <= manifest.max_holds);
  assert.ok(longest <= manifest.max_holds + 2);
});

test('seeded generation is repeatable and angle uses 90 as denominator', async () => {
  const first = seededRandom(42), second = seededRandom(42);
  assert.deepEqual(Array.from({ length: 20 }, first), Array.from({ length: 20 }, second));
  let actual;
  await completeClimb({ request: { ...request, angle: 90 }, manifest,
    predict: async (_, c) => { actual = [...c]; return new Float32Array(rank.length); } });
  assert.deepEqual(actual, [5, 1]);
});

test('validates IDs, roles, duplicates, role limits, conditions, seed, and model format', () => {
  for (const invalid of [{ angle: 91 }, { grade: NaN }, { temperature: 0 }, { seed: -1 },
    { selected: [{ id: 1, role: 0 }] }, { selected: [{ id: 100, role: 4 }] },
    { selected: [{ id: 100, role: 0 }, { id: 100, role: 2 }] },
    { selected: [{ id: 100, role: 2 }, { id: 101, role: 2 }, { id: 102, role: 2 }] }]) {
    assert.throws(() => validateRequest({ ...request, ...invalid }, manifest));
  }
  assert.throws(() => validateRequest(request, { ...manifest, format: undefined }), /kilter\.export_web/);
  assert.throws(() => validateRequest({ ...request, selected: holds.map(h => ({ id: h.id, role: 0 })) },
    { ...manifest, max_holds: 6 }), /at most 6/);
});

test('cancellation stops generation and never reports an EOS completion', async () => {
  let stop = false;
  const result = await completeClimb({ request, manifest, predict: async () => new Float32Array(rank.length),
    onStep: () => { stop = true; }, cancelled: () => stop });
  assert.equal(result.cancelled, true);
  assert.equal(result.endedWithEos, false);
  assert.equal(result.holds.length, 3);
});
