import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { COLUMNS, DIRECTIONS, FIELDS, fromCsv, holdOrder, isComplete, migrateLabel, normalizeGrip, parseCsv, rotateGrip,
  toCsv, toggleDirection } from '../src/labels.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const ids = manifest.holds.map(hold => hold.id);

test('holds are visited top row first, left to right, each exactly once', () => {
  const order = holdOrder(manifest.holds);
  assert.equal(new Set(order).size, 476);
  const byId = new Map(manifest.holds.map(hold => [hold.id, hold]));
  for (let i = 1; i < order.length; i++) {
    const [a, b] = [byId.get(order[i - 1]), byId.get(order[i])];
    assert.ok(a.y > b.y || (a.y === b.y && a.x < b.x));
  }
  assert.ok(byId.get(order.at(-1)).kickboard);
});

test('a hold is complete with every field, or with type unsure alone', () => {
  assert.equal(isComplete({}), false);
  assert.equal(isComplete({ type: 'jug', size: 'large' }), false);
  assert.equal(isComplete({ type: 'jug', size: 'large', grip: [] }), false);
  assert.equal(isComplete({ type: 'jug', size: 'large', grip: [45] }), true);
  assert.equal(isComplete({ type: 'unsure' }), true);
});

test('every shortcut key is distinct', () => {
  const keys = Object.values(FIELDS).flatMap(field => field.options.map(option => option.key));
  assert.equal(new Set(keys).size, keys.length);
  for (const reserved of ['Enter', 'u', 'n', 'a', '[', ']', 'Delete', 'Backspace', 'ArrowUp']) assert.ok(!keys.includes(reserved));
});

test('the grip compass has 16 directions every 22.5 degrees, toggled and rotated with wraparound', () => {
  assert.equal(DIRECTIONS.length, 16);
  assert.equal(DIRECTIONS[1], 22.5);
  assert.equal(DIRECTIONS.at(-1), 337.5);
  assert.deepEqual(toggleDirection([90], 45), [45, 90]);
  assert.deepEqual(toggleDirection([45, 90], 45), [90]);
  assert.deepEqual(rotateGrip([0, 337.5], 1), [0, 22.5]);
  assert.deepEqual(rotateGrip([0], -2), [315]);
  assert.deepEqual(normalizeGrip(['90', 0, 90]), [0, 90]);
  assert.throws(() => normalizeGrip([30]), /not a grip direction/);
});

test('labels saved with the old grip sides become directions', () => {
  assert.deepEqual(migrateLabel({ type: 'jug', grip: 'top' }), { type: 'jug', grip: [0] });
  assert.deepEqual(migrateLabel({ grip: 'left' }).grip, [270]);
  assert.equal(migrateLabel({ grip: 'all' }).grip.length, 16);
  assert.deepEqual(migrateLabel({ grip: [45] }), { grip: [45] });
});

test('CSV export has a row per hold and round-trips labels, including awkward notes', () => {
  const labels = {
    [ids[0]]: { type: 'pinch', size: 'small', grip: [22.5, 270], notes: 'thumb "catch", then\nmatch', updated: '2026-11-05T12:00:00.000Z' },
    [ids[5]]: { type: 'unsure' },
  };
  const csv = toCsv(manifest.holds, labels);
  const rows = parseCsv(csv);
  assert.deepEqual(rows[0], COLUMNS);
  assert.equal(rows.length, 477);
  const first = Object.fromEntries(COLUMNS.map((name, i) => [name, rows[1][i]]));
  assert.equal(first.hole_name, manifest.holds[0].name);
  assert.equal(first.set_id, String(manifest.holds[0].set));
  assert.equal(first.grip, '22.5;270');
  assert.deepEqual(fromCsv(csv, ids), labels);
});

test('import rejects holds from another board and values outside the schema', () => {
  assert.throws(() => fromCsv('hold_token,type\n99999,jug\n', ids), /not on this board/);
  assert.throws(() => fromCsv(`hold_token,type\n${ids[0]},jugg\n`, ids), /unknown type/);
  assert.throws(() => fromCsv('id,type\n', ids), /hold_token/);
  assert.throws(() => fromCsv(`hold_token,grip\n${ids[0]},0;30\n`, ids), /30 is not a grip direction/);
  assert.deepEqual(fromCsv(`hold_token,grip\n${ids[0]},bottom\n`, ids), { [ids[0]]: { grip: [180] } });
  assert.deepEqual(fromCsv(`hold_token,type,size\r\n${ids[0]},edge,\r\n${ids[1]},,\r\n`, ids), { [ids[0]]: { type: 'edge' } });
});
