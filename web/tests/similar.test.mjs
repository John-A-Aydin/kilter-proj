import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { climbHolds, closestClimb, gradeNear, loadCatalogue } from '../src/similar.mjs';

// Packs climbs the way kilter/export_web.py does: hold_id * 4 + role, widest arrays first.
function pack(climbs) {
  const columns = [
    ['ascents', Uint32Array, climbs.flatMap(c => c.angles.map(a => a.ascents))],
    ['holds', Uint16Array, climbs.flatMap(c => c.holds.map(([id, role]) => id * 4 + role))],
    ['hold_counts', Uint8Array, climbs.map(c => c.holds.length)],
    ['angle_counts', Uint8Array, climbs.map(c => c.angles.length)],
    ['angles', Uint8Array, climbs.flatMap(c => c.angles.map(a => a.angle))],
    ['grades', Uint8Array, climbs.flatMap(c => c.angles.map(a => a.grade))],
  ];
  const bytes = columns.reduce((total, [, Type, values]) => total + values.length * Type.BYTES_PER_ELEMENT, 0);
  const buffer = new ArrayBuffer(bytes);
  let offset = 0;
  for (const [, Type, values] of columns) {
    new Type(buffer, offset, values.length).set(values);
    offset += values.length * Type.BYTES_PER_ELEMENT;
  }
  const meta = { format: 'kilter-climbs-v1', climbs: climbs.length, names: climbs.map(c => c.name),
    arrays: columns.map(([name, Type, values]) => ({ name, type: Type.name.replace('Array', '').toLowerCase(), length: values.length })) };
  return loadCatalogue(meta, buffer);
}

const at = (angle, grade, ascents) => ({ angle, grade, ascents });
const catalogue = pack([
  { name: 'Popular', holds: [[1100, 2], [1200, 0], [1300, 3]], angles: [at(40, 5, 900), at(20, 3, 50)] },
  { name: 'Same holds, other roles', holds: [[1100, 0], [1200, 2], [1300, 3]], angles: [at(40, 6, 10)] },
  { name: 'Bigger', holds: [[1100, 2], [1200, 0], [1250, 0], [1300, 3], [1050, 1]], angles: [at(30, 4, 5)] },
  { name: 'Far away', holds: [[1400, 2], [1500, 3]], angles: [at(40, 9, 1)] },
]);

test('the closest climb has the most shared holds relative to both climbs', () => {
  const match = closestClimb(catalogue, [{ id: 1100, role: 2 }, { id: 1200, role: 0 }, { id: 1250, role: 0 }, { id: 1300, role: 3 }]);
  assert.equal(match.name, 'Bigger');
  assert.equal(match.shared, 4);
  assert.equal(match.overlap, 4 / 5);
  assert.equal(match.sameRole, 4);
  assert.deepEqual(match.angles, [at(30, 4, 5)]);
});

test('ties go to more holds in the same role, then to the more-ascended climb', () => {
  const roles = closestClimb(catalogue, [{ id: 1100, role: 0 }, { id: 1200, role: 2 }, { id: 1300, role: 3 }]);
  assert.equal(roles.name, 'Same holds, other roles');
  assert.equal(roles.overlap, 1);
  const popular = closestClimb(catalogue, [{ id: 1100, role: 1 }, { id: 1200, role: 1 }, { id: 1300, role: 1 }]);
  assert.equal(popular.name, 'Popular');
  assert.equal(popular.sameRole, 0);
  // Lookups leave no counts behind.
  assert.ok(catalogue.shared.every(value => value === 0) && catalogue.sameRole.every(value => value === 0));
});

test('holds no climb uses give no match, and repeated holds are rejected', () => {
  assert.equal(closestClimb(catalogue, [{ id: 1999, role: 0 }]), null);
  assert.equal(closestClimb(catalogue, []), null);
  assert.throws(() => closestClimb(catalogue, [{ id: 1100, role: 2 }, { id: 1100, role: 1 }]));
});

test('grades are read at the requested angle, or the nearest one', () => {
  const angles = closestClimb(catalogue, [{ id: 1100, role: 2 }, { id: 1200, role: 0 }, { id: 1300, role: 3 }]).angles;
  assert.deepEqual(angles, [at(20, 3, 50), at(40, 5, 900)]);
  assert.deepEqual(gradeNear(angles, 40), at(40, 5, 900));
  assert.deepEqual(gradeNear(angles, 25), at(20, 3, 50));
});

test('the exported catalogue matches the model and finds each real climb as its own match', () => {
  const meta = JSON.parse(readFileSync(new URL('../public/model/climbs.json', import.meta.url)));
  const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
  const file = readFileSync(new URL('../public/model/climbs.bin', import.meta.url));
  const real = loadCatalogue(meta, file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
  assert.equal(meta.data_fingerprint, manifest.data_fingerprint);
  assert.ok(real.count > 200000);
  const ids = new Set(manifest.holds.map(hold => hold.id));
  for (const climb of [0, 1, 1000, real.count - 1]) {
    const holds = climbHolds(real, climb);
    assert.ok(holds.every(hold => ids.has(hold.id)));
    const match = closestClimb(real, holds);
    assert.equal(match.overlap, 1);
    assert.equal(match.sameRole, holds.length);
  }
  const ascents = climb => closestClimb(real, climbHolds(real, climb)).angles.reduce((total, row) => total + row.ascents, 0);
  assert.ok(ascents(0) >= ascents(1000));
});
