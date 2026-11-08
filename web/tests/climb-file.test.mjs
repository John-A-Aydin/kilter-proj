import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseClimbFile } from '../src/climb-file.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const [a, b, c] = manifest.holds.map(hold => hold.id);

test('reads the page export, keeping its picks, grade, angle, seed, and temperature', () => {
  const climb = parseClimbFile({ v_grade: 7, angle: 45, seed: 9, temperature: 0.8, ended_with_eos: false,
    selected_holds: [{ id: b, role: 2 }], model: { checkpoint_sha256: 'abc' },
    frames: [[{ hold_token: c, hold_use_type: 3 }, { hold_token: b, hold_use_type: 2 }]] }, manifest);
  assert.deepEqual(climb.holds, [{ id: b, role: 2 }, { id: c, role: 3 }]);
  assert.deepEqual(climb.selected, [{ id: b, role: 2 }]);
  assert.equal(climb.grade, 7);
  assert.equal(climb.angle, 45);
  assert.equal(climb.seed, 9);
  assert.equal(climb.temperature, 0.8);
  assert.equal(climb.endedWithEos, false);
  assert.equal(climb.checkpoint, 'abc');
});

test('reads a dataset row, taking the grade from grade_hueco', () => {
  const row = { climb_uuid: 'X', angle: 40, grade_hueco: 'V5', name: 'pinch city7',
    frames: [[{ hold_use_type: 2, hold_token: a }, { hold_use_type: 3, hold_token: b }]] };
  const climb = parseClimbFile(row, manifest);
  assert.equal(climb.grade, 5);
  assert.equal(climb.name, 'pinch city7');
  assert.deepEqual(climb.selected, []);
  assert.equal(climb.seed, null);
  assert.equal(parseClimbFile({ ...row, grade_hueco: null }, manifest).grade, null);
});

test('rejects files that are not a climb on this board', () => {
  const frames = [[{ hold_token: a, hold_use_type: 0 }]];
  assert.throws(() => parseClimbFile([], manifest), /not a climb/);
  assert.throws(() => parseClimbFile({}, manifest), /no holds/);
  assert.throws(() => parseClimbFile({ frames: [[{ hold_token: 1, hold_use_type: 0 }]] }, manifest), /not on the Kilter/);
  assert.throws(() => parseClimbFile({ frames: [[{ hold_token: a, hold_use_type: 7 }]] }, manifest), /unknown role/);
  assert.throws(() => parseClimbFile({ frames: [[...frames[0], ...frames[0]]] }, manifest), /twice/);
  assert.throws(() => parseClimbFile({ frames: [frames[0], frames[0]] }, manifest), /more than one frame/);
  assert.throws(() => parseClimbFile({ frames, selected_holds: [{ id: a, role: 2 }] }, manifest), /not in the climb/);
  assert.throws(() => parseClimbFile({ frames, angle: 42 }, manifest), /5° steps/);
  assert.throws(() => parseClimbFile({ frames, v_grade: 17 }, manifest), /V0 to V16/);
});
