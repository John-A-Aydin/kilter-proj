import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildClimbExport, exportFileName, generatedSource, generatorCheckpoint, keptClimb, parseClimbFile,
  shouldAutoSend } from '../src/climb-file.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const [a, b, c] = manifest.holds.map(hold => hold.id);
const geometry = new Map(manifest.holds.map(hold => [hold.id, hold]));
const otherModel = { ...manifest, name: 'other', checkpoint_sha256: 'other-checkpoint' };

// Export what the page would show for a parsed climb, then read it back.
function roundTrip(climb, viewer = manifest) {
  const result = { source: climb.source, grade: climb.grade, gradeBasis: climb.gradeBasis, angle: climb.angle,
    endedWithEos: climb.endedWithEos, selected: climb.selected };
  const output = JSON.parse(JSON.stringify(buildClimbExport({ holds: climb.holds, geometry, result, manifest: viewer })));
  return { output, climb: parseClimbFile(output, manifest) };
}

test('reads a schema 1 export, keeping its picks, grade, angle, seed, and temperature', () => {
  const climb = parseClimbFile({ schema_version: 1, v_grade: 7, angle: 45, seed: 9, temperature: 0.8, ended_with_eos: false,
    selected_holds: [{ id: b, role: 2 }], model: { name: 'old', checkpoint_sha256: 'abc' },
    frames: [[{ hold_token: c, hold_use_type: 3 }, { hold_token: b, hold_use_type: 2 }]] }, manifest);
  assert.deepEqual(climb.holds, [{ id: b, role: 2 }, { id: c, role: 3 }]);
  assert.deepEqual(climb.selected, [{ id: b, role: 2 }]);
  assert.equal(climb.grade, 7);
  assert.equal(climb.gradeBasis, 'recorded');
  assert.equal(climb.angle, 45);
  assert.equal(climb.seed, 9);
  assert.equal(climb.temperature, 0.8);
  assert.equal(climb.endedWithEos, false);
  assert.equal(climb.checkpoint, 'abc');
  // Schema 1 stamped whatever model the page had, so it is kept as recorded, not as the generator.
  assert.equal(climb.source.kind, 'legacy_export');
  assert.deepEqual(climb.source.recorded_model, { name: 'old', checkpoint_sha256: 'abc' });
  assert.deepEqual(climb.source.request, { grade: 7, angle: 45, temperature: 0.8, seed: 9, selected_holds: [{ id: b, role: 2 }] });
});

test('reads a dataset row, taking the grade from grade_hueco, and keeps the row as the source', () => {
  const row = { climb_uuid: 'X', climb_angle_id: 'X:40', angle: 40, grade_hueco: 'V5', name: 'pinch city7',
    setter_username: 'someone', frames: [[{ hold_use_type: 2, hold_token: a }, { hold_use_type: 3, hold_token: b }]] };
  const climb = parseClimbFile(row, manifest);
  assert.equal(climb.grade, 5);
  assert.equal(climb.gradeBasis, 'catalogue');
  assert.equal(climb.name, 'pinch city7');
  assert.deepEqual(climb.selected, []);
  assert.equal(climb.seed, null);
  assert.equal(climb.checkpoint, null);
  assert.deepEqual(climb.source, { kind: 'catalogue', climb_uuid: 'X', climb_angle_id: 'X:40', name: 'pinch city7',
    setter_username: 'someone', angle: 40, grade_hueco: 'V5' });
  const ungraded = parseClimbFile({ ...row, grade_hueco: null }, manifest);
  assert.equal(ungraded.grade, null);
  assert.equal(ungraded.gradeBasis, null);
  assert.equal(parseClimbFile({ frames: row.frames }, manifest).source.kind, 'unknown');
});

test('an imported climb keeps its source and unknown grade when exported again, whatever model is loaded', () => {
  const row = { climb_uuid: 'X', angle: 40, grade_hueco: null, name: 'Ungraded', setter_username: 'someone',
    frames: [[{ hold_use_type: 2, hold_token: a }, { hold_use_type: 3, hold_token: b }]] };
  const first = roundTrip(parseClimbFile(row, manifest));
  assert.equal(first.output.schema_version, 2);
  assert.equal(first.output.v_grade, null);
  assert.equal(first.output.grade_hueco, null);
  assert.equal(first.output.grade_basis, null);
  assert.equal(first.output.angle, 40);
  assert.equal(first.output.source.kind, 'catalogue');
  assert.equal(first.output.source.climb_uuid, 'X');
  assert.equal(first.output.source.model, undefined);
  assert.deepEqual(first.output.viewer_model, { name: manifest.name, checkpoint_sha256: manifest.checkpoint_sha256 });
  assert.equal(first.climb.name, 'Ungraded');
  const second = roundTrip(first.climb, otherModel);
  assert.deepEqual(second.output.source, first.output.source);
  assert.equal(second.output.v_grade, null);
  assert.equal(second.output.viewer_model.checkpoint_sha256, 'other-checkpoint');
});

test('a generated climb records its model and request, and keeps them through another model\'s export', () => {
  const request = { grade: 6, angle: 25, temperature: 0.9, seed: 12, selected: [{ id: a, role: 2 }] };
  const parent = keptClimb({ source: { kind: 'catalogue', climb_uuid: 'P' }, grade: 4, angle: 30 }, [{ id: a, role: 2 }, { id: b, role: 3 }]);
  const source = generatedSource({ manifest, request, endedWithEos: true, cancelled: false, derivedFrom: parent });
  const holds = [{ id: a, role: 2 }, { id: c, role: 3 }];
  const output = buildClimbExport({ holds, geometry, manifest: otherModel,
    result: { source, grade: 6, gradeBasis: 'requested', angle: 25, endedWithEos: true, selected: request.selected } });
  assert.equal(output.source.model.checkpoint_sha256, manifest.checkpoint_sha256);
  assert.equal(output.viewer_model.checkpoint_sha256, 'other-checkpoint');
  assert.deepEqual(output.source.request, { grade: 6, angle: 25, temperature: 0.9, seed: 12, selected_holds: [{ id: a, role: 2 }] });
  assert.equal(output.source.derived_from.source.climb_uuid, 'P');
  assert.deepEqual(output.source.derived_from.frames[0].map(h => h.hold_token), [a, b]);
  assert.equal(output.grade_basis, 'requested');
  assert.equal(exportFileName(output), 'kilter-V6-25deg-12.json');
  const climb = parseClimbFile(JSON.parse(JSON.stringify(output)), manifest);
  assert.equal(generatorCheckpoint(climb.source), manifest.checkpoint_sha256);
  assert.equal(climb.seed, 12);
  assert.equal(climb.temperature, 0.9);
  assert.deepEqual(climb.selected, [{ id: a, role: 2 }]);
});

test('only finished climbs are sent to the board automatically', () => {
  assert.equal(shouldAutoSend({ cancelled: false, endedWithEos: true }), true);
  assert.equal(shouldAutoSend({ cancelled: true, endedWithEos: false }), false);
  assert.equal(shouldAutoSend({ cancelled: true, endedWithEos: true }), false);
  assert.equal(shouldAutoSend({ cancelled: false, endedWithEos: false }), false);
  assert.equal(shouldAutoSend(null), false);
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
  assert.throws(() => parseClimbFile({ schema_version: 3, frames }, manifest), /schema 3/);
  assert.throws(() => parseClimbFile({ schema_version: 2, frames }, manifest), /source/);
  assert.throws(() => parseClimbFile({ schema_version: 2, frames, source: { kind: 'unknown' }, v_grade: 3, grade_basis: 'guess' }, manifest), /grade_basis/);
});
