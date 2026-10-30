import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MAX_SHOW_HOLDS, PATTERNS, showFrame } from '../src/lightshow.mjs';
// The browser modules import the compiled Boardsesh protocol from ../vendor/, which only exists in dist/.
import { encodeClimb } from '../dist/src/bluetooth.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const mapping = JSON.parse(readFileSync(new URL('../public/vendor/boardsesh/kilter-original-12x12-leds.json', import.meta.url)));
const ids = new Set(manifest.holds.map(hold => hold.id));

test('every pattern frame is a valid, encodable climb within the LED power budget', () => {
  for (const pattern of Object.keys(PATTERNS)) {
    let lit = 0;
    for (let t = 0; t < 200; t++) {
      const frame = showFrame(pattern, manifest.holds, t);
      assert.ok(frame.length <= MAX_SHOW_HOLDS, `${pattern} frame ${t}`);
      assert.equal(new Set(frame.map(hold => hold.id)).size, frame.length);
      assert.ok(frame.every(hold => ids.has(hold.id) && [0, 1, 2, 3].includes(hold.role)));
      for (const api of [2, 3]) assert.equal(encodeClimb(frame, mapping, api).ledsWritten, frame.length);
      lit += frame.length > 0;
    }
    assert.ok(lit > 180, `${pattern} is dark for too many frames`);
  }
});

test('patterns move and change color, and are repeatable', () => {
  for (const pattern of Object.keys(PATTERNS)) {
    const frames = Array.from({ length: 60 }, (_, t) => JSON.stringify(showFrame(pattern, manifest.holds, t)));
    assert.ok(new Set(frames).size > 30, pattern);
    assert.deepEqual(showFrame(pattern, manifest.holds, 17), showFrame(pattern, manifest.holds, 17));
    const roles = new Set(Array.from({ length: 200 }, (_, t) => showFrame(pattern, manifest.holds, t)).flat().map(h => h.role));
    assert.equal(roles.size, 4, pattern);
  }
  assert.throws(() => showFrame('disco', manifest.holds, 0), /Unknown/);
});
