import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { buildBoardConfig, framesForHolds, outlinePath, BOARD_COLORS } from '../src/board-config.mjs';
import { initSync, render_overlay } from '../public/vendor/boardsesh/board_renderer_wasm.js';

const asset = name => new URL(`../public/vendor/boardsesh/${name}`, import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const geometry = JSON.parse(readFileSync(asset('kilter-original-12x12.json')));

test('Boardsesh traces map every model hold to the Original 12x12 artwork', () => {
  const config = buildBoardConfig(manifest, geometry);
  assert.equal(config.holds.length, 476);
  assert.equal(config.board_width, 1080);
  assert.equal(config.board_height, 1170);
  const hold = config.holds.find(h => h.id === 1073);
  assert.equal(hold.cx, 1020);
  assert.equal(hold.cy, 1110);
  assert.equal(hold.r, 30);
  for (const hold of config.holds) {
    assert.ok(hold.outline.length >= 6);
    assert.equal(hold.outline.length % 2, 0);
    const path = outlinePath(hold);
    assert.match(path, /^M/);
    assert.match(path, /Z$/);
    assert.ok(!path.includes('NaN'));
  }
  assert.throws(() => buildBoardConfig({ holds: manifest.holds.slice(1) }, geometry));
});

test('model roles become Kilter Original frame codes and Boardsesh display colors', () => {
  const holds = [0, 1, 2, 3].map((role, i) => ({ id: 1073 + i, role }));
  assert.equal(framesForHolds(holds), 'p1073r13p1074r15p1075r12p1076r14');
  const map = buildBoardConfig(manifest, geometry).hold_state_map;
  assert.equal(map[12].color, BOARD_COLORS[2]);
  assert.equal(map[13].role, 'hand');
  assert.equal(map[14].role, 'finish');
  assert.equal(map[15].role, 'foot');
});

test('vendored artwork, geometry and WASM match the pinned Boardsesh revision', () => {
  const provenance = JSON.parse(readFileSync(asset('provenance.json')));
  assert.equal(provenance.revision, '15f6a8588585992725d9f9ba181271a0de9165f9');
  for (const [name, record] of Object.entries(provenance.files)) {
    const hash = createHash('sha256').update(readFileSync(asset(name))).digest('hex');
    assert.equal(hash, record.sha256, name);
  }
});

test('the actual Boardsesh WASM renders traced Aura pixels rather than classic circles', () => {
  initSync({ module: readFileSync(asset('board_renderer_wasm_bg.wasm')) });
  const config = { ...buildBoardConfig(manifest, geometry), frames: 'p1073r12p1273r13' };
  const aura = render_overlay(JSON.stringify(config));
  const view = new DataView(aura.buffer, aura.byteOffset, aura.byteLength);
  assert.equal(view.getUint32(0, true), 1080);
  assert.equal(view.getUint32(4, true), 1170);
  assert.equal(aura.length, 8 + 1080 * 1170 * 4);
  assert.ok(aura.some((byte, i) => i >= 8 && (i - 8) % 4 === 3 && byte > 0));
  const classic = render_overlay(JSON.stringify({ ...config, render_mode: 'classic' }));
  assert.notDeepEqual(aura, classic);
});
