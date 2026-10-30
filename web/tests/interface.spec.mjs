import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
const manifest = JSON.parse(await readFile(new URL('../public/model/manifest.json', import.meta.url)));

async function ready(page) {
  await page.goto('/');
  await expect(page.locator('#model-status-text')).toHaveText('Model ready', { timeout: 60000 });
  await expect(page.locator('path.hold')).toHaveCount(476);
  await expect(page.locator('#svg-climb')).toHaveAttribute('data-renderer', 'boardsesh-aura');
  await expect(page.locator('#svg-climb')).toHaveAttribute('data-render-pending', 'false');
  await expect(page.locator('image.board-art')).toHaveCount(2);
}

// A stand-in for the browser device picker and a Kilter Original controller; writes land in window.bleWrites.
async function mockBluetooth(page) {
  await page.addInitScript(() => {
    window.bleWrites = [];
    const characteristic = { properties: { writeWithoutResponse: true },
      writeValueWithoutResponse: async value => { window.bleWrites.push(...value); },
      writeValueWithResponse: async value => { window.bleWrites.push(...value); } };
    const device = { id: 'fake', name: 'Kilter Board#751737@3', addEventListener() {}, removeEventListener() {},
      gatt: { connected: false, async connect() { this.connected = true; return this; }, disconnect() { this.connected = false; },
        getPrimaryService: async () => ({ getCharacteristic: async () => characteristic }) } };
    Object.defineProperty(navigator, 'bluetooth', { value: { requestDevice: async () => device } });
  });
}

test('click arbitrary holds, set grade and angle, complete with real model, and export', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await ready(page);
  const picks = [ { id: manifest.holds[200].id, role: 2 },
    { id: manifest.holds[350].id, role: 3 }, { id: manifest.holds[50].id, role: 1 } ];
  for (const pick of picks) {
    await page.locator(`[data-role="${pick.role}"]`).click();
    await page.locator(`#hold-${pick.id}`).click();
  }
  await expect(page.locator('#selected-count')).toHaveText('3');
  await page.locator('#grade').selectOption('7');
  await page.locator('#angle').evaluate(input => { input.value = 60; input.dispatchEvent(new Event('input')); });
  await page.getByText('Sampling settings', { exact: true }).click();
  await page.locator('#vary-seed').uncheck();
  await page.locator('#seed').fill('123');
  await page.locator('#generate').click();
  await expect(page.locator('#generation-status')).toContainText('Complete ·', { timeout: 30000 });
  await expect(page.locator('.hold.generated').first()).toBeVisible();
  for (const pick of picks) await expect(page.locator(`#hold-${pick.id}`)).toHaveAttribute('aria-pressed', 'true');
  const downloaded = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloaded;
  const output = JSON.parse(await readFile(await download.path(), 'utf8'));
  expect(output.v_grade).toBe(7);
  expect(output.angle).toBe(60);
  expect(output.normalized_angle).toBe(60 / 90);
  expect(output.seed).toBe(123);
  expect(output.model.checkpoint_sha256).toBe(manifest.checkpoint_sha256);
  expect(output.ended_with_eos).toBe(true);
  for (const pick of picks) expect(output.frames[0]).toContainEqual(expect.objectContaining({ hold_token: pick.id, hold_use_type: pick.role }));
  const ids = output.frames[0].map(hold => hold.hold_token);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids).toEqual([...ids].sort((a, b) => a - b));
  await expect(page.locator('#svg-climb')).toHaveAttribute('data-render-pending', 'false');
  await page.screenshot({ path: 'test-results/climb-completion.png', fullPage: true });
  await page.locator('#keep-generated').click();
  await expect(page.locator('#selected-count')).toHaveText(String(ids.length));
  await page.locator('#clear').click();
  await expect(page.locator('#selected-count')).toHaveText('0');
  expect(errors).toEqual([]);
});

test('browser WASM inference matches PyTorch fixtures, including full context', async ({ page }) => {
  await ready(page);
  const error = await page.evaluate(async () => {
    const { loadModel } = await import('/src/runtime.mjs');
    const { predict } = await loadModel();
    const fixtures = await fetch('/model/parity-fixtures.json').then(r => r.json());
    let error = 0;
    for (const fixture of fixtures) {
      const actual = await predict(fixture.tokens, fixture.conditions);
      for (let i = 0; i < actual.length; i++) error = Math.max(error, Math.abs(actual[i] - fixture.logits[i]));
    }
    return error;
  });
  expect(error).toBeLessThan(0.0002);
});

test('keyboard selection, cancellation, and mobile layout', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  const id = manifest.holds[350].id;
  await page.locator(`#hold-${id}`).focus();
  await page.keyboard.press('Enter');
  await expect(page.locator('#selected-count')).toHaveText('1');
  const hasOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  expect(hasOverflow).toBe(false);
  await page.locator('#generate').click();
  await expect(page.locator('.hold.generated').first()).toBeAttached();
  await page.locator('#stop').click();
  await expect(page.locator('#generation-status')).toContainText('Stopped ·');
  await expect(page.locator(`#hold-${id}`)).toHaveAttribute('aria-pressed', 'true');
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true });
});

test('connects to a Kilter board over Web Bluetooth and puts the completed climb on the wall', async ({ page }) => {
  await mockBluetooth(page);
  const mapping = JSON.parse(await readFile(new URL('../public/vendor/boardsesh/kilter-original-12x12-leds.json', import.meta.url)));
  await ready(page);
  await expect(page.locator('#ble-pill')).toHaveText('Offline');
  await page.locator('#ble-connect').click();
  await expect(page.locator('#ble-pill')).toHaveText('Connected');
  await expect(page.locator('#ble-send')).toBeDisabled();
  await page.locator('#ble-auto').check();
  const pick = { id: manifest.holds[300].id, role: 3 };
  await page.locator(`[data-role="${pick.role}"]`).click();
  await page.locator(`#hold-${pick.id}`).click();
  await page.locator('#generate').click();
  await expect(page.locator('#ble-pill')).toHaveText('On the wall', { timeout: 30000 });
  await expect(page.locator('#ble-send')).toHaveText('Sent ✓');

  const leds = await page.evaluate(() => {
    const bytes = window.bleWrites, leds = {};
    for (let i = 0; i < bytes.length; i += bytes[i + 1] + 5) {
      for (let j = i + 5; j < i + 4 + bytes[i + 1]; j += 3) leds[bytes[j] | bytes[j + 1] << 8] = bytes[j + 2];
    }
    return leds;
  });
  const shown = await page.locator('path.hold[stroke]:not([stroke="transparent"])').evaluateAll(paths => paths.map(path => Number(path.id.slice(5))));
  expect(Object.keys(leds).length).toBe(shown.length);
  for (const id of shown) expect(leds[mapping.positions[id]]).toBeDefined();
  expect(leds[mapping.positions[pick.id]]).toBe(0b11100011);

  await page.locator('#clear').click();
  await expect(page.locator('#ble-pill')).toHaveText('Connected');
  await page.evaluate(() => { window.bleWrites.length = 0; });
  await page.locator('#ble-clear').click();
  await expect(page.locator('#ble-pill')).toHaveText('On the wall');
  expect(await page.evaluate(() => window.bleWrites)).toEqual([1, 1, 171, 2, 84, 3]);
  await page.locator('#ble-connect').click();
  await expect(page.locator('#ble-pill')).toHaveText('Offline');
  await expect(page.locator('#ble-message')).toContainText('keeps its last LED state');
});

test('generates from nothing, and from two start holds plus a foot, with valid start and finish holds', async ({ page }) => {
  await ready(page);
  const ys = new Map(manifest.holds.map(hold => [hold.id, hold.y]));
  const low = manifest.holds.filter(hold => hold.y > 0.2 && hold.y < 0.35 && !hold.kickboard);
  async function generateAndExport() {
    await page.locator('#generate').click();
    await expect(page.locator('#generation-status')).toContainText('Complete ·', { timeout: 30000 });
    const downloaded = page.waitForEvent('download');
    await page.locator('#export').click();
    return JSON.parse(await readFile(await (await downloaded).path(), 'utf8')).frames[0];
  }
  const check = holds => {
    const ids = holds.map(hold => hold.hold_token);
    expect(new Set(ids).size).toBe(ids.length);
    const starts = holds.filter(hold => hold.hold_use_type === 2), finishes = holds.filter(hold => hold.hold_use_type === 3);
    expect(starts.length).toBeGreaterThanOrEqual(1);
    expect(starts.length).toBeLessThanOrEqual(2);
    expect(finishes.length).toBeGreaterThanOrEqual(1);
    expect(finishes.length).toBeLessThanOrEqual(2);
    return { starts, ids };
  };
  check(await generateAndExport());

  await page.locator('[data-role="2"]').click();
  for (const hold of low.slice(0, 2)) await page.locator(`#hold-${hold.id}`).click();
  await page.locator(`#hold-${low[2].id}`).click();
  await expect(page.locator('#error')).toContainText('at most 2 start');
  await expect(page.locator('#selected-count')).toHaveText('2');
  const foot = manifest.holds.find(hold => hold.y < 0.15 && !hold.kickboard);
  await page.locator('[data-role="1"]').click();
  await page.locator(`#hold-${foot.id}`).click();
  const { starts, ids } = check(await generateAndExport());
  expect(starts.map(hold => hold.hold_token).sort()).toEqual(low.slice(0, 2).map(hold => hold.id).sort());
  expect(ids).toContain(foot.id);
  expect(Math.max(...ids.map(id => ys.get(id)))).toBeGreaterThan(0.35);
});

test('light show tab previews patterns, plays them on a connected board, and clears it on stop', async ({ page }) => {
  await mockBluetooth(page);
  await ready(page);
  const mapping = JSON.parse(await readFile(new URL('../public/vendor/boardsesh/kilter-original-12x12-leds.json', import.meta.url)));
  const positions = new Set(Object.values(mapping.positions));
  await page.locator('#tab-show').click();
  await expect(page.locator('#panel-show')).toBeVisible();
  await expect(page.locator('#panel-climb')).toBeHidden();
  await expect(page.locator('#ble-send')).toBeHidden();
  await page.locator('#show-start').click();
  await expect(page.locator('#show-status')).toContainText('Previewing on screen · frame');
  await expect.poll(() => page.locator('path.hold[stroke]:not([stroke="transparent"])').count()).toBeGreaterThan(0);
  await page.locator('#show-start').click();
  await expect(page.locator('#show-start-label')).toHaveText('Start light show');

  await page.locator('#ble-connect').click();
  await expect(page.locator('#show-status')).toContainText('Ready to play on Kilter Board');
  await page.locator('[data-pattern="sparkle"]').click();
  await page.locator('#show-speed').evaluate(input => { input.value = 8; input.dispatchEvent(new Event('input')); });
  await page.locator('#show-start').click();
  await expect(page.locator('#ble-pill')).toHaveText('Light show');
  await expect(page.locator('#show-status')).toContainText('Playing on your wall · frame 4', { timeout: 10000 });
  // Several full frames went out, each lighting holds at real LED positions.
  const frames = await page.evaluate(() => {
    const bytes = window.bleWrites, frames = [];
    for (let i = 0; i < bytes.length; i += bytes[i + 1] + 5) {
      const leds = [];
      for (let j = i + 5; j < i + 4 + bytes[i + 1]; j += 3) leds.push(bytes[j] | bytes[j + 1] << 8);
      frames.push(leds);
    }
    return frames;
  });
  expect(frames.length).toBeGreaterThanOrEqual(3);
  expect(frames.slice(0, 3).every(leds => leds.length === 24 && leds.every(led => positions.has(led)))).toBe(true);
  expect(new Set(frames.slice(0, 3).map(String)).size).toBe(3);

  await page.locator('#show-start').click();
  await expect(page.locator('#ble-message')).toHaveText('Light show stopped. Board LEDs cleared.');
  expect((await page.evaluate(() => window.bleWrites)).slice(-6)).toEqual([1, 1, 171, 2, 84, 3]);
  await expect(page.locator('path.hold[stroke]:not([stroke="transparent"])')).toHaveCount(0);

  await page.locator('#tab-climb').click();
  await expect(page.locator('#panel-climb')).toBeVisible();
  await expect(page.locator('#ble-send')).toBeVisible();
  const id = manifest.holds[300].id;
  await page.locator(`#hold-${id}`).click();
  await expect(page.locator('#selected-count')).toHaveText('1');
});

test('snake tab steers with W A S D, ends on the edge, and clears the wall when left', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await mockBluetooth(page);
  await ready(page);
  const geometry = new Map(manifest.holds.map(hold => [hold.id, hold]));
  const head = () => page.locator('path.hold[stroke="#00FF00"]').evaluateAll(paths => paths.map(path => Number(path.id.slice(5))));
  await page.locator('#tab-snake').click();
  await expect(page.locator('#panel-snake')).toBeVisible();
  await expect(page.locator('#ble-send')).toBeHidden();
  await page.locator('#ble-connect').click();
  await expect(page.locator('#snake-start-label')).toHaveText('Start snake');
  await page.locator('#snake-speed').evaluate(input => { input.value = 3; input.dispatchEvent(new Event('input')); });

  // D starts a game already heading right.
  await page.keyboard.press('d');
  await expect(page.locator('#snake-start-label')).toHaveText('Stop game');
  await expect(page.locator('#ble-pill')).toHaveText('Snake');
  await expect.poll(head).toHaveLength(1);
  const [start] = await head();
  await expect.poll(async () => geometry.get((await head())[0]).x).toBeGreaterThan(geometry.get(start).x);
  expect(geometry.get((await head())[0]).y).toBeCloseTo(geometry.get(start).y);
  expect(await page.locator('path.hold[stroke="#FF00FF"]').count()).toBe(1);

  // W turns up, and the snake runs into the top of the board.
  await page.keyboard.press('w');
  await page.locator('#snake-speed').evaluate(input => { input.value = 10; input.dispatchEvent(new Event('input')); });
  await expect(page.locator('#snake-status')).toContainText('Game over. Score', { timeout: 10000 });
  await expect(page.locator('#snake-start-label')).toHaveText('Play again');
  await expect(page.locator('#ble-message')).toContainText('Snake · game over');
  expect((await page.evaluate(() => window.bleWrites)).length).toBeGreaterThan(100);

  await page.locator('#tab-climb').click();
  await expect(page.locator('#ble-message')).toHaveText('Snake stopped. Board LEDs cleared.');
  expect((await page.evaluate(() => window.bleWrites)).slice(-6)).toEqual([1, 1, 171, 2, 84, 3]);
  await expect(page.locator('path.hold[stroke]:not([stroke="transparent"])')).toHaveCount(0);
  expect(errors).toEqual([]);
});
