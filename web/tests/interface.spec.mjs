import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
const manifest = JSON.parse(await readFile(new URL('../public/model/manifest.json', import.meta.url)));

async function ready(page) {
  await page.goto('/');
  await expect(page.locator('#model-status-text')).toHaveText('Model ready', { timeout: 60000 });
  await expect(page.locator('#svg-climb path.hold')).toHaveCount(476);
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
  await expect(page.locator('#svg-climb .hold.generated').first()).toBeVisible();
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
  await expect(page.locator('#svg-climb .hold.generated').first()).toBeAttached();
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
  const shown = await page.locator('#svg-climb path.hold[stroke]:not([stroke="transparent"])').evaluateAll(paths => paths.map(path => Number(path.id.slice(5))));
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

test('shows the closest real climb below the generated one, and clears it when the request changes', async ({ page }) => {
  await ready(page);
  const panel = page.locator('#match-panel');
  await expect(panel).toBeHidden();
  await page.locator('#generate').click();
  await expect(page.locator('#generation-status')).toContainText('Complete ·', { timeout: 30000 });
  await expect(page.locator('#match-details')).toBeVisible({ timeout: 30000 });
  await expect(page.locator('#match-count')).toContainText('climbs');
  await expect(page.locator('#match-name')).not.toBeEmpty();
  await expect(page.locator('#match-overlap')).toHaveText(/^\d+%$/);
  await expect(page.locator('#svg-match')).toHaveAttribute('data-render-pending', 'false');
  const board = await page.locator('.board-panel').first().boundingBox();
  const box = await panel.boundingBox();
  expect(box.y).toBeGreaterThanOrEqual(board.y + board.height);
  // Holds shared with the generated climb are drawn solid, and only those.
  const shared = Number((await page.locator('#match-shared').textContent()).match(/^(\d+) of your/)[1]);
  const ids = locator => locator.evaluateAll(paths => paths.map(path => Number(path.id.split('-').pop())));
  const solid = await ids(page.locator('#svg-match path.hold.selected'));
  expect(solid.length).toBe(shared);
  const yours = new Set(await ids(page.locator('#svg-climb path.hold:is(.selected, .generated)')));
  expect(solid.every(id => yours.has(id))).toBe(true);
  await expect(page.locator('#svg-match path.hold[tabindex]')).toHaveCount(0);

  const downloaded = page.waitForEvent('download');
  await page.locator('#export').click();
  const output = JSON.parse(await readFile(await (await downloaded).path(), 'utf8'));
  expect(output.closest_real_climb.shared_holds).toBe(shared);
  expect(output.closest_real_climb.holds.length).toBeGreaterThan(0);

  await page.locator('#tab-show').click();
  await expect(panel).toBeHidden();
  await page.locator('#tab-climb').click();
  await expect(panel).toBeVisible();
  await page.locator('#grade').selectOption('3');
  await expect(panel).toBeHidden();
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
  await expect.poll(() => page.locator('#svg-climb path.hold[stroke]:not([stroke="transparent"])').count()).toBeGreaterThan(0);
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
  await expect(page.locator('#svg-climb path.hold[stroke]:not([stroke="transparent"])')).toHaveCount(0);

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
  const head = () => page.locator('#svg-climb path.hold[stroke="#00FF00"]').evaluateAll(paths => paths.map(path => Number(path.id.slice(5))));
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
  expect(await page.locator('#svg-climb path.hold[stroke="#FF00FF"]').count()).toBe(1);

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
  await expect(page.locator('#svg-climb path.hold[stroke]:not([stroke="transparent"])')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('the experiments tab renders the write-up and interactive charts, and links back to the tool', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('navigation', { name: 'Site' }).getByRole('link', { name: 'Experiments' }).click();
  await expect(page).toHaveURL(/experiments\.html$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Experiments');
  await expect(page.locator('#model-comparison')).toBeVisible();
  await expect(page.locator('figure[data-chart]:not([data-chart^="probe-"]) svg')).toHaveCount(2);
  const chart = page.locator('figure[data-chart="holds-by-grade"]');
  await chart.locator('svg').focus();
  await page.keyboard.press('ArrowRight');
  await expect(chart.locator('.chart-tooltip')).toContainText('V2');
  await expect(chart.locator('.chart-tooltip')).toContainText('Real held-out climbs');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  await page.getByRole('navigation', { name: 'Site' }).getByRole('link', { name: 'Climb tool' }).click();
  await expect(page.locator('#model-status-text')).toHaveText('Model ready', { timeout: 60000 });
  expect(errors).toEqual([]);
});

test('the hold-label probe figures draw both boards, switch probes and views, and list every hold', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/experiments.html');
  await expect(page.locator('body')).toHaveAttribute('data-probes-ready', 'true');
  const boards = page.locator('figure[data-chart="probe-boards"]');
  await expect(boards.locator('.probe-board svg')).toHaveCount(2);
  await expect(boards.locator('path.probe-hold')).toHaveCount(952);
  const note = boards.locator('.probe-board-head span').nth(1);
  await expect(note).toHaveText(/^\d+ of 476 right$/);
  const holdVector = await note.textContent();
  await boards.getByRole('button', { name: 'Position', exact: true }).click();
  await expect(note).not.toHaveText(holdVector);
  await boards.getByRole('button', { name: 'Grip direction' }).click();
  await expect(note).toHaveText(/^\d+ of 423 within 45°$/);
  await expect(boards.locator('.probe-arrow').first()).toBeAttached();
  await boards.getByRole('button', { name: 'Type', exact: true }).click();
  const pinch = boards.getByRole('button', { name: /^Pinch \d+$/ });
  await pinch.click();
  await expect(pinch).toHaveAttribute('aria-pressed', 'true');
  await boards.locator('path.probe-hold').first().hover();
  await expect(page.locator('.probe-tip')).toContainText('Your label');
  await boards.locator('summary', { hasText: 'Every hold' }).click();
  await expect(boards.locator('.probe-table tbody tr')).toHaveCount(476);
  for (const view of ['scores', 'projection', 'grip']) await expect(page.locator(`figure[data-chart="probe-${view}"] svg`).first()).toBeVisible();
  await expect(page.locator('figure[data-chart="probe-confusion"] .probe-matrix')).toHaveCount(3);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('how it works page draws the figures and charts from the exported results', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('link', { name: 'How it works' }).first().click();
  await expect(page).toHaveURL(/how\.html$/);
  await expect(page.locator('body')).toHaveAttribute('data-charts-ready', 'true');
  await expect(page.locator('#sequence-tokens .tok')).toHaveCount(10);
  for (const id of ['mixture-chart', 'lr-chart', 'curves-chart', 'ablation-chart', 'validity-chart', 'grade-chart', 'grading-chart']) {
    await expect(page.locator(`#${id} svg`)).toHaveCount(1);
  }
  await expect(page.locator('#ablation-chart .c-row')).toHaveCount(7);
  await expect(page.locator('#stat-nll')).toHaveText(/^\d+\.\d\d$/);
  await page.locator('#ablation-chart .c-row').last().hover();
  await expect(page.locator('#chart-tooltip')).toBeVisible();
  await expect(page.locator('#chart-tooltip')).toContainText('rule loss, 16 epochs');
  await page.locator('#curves-chart svg').focus();
  await expect(page.locator('#chart-tooltip')).toContainText('Epoch');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(errors).toEqual([]);
});

test('the hold labeler labels holds by keyboard and compass, keeps them across reloads, and exports a CSV', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await page.getByRole('navigation', { name: 'Site' }).getByRole('link', { name: 'Hold labels' }).click();
  await expect(page).toHaveURL(/label\.html$/);
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#svg-board path.label-hold')).toHaveCount(476);
  await expect(page.locator('#svg-compass path.wedge')).toHaveCount(16);
  const first = await page.locator('#hold-title').textContent();
  await page.keyboard.press('1');
  await page.keyboard.press('x');
  // A typed hold with no grip direction is outlined in red until one is picked.
  await expect(page.locator('#svg-board path.label-hold.no-grip')).toHaveCount(1);
  await expect(page.locator('#legend')).toContainText('No grip direction 1');
  // Up then rotate twice is 45 degrees; a click on the compass adds 270.
  await page.keyboard.press('ArrowUp');
  await expect(page.locator('#svg-board path.label-hold.no-grip')).toHaveCount(0);
  await page.keyboard.press(']');
  await page.keyboard.press(']');
  await page.locator('#svg-compass path[data-direction="270"]').click();
  await expect(page.locator('#grip-summary')).toHaveText('45°, 270° (left)');
  await expect(page.locator('#svg-compass path.wedge.on')).toHaveCount(2);
  // Several directions are allowed, so the hold stays put until Enter.
  await expect(page.locator('#hold-title')).toHaveText(first);
  await expect(page.locator('#progress-count')).toHaveText('1 / 476');
  await page.keyboard.press('Enter');
  await expect(page.locator('#hold-title')).not.toHaveText(first);
  await expect(page.locator('#svg-compass path.wedge.on')).toHaveCount(0);
  await page.keyboard.press('7');
  await expect(page.locator('#progress-count')).toHaveText('2 / 476');
  await page.keyboard.press('Enter');
  await page.keyboard.press('n');
  await page.keyboard.type('crimpy, "sharp" edge');
  await page.keyboard.press('Escape');
  await page.keyboard.press('4');
  await page.keyboard.press('a');
  await expect(page.locator('#grip-summary')).toHaveText('All round');
  await expect(page.getByRole('button', { name: /Pinch/ })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#svg-board path.label-hold.partial')).toHaveCount(1);

  await page.reload();
  await expect(page.locator('body')).toHaveAttribute('data-ready', 'true');
  await expect(page.locator('#progress-count')).toHaveText('2 / 476');
  await expect(page.locator('#notes')).toHaveValue('crimpy, "sharp" edge');
  await expect(page.locator('#svg-compass path.wedge.on')).toHaveCount(16);
  const id = (await page.locator('#hold-title').textContent()).replace('Hold ', '');
  await page.keyboard.press('Shift+Enter');
  await page.keyboard.press('Enter');
  await expect(page.locator('#hold-title')).toHaveText(`Hold ${id}`);

  const downloaded = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloaded;
  expect(download.suggestedFilename()).toBe('hold_labels.csv');
  const csv = await readFile(await download.path(), 'utf8');
  expect(csv.trim().split('\n').length).toBe(477);
  expect(csv).toContain(`${first.replace('Hold ', '')},`);
  expect(csv).toContain(',jug,medium,45;270,');
  expect(csv).toContain('"crimpy, ""sharp"" edge"');
  expect(csv.split('\n')[0]).toBe('hold_token,hole_name,hold_x,hold_y,set_id,on_kickboard,type,size,grip,notes,updated_at');

  await page.locator('#clear-label').click();
  await expect(page.locator('#svg-board path.label-hold.partial')).toHaveCount(0);
  page.once('dialog', dialog => dialog.accept());
  await page.locator('#import-file').setInputFiles({ name: 'hold_labels.csv', mimeType: 'text/csv', buffer: Buffer.from(csv) });
  await expect(page.locator('#save-status-text')).toHaveText('Imported 3 labeled holds');
  await expect(page.locator('#notes')).toHaveValue('crimpy, "sharp" edge');
  await expect(page.locator('#grip-summary')).toHaveText('All round');
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth)).toBe(false);
  expect(errors).toEqual([]);
});

test('imports an exported climb or a dataset row, restoring its holds, picks, grade, and angle', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await ready(page);
  const pick = manifest.holds[200].id;
  await page.locator('#svg-climb #hold-' + pick).click();
  await page.locator('#grade').selectOption('6');
  await page.locator('#angle').evaluate(input => { input.value = 25; input.dispatchEvent(new Event('input')); });
  await page.locator('#generate').click();
  await expect(page.locator('#generation-status')).toContainText('Complete ·', { timeout: 30000 });
  const shown = async () => page.locator('#svg-climb path.hold[stroke]:not([stroke="transparent"])')
    .evaluateAll(paths => paths.map(path => `${path.id}:${path.getAttribute('stroke')}:${path.getAttribute('stroke-dasharray')}`));
  const before = await shown();
  const downloaded = page.waitForEvent('download');
  await page.locator('#export').click();
  const exported = await readFile(await (await downloaded).path(), 'utf8');

  await page.locator('#clear').click();
  await page.locator('#grade').selectOption('2');
  await page.locator('#angle').evaluate(input => { input.value = 60; input.dispatchEvent(new Event('input')); });
  await page.locator('#import-climb-file').setInputFiles({ name: 'mine.json', mimeType: 'application/json', buffer: Buffer.from(exported) });
  await expect(page.locator('#generation-status')).toContainText('Imported mine.json');
  expect(await shown()).toEqual(before);
  await expect(page.locator('#selected-count')).toHaveText('1');
  await expect(page.locator('#grade')).toHaveValue('6');
  await expect(page.locator('#angle-value')).toHaveText('25°');
  await expect(page.locator('#match-panel')).toBeVisible();
  await expect(page.locator('#result-actions')).toBeVisible();

  const row = { climb_uuid: 'X', angle: 40, grade_hueco: 'V5', name: 'Imported row',
    frames: [[{ hold_use_type: 2, hold_token: manifest.holds[300].id }, { hold_use_type: 3, hold_token: manifest.holds[400].id }]] };
  await page.locator('#import-climb-file').setInputFiles({ name: 'row.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(row)) });
  await expect(page.locator('#generation-status')).toHaveText('Imported “Imported row” · 2 holds · V5 / 40°');
  await expect(page.locator('#selected-count')).toHaveText('0');
  expect((await shown()).length).toBe(2);

  await page.locator('#import-climb-file').setInputFiles({ name: 'broken.json', mimeType: 'application/json', buffer: Buffer.from('{oops') });
  await expect(page.locator('#error')).toHaveText('Could not import broken.json: it is not valid JSON.');
  expect((await shown()).length).toBe(2);
  expect(errors).toEqual([]);
});
