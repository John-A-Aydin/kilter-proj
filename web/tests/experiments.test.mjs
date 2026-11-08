import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { PROBE_FIGURES, REPOSITORY, chartData, parseCsv, probeData, renderMarkdown, repositoryLink } from '../experiments-page.mjs';
import { niceTicks } from '../src/experiments.mjs';
import { angleError, shareWithin } from '../src/probes.mjs';

test('relative README links point at the repository on GitHub', () => {
  assert.equal(repositoryLink('compare_models.py'), `${REPOSITORY}/blob/main/experiments/compare_models.py`);
  assert.equal(repositoryLink('../docs/model.md#model'), `${REPOSITORY}/blob/main/docs/model.md#model`);
  assert.equal(repositoryLink('.'), `${REPOSITORY}/blob/main/experiments/`);
  for (const href of ['#model-comparison', 'https://example.com/a', 'mailto:a@b.c']) assert.equal(repositoryLink(href), href);
});

test('markdown gets heading anchors, wrapped tables, and chart placeholders', () => {
  const { html, headings } = renderMarkdown('## Null grade\n\n### Null grade\n\n| a | b |\n| --- | ---: |\n| x | 1 |\n\n<!-- chart: holds-by-grade -->\n');
  assert.deepEqual(headings.map(h => h.id), ['null-grade', 'null-grade-2']);
  assert.match(html, /<div class="table-scroll"><table>/);
  assert.match(html, /<figure class="chart" data-chart="holds-by-grade"><\/figure>/);
});

test('every chart the experiments README places exists, with finite values for every grade', async () => {
  const readme = await readFile(new URL('../../experiments/README.md', import.meta.url), 'utf8');
  const placed = [...readme.matchAll(/<!--\s*chart:\s*([a-z0-9-]+)\s*-->/g)].map(m => m[1]);
  const charts = await chartData();
  assert.ok(placed.length > 0);
  for (const name of placed) {
    const chart = charts[name];
    assert.ok(chart, name);
    if (chart.kind === 'probe') continue;
    assert.ok(chart.series.length >= 2 && chart.series.length <= 4, `${name} uses the four validated colors at most`);
    for (const series of chart.series) {
      assert.equal(series.values.length, chart.x.length);
      assert.ok(series.values.every(Number.isFinite), `${name}: ${series.name}`);
    }
  }
});

test('axis ticks are round and cover the data', () => {
  assert.deepEqual(niceTicks(10.38, 13.57), [10, 11, 12, 13, 14]);
  assert.deepEqual(niceTicks(0, 9.74), [0, 2, 4, 6, 8, 10]);
});

test('the built experiments page links every relative path to GitHub', async () => {
  const page = await readFile(new URL('../dist/experiments.html', import.meta.url), 'utf8');
  const hrefs = [...page.matchAll(/href="([^"]+)"/g)].map(m => m[1]);
  const local = hrefs.filter(h => !/^(https:|#|data:|\.\/(|experiments\.html|how\.html|label\.html|src\/style\.css|src\/probes\.css)$)/.test(h));
  assert.deepEqual(local, []);
  assert.doesNotMatch(page, /<!-- (TOC|EXPERIMENTS|CHART DATA|PROBE DATA) -->|\{\{REPOSITORY\}\}/);
});

test('CSV rows parse the same with Windows line endings, as Python writes them', () => {
  assert.deepEqual(parseCsv('a,b\r\n1,2\r\n'), [{ a: '1', b: '2' }]);
  assert.deepEqual(parseCsv('a,b\n1,2\n'), [{ a: '1', b: '2' }]);
});

test('the README places every probe figure, and the probe data covers every hold', async () => {
  const readme = await readFile(new URL('../../experiments/README.md', import.meta.url), 'utf8');
  for (const view of PROBE_FIGURES) assert.match(readme, new RegExp(`<!-- chart: probe-${view} -->`));
  const data = await probeData();
  assert.equal(data.holds.length, 476);
  assert.equal(new Set(data.holds.map(h => h.id)).size, 476);
  for (const hold of data.holds) {
    assert.match(hold.d, /^M[\d.]+,[\d.]+(L[\d.]+,[\d.]+)+Z$/);
    assert.ok(data.types.includes(hold.type));
    for (const prediction of Object.values(hold.pred)) {
      assert.ok(data.types.includes(prediction.type) && prediction.p >= 0 && prediction.p <= 1);
      // Exactly the single-direction holds carry a finite grip prediction.
      assert.equal(prediction.grip !== null, hold.grip.length === 1);
      if (prediction.grip !== null) assert.ok(Number.isFinite(prediction.grip));
    }
  }
  assert.equal(data.holds.filter(h => h.grip.length === 1).length, 423);
  for (const name of ['position', 'usage', 'untrained', 'identity', 'hold_vector', 'context']) {
    assert.equal(data.mean[name].hand_type_balanced_accuracy.values.length, 3);
    assert.equal(data.handRecall[name].length, 3);
  }
});

test('grip angles wrap around the circle', () => {
  assert.equal(angleError(350, 10), 20);
  assert.equal(angleError(0, 180), 180);
  assert.equal(angleError(90, 90), 0);
  assert.equal(shareWithin([0, 10, 50, 90], 45), 0.5);
});
