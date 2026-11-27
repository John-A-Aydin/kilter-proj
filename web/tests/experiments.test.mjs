import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import test from 'node:test';
import { GRADING_METHODS, PROBE_FIGURES, REPOSITORY, TAG_FIGURES, chartData, gradingConfusion, parseCsv, probeData, renderMarkdown, repositoryLink, tagData } from '../experiments-page.mjs';
import { niceTicks } from '../src/experiments.mjs';
import { SCALE_MAX, agreement, cellMix, count } from '../src/grading-figure.mjs';
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
    if (chart.kind) continue;
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
  assert.doesNotMatch(page, /<!-- (TOC|EXPERIMENTS|CHART DATA|PROBE DATA|TAG DATA) -->|\{\{REPOSITORY\}\}/);
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
  assert.equal(data.holds.filter(h => h.grip.length === 1).length, 476);
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

test('the README places every style-tag figure, and the tag data covers exactly the current tags', async () => {
  const readme = await readFile(new URL('../../experiments/README.md', import.meta.url), 'utf8');
  for (const view of TAG_FIGURES) assert.match(readme, new RegExp(`<!-- chart: tags-${view} -->`));
  const data = await tagData();
  // The tag file's header lists the tags in use; results from an older tag set would differ.
  const tagFile = gunzipSync(await readFile(new URL('../../data/training/climb_tags.csv.gz', import.meta.url))).toString();
  assert.deepEqual(data.tags, tagFile.slice(0, tagFile.indexOf('\n')).trim().split(',').slice(1));
  assert.deepEqual(data.seeds, [42, 43, 44]);
  for (const tag of data.tags) {
    const s = data.steering[tag];
    assert.ok(s.off >= 0 && s.off <= 1 && s.on >= 0 && s.on <= 1, tag);
    assert.equal(s.shiftSeeds.length, 3);
    assert.equal(s.statDiff.length, data.stats.length);
    assert.ok(data.auc[tag] > 0 && data.auc[tag] < 1, tag);
  }
  for (const block of Object.values(data.likelihood)) {
    assert.equal(block.seeds.length, 3);
    assert.ok(block.rows >= 10);
  }
  assert.ok(data.correlation > -1 && data.correlation < 1);
});

test('the grading heatmap has a true-by-predicted grade matrix for every method and test subset', async () => {
  const data = await gradingConfusion();
  assert.equal(data.kind, 'grading');
  assert.deepEqual(data.grades, [...Array(17).keys()]);
  assert.deepEqual(data.methods.map(m => m.key), GRADING_METHODS.map(([key]) => key));
  assert.deepEqual(data.subsets.map(s => s.key), ['all', 'more_than_5_ascents', 'more_than_10_ascents', 'more_than_25_ascents']);
  assert.ok(data.methods.some(m => m.key === data.defaultMethod));
  for (const method of data.methods) {
    for (const subset of data.subsets) {
      const matrix = data.counts[method.key][subset.key];
      assert.equal(matrix.length, 17);
      assert.ok(Math.abs(matrix.flat().reduce((a, b) => a + b, 0) - subset.rows) < 0.1, `${method.key} ${subset.key}`);
      assert.ok(Number.isFinite(data.rmse[method.key][subset.key]));
      assert.ok(Number.isFinite(data.mae[method.key][subset.key]));
    }
    // Every method sees the same climbs, so each true-grade row has the same total.
    const totals = data.counts[method.key].all.map(row => row.reduce((a, b) => a + b, 0));
    assert.deepEqual(totals.map(Math.round), data.counts.model_mlp.all.map(row => Math.round(row.reduce((a, b) => a + b, 0))));
  }
  // The best grader is right more often than the training mean.
  assert.ok(agreement(data.counts.model_mlp.all).exact > agreement(data.counts.mean_at_angle.all).exact);
  const readme = await readFile(new URL('../../experiments/README.md', import.meta.url), 'utf8');
  assert.match(readme, /<!-- chart: grading-confusion -->/);
});

test('heatmap cells use one fixed scale, and agreement counts exact and off-by-one predictions', () => {
  assert.equal(cellMix(0), 0);
  assert.equal(cellMix(SCALE_MAX), 100);
  assert.equal(cellMix(0.9), 100);
  assert.ok(cellMix(0.01) > 0 && cellMix(0.01) < cellMix(0.2));
  const result = agreement([[3, 1, 0], [0, 2, 2], [1, 0, 1]]);
  assert.equal(result.total, 10);
  assert.equal(result.exact, 0.6);
  assert.equal(result.withinOne, 0.9);
  assert.equal(count(1900.99999), '1,901');
  assert.equal(count(12.34), '12.3');
});
