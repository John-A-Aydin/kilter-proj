import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { REPOSITORY, chartData, renderMarkdown, repositoryLink } from '../experiments-page.mjs';
import { niceTicks } from '../src/experiments.mjs';

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
  const local = hrefs.filter(h => !/^(https:|#|data:|\.\/(|experiments\.html|how\.html|src\/style\.css)$)/.test(h));
  assert.deepEqual(local, []);
  assert.doesNotMatch(page, /<!-- (TOC|EXPERIMENTS|CHART DATA) -->|\{\{REPOSITORY\}\}/);
});
