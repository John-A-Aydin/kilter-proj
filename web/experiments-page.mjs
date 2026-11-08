// Builds experiments.html from experiments/README.md and the committed results/, so the
// page always matches the write-up. A `<!-- chart: name -->` line in the README places a chart.
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Marked } from 'marked';
import { BOARD_HEIGHT, BOARD_WIDTH, outlinePath } from './src/board-config.mjs';

const web = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(web);
export const REPOSITORY = 'https://github.com/John-A-Aydin/kilter-proj';

// Relative links in the README point into the repository; on the site they go to GitHub.
export function repositoryLink(href, from = 'experiments/') {
  if (/^([a-z][a-z0-9+.-]*:|#|\/)/i.test(href)) return href;
  const url = new URL(href, `https://repository.invalid/${from}`);
  return `${REPOSITORY}/blob/main${url.pathname}${url.hash}`;
}

export function slug(text) {
  return text.toLowerCase().replace(/<[^>]+>/g, '').replace(/&[a-z]+;|&#\d+;/g, '')
    .replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export function renderMarkdown(markdown) {
  const headings = [];
  const used = new Set();
  const marked = new Marked({
    renderer: {
      heading({ tokens, depth }) {
        const html = this.parser.parseInline(tokens);
        let id = slug(html);
        for (let n = 2; used.has(id); n++) id = `${slug(html)}-${n}`;
        used.add(id);
        if (depth <= 3) headings.push({ id, depth, html });
        return `<h${depth} id="${id}">${html}</h${depth}>\n`;
      },
      link({ href, title, tokens }) {
        const titled = title ? ` title="${title}"` : '';
        return `<a href="${repositoryLink(href)}"${titled}>${this.parser.parseInline(tokens)}</a>`;
      },
    },
  });
  const html = marked.parse(markdown)
    .replace(/<!--\s*chart:\s*([a-z0-9-]+)\s*-->/g, '<figure class="chart" data-chart="$1"></figure>')
    .replace(/<table>/g, '<div class="table-scroll"><table>').replace(/<\/table>/g, '</table></div>');
  return { html, headings };
}

// Python's csv module ends lines with \r\n, so both endings are accepted.
export function parseCsv(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  if (text.includes('"')) throw new Error('Quoted CSV fields are not supported');
  const keys = header.split(',');
  return lines.map(line => Object.fromEntries(line.split(',').map((value, i) => [keys[i], value])));
}

const mean = values => values.reduce((a, b) => a + b, 0) / values.length;

// Mean of `field` for each [method, name] series at each grade, over every matching row (the seeds).
function byGrade(rows, series, field) {
  const xs = [...new Set(rows.map(r => Number(r.grade)))].sort((a, b) => a - b);
  return {
    x: xs,
    series: series.map(([method, name]) => ({
      name,
      values: xs.map(grade => {
        const matching = rows.filter(r => r.method === method && Number(r.grade) === grade);
        if (!matching.length) throw new Error(`No ${field} for ${method} at V${grade}`);
        return mean(matching.map(r => Number(r[field])));
      }),
    })),
  };
}

export async function chartData(root = repo) {
  const read = file => readFile(path.join(root, 'results', file), 'utf8').then(parseCsv);
  const samples = (await read('compare_models/samples_by_grade.csv'))
    .filter(r => r.model === 'climb_v2_d192' && r.split === 'test');
  const holdsSeries = [
    ['real', 'Real held-out climbs'], ['v2_scratch_unconstrained', 'Large model, no rules'],
    ['v2_scratch', 'Large model, with rules'], ['baseline_scratch', 'Baseline'],
  ];
  const grading = (await read('grade_climbs/by_grade.csv')).filter(r => Number(r.grade) <= 12);
  const gradingSeries = [
    ['model_mlp', 'null_rule_16, MLP head'], ['bayes', 'null_rule_16, Bayes'],
    ['bag_mlp', 'Bag of holds, MLP head'], ['untrained_mlp', 'Untrained model, MLP head'],
  ];
  return {
    'holds-by-grade': {
      title: 'Holds per climb by grade',
      subtitle: 'Mean holds in sampled and real climbs, test split, mean of three seeds',
      xLabel: grade => `V${grade}`, digits: 1, zero: false,
      ...byGrade(samples, holdsSeries, 'holds'),
    },
    // Hold-label probe figures, drawn by src/probes.mjs from probeData() below.
    ...Object.fromEntries(PROBE_FIGURES.map(view => [`probe-${view}`, { kind: 'probe', view }])),
    'grading-error': {
      title: 'Grade prediction error by true grade',
      subtitle: 'Test MSE in grades², mean of three seeds; V13 and above have under 20 climbs and are left out',
      xLabel: grade => `V${grade}`, digits: 2, zero: true,
      ...byGrade(grading, gradingSeries, 'mse'),
    },
  };
}

export const PROBE_FIGURES = ['scores', 'boards', 'confusion', 'projection', 'grip'];
const PROBE_REPRESENTATIONS = ['position', 'usage', 'hold_vector', 'context'];

// One record per hold for the probe figures: its outline on the board, labels, training uses,
// position in the hold-vector projection, and each probe's held-out prediction (seed 43).
// The outline is the Boardsesh artwork the climb tool draws, placed with the same scale.
export async function probeData(root = repo) {
  const results = path.join(root, 'results/hold_label_probes');
  const summary = JSON.parse(await readFile(path.join(results, 'summary.json'), 'utf8'));
  const rows = parseCsv(await readFile(path.join(results, 'holds.csv'), 'utf8'));
  const { outlines } = JSON.parse(await readFile(path.join(web, 'public/vendor/boardsesh/kilter-original-12x12.json'), 'utf8'));
  const r = BOARD_WIDTH / 144 * 4;
  const number = (row, key) => {
    const value = Number(row[key]);
    if (row[key] === undefined || row[key] === '' || !Number.isFinite(value)) throw new Error(`holds.csv: ${key} is not a number for hold ${row.hold}`);
    return value;
  };
  const holds = rows.map(row => {
    const shape = { cx: Number(row.x) * BOARD_WIDTH, cy: (1 - Number(row.y)) * BOARD_HEIGHT, r, outline: outlines[row.hold] };
    if (!shape.outline) throw new Error(`No outline for hold ${row.hold}`);
    return {
      id: Number(row.hold), d: outlinePath(shape).replace(/(\.\d)\d+/g, '$1'),
      cx: Math.round(shape.cx * 10) / 10, cy: Math.round(shape.cy * 10) / 10,
      type: row.type, grip: row.grip ? row.grip.split(';').map(Number) : [],
      uses: ['hand', 'foot', 'start', 'finish'].map(role => number(row, `uses_${role}`)),
      pc: [number(row, 'pc1'), number(row, 'pc2')],
      // Only holds with exactly one labeled grip direction have a grip prediction.
      pred: Object.fromEntries(PROBE_REPRESENTATIONS.map(name => [name, {
        type: row[`${name}_type`], p: number(row, `${name}_type_probability`),
        grip: row.grip && !row.grip.includes(';') ? number(row, `${name}_grip`) : null,
      }])),
    };
  });
  const plot = summary.by_seed[summary.plot_seed];
  const seeds = summary.seeds.map(String);
  return {
    types: summary.types, handTypes: summary.hand_types, seeds: summary.seeds, plotSeed: summary.plot_seed,
    repeats: summary.repeats, mean: summary.mean,
    confusion: Object.fromEntries(PROBE_REPRESENTATIONS.map(name => [name, plot[name].type_confusion])),
    handRecall: Object.fromEntries(summary.representations.map(name =>
      [name, seeds.map(seed => summary.by_seed[seed][name].hand_type_recall)])),
    constantDirection: plot.null.grip_constant_mean_direction,
    holds,
  };
}

const escapeJson = value => JSON.stringify(value).replace(/</g, '\\u003c');

export async function buildExperimentsPage(output, root = repo) {
  const markdown = await readFile(path.join(root, 'experiments/README.md'), 'utf8');
  // The README's own H1 is replaced by the page header.
  const { html, headings } = renderMarkdown(markdown.replace(/^# .*\n/, ''));
  const charts = await chartData(root);
  for (const [, name] of html.matchAll(/data-chart="([^"]+)"/g)) {
    if (!charts[name]) throw new Error(`experiments/README.md places an unknown chart: ${name}`);
  }
  const serialized = Object.fromEntries(Object.entries(charts).map(([name, { xLabel, ...chart }]) =>
    [name, chart.kind ? chart : { ...chart, xLabels: chart.x.map(xLabel) }]));
  const usesProbes = /data-chart="probe-/.test(html);
  const toc = headings.filter(h => h.depth === 2)
    .map(h => `<li><a href="#${h.id}">${h.html}</a></li>`).join('');
  const template = await readFile(path.join(web, 'experiments.html'), 'utf8');
  const page = template.replace('<!-- TOC -->', `<ol>${toc}</ol>`)
    .replace('<!-- EXPERIMENTS -->', html)
    .replace('<!-- CHART DATA -->', escapeJson(serialized))
    .replace('<!-- PROBE DATA -->', usesProbes ? escapeJson(await probeData(root)) : 'null')
    .replaceAll('{{REPOSITORY}}', REPOSITORY);
  await writeFile(output, page);
}
