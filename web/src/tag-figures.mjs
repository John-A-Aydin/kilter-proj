// Figures for the "Style tags from climb names and descriptions" experiment on experiments.html.
// experiments-page.mjs embeds results/tag_conditioning/summary.json, trimmed by tagData(), as
// #tag-data; experiments.mjs calls renderTagFigure for each <figure data-chart="tags-<view>">.
// The charts share the probe figures' helpers and styles, and every number is also in the
// README's tables or the figure's data table.
import { caption, hideTip, html, legendKey, percent, responsive, showTip, svg, tipRow } from './probes.mjs';

export const TAG_FIGURES = ['gain', 'steering', 'scatter'];
const label = tag => tag.replace(/_/g, ' ');
const signed = (value, digits = 2) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
// Below this ROC AUC the recognizer cannot find a tag in real climbs, so steering cannot be judged by it.
export const VISIBLE_AUC = 0.6;
// Tags with fewer test climbs than this give noisy likelihood numbers.
const FEW = 20;

function table(rows, head) {
  const details = html('details', { class: 'chart-table' });
  details.innerHTML = `<summary>Data table</summary><div class="table-scroll"><table><thead><tr>${head.map(h => `<th>${h}</th>`).join('')}</tr></thead><tbody>${
    rows.map(r => `<tr>${r.map((v, i) => i ? `<td>${v}</td>` : `<th scope="row">${v}</th>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  return details;
}

// Nats per test climb saved by the true tags against tags off, per tag, with each seed.
function gain(figure, data) {
  const rows = data.tags.filter(tag => data.likelihood[tag]).map(tag => ({ tag, ...data.likelihood[tag], gain: mean(data.likelihood[tag].seeds) }))
    .sort((a, b) => b.gain - a.gain);
  caption(figure, 'What each tag is worth to the model',
    `Nats per test climb saved by giving null_rule_16_tags each climb's true tags instead of none, on test climbs with that tag. Dots are the seeds. Tags with fewer than ${FEW} test climbs are drawn lighter.`);
  const plot = html('div', { class: 'probe-plot' });
  const legend = html('div', { class: 'probe-legend' });
  legend.append(legendKey('Saves nats: the tag helps', '', 'var(--probe-learned)'), legendKey('Costs nats', '', 'var(--probe-neg)'));
  figure.append(legend, plot, table(rows.map(r => [label(r.tag), r.rows, signed(r.gain), ...r.seeds.map(v => signed(v))]),
    ['Tag', 'Test climbs', 'Mean', ...data.seeds.map(s => `Seed ${s}`)]));
  responsive(plot, width => {
    plot.replaceChildren();
    const left = 104, right = width - 96, rowHeight = 21, top = 4, height = top + rowHeight * rows.length + 24;
    const values = rows.flatMap(r => r.seeds);
    const lo = Math.min(-1, Math.floor(Math.min(...values))), hi = Math.max(1, Math.ceil(Math.max(...values)));
    const x = v => left + (v - lo) / (hi - lo) * (right - left);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
      'aria-label': 'Likelihood gain from each tag. The data table below lists every value.' }, plot);
    for (let t = lo; t <= hi; t++) {
      svg('line', { class: t === 0 ? 'probe-axis' : 'probe-grid', x1: x(t), x2: x(t), y1: top, y2: height - 24 }, root);
      svg('text', { class: 'probe-tick', x: x(t), y: height - 8, 'text-anchor': 'middle' }, root).textContent = t;
    }
    rows.forEach((r, i) => {
      const cy = top + i * rowHeight + rowHeight / 2, few = r.rows < FEW;
      svg('text', { class: `probe-row-label${few ? ' dim' : ''}`, x: left - 10, y: cy + 4, 'text-anchor': 'end' }, root).textContent = label(r.tag);
      const x0 = x(0), x1 = x(r.gain);
      svg('rect', { x: Math.min(x0, x1), y: cy - 6, width: Math.max(1, Math.abs(x1 - x0)), height: 12, rx: 3, opacity: few ? 0.4 : 1,
        fill: r.gain >= 0 ? 'var(--probe-learned)' : 'var(--probe-neg)' }, root);
      for (const value of r.seeds) svg('circle', { cx: x(value), cy, r: 2.4, class: 'probe-seed' }, root);
      svg('text', { class: 'probe-value', x: x(Math.max(r.gain, 0, ...r.seeds)) + 6, y: cy + 4 }, root).textContent = `${signed(r.gain)} · n ${r.rows}`;
      const hit = svg('rect', { x: 0, y: cy - rowHeight / 2, width, height: rowHeight, fill: 'transparent' }, root);
      hit.addEventListener('pointermove', event => showTip(event, `<strong>${label(r.tag)}</strong>${tipRow('Test climbs with the tag', r.rows)}`
        + tipRow('Nats saved, mean', signed(r.gain)) + r.seeds.map((v, k) => tipRow(`Seed ${data.seeds[k]}`, signed(v))).join('')));
      hit.addEventListener('pointerleave', hideTip);
    });
  });
}

// Recognizer probability of each tag for generated climbs with the tag off and on.
function steering(figure, data) {
  const rows = data.tags.map(tag => ({ tag, ...data.steering[tag], auc: data.auc[tag] })).sort((a, b) => (b.on - b.off) - (a.on - a.off));
  caption(figure, 'How tag-like the generated climbs look',
    `A recognizer trained on real climbs scores ${data.samples} generated climbs per tag and seed with the tag off (hollow) and on (filled), at the same grades and angles. `
    + `The right column is its ROC AUC on real test climbs; below ${VISIBLE_AUC} it cannot see the tag, so those rows are drawn lighter.`);
  const plot = html('div', { class: 'probe-plot' });
  const legend = html('div', { class: 'probe-legend' });
  legend.append(legendKey('Tag off', 'hollow'), legendKey('Tag on', '', 'var(--probe-learned)'), legendKey('Recognizer cannot see the tag', '', 'var(--probe-control)'));
  figure.append(legend, plot, table(rows.map(r => [label(r.tag), percent(r.off), percent(r.on), signed(mean(r.shiftSeeds)), r.auc.toFixed(2)]),
    ['Tag', 'Probability, off', 'Probability, on', 'Logit shift', 'Recognizer AUC']));
  responsive(plot, width => {
    plot.replaceChildren();
    const left = 104, right = width - 48, rowHeight = 20, top = 20, height = top + rowHeight * rows.length + 24;
    const x = v => left + v * (right - left);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
      'aria-label': 'Recognizer probability with each tag off and on. The data table below lists every value.' }, plot);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      svg('line', { class: 'probe-grid', x1: x(t), x2: x(t), y1: top, y2: height - 24 }, root);
      svg('text', { class: 'probe-tick', x: x(t), y: height - 8, 'text-anchor': 'middle' }, root).textContent = percent(t);
    }
    svg('text', { class: 'probe-tick', x: width - 2, y: 11, 'text-anchor': 'end' }, root).textContent = 'AUC';
    rows.forEach((r, i) => {
      const cy = top + i * rowHeight + rowHeight / 2, dim = r.auc < VISIBLE_AUC;
      const color = dim ? 'var(--probe-control)' : 'var(--probe-learned)';
      svg('text', { class: `probe-row-label${dim ? ' dim' : ''}`, x: left - 10, y: cy + 4, 'text-anchor': 'end' }, root).textContent = label(r.tag);
      svg('line', { x1: x(r.off), x2: x(r.on), y1: cy, y2: cy, stroke: color, 'stroke-width': 2.5, 'stroke-linecap': 'round' }, root);
      svg('circle', { cx: x(r.off), cy, r: 4.5, fill: '#fdfdf8', stroke: dim ? 'var(--probe-control)' : 'var(--probe-ink-2)', 'stroke-width': 1.8 }, root);
      svg('circle', { cx: x(r.on), cy, r: 4.5, fill: color, stroke: '#fdfdf8', 'stroke-width': 1.5 }, root);
      svg('text', { class: 'probe-value', x: width - 2, y: cy + 4, 'text-anchor': 'end' }, root).textContent = r.auc.toFixed(2);
      const hit = svg('rect', { x: 0, y: cy - rowHeight / 2, width, height: rowHeight, fill: 'transparent' }, root);
      hit.addEventListener('pointermove', event => showTip(event, `<strong>${label(r.tag)}</strong>${tipRow('Probability, tag off', percent(r.off))}`
        + tipRow('Probability, tag on', percent(r.on)) + tipRow('Recognizer AUC, real climbs', r.auc.toFixed(2))
        + r.shiftSeeds.map((v, k) => tipRow(`Logit shift, seed ${data.seeds[k]}`, signed(v))).join('')
        + `<em>${dim ? 'The recognizer cannot find this tag in real climbs, so this says little.' : 'Higher with the tag on means the tag steers generation.'}</em>`));
      hit.addEventListener('pointerleave', hideTip);
    });
  });
}

// One point per tag and statistic: generated change against the real difference, both in standard deviations.
function scatter(figure, data) {
  const points = [];
  for (const tag of data.tags) {
    if (!data.realDiff[tag]) continue;
    data.stats.forEach((stat, k) => {
      const generated = data.steering[tag].statDiff[k], real = data.realDiff[tag][k];
      if (Number.isFinite(generated) && Number.isFinite(real)) points.push({ tag, stat, generated, real });
    });
  }
  caption(figure, `Generated changes against real differences, r = ${data.correlation.toFixed(2)}`,
    `Each point is one tag and one statistic: the change from turning the tag on (up) against how real climbs with the tag differ from those without it at the same grade and angle (across), in standard deviations. Points on the diagonal match.`);
  const plot = html('div', { class: 'probe-plot' });
  const notable = points.filter(p => Math.abs(p.real) > 0.8 || Math.abs(p.generated) > 0.8).sort((a, b) => Math.abs(b.real) - Math.abs(a.real));
  figure.append(plot, table(notable.map(p => [`${label(p.tag)} · ${label(p.stat)}`, signed(p.real), signed(p.generated)]),
    ['Tag and statistic (beyond 0.8 sd)', 'Real difference (sd)', 'Generated change (sd)']));
  responsive(plot, width => {
    plot.replaceChildren();
    const height = Math.min(440, Math.round(width * 0.72)), pad = 40;
    const limit = Math.ceil(Math.max(...points.map(p => Math.max(Math.abs(p.generated), Math.abs(p.real)))));
    const x = v => pad + (v + limit) / (2 * limit) * (width - pad - 12);
    const y = v => height - pad - (v + limit) / (2 * limit) * (height - pad - 14);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
      'aria-label': `Generated against real differences, correlation ${data.correlation.toFixed(2)}. The data table below lists the largest.` }, plot);
    for (let t = -limit; t <= limit; t++) {
      svg('line', { class: t === 0 ? 'probe-axis' : 'probe-grid', x1: x(t), x2: x(t), y1: 14, y2: height - pad }, root);
      svg('line', { class: t === 0 ? 'probe-axis' : 'probe-grid', x1: pad, x2: width - 12, y1: y(t), y2: y(t) }, root);
      svg('text', { class: 'probe-tick', x: x(t), y: height - pad + 15, 'text-anchor': 'middle' }, root).textContent = t;
      svg('text', { class: 'probe-tick', x: pad - 8, y: y(t) + 4, 'text-anchor': 'end' }, root).textContent = t;
    }
    svg('line', { x1: x(-limit), y1: y(-limit), x2: x(limit), y2: y(limit), stroke: 'var(--probe-muted)', 'stroke-width': 1, opacity: 0.6 }, root);
    svg('text', { class: 'probe-tick', x: width - 12, y: height - 6, 'text-anchor': 'end' }, root).textContent = 'real difference (sd) →';
    svg('text', { class: 'probe-tick', x: pad, y: 10 }, root).textContent = '↑ generated change (sd)';
    for (const p of points) {
      const big = Math.abs(p.real) > 0.8 || Math.abs(p.generated) > 0.8;
      const dot = svg('circle', { cx: x(p.real), cy: y(p.generated), r: big ? 5 : 3.5, fill: big ? 'var(--probe-learned)' : 'var(--probe-control)',
        stroke: '#fdfdf8', 'stroke-width': 1.5 }, root);
      dot.addEventListener('pointermove', event => showTip(event, `<strong>${label(p.tag)} · ${label(p.stat)}</strong>`
        + tipRow('Real difference', `${signed(p.real)} sd`) + tipRow('Generated change', `${signed(p.generated)} sd`)));
      dot.addEventListener('pointerleave', hideTip);
    }
    if (width >= 520) {
      // Label the largest effects, skipping any label that would collide with one already placed.
      const placed = [];
      for (const p of points.filter(p => Math.abs(p.real) > 1.3 || Math.abs(p.generated) > 1.3)
        .sort((a, b) => Math.abs(b.real) - Math.abs(a.real))) {
        const lx = x(p.real) + 7, ly = y(p.generated) + 4;
        if (placed.some(([px, py]) => Math.abs(px - lx) < 130 && Math.abs(py - ly) < 14)) continue;
        placed.push([lx, ly]);
        svg('text', { class: 'probe-value', x: lx, y: ly }, root)
          .textContent = `${label(p.tag)}: ${label(p.stat.replace('_share', '').replace('_cm', ''))}`;
      }
    }
  });
}

const VIEWS = { gain, steering, scatter };

export function renderTagFigure(figure, view, data) {
  if (!VIEWS[view]) throw new Error(`Unknown tag figure: ${view}`);
  figure.classList.add('probe-figure');
  figure.replaceChildren();
  VIEWS[view](figure, data);
}
