// Line charts for experiments.html. build.mjs embeds each chart's data; a figure with
// data-chart="name" is drawn from it, with a hover/keyboard readout and a data table.
const SVG = 'http://www.w3.org/2000/svg';
const MARGIN = { top: 14, right: 18, bottom: 30, left: 44 };

function element(name, attributes = {}, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent?.append(node);
  return node;
}

// Round tick steps (1, 2 or 5 × 10^n) covering [low, high].
export function niceTicks(low, high, count = 5) {
  const raw = (high - low) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(m => m * power).find(s => s >= raw);
  const ticks = [];
  for (let t = Math.floor(low / step) * step; t <= high + step * 1e-9; t += step) ticks.push(+t.toFixed(10));
  if (ticks.at(-1) < high) ticks.push(+(ticks.at(-1) + step).toFixed(10));
  return ticks;
}

function table(spec) {
  const details = document.createElement('details');
  details.className = 'chart-table';
  const rows = spec.series.map(s => `<tr><th scope="row">${s.name}</th>${s.values.map(v =>
    `<td>${v.toFixed(spec.digits)}</td>`).join('')}</tr>`).join('');
  details.innerHTML = `<summary>Data table</summary><div class="table-scroll"><table><thead><tr><th></th>${
    spec.xLabels.map(x => `<th scope="col">${x}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div>`;
  return details;
}

function lineChart(figure, spec) {
  figure.innerHTML = `<figcaption><strong>${spec.title}</strong><span>${spec.subtitle}</span></figcaption>
    <ul class="chart-legend">${spec.series.map((s, i) => `<li><i style="--color:var(--series-${i + 1})"></i>${s.name}</li>`).join('')}</ul>
    <div class="chart-plot"><div class="chart-tooltip" hidden></div></div>`;
  figure.append(table(spec));
  const plot = figure.querySelector('.chart-plot');
  const tooltip = figure.querySelector('.chart-tooltip');
  const values = spec.series.flatMap(s => s.values);
  const ticks = niceTicks(spec.zero ? 0 : Math.min(...values), Math.max(...values));
  let svg, active = -1, geometry;

  function draw() {
    const width = Math.max(260, plot.clientWidth);
    const height = width < 480 ? 220 : 260;
    const left = MARGIN.left, right = width - MARGIN.right, top = MARGIN.top, bottom = height - MARGIN.bottom;
    const x0 = spec.x[0], x1 = spec.x.at(-1);
    const x = value => left + (value - x0) / (x1 - x0) * (right - left);
    const y = value => bottom - (value - ticks[0]) / (ticks.at(-1) - ticks[0]) * (bottom - top);
    geometry = { x, y, top, bottom, width };
    svg?.remove();
    svg = element('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', tabindex: 0,
      'aria-label': `${spec.title}. Use the left and right arrow keys to read values; the data table below lists them all.` });
    plot.prepend(svg);
    for (const tick of ticks) {
      element('line', { x1: left, x2: right, y1: y(tick), y2: y(tick), class: tick === ticks[0] ? 'axis' : 'grid' }, svg);
      element('text', { x: left - 8, y: y(tick), class: 'tick y' }, svg).textContent = tick.toLocaleString();
    }
    spec.x.forEach((value, i) => {
      element('text', { x: x(value), y: bottom + 18, class: 'tick x' }, svg).textContent = spec.xLabels[i];
    });
    element('line', { class: 'crosshair', y1: top, y2: bottom, visibility: 'hidden' }, svg);
    spec.series.forEach((series, s) => {
      const color = `var(--series-${s + 1})`;
      element('path', { class: 'series-line', stroke: color,
        d: series.values.map((v, i) => `${i ? 'L' : 'M'}${x(spec.x[i]).toFixed(1)},${y(v).toFixed(1)}`).join('') }, svg);
      series.values.forEach((v, i) => element('circle', { class: 'marker', cx: x(spec.x[i]), cy: y(v), r: 4, fill: color,
        'data-index': i }, svg));
    });
    const hit = element('rect', { x: left - 12, y: top, width: right - left + 24, height: bottom - top, class: 'hit' }, svg);
    hit.addEventListener('pointermove', event => {
      const box = svg.getBoundingClientRect();
      const px = (event.clientX - box.left) * width / box.width;
      show(spec.x.reduce((best, v, i) => Math.abs(x(v) - px) < Math.abs(x(spec.x[best]) - px) ? i : best, 0));
    });
    hit.addEventListener('pointerleave', () => show(-1));
    svg.addEventListener('keydown', event => {
      const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
      if (!step) return;
      event.preventDefault();
      show(Math.min(spec.x.length - 1, Math.max(0, (active < 0 ? (step > 0 ? -1 : spec.x.length) : active) + step)));
    });
    svg.addEventListener('blur', () => show(-1));
    if (active >= 0) show(active);
  }

  function show(index) {
    active = index;
    const crosshair = svg.querySelector('.crosshair');
    for (const marker of svg.querySelectorAll('.marker')) marker.setAttribute('r', Number(marker.dataset.index) === index ? 5.5 : 4);
    if (index < 0) {
      crosshair.setAttribute('visibility', 'hidden');
      tooltip.hidden = true;
      return;
    }
    const cx = geometry.x(spec.x[index]);
    crosshair.setAttribute('x1', cx);
    crosshair.setAttribute('x2', cx);
    crosshair.setAttribute('visibility', 'visible');
    const order = spec.series.map((s, i) => [s, i]).sort((a, b) => b[0].values[index] - a[0].values[index]);
    tooltip.innerHTML = `<strong>${spec.xLabels[index]}</strong>${order.map(([s, i]) =>
      `<span><i style="--color:var(--series-${i + 1})"></i>${s.name}<b>${s.values[index].toFixed(spec.digits)}</b></span>`).join('')}`;
    tooltip.hidden = false;
    const scale = svg.getBoundingClientRect().width / geometry.width;
    const room = plot.clientWidth - tooltip.offsetWidth - 4;
    const leftOf = cx * scale - tooltip.offsetWidth - 14;
    tooltip.style.left = `${Math.max(4, Math.min(room, cx * scale + 14 > room ? leftOf : cx * scale + 14))}px`;
    tooltip.style.top = `${geometry.top * scale}px`;
  }

  draw();
  new ResizeObserver(() => { if (Math.abs(plot.clientWidth - geometry.width) > 1) draw(); }).observe(plot);
}

// Sequential blue ramp (light to dark = small to large), in seven steps chosen so the value
// printed in a cell keeps at least 4.5:1 contrast: dark ink on the first four, white on the rest.
export const MATRIX_STEPS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#5598e7', '#256abf', '#184f95', '#0d366b'];
const MATRIX_LIGHT_STEPS = 4;

// The ramp step for `value` within [low, high].
export function matrixStep(value, low, high) {
  const share = high > low ? (value - low) / (high - low) : 0;
  return Math.max(0, Math.min(MATRIX_STEPS.length - 1, Math.round(share * (MATRIX_STEPS.length - 1))));
}

// A heatmap table: one row per method, one column per (metric, subset). Each metric has its own
// color scale, shared by its subsets so they can be compared directly.
function matrixChart(figure, spec) {
  figure.classList.add('chart-matrix');
  figure.innerHTML = `<figcaption><strong>${spec.title}</strong><span>${spec.subtitle}</span></figcaption>`;
  const scales = Object.fromEntries(spec.metrics.map(metric => {
    const values = spec.rows.flatMap(row => spec.columns.filter(c => c.metric === metric.key).map(c => row.values[c.key].mean));
    return [metric.key, [Math.min(...values), Math.max(...values)]];
  }));
  const keys = document.createElement('div');
  keys.className = 'matrix-keys';
  for (const metric of spec.metrics) {
    const [low, high] = scales[metric.key];
    keys.insertAdjacentHTML('beforeend', `<span class="matrix-key"><b>${metric.label}</b>${low.toFixed(spec.digits)}<span class="matrix-ramp">${
      MATRIX_STEPS.map(color => `<i style="background:${color}"></i>`).join('')}</span>${high.toFixed(spec.digits)}</span>`);
  }
  figure.append(keys);
  const scroll = document.createElement('div');
  scroll.className = 'table-scroll';
  const head = `<tr><th scope="col" rowspan="2" class="matrix-corner">Method</th>${spec.metrics.map(metric =>
    `<th scope="colgroup" colspan="${spec.columns.filter(c => c.metric === metric.key).length}">${metric.label}</th>`).join('')}</tr>
    <tr>${spec.columns.map(column => `<th scope="col" class="matrix-subset">${column.subsetLabel}</th>`).join('')}</tr>`;
  const body = spec.rows.map(row => `<tr><th scope="row">${row.label}</th>${spec.columns.map(column => {
    const { mean } = row.values[column.key];
    const step = matrixStep(mean, ...scales[column.metric]);
    return `<td tabindex="0" data-row="${row.key}" data-column="${column.key}" style="background:${MATRIX_STEPS[step]};color:${
      step < MATRIX_LIGHT_STEPS ? 'var(--matrix-ink)' : '#ffffff'}">${mean.toFixed(spec.digits)}</td>`;
  }).join('')}</tr>`).join('');
  scroll.innerHTML = `<table class="matrix-table"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  const tooltip = document.createElement('div');
  tooltip.className = 'chart-tooltip';
  tooltip.hidden = true;
  const plot = document.createElement('div');
  plot.className = 'matrix-plot';
  plot.append(scroll, tooltip);
  figure.append(plot);

  const show = cell => {
    const row = spec.rows.find(r => r.key === cell.dataset.row);
    const column = spec.columns.find(c => c.key === cell.dataset.column);
    const { mean, std } = row.values[column.key];
    tooltip.innerHTML = `<strong>${row.label}</strong><span>${column.metricLabel}, ${column.subsetLabel.toLowerCase()}<b>${
      mean.toFixed(spec.digits + 1)} ± ${std.toFixed(spec.digits + 1)}</b></span><span>Test rows<b>${column.rows.toLocaleString()}</b></span>`;
    tooltip.hidden = false;
    const box = plot.getBoundingClientRect(), target = cell.getBoundingClientRect();
    const left = target.right - box.left + 8;
    tooltip.style.left = `${Math.max(4, left + tooltip.offsetWidth > box.width ? target.left - box.left - tooltip.offsetWidth - 8 : left)}px`;
    // Above the cell, so it never hides the rows being compared; below it only for the top rows.
    const above = target.top - box.top - tooltip.offsetHeight - 6;
    tooltip.style.top = `${above >= 0 ? above : target.bottom - box.top + 6}px`;
  };
  for (const cell of scroll.querySelectorAll('td')) {
    cell.addEventListener('pointerenter', () => show(cell));
    cell.addEventListener('focus', () => show(cell));
    cell.addEventListener('pointerleave', () => { tooltip.hidden = true; });
    cell.addEventListener('blur', () => { tooltip.hidden = true; });
  }
}

const source = globalThis.document?.getElementById('chart-data');
if (source) {
  const charts = JSON.parse(source.textContent);
  const probes = [], tags = [];
  for (const figure of document.querySelectorAll('figure[data-chart]')) {
    const spec = charts[figure.dataset.chart];
    if (spec?.kind === 'probe') probes.push([figure, spec.view]);
    else if (spec?.kind === 'tags') tags.push([figure, spec.view]);
    else if (spec?.kind === 'matrix') matrixChart(figure, spec);
    else if (spec) lineChart(figure, spec);
  }
  // The style-tag figures have their own data block and module.
  if (tags.length) {
    const data = JSON.parse(document.getElementById('tag-data').textContent);
    import('./tag-figures.mjs').then(({ renderTagFigure }) => {
      for (const [figure, view] of tags) renderTagFigure(figure, view, data);
      document.body.dataset.tagsReady = 'true';
    });
  }
  // The hold-label probe figures share one data block and their own module.
  if (probes.length) {
    const data = JSON.parse(document.getElementById('probe-data').textContent);
    import('./probes.mjs').then(({ renderProbeFigure }) => {
      for (const [figure, view] of probes) renderProbeFigure(figure, view, data);
      document.body.dataset.probesReady = 'true';
    });
  }
}
