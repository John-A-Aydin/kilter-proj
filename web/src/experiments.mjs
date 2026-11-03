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

const source = globalThis.document?.getElementById('chart-data');
if (source) {
  const charts = JSON.parse(source.textContent);
  for (const figure of document.querySelectorAll('figure[data-chart]')) {
    if (charts[figure.dataset.chart]) lineChart(figure, charts[figure.dataset.chart]);
  }
}
