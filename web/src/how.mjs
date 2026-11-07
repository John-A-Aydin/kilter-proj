// "How it works" page: draws the sequence figure and the charts from training.json (experiments/export_training_page.py).
const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);
const COLOR = { s1: 'var(--series-1)', s2: 'var(--series-2)', s3: 'var(--series-3)', s4: 'var(--series-4)',
  quiet: 'var(--series-quiet)', ink: 'var(--ink)' };
const MODEL_LABELS = {
  climb_v2_null_grade: ['Starting point', '8 epochs, 2.9M params'],
  null_ascent_weighted: ['+ ascent weighting', 'log₂(1 + ascents), capped at 6'],
  null_d256: ['+ bigger model', 'width 256, 8 layers, 6.6M params'],
  null_16_epochs: ['+ 16 epochs', ''],
  null_rule_loss: ['+ rule loss', 'illegal tokens masked in the loss'],
  null_rule_16_d256: ['+ rule loss, 16 epochs, bigger', '6.6M params'],
  null_rule_16: ['+ rule loss, 16 epochs', 'chosen, served on this site'],
};
const fixed = digits => value => value.toFixed(digits);
const percent = value => `${(value * 100).toFixed(1)}%`;

function node(tag, attrs = {}, parent) {
  const element = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value !== undefined) element.setAttribute(key, value);
  if (parent) parent.append(element);
  return element;
}

function label(parent, x, y, content, attrs = {}) {
  const element = node('text', { x, y, ...attrs }, parent);
  element.textContent = content;
  return element;
}

function html(tag, attrs = {}, content) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (content !== undefined) element.textContent = content;
  return element;
}

function scale([d0, d1], [r0, r1]) {
  return value => r0 + (value - d0) / (d1 - d0) * (r1 - r0);
}

function niceTicks(min, max, count = 5) {
  const raw = (max - min) / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * power).find(s => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(min / step) * step; t <= max + step * 1e-6; t += step) ticks.push(+t.toFixed(10));
  return ticks;
}

// One tooltip for the page. Content is built with textContent only.
const tooltip = {
  show({ head, rows }, x, y) {
    const box = $('chart-tooltip');
    box.replaceChildren();
    if (head) box.append(html('div', { class: 'tt-head' }, head));
    for (const row of rows) {
      const line = html('div', { class: 'tt-row' });
      const key = html('i');
      key.style.setProperty('--c', row.color);
      line.append(key, html('strong', {}, row.value), html('span', {}, row.label));
      box.append(line);
    }
    box.hidden = false;
    const { width, height } = box.getBoundingClientRect();
    const left = x + 16 + width > innerWidth - 8 ? x - width - 16 : x + 16;
    box.style.left = `${Math.max(8, left)}px`;
    box.style.top = `${Math.min(innerHeight - height - 8, Math.max(8, y - height / 2))}px`;
  },
  hide() { $('chart-tooltip').hidden = true; },
};

function legend(container, items) {
  container.replaceChildren();
  for (const item of items) {
    const entry = html('span');
    const key = html('i', { class: `key-${item.kind || 'line'}` });
    key.style.setProperty('--c', item.color);
    entry.append(key, document.createTextNode(item.name));
    container.append(entry);
  }
}

function table(container, headers, rows, highlight = () => false) {
  const wrap = html('div', { class: 'table-scroll' });
  const element = html('table');
  const head = html('tr');
  for (const header of headers) head.append(html('th', { scope: 'col' }, header));
  element.append(html('thead'), html('tbody'));
  element.tHead.append(head);
  for (const row of rows) {
    const line = html('tr');
    if (highlight(row)) line.className = 'highlight';
    for (const cell of row) line.append(html('td', {}, cell));
    element.tBodies[0].append(line);
  }
  wrap.append(element);
  container.replaceChildren(wrap);
}

// Keeps direct labels at least `gap` pixels apart, preserving their order.
function spread(items, gap, min, max) {
  items.sort((a, b) => a.y - b.y);
  for (let i = 1; i < items.length; i++) items[i].y = Math.max(items[i].y, items[i - 1].y + gap);
  const overflow = items.length ? items[items.length - 1].y - max : 0;
  if (overflow > 0) for (const item of items) item.y -= overflow;
  for (let i = items.length - 2; i >= 0; i--) items[i].y = Math.min(items[i].y, items[i + 1].y - gap);
  for (const item of items) item.y = Math.max(min, item.y);
  return items;
}

// Responsive charts redraw at the container's width.
function responsive(container, draw) {
  let width = 0;
  const render = () => {
    const next = Math.round(container.clientWidth);
    if (next && next !== width) { width = next; container.replaceChildren(); draw(width); }
  };
  new ResizeObserver(render).observe(container);
  render();
}

function lineChart(container, options) {
  const { series, xDomain, yDomain, xTicks, yTicks, xFormat = String, yFormat = String, height = 280,
    xLabel, yLabel, area = false, endLabels = true, headFormat = xFormat, valueFormat = yFormat,
    annotations = [], ariaLabel } = options;
  responsive(container, width => {
    const narrow = width < 520;
    const showEnds = endLabels && !narrow;
    const margin = { top: yLabel ? 28 : 14, right: showEnds ? 150 : 14, bottom: xLabel ? 42 : 28, left: 46 };
    const svg = node('svg', { viewBox: `0 0 ${width} ${height}`, width, height, tabindex: 0, role: 'img',
      'aria-label': ariaLabel }, container);
    const x = scale(xDomain, [margin.left, width - margin.right]);
    const y = scale(yDomain, [height - margin.bottom, margin.top]);
    for (const tick of yTicks) {
      node('line', { class: 'c-grid', x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick) }, svg);
      label(svg, margin.left - 8, y(tick) + 4, yFormat(tick), { class: 'c-tick', 'text-anchor': 'end' });
    }
    node('line', { class: 'c-axis', x1: margin.left, x2: width - margin.right, y1: height - margin.bottom,
      y2: height - margin.bottom }, svg);
    for (const tick of xTicks) label(svg, x(tick), height - margin.bottom + 17, xFormat(tick), { class: 'c-tick', 'text-anchor': 'middle' });
    if (xLabel) label(svg, (margin.left + width - margin.right) / 2, height - 6, xLabel, { class: 'c-axis-label', 'text-anchor': 'middle' });
    if (yLabel) label(svg, margin.left - 40, 11, yLabel, { class: 'c-axis-label' });
    for (const note of annotations) {
      node('line', { class: 'c-annot-line', x1: x(note.x), x2: x(note.x), y1: margin.top, y2: height - margin.bottom }, svg);
      const right = x(note.x) > (margin.left + width - margin.right) / 2;
      label(svg, x(note.x) + (right ? -6 : 6), margin.top + 10 + (note.dy || 0), note.label,
        { class: 'c-annot', 'text-anchor': right ? 'end' : 'start' });
    }
    for (const line of series) {
      const d = line.points.map(([px, py], i) => `${i ? 'L' : 'M'}${x(px).toFixed(1)} ${y(py).toFixed(1)}`).join('');
      if (area) {
        const [first, last] = [line.points[0], line.points[line.points.length - 1]];
        node('path', { class: 'c-area', fill: line.color,
          d: `${d}L${x(last[0])} ${y(yDomain[0])}L${x(first[0])} ${y(yDomain[0])}Z` }, svg);
      }
      node('path', { class: 'c-line', stroke: line.color, d }, svg);
      if (line.markers) for (const [px, py] of line.points) node('circle', { class: 'c-dot', cx: x(px), cy: y(py), r: 4, fill: line.color }, svg);
      if (line.best) node('circle', { class: 'c-dot', cx: x(line.best[0]), cy: y(line.best[1]), r: 5, fill: line.color }, svg);
    }
    if (showEnds) {
      const ends = spread(series.map(line => {
        const [px, py] = line.points[line.points.length - 1];
        return { line, x: x(px), y: y(py) + 4 };
      }), 14, margin.top + 4, height - margin.bottom);
      for (const end of ends) label(svg, end.x + 8, end.y, end.line.name, { class: 'c-label' });
    }
    // Crosshair: snaps to the nearest x with data, and lists every series there.
    const xs = [...new Set(series.flatMap(line => line.points.map(p => p[0])))].sort((a, b) => a - b);
    const cross = node('line', { class: 'c-cross', y1: margin.top, y2: height - margin.bottom, visibility: 'hidden' }, svg);
    const dots = series.map(line => node('circle', { class: 'c-dot', r: 5, fill: line.color, visibility: 'hidden' }, svg));
    let index = -1;
    const focus = (i, clientX, clientY) => {
      index = Math.max(0, Math.min(xs.length - 1, i));
      const at = xs[index];
      cross.setAttribute('x1', x(at)); cross.setAttribute('x2', x(at)); cross.setAttribute('visibility', 'visible');
      const rows = [];
      series.forEach((line, n) => {
        const point = line.points.find(p => p[0] === at);
        dots[n].setAttribute('visibility', point ? 'visible' : 'hidden');
        if (!point) return;
        dots[n].setAttribute('cx', x(at)); dots[n].setAttribute('cy', y(point[1]));
        rows.push({ color: line.color, value: valueFormat(point[1]), label: line.name, y: point[1] });
      });
      if (options.sortTooltip) rows.sort((a, b) => a.y - b.y);
      if (clientX === undefined) {
        const box = svg.getBoundingClientRect();
        clientX = box.left + x(at); clientY = box.top + margin.top + 40;
      }
      tooltip.show({ head: headFormat(at), rows }, clientX, clientY);
    };
    const clear = () => {
      cross.setAttribute('visibility', 'hidden');
      for (const dot of dots) dot.setAttribute('visibility', 'hidden');
      tooltip.hide();
    };
    const hit = node('rect', { class: 'c-hit', x: margin.left, y: margin.top, width: width - margin.left - margin.right + 8,
      height: height - margin.top - margin.bottom }, svg);
    hit.addEventListener('pointermove', event => {
      const box = svg.getBoundingClientRect();
      const px = (event.clientX - box.left) * width / box.width;
      let best = 0;
      xs.forEach((value, i) => { if (Math.abs(x(value) - px) < Math.abs(x(xs[best]) - px)) best = i; });
      focus(best, event.clientX, event.clientY);
    });
    hit.addEventListener('pointerleave', clear);
    svg.addEventListener('keydown', event => {
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault();
        focus(index < 0 ? 0 : index + (event.key === 'ArrowRight' ? 1 : -1));
      } else if (event.key === 'Escape') clear();
    });
    svg.addEventListener('focus', () => focus(index < 0 ? xs.length - 1 : index));
    svg.addEventListener('blur', clear);
  });
}

// Horizontal rows: bars from zero, or dots with error bars. Labels sit beside rows, or above them when narrow.
function rowChart(container, options) {
  const { rows, domain, ticks, format, valueFormat = format, axisLabel, kind = 'bar', reference, ariaLabel } = options;
  responsive(container, width => {
    const narrow = width < 600;
    const labelWidth = narrow ? 0 : Math.min(250, width * 0.36);
    const rowHeight = narrow ? 54 : 42;
    const margin = { top: reference ? 22 : 6, right: 58, bottom: axisLabel ? 44 : 28, left: labelWidth + (narrow ? 4 : 16) };
    const height = margin.top + rows.length * rowHeight + margin.bottom;
    const svg = node('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': ariaLabel }, container);
    const x = scale(domain, [margin.left, width - margin.right]);
    const bottom = height - margin.bottom;
    for (const tick of ticks) {
      node('line', { class: 'c-grid', x1: x(tick), x2: x(tick), y1: margin.top, y2: bottom }, svg);
      label(svg, x(tick), bottom + 17, format(tick), { class: 'c-tick', 'text-anchor': 'middle' });
    }
    if (kind === 'bar') node('line', { class: 'c-axis', x1: x(domain[0]), x2: x(domain[0]), y1: margin.top, y2: bottom }, svg);
    if (axisLabel) label(svg, (margin.left + width - margin.right) / 2, height - 6, axisLabel, { class: 'c-axis-label', 'text-anchor': 'middle' });
    if (reference) {
      node('line', { class: 'c-annot-line', x1: x(reference.value), x2: x(reference.value), y1: margin.top, y2: bottom }, svg);
      label(svg, x(reference.value), 12, reference.label, { class: 'c-annot', 'text-anchor': 'middle' });
    }
    rows.forEach((row, i) => {
      const top = margin.top + i * rowHeight;
      const center = top + (narrow ? 38 : rowHeight / 2);
      const group = node('g', { class: 'c-row', tabindex: 0, role: 'listitem',
        'aria-label': `${row.label}: ${row.marks.map(m => `${m.name ? m.name + ' ' : ''}${valueFormat(m.value)}`).join(', ')}` }, svg);
      node('rect', { class: 'c-rowbg', x: 0, y: top + 1, width, height: rowHeight - 2, rx: 6 }, group);
      if (narrow) {
        label(group, 4, top + 18, row.label, { class: `c-label${row.strong ? ' strong' : ''}` });
        if (row.sub) label(group, width - 4, top + 18, row.sub, { class: 'c-tick', 'text-anchor': 'end' });
      } else {
        label(group, labelWidth, center + (row.sub ? -2 : 4), row.label, { class: `c-label${row.strong ? ' strong' : ''}`, 'text-anchor': 'end' });
        if (row.sub) label(group, labelWidth, center + 12, row.sub, { class: 'c-tick', 'text-anchor': 'end' });
      }
      for (const mark of row.marks) {
        if (kind === 'bar') {
          const x0 = x(domain[0]), x1 = Math.max(x0 + 1, x(mark.value)), h = 14, r = Math.min(4, x1 - x0);
          node('path', { class: 'c-bar', fill: mark.color,
            d: `M${x0} ${center - h / 2}H${x1 - r}Q${x1} ${center - h / 2} ${x1} ${center - h / 2 + r}V${center + h / 2 - r}Q${x1} ${center + h / 2} ${x1 - r} ${center + h / 2}H${x0}Z` }, group);
        } else {
          if (mark.err) node('line', { class: 'c-err', stroke: mark.color, x1: x(mark.value - mark.err), x2: x(mark.value + mark.err),
            y1: center + (mark.offset || 0), y2: center + (mark.offset || 0) }, group);
          node('circle', { class: 'c-dot', cx: x(mark.value), cy: center + (mark.offset || 0), r: 5,
            style: mark.ring ? `fill:var(--card);stroke:${mark.color};stroke-width:2` : `fill:${mark.color}` }, group);
        }
      }
      const lead = row.marks[0];
      const end = kind === 'bar' ? x(lead.value) + 6 : width - margin.right + 10;
      label(group, end, center + 4, valueFormat(lead.value), { class: 'c-value' });
      const show = (clientX, clientY) => tooltip.show({ head: row.label, rows: row.marks.map(mark => ({
        color: mark.color, value: `${valueFormat(mark.value)}${mark.err ? ` ± ${valueFormat(mark.err)}` : ''}`, label: mark.name || '' })) },
      clientX, clientY);
      group.addEventListener('pointermove', event => show(event.clientX, event.clientY));
      group.addEventListener('pointerleave', () => tooltip.hide());
      group.addEventListener('focus', () => {
        const box = group.getBoundingClientRect();
        show(box.left + box.width * 0.6, box.top + box.height / 2);
      });
      group.addEventListener('blur', () => tooltip.hide());
    });
    svg.setAttribute('role', 'list');
  });
}

function sequenceFigure() {
  const group = $('sequence-tokens');
  const svg = group.ownerSVGElement;
  const tokens = [
    { text: 'BOS', kind: 'special', pos: 0, seg: 0 },
    { text: '1176', sub: 'start', role: 'start', kind: 'given', pos: 1, seg: 0 },
    { text: '1192', sub: 'start', role: 'start', kind: 'given', pos: 2, seg: 0 },
    { text: '1091', sub: 'foot', role: 'foot', kind: 'given', pos: 3, seg: 0 },
    { text: 'SEP', kind: 'special', pos: 0, seg: 1 },
    { text: 'hand', role: 'hand', kind: 'rest', pos: 1, seg: 1, scored: true },
    { text: 'hand', role: 'hand', kind: 'rest', pos: 2, seg: 1, scored: true },
    { text: 'finish', role: 'finish', kind: 'rest', pos: 3, seg: 1, scored: true },
    { text: 'foot', role: 'foot', kind: 'rest', pos: 4, seg: 1, scored: true },
    { text: 'EOS', kind: 'special', pos: 5, seg: 1, scored: true },
  ];
  const x0 = 86, w = 66, gap = 8;
  tokens.forEach((token, i) => {
    const x = x0 + i * (w + gap);
    label(group, x + w / 2, 30, token.pos, { class: 'tok-pos', 'text-anchor': 'middle' });
    label(group, x + w / 2, 58, token.seg ? 'rest' : 'given', { class: 'tok-seg', 'text-anchor': 'middle' });
    const chip = node('g', { class: `tok ${token.kind}` }, group);
    node('rect', { x, y: 74, width: w, height: 54, rx: 8 }, chip);
    if (token.role) node('rect', { x: x + 6, y: 84, width: 4, height: 34, rx: 2, style: `fill:var(--role-${token.role});stroke:none` }, chip);
    const textX = token.role ? x + 16 : x + w / 2;
    const anchor = token.role ? 'start' : 'middle';
    label(chip, textX, token.sub ? 98 : 106, token.text, { 'text-anchor': anchor });
    if (token.sub) label(chip, textX, 114, token.sub, { class: 'tok-sub', 'text-anchor': anchor });
    if (token.scored) node('circle', { class: 'tok-loss', cx: x + w / 2, cy: 140, r: 3 }, group);
  });
  const brace = (from, to, text) => {
    const a = x0 + from * (w + gap), b = x0 + to * (w + gap) + w;
    node('path', { d: `M${a} 150v8H${b}v-8`, class: 'd-brace-path' }, group);
    label(group, (a + b) / 2, 178, text, { class: 'd-brace-text', 'text-anchor': 'middle' });
  };
  brace(1, 3, 'given: context only, no loss');
  brace(5, 9, 'rest + EOS: predicted and scored (●)');
  svg.querySelector('.d-brace')?.remove();
}

function mixture(train) {
  const parts = [
    { name: 'Nothing given', share: train.p_scratch, color: COLOR.s1, ink: '#fff',
      text: 'Generate the whole climb: start, hands, finish, feet.' },
    { name: 'Starts + some feet', share: train.p_starts, color: COLOR.s2, ink: '#fff',
      text: 'Every start hold, plus each foot with a probability drawn per climb from 0 to 1.' },
    { name: 'Random subset', share: train.p_random, color: COLOR.s3, ink: '#14281f',
      text: 'Each hold in any role, with a probability drawn per climb from 0 to 0.8.' },
  ];
  responsive($('mixture-chart'), width => {
    const height = 34;
    const svg = node('svg', { viewBox: `0 0 ${width} ${height}`, width, height }, $('mixture-chart'));
    let at = 0;
    parts.forEach((part, i) => {
      const x = at * width + (i ? 1 : 0), w = part.share * width - (i ? 1 : 0) - (i < parts.length - 1 ? 1 : 0);
      at += part.share;
      const group = node('g', { class: 'c-row', tabindex: 0, 'aria-label': `${part.name}: ${percent(part.share)}` }, svg);
      const r = 6;
      const left = i === 0 ? r : 0, right = i === parts.length - 1 ? r : 0;
      node('path', { class: 'c-bar', fill: part.color, d: `M${x + left} 2H${x + w - right}${right ? `Q${x + w} 2 ${x + w} ${2 + right}V${32 - right}Q${x + w} 32 ${x + w - right} 32` : `V32`}H${x + left}${left ? `Q${x} 32 ${x} ${32 - left}V${2 + left}Q${x} 2 ${x + left} 2` : `V2`}Z` }, group);
      label(group, x + 12, 22, `${Math.round(part.share * 100)}% · ${part.name}`, { class: 'c-label', fill: part.ink, style: `fill:${part.ink}` });
      const show = (cx, cy) => tooltip.show({ head: 'Given set', rows: [{ color: part.color, value: percent(part.share), label: part.name }] }, cx, cy);
      group.addEventListener('pointermove', event => show(event.clientX, event.clientY));
      group.addEventListener('pointerleave', () => tooltip.hide());
      group.addEventListener('focus', () => { const b = group.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
      group.addEventListener('blur', () => tooltip.hide());
    });
    // Hide inline labels that don't fit their segment; the cards below carry them.
    for (const text of svg.querySelectorAll('text')) {
      const segment = text.previousElementSibling.getBBox();
      if (text.getComputedTextLength() + 20 > segment.width) text.textContent = text.textContent.split(' · ')[0];
    }
  });
  const cards = $('mixture-cards');
  for (const part of parts) {
    const card = html('div', { class: 'mixture-card' });
    const head = html('div', { class: 'mix-head' });
    const key = html('i');
    key.style.setProperty('--c', part.color);
    head.append(key, document.createTextNode(part.name), html('strong', {}, `${Math.round(part.share * 100)}%`));
    card.append(head, html('p', {}, part.text));
    cards.append(card);
  }
}

function protocol(schedule, servedBest) {
  const t = schedule.train;
  const items = [
    ['Optimizer', `AdamW, β = (${t.beta1}, ${t.beta2}), weight decay ${t.weight_decay}`],
    ['Learning rate', `${t.learning_rate} peak, ${t.warmup_fraction * 100}% warmup, cosine to ${t.min_lr_ratio * 100}%`],
    ['Batch', `${t.batch_size} climbs · ${schedule.steps_per_epoch.toLocaleString()} steps per epoch`],
    ['Epochs', `${t.epochs}, stopping after ${t.patience} without improvement`],
    ['Regularization', `dropout ${schedule.model.dropout}, gradient clip ${t.grad_clip}`],
    ['Loss', `cross-entropy on the rest + EOS${t.rule_loss ? ', illegal tokens masked' : ''}`],
    ['Given sets', `${t.p_scratch * 100}% / ${t.p_starts * 100}% / ${t.p_random * 100}%, redrawn every epoch`],
    ['No-grade rows', `${t.p_null_grade * 100}%, redrawn every batch`],
    ['Row weights', t.ascent_weighting === 'none' ? 'equal' : t.ascent_weighting],
    ['Checkpoint', `lowest validation nats per climb · served: seed 43, epoch ${servedBest}`],
  ];
  const list = $('protocol-list');
  for (const [term, value] of items) list.append(html('dt', {}, term), html('dd', {}, value));
}

function lrChart(schedule, servedBest) {
  const per = schedule.steps_per_epoch;
  const points = schedule.points.map(([step, lr]) => [step / per, lr * 1000]);
  lineChart($('lr-chart'), {
    series: [{ name: 'Learning rate', color: COLOR.s1, points }], area: true, endLabels: false, height: 220,
    xDomain: [0, schedule.epochs], yDomain: [0, 1.05], xTicks: niceTicks(0, schedule.epochs, 4), yTicks: [0, 0.25, 0.5, 0.75, 1],
    xLabel: 'epoch', yLabel: '× 10⁻³', yFormat: v => v.toFixed(2).replace(/0$/, ''), valueFormat: v => `${(v / 1000).toExponential(2)}`,
    headFormat: v => `Epoch ${v.toFixed(2)} · step ${Math.round(v * per).toLocaleString()}`,
    annotations: [{ x: schedule.warmup_steps / per, label: `warmup ends, step ${schedule.warmup_steps}`, dy: 100 },
      { x: servedBest, label: `served checkpoint: epoch ${servedBest}`, dy: 0 }],
    ariaLabel: 'Learning rate rises linearly to 0.001 over the first 3% of steps, then decays along a cosine to 0.00005.',
  });
  table($('lr-table'), ['Epoch', 'Step', 'Learning rate'],
    schedule.points.filter((_, i) => i % 20 === 0 || i === schedule.points.length - 1)
      .map(([step, lr]) => [(step / per).toFixed(2), step.toLocaleString(), lr.toExponential(2)]));
}

function curvesChart(curves) {
  const order = [['null_rule_16', COLOR.s1], ['null_rule_16_d256', COLOR.s2], ['null_rule_loss', COLOR.s3], ['null_16_epochs', COLOR.s4]];
  const series = order.map(([model, color]) => {
    const curve = curves[model];
    const points = curve.epochs.map((epoch, i) => [epoch, curve.selection_nll_rules[i]]);
    const best = points.reduce((a, b) => (b[1] < a[1] ? b : a));
    return { name: MODEL_LABELS[model][0].replace(/^\+ /, ''), model, color, points, best };
  });
  legend($('curves-legend'), series.map(s => ({ name: s.name, color: s.color })));
  const values = series.flatMap(s => s.points.map(p => p[1]));
  const lo = Math.floor(Math.min(...values) * 2) / 2 - 0.25, hi = Math.ceil(Math.max(...values));
  lineChart($('curves-chart'), {
    series, xDomain: [1, 16], yDomain: [lo, hi], xTicks: [1, 4, 8, 12, 16], yTicks: niceTicks(lo, hi, 5),
    xLabel: 'epoch', yLabel: 'nats / climb', yFormat: fixed(1), valueFormat: fixed(2), headFormat: v => `Epoch ${v}`,
    sortTooltip: true, height: 300, endLabels: false,
    ariaLabel: 'Validation nats per climb by epoch for four runs. Rule loss with 16 epochs reaches the lowest value.',
  });
  const epochs = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
  table($('curves-table'), ['Epoch', ...series.map(s => s.name)],
    epochs.map(epoch => [epoch, ...series.map(s => s.points.find(p => p[0] === epoch)?.[1].toFixed(2) ?? '–')]));
  const chosen = curves.null_rule_16;
  return { min: Math.min(...chosen.best_epochs), max: Math.max(...chosen.best_epochs) };
}

function ablationChart(rows) {
  const start = rows.find(r => r.model === 'climb_v2_null_grade');
  const sorted = [start, ...rows.filter(r => r !== start).sort((a, b) => b.validation.mean - a.validation.mean)];
  legend($('ablation-legend'), [{ name: 'Validation (used to choose)', color: COLOR.s1, kind: 'dot' },
    { name: 'Test', color: COLOR.s2, kind: 'ring' }]);
  const values = sorted.flatMap(r => [r.validation.mean, r.test.mean]);
  const lo = Math.floor(Math.min(...values) * 4) / 4 - 0.1, hi = Math.ceil(Math.max(...values) * 4) / 4 + 0.05;
  rowChart($('ablation-chart'), {
    kind: 'dot', domain: [lo, hi], ticks: niceTicks(lo, hi, 4), format: fixed(2), valueFormat: fixed(2),
    axisLabel: 'mean nats per climb, rules applied (lower is better)', reference: { value: start.validation.mean, label: 'starting point' },
    rows: sorted.map(r => ({
      label: MODEL_LABELS[r.model][0], sub: MODEL_LABELS[r.model][1], strong: r.model === 'null_rule_16',
      marks: [{ name: 'validation', value: r.validation.mean, err: r.validation.std, color: COLOR.s1, offset: -4 },
        { name: 'test', value: r.test.mean, err: r.test.std, color: COLOR.s2, ring: true, offset: 4 }],
    })),
    ariaLabel: 'Validation and test nats per climb for each single change from the starting model.',
  });
  table($('ablation-table'), ['Model', 'Validation', 'Test', 'Test scratch', 'Test starts', 'Test starts + feet', 'Valid samples (rules on)'],
    sorted.map(r => [r.model, `${r.validation.mean.toFixed(2)} ± ${r.validation.std.toFixed(2)}`,
      `${r.test.mean.toFixed(2)} ± ${r.test.std.toFixed(2)}`, r.scratch.mean.toFixed(2), r.starts.mean.toFixed(2),
      r.starts_feet.mean.toFixed(2), percent(r.valid.mean)]), row => row[0] === 'null_rule_16');
}

function nullTable(rows) {
  table($('null-table'), ['Grade during training / at test', 'Scratch', 'Starts', 'Starts + feet'],
    rows.map(r => [r.label, r.scratch.mean.toFixed(2), r.starts.mean.toFixed(2), r.starts_feet.mean.toFixed(2)]));
  $('null-table').querySelector('table').classList.add('data-table');
}

function validityChart(rows) {
  const color = r => (r.label.startsWith('Real') ? COLOR.ink : r.label.startsWith('Baseline') ? COLOR.s2 : COLOR.s1);
  legend($('validity-legend'), [{ name: 'v2', color: COLOR.s1, kind: 'rect' }, { name: 'Baseline', color: COLOR.s2, kind: 'rect' },
    { name: 'Real climbs', color: COLOR.ink, kind: 'rect' }]);
  rowChart($('validity-chart'), {
    domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], format: v => `${v * 100}%`, valueFormat: percent,
    rows: rows.map(r => ({ label: r.label.replace(' (large, no rule loss)', ''), sub: r.label.includes('no rules') ? 'large model, trained without rule loss' : undefined,
      marks: [{ name: `valid · ${percent(r.starts_1_2.mean)} have 1–2 starts`, value: r.valid.mean, color: color(r) }] })),
    ariaLabel: 'Share of structurally valid samples for real climbs, v2 and the baseline.',
  });
  table($('validity-table'), ['Sampler', 'Valid', '1–2 starts'],
    rows.map(r => [r.label, `${percent(r.valid.mean)} ± ${percent(r.valid.std)}`, percent(r.starts_1_2.mean)]));
}

function gradeChart(samples) {
  const defs = [['real', 'Real climbs', COLOR.ink], ['v2_scratch', 'v2 from nothing', COLOR.s1],
    ['v2_given_starts_feet', 'v2 completing starts + feet', COLOR.s2], ['baseline_scratch', 'Baseline from nothing', COLOR.s3]];
  const series = defs.map(([key, name, color]) => ({ name, color, markers: true,
    points: Object.entries(samples[key]).map(([g, v]) => [Number(g), v]).sort((a, b) => a[0] - b[0]) }));
  legend($('grade-legend'), series.map(s => ({ name: s.name, color: s.color })));
  const values = series.flatMap(s => s.points.map(p => p[1]));
  const lo = Math.floor(Math.min(...values)) - 0.5, hi = Math.ceil(Math.max(...values)) + 0.5;
  lineChart($('grade-chart'), {
    series, xDomain: [1.5, 11.5], yDomain: [lo, hi], xTicks: [2, 5, 8, 11], yTicks: niceTicks(lo, hi, 4),
    xFormat: v => `V${v}`, yFormat: fixed(0), valueFormat: fixed(1), yLabel: 'holds', endLabels: false, sortTooltip: true,
    ariaLabel: 'Mean holds per climb at V2, V5, V8 and V11 for real climbs and three samplers.',
  });
  table($('grade-table'), ['Grade', ...series.map(s => s.name)],
    [2, 5, 8, 11].map(g => [`V${g}`, ...series.map(s => s.points.find(p => p[0] === g)?.[1].toFixed(2) ?? '–')]));
}

function gradingChart(grading) {
  const group = method => (method.startsWith('model') || method === 'bayes' ? 'trained' : method.startsWith('untrained') ? 'untrained' : 'simple');
  const colors = { trained: COLOR.s1, untrained: COLOR.s3, simple: COLOR.quiet };
  legend($('grading-legend'), [{ name: 'Trained generator', color: colors.trained, kind: 'rect' },
    { name: 'Same network, untrained', color: colors.untrained, kind: 'rect' }, { name: 'No transformer', color: colors.simple, kind: 'rect' }]);
  const rows = [...grading.rows].sort((a, b) => b.mse - a.mse);
  const hi = Math.ceil(Math.max(...rows.map(r => r.mse)));
  rowChart($('grading-chart'), {
    domain: [0, hi], ticks: niceTicks(0, hi, 5), format: fixed(0), valueFormat: fixed(2), axisLabel: 'test MSE (grades²)',
    rows: rows.map(r => ({ label: r.label, strong: group(r.method) === 'trained',
      marks: [{ name: `MSE · ${percent(r.within_one)} within 1 grade`, value: r.mse, color: colors[group(r.method)] }] })),
    ariaLabel: 'Test mean squared error of grade predictions for each method.',
  });
  table($('grading-table'), ['Method', 'MSE', 'Within 1 grade'], rows.map(r => [r.label, r.mse.toFixed(2), percent(r.within_one)]));
}

sequenceFigure();
fetch('./training.json').then(response => {
  if (!response.ok) throw new Error(`training.json: ${response.status}`);
  return response.json();
}).then(data => {
  const head = data.headline;
  $('stat-nll').textContent = head.chosen_test.scratch.mean.toFixed(2);
  if (head.baseline_scratch_test) $('stat-nll-note').textContent = `nats per climb, vs ${head.baseline_scratch_test.toFixed(2)} for the baseline (lower is better)`;
  const servedBest = data.served_best_epoch;
  mixture(data.schedule.train);
  protocol(data.schedule, servedBest);
  lrChart(data.schedule, servedBest);
  const best = curvesChart(data.curves);
  $('stat-epochs').textContent = `best at epoch ${best.min}–${best.max} across seeds, picked on validation`;
  ablationChart(data.ablations);
  nullTable(data.null_grade);
  validityChart(data.validity);
  gradeChart(data.samples_by_grade);
  gradingChart(data.grading);
  document.body.dataset.chartsReady = 'true';
}).catch(error => {
  for (const chart of document.querySelectorAll('.chart')) chart.textContent = `Couldn’t load the training results (${error.message}).`;
  document.body.dataset.chartsReady = 'error';
});
