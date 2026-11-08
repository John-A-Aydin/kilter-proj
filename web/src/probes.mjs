// Figures for the "Hold labels in the learned representations" experiment on experiments.html.
// experiments-page.mjs embeds results/hold_label_probes as #probe-data; experiments.mjs calls
// renderProbeFigure for each <figure data-chart="probe-<view>">. Every chart is redrawn at its
// container's width, and every value is also in a table or the README's results table.
const SVG = 'http://www.w3.org/2000/svg';
export const TYPE_ORDER = ['edge', 'jug', 'pinch', 'sloper', 'foot'];
export const TYPE_NAMES = { edge: 'Edge / crimp', jug: 'Jug', pinch: 'Pinch', sloper: 'Sloper', foot: 'Foot chip' };
const HAND = ['edge', 'jug', 'pinch', 'sloper'];
export const REPRESENTATIONS = {
  position: { name: 'Position', learned: false, note: 'x, y, kickboard and screw-on flags, Fourier terms' },
  usage: { name: 'Usage counts', learned: false, note: 'uses as hand, foot, start and finish in training climbs' },
  untrained: { name: 'Untrained model', learned: false, note: 'the hold vector with random weights' },
  identity: { name: 'Hold embedding', learned: true, note: 'the learned per-hold vector, no position input' },
  hold_vector: { name: 'Hold vector', learned: true, note: 'hold embedding plus the learned geometry network' },
  context: { name: 'In-context state', learned: true, note: 'final hidden state where the hold appears in a climb' },
};
const ORDER = ['position', 'usage', 'untrained', 'identity', 'hold_vector', 'context'];
const BOARD_REPRESENTATIONS = ['position', 'usage', 'hold_vector', 'context'];
const typeColor = type => `var(--probe-${type})`;

export const percent = (value, digits = 0) => `${(value * 100).toFixed(digits)}%`;
export const degrees = value => `${value.toFixed(1)}°`;
// Angle between two directions in degrees, from 0 (same) to 180 (opposite).
export const angleError = (a, b) => Math.abs(((a - b + 540) % 360) - 180);
export const shareWithin = (errors, limit) => errors.filter(e => e <= limit).length / errors.length;

export function svg(name, attributes = {}, parent) {
  const node = document.createElementNS(SVG, name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  parent?.append(node);
  return node;
}

export function html(name, attributes = {}, text) {
  const node = document.createElement(name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

let tipNode;
function tip() {
  tipNode ??= document.body.appendChild(html('div', { class: 'probe-tip', hidden: '' }));
  return tipNode;
}
export function showTip(event, content) {
  const node = tip();
  node.innerHTML = content;
  node.hidden = false;
  const pad = 14;
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + node.offsetWidth > innerWidth - 8) x = event.clientX - node.offsetWidth - pad;
  if (y + node.offsetHeight > innerHeight - 8) y = event.clientY - node.offsetHeight - pad;
  node.style.left = `${Math.max(8, x)}px`;
  node.style.top = `${Math.max(8, y)}px`;
}
export const hideTip = () => { if (tipNode) tipNode.hidden = true; };
export const tipRow = (label, value, swatch) =>
  `<span>${swatch ? `<i style="background:${swatch}"></i>` : ''}${label}<b>${value}</b></span>`;

export function caption(figure, title, subtitle) {
  const node = html('figcaption');
  node.append(html('strong', {}, title), html('span', {}, subtitle));
  figure.append(node);
}

// Redraw when the container width changes, so text keeps its size on any screen.
export function responsive(container, draw) {
  let width = 0;
  const run = () => {
    const next = Math.max(260, Math.floor(container.clientWidth));
    if (Math.abs(next - width) < 2) return;
    width = next;
    draw(width);
  };
  run();
  new ResizeObserver(run).observe(container);
}

function segmented(label, options, current, onChange) {
  const group = html('div', { class: 'probe-seg', role: 'group', 'aria-label': label });
  for (const [value, text] of options) {
    const button = html('button', { type: 'button', 'aria-pressed': String(value === current) }, text);
    button.addEventListener('click', () => {
      for (const other of group.children) other.setAttribute('aria-pressed', String(other === button));
      onChange(value);
    });
    group.append(button);
  }
  const wrap = html('div', { class: 'probe-control' });
  wrap.append(html('span', {}, label), group);
  return wrap;
}

export function legendKey(text, swatchClass, color) {
  const key = html('span', { class: 'probe-key' });
  const swatch = html('i', { class: swatchClass || '' });
  if (color && swatchClass === 'line') swatch.style.borderTopColor = color;
  else if (color) swatch.style.background = color;
  key.append(swatch, document.createTextNode(text));
  return key;
}

// ---------- probe scores: one bar per representation, three tasks ----------
function scores(figure, data) {
  const M = data.mean;
  caption(figure, 'Probe scores on held-out holds',
    'Bars are the mean of seeds 42, 43 and 44 and dots are the seeds. Green representations were learned from climbs; grey ones are controls.');
  const tasks = [
    { title: 'Hand-hold type', note: 'balanced accuracy, 306 holds', metric: 'hand_type_balanced_accuracy', max: 1, ticks: [0, 0.5, 1],
      format: v => percent(v), refs: [[0.25, 'chance'], [M.null.hand_type_shuffled_95th, 'shuffled, 95th pct']] },
    { title: 'All five types', note: 'balanced accuracy, 476 holds', metric: 'type_balanced_accuracy', max: 1, ticks: [0, 0.5, 1],
      format: v => percent(v), refs: [[0.2, 'chance'], [M.null.type_shuffled_95th, 'shuffled, 95th pct']] },
    { title: 'Grip direction', note: 'mean error, lower is better', metric: 'grip_mean_error_degrees', max: 48, ticks: [0, 15, 30, 45],
      format: v => degrees(v), tick: v => `${v}°`, refs: [[M.null.grip_constant_mean_direction_error, 'constant guess']] },
  ];
  const plot = html('div', { class: 'probe-plot' });
  figure.append(plot);
  const legend = html('div', { class: 'probe-legend' });
  legend.append(legendKey('Learned from climbs', '', 'var(--probe-learned)'), legendKey('Control', '', 'var(--probe-control)'),
    legendKey('Reference line', 'line'));
  figure.append(legend);
  responsive(plot, width => {
    plot.replaceChildren();
    const wide = width >= 620;
    const labelWidth = 118, rowHeight = 26, head = 52, foot = 24, gap = 22;
    const facetWidth = wide ? (width - labelWidth - gap * 2) / 3 : width - labelWidth;
    const facetHeight = head + rowHeight * ORDER.length + foot;
    const height = wide ? facetHeight : facetHeight * 3 + 12;
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
      'aria-label': 'Probe scores for each representation on the three tasks. The README table lists every value.' }, plot);
    tasks.forEach((task, t) => {
      const x0 = wide ? labelWidth + t * (facetWidth + gap) : labelWidth;
      const y0 = wide ? 0 : t * (facetHeight + 6);
      const right = x0 + facetWidth - 34;
      const x = v => x0 + v / task.max * (right - x0);
      svg('text', { class: 'probe-facet-title', x: x0, y: y0 + 14 }, root).textContent = task.title;
      svg('text', { class: 'probe-tick', x: x0, y: y0 + 29 }, root).textContent = task.note;
      const top = y0 + head, bottom = top + rowHeight * ORDER.length;
      for (const tick of task.ticks) {
        svg('line', { class: 'probe-grid', x1: x(tick), x2: x(tick), y1: top, y2: bottom }, root);
        svg('text', { class: 'probe-tick', x: x(tick), y: bottom + 15, 'text-anchor': 'middle' }, root).textContent = (task.tick || task.format)(tick);
      }
      svg('line', { class: 'probe-axis', x1: x0, x2: x0, y1: top, y2: bottom }, root);
      ORDER.forEach((name, i) => {
        const stats = M[name][task.metric];
        const cy = top + rowHeight * i + rowHeight / 2;
        if (t === 0 || !wide) svg('text', { class: 'probe-row-label', x: labelWidth - 10, y: cy + 4, 'text-anchor': 'end' }, root)
          .textContent = REPRESENTATIONS[name].name;
        const length = Math.max(0, x(stats.mean) - x0), r = Math.min(4, length / 2);
        svg('path', { d: `M${x0},${cy - 7}h${length - r}a${r},${r} 0 0 1 ${r},${r}v${14 - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${length - r}z`,
          fill: REPRESENTATIONS[name].learned ? 'var(--probe-learned)' : 'var(--probe-control)' }, root);
        for (const value of stats.values) svg('circle', { cx: x(value), cy, r: 2.5, class: 'probe-seed' }, root);
        svg('text', { class: 'probe-value', x: x(Math.max(stats.mean, ...stats.values)) + 6, y: cy + 4 }, root).textContent = task.format(stats.mean);
        const hit = svg('rect', { x: x0 - (t === 0 || !wide ? labelWidth : 0), y: cy - rowHeight / 2, width: facetWidth + (t === 0 || !wide ? labelWidth : 0),
          height: rowHeight, fill: 'transparent' }, root);
        hit.addEventListener('pointermove', event => showTip(event, `<strong>${REPRESENTATIONS[name].name} · ${task.title}</strong>`
          + `<em>${REPRESENTATIONS[name].note}</em>${tipRow('Mean of seeds', task.format(stats.mean))}`
          + stats.values.map((v, s) => tipRow(`Seed ${data.seeds[s]}`, task.format(v))).join('')));
        hit.addEventListener('pointerleave', hideTip);
      });
      task.refs.forEach(([value, label], k) => {
        svg('line', { class: 'probe-ref', x1: x(value), x2: x(value), y1: top - 4, y2: bottom }, root);
        // The first reference is labeled to the left of its line, a second one to the right.
        svg('text', { class: 'probe-ref-label', x: x(value) + (k ? 4 : -4), y: top - 8, 'text-anchor': k ? 'start' : 'end' }, root)
          .textContent = label;
      });
    });
  });
}

// ---------- boards: your labels next to held-out predictions ----------
function boards(figure, data) {
  const state = { view: 'type', rep: 'hold_vector', only: null };
  caption(figure, 'Your labels and the probe’s held-out predictions, seed 43',
    'Each hold on the right was predicted by a probe fit without it. Hover a hold for its numbers; pick a type to show only that type.');
  const controls = html('div', { class: 'probe-controls' });
  figure.append(controls);
  const legend = html('div', { class: 'probe-legend' });
  figure.append(legend);
  const pair = html('div', { class: 'probe-boards' });
  figure.append(pair);
  const panels = {};
  for (const side of ['left', 'right']) {
    const card = html('div', { class: 'probe-board' });
    const head = html('div', { class: 'probe-board-head' });
    const title = html('strong', {}, side === 'left' ? 'Your labels' : '');
    const note = html('span');
    head.append(title, note);
    const board = svg('svg', { viewBox: '0 0 1080 1170', role: 'img', 'aria-label': side === 'left'
      ? 'Board colored by your hold labels' : 'Board colored by the probe’s held-out predictions' });
    svg('rect', { width: 1080, height: 1170, fill: 'var(--probe-board)' }, board);
    const holdLayer = svg('g', {}, board), marks = svg('g', {}, board);
    const paths = new Map();
    for (const hold of data.holds) {
      const path = svg('path', { d: hold.d, class: 'probe-hold' }, holdLayer);
      path.addEventListener('pointermove', event => { showTip(event, holdTip(hold, state.rep)); focus(hold.id, true); });
      path.addEventListener('pointerleave', () => { hideTip(); focus(hold.id, false); });
      paths.set(hold.id, path);
    }
    card.append(head, board);
    pair.append(card);
    panels[side] = { title, note, paths, marks };
  }
  const focus = (id, on) => { for (const side of ['left', 'right']) panels[side].paths.get(id).classList.toggle('focus', on); };
  const table = html('details', { class: 'chart-table' });
  table.append(html('summary', {}, 'Every hold'));
  const tableScroll = html('div', { class: 'table-scroll probe-table' });
  table.append(tableScroll);
  figure.append(table);

  controls.append(
    segmented('Show', [['type', 'Type'], ['grip', 'Grip direction']], state.view, value => { state.view = value; render(); }),
    segmented('Probe on', BOARD_REPRESENTATIONS.map(name => [name, REPRESENTATIONS[name].name]), state.rep, value => { state.rep = value; render(); }),
  );

  function arrow(layer, hold, angle, off) {
    const radians = angle * Math.PI / 180, dx = Math.sin(radians), dy = -Math.cos(radians);
    const x2 = hold.cx + dx * 46, y2 = hold.cy + dy * 46, s = 13;
    svg('line', { class: `probe-arrow${off ? ' off' : ''}`, x1: hold.cx, y1: hold.cy, x2, y2 }, layer);
    svg('path', { class: `probe-arrow-head${off ? ' off' : ''}`,
      d: `M${x2 + dx * s},${y2 + dy * s}L${x2 - dy * s * 0.7},${y2 + dx * s * 0.7}L${x2 + dy * s * 0.7},${y2 - dx * s * 0.7}Z` }, layer);
  }

  function render() {
    let right = 0, within = 0, judged = 0;
    for (const side of ['left', 'right']) {
      const { paths, marks } = panels[side];
      marks.replaceChildren();
      for (const hold of data.holds) {
        const path = paths.get(hold.id);
        const prediction = hold.pred[state.rep];
        if (state.view === 'type') {
          const shown = side === 'left' ? hold.type : prediction.type;
          const dim = state.only && hold.type !== state.only && !(side === 'right' && prediction.type === state.only);
          path.style.fill = typeColor(shown);
          path.style.opacity = dim ? 0.13 : 1;
          if (shown === 'sloper' && !dim) svg('circle', { class: 'probe-sloper-dot', cx: hold.cx, cy: hold.cy, r: 7 }, marks);
          if (side === 'right') {
            if (prediction.type === hold.type) right++;
            else if (!dim) svg('path', { d: hold.d, class: 'probe-wrong' }, marks);
          }
        } else {
          path.style.fill = 'var(--probe-idle)';
          path.style.opacity = 1;
          if (side === 'left') for (const direction of hold.grip) arrow(marks, hold, direction, false);
          else if (prediction.grip !== null) {
            const error = angleError(prediction.grip, hold.grip[0]);
            judged++;
            if (error <= 45) within++;
            arrow(marks, hold, prediction.grip, error > 45);
          }
        }
      }
    }
    panels.right.title.textContent = `${REPRESENTATIONS[state.rep].name} probe`;
    if (state.view === 'type') {
      panels.left.note.textContent = `${data.holds.length} holds`;
      panels.right.note.textContent = `${right} of ${data.holds.length} right`;
    } else {
      panels.left.note.textContent = 'every labeled direction';
      panels.right.note.textContent = `${within} of ${judged} within 45°`;
    }
    renderLegend();
    renderTable();
  }

  function renderLegend() {
    legend.replaceChildren();
    if (state.view === 'type') {
      for (const type of TYPE_ORDER) {
        const count = data.holds.filter(hold => hold.type === type).length;
        const key = html('button', { type: 'button', class: `probe-key${state.only && state.only !== type ? ' dim' : ''}`,
          'aria-pressed': String(state.only === type), title: `Show only ${TYPE_NAMES[type].toLowerCase()} holds` });
        const swatch = html('i', { class: type === 'sloper' ? 'dot' : '' });
        swatch.style.background = typeColor(type);
        key.append(swatch, document.createTextNode(`${TYPE_NAMES[type]} ${count}`));
        key.addEventListener('click', () => { state.only = state.only === type ? null : type; render(); });
        legend.append(key);
      }
      legend.append(legendKey('Probe got it wrong', 'ring'));
    } else {
      legend.append(legendKey('Direction you pull, 0° is up', 'arrow'), legendKey('Prediction more than 45° off', 'arrow off'),
        html('span', { class: 'probe-note' }, 'The right board shows holds with exactly one labeled direction.'));
    }
  }

  function renderTable() {
    const rows = data.holds.map(hold => {
      const p = hold.pred[state.rep];
      const total = hold.uses.reduce((a, b) => a + b, 0);
      return `<tr><td>${hold.id}</td><td>${TYPE_NAMES[hold.type]}</td><td>${p.type === hold.type ? '' : '✕ '}${TYPE_NAMES[p.type]}</td>`
        + `<td>${percent(p.p)}</td><td>${hold.grip.map(g => `${g}°`).join(', ')}</td><td>${p.grip === null ? '' : `${p.grip.toFixed(0)}°`}</td>`
        + `<td>${total.toLocaleString()}</td><td>${total ? percent(hold.uses[1] / total) : ''}</td></tr>`;
    }).join('');
    tableScroll.innerHTML = `<table><thead><tr><th>Hold</th><th>Your type</th><th>${REPRESENTATIONS[state.rep].name} probe</th>`
      + `<th>P(your type)</th><th>Your grip</th><th>Predicted grip</th><th>Uses</th><th>Foot share</th></tr></thead><tbody>${rows}</tbody></table>`;
  }
  render();
}

function holdTip(hold, rep) {
  const p = hold.pred[rep];
  const total = hold.uses.reduce((a, b) => a + b, 0);
  let out = `<strong>Hold ${hold.id}</strong>${tipRow('Your label', TYPE_NAMES[hold.type], typeColor(hold.type))}`
    + tipRow(`${REPRESENTATIONS[rep].name} probe`, TYPE_NAMES[p.type], typeColor(p.type))
    + tipRow('Probability of your label', percent(p.p))
    + tipRow('Grip labeled', hold.grip.length ? hold.grip.map(g => `${g}°`).join(', ') : 'none');
  if (p.grip !== null) out += tipRow('Grip predicted', `${p.grip.toFixed(0)}°, ${angleError(p.grip, hold.grip[0]).toFixed(0)}° off`);
  return `${out}<em>Training uses ${total.toLocaleString()}: hand ${hold.uses[0].toLocaleString()}, foot ${hold.uses[1].toLocaleString()}, `
    + `start ${hold.uses[2].toLocaleString()}, finish ${hold.uses[3].toLocaleString()}</em>`;
}

// ---------- confusion matrices ----------
function confusion(figure, data) {
  caption(figure, 'Which types get confused, seed 43',
    'Rows are your labels and columns the held-out prediction; each row sums to 100%. Averaged over the 5 cross-validation repeats.');
  const grid = html('div', { class: 'probe-matrices' });
  figure.append(grid);
  for (const name of ['hold_vector', 'usage', 'position']) {
    const counts = data.confusion[name];
    const card = html('div', { class: 'probe-matrix' });
    card.append(html('strong', {}, REPRESENTATIONS[name].name));
    const cells = html('div', { class: 'probe-matrix-grid', role: 'table', 'aria-label': `${REPRESENTATIONS[name].name} confusion matrix` });
    cells.append(html('span', { class: 'corner' }, 'label ↓'));
    for (const type of TYPE_ORDER) cells.append(html('span', { class: 'col', title: TYPE_NAMES[type] }, TYPE_NAMES[type].split(' ')[0]));
    for (const type of TYPE_ORDER) {
      const row = counts[data.types.indexOf(type)];
      const total = row.reduce((a, b) => a + b, 0);
      const head = html('span', { class: 'row' });
      const swatch = html('i');
      swatch.style.background = typeColor(type);
      head.append(swatch, document.createTextNode(TYPE_NAMES[type].split(' ')[0]));
      cells.append(head);
      for (const predicted of TYPE_ORDER) {
        const count = row[data.types.indexOf(predicted)], share = count / total;
        const cell = html('span', { class: `cell${type === predicted ? ' diagonal' : ''}` }, share >= 0.005 ? percent(share) : '');
        cell.style.background = `color-mix(in oklab, var(--probe-learned) ${Math.round(share * 100)}%, var(--probe-cell))`;
        cell.style.color = share > 0.55 ? '#ffffff' : 'var(--probe-ink-2)';
        cell.addEventListener('pointermove', event => showTip(event, `<strong>${TYPE_NAMES[type]} predicted as ${TYPE_NAMES[predicted]}</strong>`
          + tipRow('Holds', `${count.toFixed(1)} of ${total.toFixed(0)}`) + tipRow('Share of row', percent(share, 1))));
        cell.addEventListener('pointerleave', hideTip);
        cells.append(cell);
      }
    }
    card.append(cells);
    grid.append(card);
  }
}

// ---------- projection and recall per type ----------
function projection(figure, data) {
  caption(figure, 'The hold vectors in two dimensions, and recall per hand-hold type',
    'Left: the top two principal components of seed 43’s hold vectors, found without labels. Right: share of each type predicted correctly, mean of three seeds.');
  const footShare = hold => { const total = hold.uses.reduce((a, b) => a + b, 0); return total ? hold.uses[1] / total : 0; };
  const layout = html('div', { class: 'probe-two' });
  figure.append(layout);
  const scatterCard = html('div', { class: 'probe-scatter' });
  const scatterState = { color: 'type' };
  const scatterLegend = html('div', { class: 'probe-legend dark' });
  const plot = svg('svg', { viewBox: '0 0 560 440', role: 'img', 'aria-label': 'Hold vectors projected on two principal components' });
  scatterCard.append(segmented('Color by', [['type', 'Your label'], ['foot', 'Foot share of uses']], 'type',
    value => { scatterState.color = value; drawScatter(); }), plot, scatterLegend);
  layout.append(scatterCard);
  function drawScatter() {
    plot.replaceChildren();
    const pad = 24, xs = data.holds.map(h => h.pc[0]), ys = data.holds.map(h => h.pc[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const sx = v => pad + (v - x0) / (x1 - x0) * (560 - 2 * pad), sy = v => 440 - pad - (v - y0) / (y1 - y0) * (440 - 2 * pad);
    svg('text', { class: 'probe-dark-tick', x: 560 - pad, y: 436, 'text-anchor': 'end' }, plot).textContent = 'component 1 →';
    svg('text', { class: 'probe-dark-tick', x: 4, y: 14 }, plot).textContent = '↑ component 2';
    const ordered = [...data.holds].sort((a, b) => (b.type === 'foot') - (a.type === 'foot'));
    for (const hold of ordered) {
      const fill = scatterState.color === 'type' ? typeColor(hold.type)
        : `color-mix(in oklab, #f2f6ec ${Math.round(footShare(hold) * 100)}%, #3c463c)`;
      const dot = svg('circle', { class: 'probe-point', cx: sx(hold.pc[0]), cy: sy(hold.pc[1]), r: hold.type === 'sloper' && scatterState.color === 'type' ? 6.5 : 5, fill }, plot);
      if (hold.type === 'sloper' && scatterState.color === 'type') svg('circle', { cx: sx(hold.pc[0]), cy: sy(hold.pc[1]), r: 2, fill: '#ffffff' }, plot);
      dot.addEventListener('pointermove', event => showTip(event, holdTip(hold, 'hold_vector') + tipRow('Foot share of uses', percent(footShare(hold)))));
      dot.addEventListener('pointerleave', hideTip);
    }
    scatterLegend.replaceChildren();
    if (scatterState.color === 'type') {
      for (const type of TYPE_ORDER) scatterLegend.append(legendKey(TYPE_NAMES[type], type === 'sloper' ? 'dot' : '', typeColor(type)));
    } else {
      const ramp = html('span', { class: 'probe-key' });
      ramp.append(document.createTextNode('never a foot'), html('i', { class: 'ramp' }), document.createTextNode('always a foot'));
      scatterLegend.append(ramp);
    }
  }
  drawScatter();

  const recallCard = html('div', { class: 'probe-recall' });
  const recallPlot = html('div', { class: 'probe-plot' });
  const reps = [['position', 'var(--probe-control-2)'], ['usage', 'var(--probe-control)'], ['hold_vector', 'var(--probe-learned)']];
  const recallLegend = html('div', { class: 'probe-legend' });
  for (const [name, color] of reps) recallLegend.append(legendKey(REPRESENTATIONS[name].name, '', color));
  recallCard.append(recallLegend, recallPlot);
  layout.append(recallCard);
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  responsive(recallPlot, width => {
    recallPlot.replaceChildren();
    const left = 70, right = width - 40, rowHeight = 52, top = 4, height = top + rowHeight * HAND.length + 22;
    const x = v => left + v * (right - left);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': 'Recall per hand-hold type' }, recallPlot);
    for (const tick of [0, 0.5, 1]) {
      svg('line', { class: 'probe-grid', x1: x(tick), x2: x(tick), y1: top, y2: height - 22 }, root);
      svg('text', { class: 'probe-tick', x: x(tick), y: height - 6, 'text-anchor': 'middle' }, root).textContent = percent(tick);
    }
    svg('line', { class: 'probe-axis', x1: left, x2: left, y1: top, y2: height - 22 }, root);
    HAND.forEach((type, i) => {
      const y0 = top + i * rowHeight;
      svg('rect', { x: 0, y: y0 + rowHeight / 2 - 4, width: 8, height: 8, rx: 2, fill: typeColor(type) }, root);
      svg('text', { class: 'probe-row-label', x: 14, y: y0 + rowHeight / 2 + 4 }, root).textContent = TYPE_NAMES[type].split(' ')[0];
      reps.forEach(([name, color], j) => {
        const values = data.handRecall[name].map(recall => recall[type]);
        const value = mean(values), cy = y0 + 8 + j * 13, length = Math.max(0.5, x(value) - left);
        svg('rect', { x: left, y: cy, width: length, height: 10, rx: Math.min(3, length / 2), fill: color }, root);
        if (name === 'hold_vector') svg('text', { class: 'probe-value', x: x(value) + 6, y: cy + 9 }, root).textContent = percent(value);
        const hit = svg('rect', { x: 0, y: cy - 1.5, width, height: 13, fill: 'transparent' }, root);
        hit.addEventListener('pointermove', event => showTip(event, `<strong>${TYPE_NAMES[type]} · ${REPRESENTATIONS[name].name}</strong>`
          + tipRow('Mean of seeds', percent(value)) + values.map((v, s) => tipRow(`Seed ${data.seeds[s]}`, percent(v))).join('')));
        hit.addEventListener('pointerleave', hideTip);
      });
    });
  });
}

// ---------- grip: cumulative angular error ----------
function grip(figure, data) {
  const single = data.holds.filter(hold => hold.pred.hold_vector.grip !== null);
  caption(figure, 'How close each grip probe gets, seed 43',
    `Share of the ${single.length} single-direction holds predicted within a given angle of the labeled direction. Higher and further left is better.`);
  const series = [
    { name: 'Hold vector', color: 'var(--probe-learned)', width: 2.5, errors: single.map(h => angleError(h.pred.hold_vector.grip, h.grip[0])) },
    { name: 'In-context state', color: 'var(--probe-learned-2)', width: 2, errors: single.map(h => angleError(h.pred.context.grip, h.grip[0])) },
    { name: 'Usage counts', color: 'var(--probe-control)', width: 2.5, errors: single.map(h => angleError(h.pred.usage.grip, h.grip[0])) },
    { name: 'Position', color: 'var(--probe-control-2)', width: 2.5, errors: single.map(h => angleError(h.pred.position.grip, h.grip[0])) },
    { name: 'Always the average direction', color: 'var(--probe-muted)', width: 1.5, errors: single.map(h => angleError(data.constantDirection, h.grip[0])) },
  ];
  const legend = html('div', { class: 'probe-legend' });
  for (const s of series) legend.append(legendKey(s.name, 'line', s.color));
  const plot = html('div', { class: 'probe-plot' });
  figure.append(legend, plot);
  const limits = [10, 22.5, 45, 90];
  const table = html('details', { class: 'chart-table' });
  table.innerHTML = `<summary>Data table</summary><div class="table-scroll"><table><thead><tr><th></th>${limits.map(l => `<th>Within ${l}°</th>`).join('')}<th>Mean error</th></tr></thead><tbody>${
    series.map(s => `<tr><th scope="row">${s.name}</th>${limits.map(l => `<td>${percent(shareWithin(s.errors, l))}</td>`).join('')}<td>${
      degrees(s.errors.reduce((a, b) => a + b, 0) / s.errors.length)}</td></tr>`).join('')}</tbody></table></div>`;
  figure.append(table);
  responsive(plot, width => {
    plot.replaceChildren();
    const height = width < 480 ? 230 : 280, left = 42, right = width - 14, top = 10, bottom = height - 34;
    const x = v => left + v / 180 * (right - left), y = v => bottom - v * (bottom - top);
    const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img',
      'aria-label': 'Cumulative grip error for each probe. The data table below lists the values.' }, plot);
    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      svg('line', { class: tick ? 'probe-grid' : 'probe-axis', x1: left, x2: right, y1: y(tick), y2: y(tick) }, root);
      svg('text', { class: 'probe-tick', x: left - 8, y: y(tick) + 4, 'text-anchor': 'end' }, root).textContent = percent(tick);
    }
    for (const tick of (width < 480 ? [0, 45, 90, 135, 180] : [0, 22.5, 45, 90, 135, 180])) {
      svg('text', { class: 'probe-tick', x: x(tick), y: bottom + 16, 'text-anchor': 'middle' }, root).textContent = `${tick}°`;
    }
    svg('text', { class: 'probe-tick', x: right, y: height - 2, 'text-anchor': 'end' }, root).textContent = 'angle from the labeled direction';
    for (const s of [...series].reverse()) {
      let d = '';
      for (let v = 0; v <= 180; v++) d += `${v ? 'L' : 'M'}${x(v).toFixed(1)},${y(shareWithin(s.errors, v)).toFixed(1)}`;
      svg('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, root);
    }
    const cross = svg('line', { class: 'probe-axis', y1: top, y2: bottom, visibility: 'hidden' }, root);
    const hit = svg('rect', { x: left, y: top, width: right - left, height: bottom - top, fill: 'transparent' }, root);
    hit.addEventListener('pointermove', event => {
      const box = root.getBoundingClientRect();
      const v = Math.max(0, Math.min(180, Math.round(((event.clientX - box.left) * width / box.width - left) / (right - left) * 180)));
      cross.setAttribute('x1', x(v));
      cross.setAttribute('x2', x(v));
      cross.setAttribute('visibility', 'visible');
      showTip(event, `<strong>Within ${v}°</strong>${series.map(s => tipRow(s.name, percent(shareWithin(s.errors, v)), s.color)).join('')}`);
    });
    hit.addEventListener('pointerleave', () => { hideTip(); cross.setAttribute('visibility', 'hidden'); });
  });
}

const VIEWS = { scores, boards, confusion, projection, grip };

export function renderProbeFigure(figure, view, data) {
  if (!VIEWS[view]) throw new Error(`Unknown probe figure: ${view}`);
  figure.classList.add('probe-figure');
  figure.replaceChildren();
  VIEWS[view](figure, data);
}
