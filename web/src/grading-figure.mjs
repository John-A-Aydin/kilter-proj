// The predicted-against-true grade heatmap on experiments.html, drawn from the "grading-confusion"
// chart data (experiments-page.mjs gradingConfusion). Rows are true grades and columns predicted
// grades, rounded; each row shows how that grade's test climbs were predicted, as shares of the row.
// It uses the hold-label probe figures' theme and controls (src/probes.mjs, src/probes.css).
import { caption, hideTip, html, percent, segmented, showTip, tipRow } from './probes.mjs';

// One fixed color scale for every method and subset, so switching keeps colors comparable.
// Shares at or above SCALE_MAX get the full color.
export const SCALE_MAX = 0.5;
const MIN_SHARE_MIX = 8;

export function cellMix(share) {
  if (!(share > 0)) return 0;
  return Math.round(MIN_SHARE_MIX + (100 - MIN_SHARE_MIX) * Math.min(1, share / SCALE_MAX));
}

// Shares of exactly right and within one grade, over every climb in the matrix.
export function agreement(counts) {
  let total = 0, exact = 0, withinOne = 0;
  counts.forEach((row, truth) => row.forEach((count, predicted) => {
    total += count;
    if (predicted === truth) exact += count;
    if (Math.abs(predicted - truth) <= 1) withinOne += count;
  }));
  return { total, exact: exact / total, withinOne: withinOne / total };
}

// Counts are averaged over seeds, so they can have a fraction; whole numbers show without one.
export const count = value => (Math.round(value * 10) / 10).toLocaleString('en-US', { maximumFractionDigits: 1 });

export function renderGradingFigure(figure, spec) {
  figure.classList.add('probe-figure', 'grading-figure');
  figure.replaceChildren();
  caption(figure, spec.title, spec.subtitle);
  const state = { method: spec.defaultMethod, subset: spec.subsets[0].key, active: null };
  const controls = html('div', { class: 'probe-controls' });
  controls.append(
    segmented('Method', spec.methods.map(m => [m.key, m.short]), state.method, value => { state.method = value; render(); }),
    segmented('Test climbs', spec.subsets.map(s => [s.key, s.label]), state.subset, value => { state.subset = value; render(); }));
  const stats = html('p', { class: 'grading-stats', 'aria-live': 'polite' });
  const key = html('div', { class: 'grading-key' });
  key.innerHTML = `<span>0%</span><i class="grading-ramp"></i><span>${percent(SCALE_MAX)}+ of the true grade’s climbs</span>`
    + `<span class="grading-key-diagonal"><i></i>prediction equals the true grade</span>`;
  const grid = html('div', { class: 'grading-grid', role: 'grid', tabindex: '0' });
  grid.style.setProperty('--grades', spec.grades.length);
  const table = html('details', { class: 'chart-table' });
  figure.append(controls, stats, key, grid, table);

  const counts = () => spec.counts[state.method][state.subset];
  const method = () => spec.methods.find(m => m.key === state.method);
  const subset = () => spec.subsets.find(s => s.key === state.subset);

  function cellTip(truth, predicted) {
    const row = counts()[truth], total = row.reduce((a, b) => a + b, 0), value = row[predicted];
    return `<strong>True V${spec.grades[truth]}, predicted V${spec.grades[predicted]}</strong>`
      + tipRow(`Share of V${spec.grades[truth]} climbs`, total ? percent(value / total, 1) : '—')
      + tipRow('Test rows', `${count(value)} of ${count(total)}`)
      + `<em>${method().label} · ${subset().label.toLowerCase()} · mean of ${spec.seeds[state.method]} seed${spec.seeds[state.method] === 1 ? '' : 's'}</em>`;
  }

  function render() {
    hideTip();
    const matrix = counts();
    const { exact, withinOne } = agreement(matrix);
    stats.innerHTML = `<b>${method().label}</b>, ${subset().label.toLowerCase()}: rounded prediction exact for `
      + `<b>${percent(exact)}</b>, within one grade for <b>${percent(withinOne)}</b>; RMSE <b>${spec.rmse[state.method][state.subset].toFixed(2)}</b> grades.`;
    grid.setAttribute('aria-label', `${method().label}, ${subset().label}: share of each true grade's test climbs by predicted grade. `
      + 'Use the arrow keys to read cells; the data table below lists them all.');
    const cells = [html('span', { class: 'grading-corner' }, 'True ↓ Predicted →')];
    for (const grade of spec.grades) {
      const head = html('span', { class: 'grading-col', title: `Predicted V${grade}` });
      head.append(html('span', { class: 'grading-v' }, 'V'), document.createTextNode(String(grade)));
      cells.push(head);
    }
    cells.push(html('span', { class: 'grading-col grading-n' }, 'Rows'));
    matrix.forEach((row, truth) => {
      const total = row.reduce((a, b) => a + b, 0);
      const sparse = total < spec.minRows;
      cells.push(html('span', { class: `grading-row${sparse ? ' sparse' : ''}` }, `V${spec.grades[truth]}`));
      row.forEach((value, predicted) => {
        const share = total ? value / total : 0;
        const cell = html('span', { class: `grading-cell${predicted === truth ? ' diagonal' : ''}${sparse ? ' sparse' : ''}`,
          'data-truth': truth, 'data-predicted': predicted });
        cell.style.setProperty('--mix', `${cellMix(share)}%`);
        cell.addEventListener('pointermove', event => { setActive(null); showTip(event, cellTip(truth, predicted)); });
        cell.addEventListener('pointerleave', hideTip);
        cells.push(cell);
      });
      cells.push(html('span', { class: `grading-n${sparse ? ' sparse' : ''}` }, Math.round(total).toLocaleString()));
    });
    grid.replaceChildren(...cells);
    table.innerHTML = `<summary>Data table: share of each true grade’s climbs, by predicted grade</summary><div class="table-scroll"><table><thead><tr><th scope="col">True</th>${
      spec.grades.map(g => `<th scope="col">V${g}</th>`).join('')}<th scope="col">Rows</th></tr></thead><tbody>${matrix.map((row, truth) => {
      const total = row.reduce((a, b) => a + b, 0);
      return `<tr><th scope="row">V${spec.grades[truth]}</th>${row.map(v => `<td>${total && v ? percent(v / total) : ''}</td>`).join('')}<td>${Math.round(total).toLocaleString()}</td></tr>`;
    }).join('')}</tbody></table></div>`;
    if (state.active) setActive(state.active);
  }

  // Keyboard reading: the grid is one tab stop; arrow keys move between cells.
  function setActive(position) {
    grid.querySelector('.grading-cell.active')?.classList.remove('active');
    state.active = position;
    if (!position) return;
    const cell = grid.querySelector(`[data-truth="${position[0]}"][data-predicted="${position[1]}"]`);
    cell.classList.add('active');
    const box = cell.getBoundingClientRect();
    showTip({ clientX: box.right, clientY: box.bottom }, cellTip(...position));
  }
  grid.addEventListener('keydown', event => {
    const move = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[event.key];
    if (!move) return;
    event.preventDefault();
    const last = spec.grades.length - 1;
    const [r, c] = state.active ?? [Math.min(5, last), Math.min(5, last)];
    setActive(state.active ? [Math.max(0, Math.min(last, r + move[0])), Math.max(0, Math.min(last, c + move[1]))] : [r, c]);
  });
  grid.addEventListener('blur', () => { setActive(null); hideTip(); });
  render();
}
