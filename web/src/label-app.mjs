import { addBoardArt, loadBoardConfig } from './board.mjs';
import { BOARD_WIDTH, BOARD_HEIGHT, outlinePath } from './board-config.mjs';
import { CARDINALS, DIRECTIONS, DIRECTION_STEP, FIELDS, FIELD_NAMES, directionLabel, fromCsv, hasLabel, holdOrder,
  isComplete, migrateLabel, rotateGrip, toCsv, toggleDirection } from './labels.mjs';

const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const STORAGE_KEY = 'kilter-hold-labels-v1';
const SETTINGS_KEY = 'kilter-hold-labels-settings';
const TYPE_COLORS = Object.fromEntries(FIELDS.type.options.map(option => [option.value, option.color]));

let holds;        // manifest holds by id
let shapes;       // board-config holds by id: cx, cy, r, outline
let order = [];
let index = 0;
let labels = {};
const paths = new Map();
const fieldButtons = new Map();
const wedges = new Map();
let zoomOutline;

function load(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}

function setStatus(text, state = 'ready') {
  $('save-status-text').textContent = text;
  $('save-status').className = `model-status ${state}`;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}

// Storage can be blocked or cleared, so the status says when only an export will keep the work.
function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(labels));
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ index, skipLabeled: $('skip-labeled').checked }));
    setStatus(`Saved in this browser · ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`);
  } catch {
    setStatus('Browser storage is blocked. Export to keep your labels', 'running');
  }
}

const currentId = () => order[index];

function setLabel(id, label) {
  if (hasLabel(label)) labels[id] = { ...label, updated: new Date().toISOString() };
  else delete labels[id];
  save();
  paintHold(id);
  renderProgress();
}

function select(field, value) {
  const label = { ...labels[currentId()] };
  if (label[field] === value) delete label[field];
  else label[field] = value;
  setLabel(currentId(), label);
  renderFields();
}

function setGrip(grip) {
  const label = { ...labels[currentId()], grip };
  if (!grip.length) delete label.grip;
  setLabel(currentId(), label);
  renderFields();
}

const currentGrip = () => labels[currentId()]?.grip || [];
const toggleAllRound = () => setGrip(currentGrip().length === DIRECTIONS.length ? [] : [...DIRECTIONS]);

function step(direction, onlyUnlabeled = $('skip-labeled').checked) {
  if (onlyUnlabeled) {
    for (let i = 1; i <= order.length; i++) {
      const candidate = (index + direction * i + order.length) % order.length;
      if (!isComplete(labels[order[candidate]])) return show(candidate);
    }
  }
  show((index + direction + order.length) % order.length);
}

function paintHold(id) {
  const path = paths.get(id);
  const label = labels[id];
  const color = $('show-labels').checked && label?.type ? TYPE_COLORS[label.type] : null;
  path.style.setProperty('--type-color', color || 'transparent');
  path.classList.toggle('typed', Boolean(color));
  path.classList.toggle('partial', Boolean(label?.type) && !isComplete(label));
  path.setAttribute('aria-label', `Hold ${id}, ${holds.get(id).name}${label?.type ? `, ${label.type}` : ', unlabeled'}`);
}

function show(newIndex) {
  paths.get(currentId())?.classList.remove('current');
  index = newIndex;
  const id = currentId();
  const hold = holds.get(id);
  const shape = shapes.get(id);
  paths.get(id).classList.add('current');
  const extent = Math.max(...shape.outline.map(Math.abs)) * shape.r;
  const half = Math.max(extent * 1.7, 45);
  $('svg-zoom').setAttribute('viewBox', `${shape.cx - half} ${shape.cy - half} ${2 * half} ${2 * half}`);
  zoomOutline.setAttribute('d', outlinePath(shape));
  $('hold-title').textContent = `Hold ${id}`;
  $('hold-meta').textContent = [hold.name, hold.set === 20 ? 'screw-on' : 'bolt-on',
    ...(hold.kickboard ? ['kickboard'] : []), `${index + 1} of ${order.length}`].join(' · ');
  $('notes').value = labels[id]?.notes || '';
  renderFields();
  save();
}

function renderFields() {
  const label = labels[currentId()] || {};
  for (const [key, button] of fieldButtons) {
    const [field, value] = key.split(':');
    button.setAttribute('aria-pressed', String(label[field] === value));
  }
  const grip = label.grip || [];
  for (const [direction, wedge] of wedges) {
    wedge.classList.toggle('on', grip.includes(direction));
    wedge.setAttribute('aria-checked', String(grip.includes(direction)));
  }
  $('grip-summary').textContent = grip.length === DIRECTIONS.length ? 'All round'
    : grip.length ? grip.map(directionLabel).join(', ') : 'None yet';
  $('grip-all').setAttribute('aria-pressed', String(grip.length === DIRECTIONS.length));
  const needsMore = label.type && label.type !== 'unsure';
  $('fields').classList.toggle('unsure', label.type === 'unsure');
  $('zoom-view').classList.toggle('unsure', label.type === 'unsure');
  $('fields').dataset.state = isComplete(label) ? 'complete' : needsMore ? 'partial' : 'empty';
}

function renderProgress() {
  const done = order.filter(id => isComplete(labels[id])).length;
  $('progress-count').textContent = `${done} / ${order.length}`;
  $('progress-fill').style.width = `${done / order.length * 100}%`;
  const counts = Object.fromEntries(FIELDS.type.options.map(option => [option.value, 0]));
  for (const label of Object.values(labels)) if (label.type) counts[label.type]++;
  $('legend').replaceChildren(...FIELDS.type.options.map(option => {
    const item = document.createElement('span');
    item.innerHTML = '<i class="swatch"></i><span></span>';
    item.firstChild.style.background = option.color;
    item.lastChild.textContent = `${option.label} ${counts[option.value]}`;
    return item;
  }));
}

function buildFields() {
  const card = $('fields');
  for (const name of FIELD_NAMES) {
    const group = document.createElement('div');
    group.className = `field field-${name}`;
    group.setAttribute('role', 'group');
    const heading = document.createElement('h3');
    heading.textContent = FIELDS[name].title;
    group.setAttribute('aria-label', FIELDS[name].title);
    const options = document.createElement('div');
    options.className = 'field-options';
    for (const option of FIELDS[name].options) {
      const button = document.createElement('button');
      button.className = 'option-button';
      button.innerHTML = '<i></i><span></span><kbd></kbd>';
      button.querySelector('span').textContent = option.label;
      button.querySelector('kbd').textContent = option.key.toUpperCase();
      if (option.color) button.style.setProperty('--color', option.color);
      else button.querySelector('i').remove();
      button.onclick = () => select(name, option.value);
      fieldButtons.set(`${name}:${option.value}`, button);
      options.append(button);
    }
    group.append(heading, options);
    card.insertBefore(group, $('grip-field'));
  }
  const shortcuts = [...FIELD_NAMES.map(name => [FIELDS[name].options.map(o => o.key.toUpperCase()).join(' '), FIELDS[name].title]),
    ['↑ → ↓ ←', 'Toggle grip at 0°, 90°, 180°, 270°'], ['[ ]', 'Rotate the grip by 22.5°'], ['A', 'Grip all round'],
    ['Enter / Shift+Enter', 'Next / previous hold'], ['U', 'Next unlabeled hold'], ['N', 'Write a note'],
    ['Esc', 'Leave the note'], ['Delete', 'Clear this hold']];
  $('shortcuts').replaceChildren(...shortcuts.flatMap(([keys, action]) => {
    const dt = document.createElement('dt'); dt.textContent = keys;
    const dd = document.createElement('dd'); dd.textContent = action;
    return [dt, dd];
  }));
}

function buildBoards() {
  const board = $('svg-board');
  board.setAttribute('viewBox', `0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`);
  addBoardArt(board);
  for (const id of order) {
    const path = document.createElementNS(NS, 'path');
    path.id = `hold-${id}`;
    path.classList.add('hold', 'label-hold');
    path.setAttribute('d', outlinePath(shapes.get(id)));
    path.onclick = () => show(order.indexOf(id));
    paths.set(id, path);
    board.append(path);
    paintHold(id);
  }
  const zoom = $('svg-zoom');
  addBoardArt(zoom);
  zoomOutline = document.createElementNS(NS, 'path');
  zoomOutline.classList.add('zoom-outline');
  zoom.append(zoomOutline);
  buildCompass();
}

const INNER = 0.72;
const OUTER = 0.98;
const point = (degrees, radius) => {
  const angle = degrees * Math.PI / 180;
  return `${(radius * Math.sin(angle)).toFixed(4)},${(-radius * Math.cos(angle)).toFixed(4)}`;
};

// A ring of 16 wedges around the close-up; each toggles one grip direction.
function buildCompass() {
  const compass = $('svg-compass');
  const half = DIRECTION_STEP / 2 - 0.8;
  for (const direction of DIRECTIONS) {
    const wedge = document.createElementNS(NS, 'path');
    const [a, b] = [direction - half, direction + half];
    wedge.setAttribute('d', `M${point(a, OUTER)}A${OUTER},${OUTER} 0 0 1 ${point(b, OUTER)}`
      + `L${point(b, INNER)}A${INNER},${INNER} 0 0 0 ${point(a, INNER)}Z`);
    wedge.classList.add('wedge');
    wedge.dataset.direction = direction;
    wedge.setAttribute('tabindex', '0');
    wedge.setAttribute('role', 'checkbox');
    wedge.setAttribute('aria-label', `Grip ${directionLabel(direction)}`);
    wedge.onclick = () => setGrip(toggleDirection(currentGrip(), direction));
    wedges.set(direction, wedge);
    compass.append(wedge);
  }
  for (const direction of Object.keys(CARDINALS).map(Number)) {
    const text = document.createElementNS(NS, 'text');
    const [x, y] = point(direction, (INNER + OUTER) / 2).split(',');
    Object.entries({ x, y, 'text-anchor': 'middle', 'dominant-baseline': 'central' }).forEach(([k, v]) => text.setAttribute(k, v));
    text.classList.add('compass-label');
    text.textContent = `${direction}°`;
    compass.append(text);
  }
}

function onKey(event) {
  if (event.target.id === 'notes') {
    if (event.key === 'Escape' || (event.key === 'Enter' && !event.shiftKey)) { event.preventDefault(); event.target.blur(); }
    return;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  for (const name of FIELD_NAMES) {
    const option = FIELDS[name].options.find(o => o.key === key);
    if (option) { event.preventDefault(); return select(name, option.value); }
  }
  if (key === ' ' && event.target.dataset?.direction) {
    event.preventDefault();
    return setGrip(toggleDirection(currentGrip(), Number(event.target.dataset.direction)));
  }
  const toggle = direction => () => setGrip(toggleDirection(currentGrip(), direction));
  const actions = {
    ArrowUp: toggle(0), ArrowRight: toggle(90), ArrowDown: toggle(180), ArrowLeft: toggle(270),
    '[': () => setGrip(rotateGrip(currentGrip(), -1)), ']': () => setGrip(rotateGrip(currentGrip(), 1)),
    a: toggleAllRound,
    Enter: () => step(event.shiftKey ? -1 : 1), u: () => step(1, true),
    n: () => $('notes').focus(), Delete: () => clear(), Backspace: () => clear(),
  };
  if (actions[key]) { event.preventDefault(); actions[key](); }
}

function clear() {
  setLabel(currentId(), {});
  $('notes').value = '';
  renderFields();
}

function exportCsv() {
  const url = URL.createObjectURL(new Blob([toCsv(order.map(id => holds.get(id)), labels)], { type: 'text/csv' }));
  const link = Object.assign(document.createElement('a'), { href: url, download: 'hold_labels.csv' });
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function importCsv(file) {
  showError('');
  try {
    const imported = fromCsv(await file.text(), order);
    const overlap = Object.keys(imported).filter(id => labels[id]).length;
    if (overlap && !confirm(`${overlap} of the imported holds are already labeled here. Replace them with the file's labels?`)) return;
    labels = { ...labels, ...imported };
    save();
    order.forEach(paintHold);
    renderProgress();
    show(index);
    setStatus(`Imported ${Object.keys(imported).length} labeled holds`);
  } catch (error) {
    showError(`Could not import ${file.name}: ${error.message}`);
  }
}

async function start() {
  const manifest = await (await fetch('./model/manifest.json')).json();
  holds = new Map(manifest.holds.map(hold => [hold.id, hold]));
  shapes = new Map((await loadBoardConfig(manifest)).holds.map(hold => [hold.id, hold]));
  order = holdOrder(manifest.holds);
  labels = load(STORAGE_KEY, {});
  for (const id of Object.keys(labels)) {
    if (holds.has(Number(id))) labels[id] = migrateLabel(labels[id]);
    else delete labels[id];
  }
  const settings = load(SETTINGS_KEY, {});
  $('skip-labeled').checked = settings.skipLabeled ?? true;
  buildFields();
  buildBoards();
  renderProgress();
  const firstOpen = order.findIndex(id => !isComplete(labels[id]));
  show(Number.isInteger(settings.index) && settings.index < order.length ? settings.index : Math.max(firstOpen, 0));
  $('prev').onclick = () => step(-1);
  $('next').onclick = () => step(1);
  $('next-unlabeled').onclick = () => step(1, true);
  $('clear-label').onclick = clear;
  $('notes').oninput = event => setLabel(currentId(), { ...labels[currentId()], notes: event.target.value });
  $('show-labels').onchange = () => order.forEach(paintHold);
  $('grip-all').onclick = toggleAllRound;
  $('grip-clear').onclick = () => setGrip([]);
  $('skip-labeled').onchange = save;
  $('export').onclick = exportCsv;
  $('import').onclick = () => $('import-file').click();
  $('import-file').onchange = event => { if (event.target.files[0]) importCsv(event.target.files[0]); event.target.value = ''; };
  document.addEventListener('keydown', onKey);
  document.body.dataset.ready = 'true';
}

start().catch(error => {
  setStatus('Could not load the board', 'running');
  showError(error.message);
});
