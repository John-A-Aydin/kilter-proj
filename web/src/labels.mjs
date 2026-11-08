// Hold labels: the schema, the order holds are visited in, and CSV export/import.
// Kept free of the DOM so the Node tests can check it.

// Fields with exactly one choice per hold.
export const FIELDS = {
  type: { title: 'Type', options: [
    { value: 'jug', label: 'Jug', key: '1', color: '#7bd389' },
    { value: 'edge', label: 'Edge / crimp', key: '2', color: '#5aa9e6' },
    { value: 'sloper', label: 'Sloper', key: '3', color: '#f2c14e' },
    { value: 'pinch', label: 'Pinch', key: '4', color: '#f78154' },
    { value: 'pocket', label: 'Pocket', key: '5', color: '#b48ee6' },
    { value: 'foot', label: 'Foot chip', key: '6', color: '#2ad4c0' },
    { value: 'unsure', label: 'Unsure', key: '7', color: '#ff5d8f' },
  ] },
  size: { title: 'Size', options: [
    { value: 'small', label: 'Small', key: 'z' },
    { value: 'medium', label: 'Medium', key: 'x' },
    { value: 'large', label: 'Large', key: 'c' },
  ] },
};

export const FIELD_NAMES = Object.keys(FIELDS);

// Grip: every side of the hold you can pull on, in degrees clockwise from the top of
// the hold as you face the wall. 0 is a normal hold, 90 its right side, 180 an undercling.
export const DIRECTION_STEP = 22.5;
export const DIRECTIONS = Array.from({ length: 360 / DIRECTION_STEP }, (_, i) => i * DIRECTION_STEP);
export const CARDINALS = { 0: 'top', 90: 'right', 180: 'bottom', 270: 'left' };
// Labels saved before the compass used these names.
const LEGACY_GRIP = { top: [0], right: [90], bottom: [180], left: [270], all: DIRECTIONS };

export const COLUMNS = ['hold_token', 'hole_name', 'hold_x', 'hold_y', 'set_id', 'on_kickboard',
  ...FIELD_NAMES, 'grip', 'notes', 'updated_at'];

const wrap = degrees => ((degrees % 360) + 360) % 360;

// Sorted, without repeats. Anything off the 22.5° compass is an error.
export function normalizeGrip(grip) {
  if (typeof grip === 'string' && LEGACY_GRIP[grip]) return [...LEGACY_GRIP[grip]];
  const values = [...new Set((grip || []).map(Number))].sort((a, b) => a - b);
  const bad = values.find(value => !DIRECTIONS.includes(value));
  if (bad !== undefined) throw new Error(`${bad} is not a grip direction`);
  return values;
}

export function toggleDirection(grip = [], direction) {
  return grip.includes(direction) ? grip.filter(d => d !== direction) : normalizeGrip([...grip, direction]);
}

export function rotateGrip(grip = [], steps) {
  return normalizeGrip(grip.map(d => wrap(d + steps * DIRECTION_STEP)));
}

export function directionLabel(direction) {
  return CARDINALS[direction] ? `${direction}° (${CARDINALS[direction]})` : `${direction}°`;
}

// Old saved labels name a grip side; turn that into directions.
export function migrateLabel(label) {
  return typeof label?.grip === 'string' ? { ...label, grip: normalizeGrip(label.grip) } : label;
}

// An unsure hold needs nothing else; any other hold needs a type, a size, and a grip direction.
export function isComplete(label) {
  if (!label?.type) return false;
  return label.type === 'unsure' || Boolean(label.size && label.grip?.length);
}

// A typed hold still waiting for its grip directions. Unsure holds need none.
export function missingGrip(label) {
  return Boolean(label?.type) && label.type !== 'unsure' && !label.grip?.length;
}

export function hasLabel(label) {
  return Boolean(label && (FIELD_NAMES.some(name => label[name]) || label.grip?.length || label.notes));
}

// Reading order: top row first, left to right, so the kickboard comes last.
export function holdOrder(holds) {
  return [...holds].sort((a, b) => (b.y - a.y) || (a.x - b.x)).map(hold => hold.id);
}

function csvField(value) {
  const text = value === undefined || value === null ? '' : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

// One row per hold, labeled or not, so the file joins straight onto hold_vocabulary.csv.
// Grip directions are separated by semicolons, e.g. "0;22.5".
export function toCsv(holds, labels) {
  const rows = holds.map(hold => {
    const label = labels[hold.id] || {};
    return [hold.id, hold.name, hold.x, hold.y, hold.set, Number(hold.kickboard),
      ...FIELD_NAMES.map(name => label[name] || ''), (label.grip || []).join(';'),
      label.notes || '', label.updated || ''];
  });
  return [COLUMNS, ...rows].map(row => row.map(csvField).join(',')).join('\n') + '\n';
}

export function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field); rows.push(row); row = []; field = '';
    } else field += c;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows;
}

// Reads labels back from an exported CSV. Unknown holds and invalid values are rejected
// rather than silently dropped, so a file from another board or schema is noticed.
export function fromCsv(text, holdIds) {
  const [header, ...rows] = parseCsv(text.replace(/^﻿/, ''));
  const column = Object.fromEntries((header || []).map((name, i) => [name.trim(), i]));
  if (column.hold_token === undefined) throw new Error('The file has no hold_token column.');
  const cell = (row, name) => column[name] === undefined ? '' : row[column[name]] || '';
  const known = new Set(holdIds);
  const labels = {};
  for (const row of rows) {
    if (row.length === 1 && row[0] === '') continue;
    const id = Number(row[column.hold_token]);
    if (!known.has(id)) throw new Error(`Hold ${row[column.hold_token]} is not on this board.`);
    const label = {};
    for (const name of FIELD_NAMES) {
      const value = cell(row, name);
      if (value && !FIELDS[name].options.some(option => option.value === value)) {
        throw new Error(`Hold ${id} has an unknown ${name} "${value}".`);
      }
      if (value) label[name] = value;
    }
    const grip = cell(row, 'grip');
    if (grip) {
      try { label.grip = normalizeGrip(LEGACY_GRIP[grip] ? grip : grip.split(';').filter(Boolean)); }
      catch (error) { throw new Error(`Hold ${id}: ${error.message}.`); }
    }
    if (cell(row, 'notes')) label.notes = cell(row, 'notes');
    if (cell(row, 'updated_at')) label.updated = cell(row, 'updated_at');
    if (hasLabel(label)) labels[id] = label;
  }
  return labels;
}
