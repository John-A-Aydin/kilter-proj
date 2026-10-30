import { BOARD_COLORS, BOARD_WIDTH, BOARD_HEIGHT, buildBoardConfig, framesForHolds, outlinePath } from './board-config.mjs';
import { ROLE_NAMES } from './decoding.mjs';

const NS = 'http://www.w3.org/2000/svg';
const ASSETS = new URL('../vendor/boardsesh/', import.meta.url);

// Image layering and placement projection follow Boardsesh's BoardRenderer and
// BoardLitupHolds. Traced paths give each hold an accurate, keyboard-accessible
// hit area; the actual Boardsesh WASM renderer supplies the Aura overlay.
export async function createBoard(svg, manifest, onHoldClick, onError) {
  const response = await fetch(new URL('kilter-original-12x12.json', ASSETS));
  if (!response.ok) throw new Error('Could not load the Kilter hold outlines.');
  const config = buildBoardConfig(manifest, await response.json());
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${BOARD_WIDTH} ${BOARD_HEIGHT}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.dataset.renderer = 'boardsesh-aura';
  for (const filename of ['45-1.webp', '46-1.webp']) {
    const image = document.createElementNS(NS, 'image');
    image.setAttribute('href', new URL(filename, ASSETS));
    image.setAttribute('width', BOARD_WIDTH);
    image.setAttribute('height', BOARD_HEIGHT);
    image.setAttribute('preserveAspectRatio', 'none');
    image.classList.add('board-art');
    svg.append(image);
  }
  const overlay = document.createElementNS(NS, 'image');
  overlay.id = 'boardsesh-aura';
  overlay.setAttribute('width', BOARD_WIDTH);
  overlay.setAttribute('height', BOARD_HEIGHT);
  overlay.setAttribute('pointer-events', 'none');
  svg.append(overlay);

  const paths = new Map();
  for (const hold of config.holds) {
    const path = document.createElementNS(NS, 'path');
    path.id = `hold-${hold.id}`;
    path.classList.add('hold');
    path.setAttribute('d', outlinePath(hold));
    path.setAttribute('fill', 'transparent');
    path.setAttribute('stroke', 'transparent');
    path.setAttribute('tabindex', '0');
    path.setAttribute('role', 'button');
    path.onclick = onHoldClick;
    path.onkeydown = event => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onHoldClick(event); }
    };
    paths.set(hold.id, path);
    svg.append(path);
  }
  const worker = new Worker(new URL('./board-worker.mjs', import.meta.url), { type: 'module' });
  let version = 0;
  let currentUrl;
  let resolveReady, rejectReady;
  const ready = new Promise((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  worker.onmessage = ({ data }) => {
    if (data.type === 'ready') resolveReady();
    else if (data.type === 'error') {
      const error = new Error(`Board rendering failed: ${data.message}`);
      rejectReady(error); onError(error.message);
    } else if (data.type === 'rendered' && data.id === version) {
      const oldUrl = currentUrl;
      currentUrl = URL.createObjectURL(data.blob);
      overlay.onload = () => {
        if (data.id !== version) return;
        svg.dataset.renderedVersion = data.id;
        svg.dataset.renderPending = 'false';
      };
      overlay.setAttribute('href', currentUrl);
      if (oldUrl) URL.revokeObjectURL(oldUrl);
    }
  };
  worker.onerror = event => { rejectReady(new Error(event.message)); onError(event.message); };
  worker.postMessage({ type: 'init', config });
  await ready;
  const lookup = new Map(manifest.holds.map(hold => [hold.id, hold]));
  let lastFrames = null;
  function update(holds, selected) {
    const active = new Map(holds.map(hold => [hold.id, hold]));
    for (const [id, path] of paths) {
      const entry = active.get(id);
      const locked = selected.has(id);
      path.classList.toggle('selected', locked);
      path.classList.toggle('generated', Boolean(entry && !locked));
      path.setAttribute('stroke', entry ? BOARD_COLORS[entry.role] : 'transparent');
      path.setAttribute('stroke-width', locked ? '2.5' : '1.5');
      path.setAttribute('stroke-dasharray', entry && !locked ? '4 4' : 'none');
      path.setAttribute('aria-pressed', String(locked));
      path.setAttribute('aria-label', `Hold ${id}, ${lookup.get(id).name}${entry ? `, ${ROLE_NAMES[entry.role]}${locked ? ', locked' : ', generated'}` : ''}`);
    }
    const frames = framesForHolds(holds);
    if (frames === lastFrames) return;
    lastFrames = frames;
    svg.dataset.renderPending = 'true';
    svg.dataset.activeHolds = holds.length;
    worker.postMessage({ type: 'render', id: ++version, frames });
  }
  return { update, dispose() { worker.terminate(); if (currentUrl) URL.revokeObjectURL(currentUrl); } };
}
