import { ROLE_NAMES, validateRequest } from './decoding.mjs';
import { createBoard } from './board.mjs';
import { BOARD_COLORS } from './board-config.mjs';
import { KilterBluetooth, climbSignature, validateLedMap } from './bluetooth.mjs';
import { PATTERNS, showFrame } from './lightshow.mjs';
import { KEYS, newGame, snakeFrame, snakeGrid, step, turn } from './snake.mjs';

const $ = id => document.getElementById(id);
const COLORS = BOARD_COLORS;
const selected = new Map();
let generated = [];
let role = 2;
let busy = false;
let ready = false;
let manifest;
let board;
let activeRequest;
let activeId = 0;
let lastResult;
let zoom = 100;
let ble;
let bleState;
let resendToBoard = false;
let tab = 'climb';
let pattern = 'wave';
let show = null;
let snake = null;
let snakeBest = 0;
try { snakeBest = Number(localStorage.getItem('snake-best')) || 0; } catch { /* Storage can be blocked. */ }
const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });

for (let grade = 0; grade <= 16; grade++) {
  const option = new Option(`V${grade}`, grade, false, grade === 5);
  $('grade').append(option);
}

function setStatus(text, state = 'ready') {
  $('model-status-text').textContent = text;
  $('model-status').className = `model-status ${state}`;
}

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll('[data-role], .hold-chip, #clear, #grade, #angle, #temperature, #seed, #vary-seed, #generate, [role=tab]').forEach(element => {
    element.disabled = value;
  });
  $('clear').disabled = value || selected.size === 0;
  $('generate').disabled = value || !ready || !board;
  $('generate-label').textContent = value ? 'Growing your climb…' : ready && board ? 'Complete climb' : 'Loading model…';
  $('stop').hidden = !value;
  $('board-scroll').classList.toggle('busy', value);
  renderBoardLink();
}

function allHolds() {
  const holds = new Map(generated.map(hold => [hold.id, hold]));
  for (const [id, role] of selected) holds.set(id, { id, role, locked: true });
  return [...holds.values()].sort((a, b) => a.id - b.id);
}

function updateBoard() {
  board?.update(allHolds(), selected);
  renderBoardLink();
}

function renderBoardLink() {
  if (!bleState) return;
  const { phase, connected, available, apiLevel, deviceName, lastSent } = bleState;
  const holds = allHolds();
  const synced = connected && lastSent !== null && lastSent === climbSignature(holds);
  const pill = $('ble-pill');
  const playing = Boolean(show || snake?.running);
  pill.textContent = playing && connected ? (show ? 'Light show' : 'Snake')
    : { disconnected: 'Offline', connecting: 'Connecting', sending: 'Sending' }[phase] ?? (synced ? 'On the wall' : 'Connected');
  pill.className = `link-pill ${playing && connected ? 'busy' : phase === 'connected' ? (synced ? 'synced' : 'connected') : phase === 'disconnected' ? '' : 'busy'}`;
  const linux = /Linux/.test(navigator.userAgent) && !/Android/.test(navigator.userAgent);
  $('ble-message').textContent = available ? bleState.message
    : 'Web Bluetooth needs Chrome or Edge on desktop or Android, opened from localhost or HTTPS.'
      + (linux ? ' On Linux, also enable chrome://flags/#enable-experimental-web-platform-features.' : '');
  $('ble-connect').textContent = phase === 'connecting' ? 'Connecting…' : connected ? 'Disconnect' : 'Connect board';
  // During a show or game the link is busy between every frame, so disconnecting stays possible.
  $('ble-connect').disabled = !available || phase === 'connecting' || (phase === 'sending' && !playing);
  $('ble-send').textContent = synced ? 'Sent ✓' : 'Send climb';
  $('ble-send').disabled = phase !== 'connected' || busy || !holds.length;
  $('ble-clear').disabled = playing ? !connected : phase !== 'connected';
  $('ble-protocol').disabled = phase === 'connecting' || phase === 'sending' || playing;
  $('ble-protocol').options[0].textContent = connected ? `Auto (API v${apiLevel} from ${deviceName})` : 'Auto';
}

async function sendToBoard(holds = allHolds()) {
  if (ble.phase === 'sending') { resendToBoard = true; return; }
  try { await ble.send(holds); } catch { /* KilterBluetooth reports the failure in its status. */ }
  if (resendToBoard) {
    resendToBoard = false;
    if (ble.phase === 'connected' && !busy) sendToBoard();
  }
}

function updateSelections() {
  $('selected-count').textContent = selected.size;
  $('clear').disabled = busy || !selected.size;
  const container = $('selected-holds');
  container.replaceChildren();
  if (!selected.size) {
    const empty = document.createElement('p');
    empty.className = 'empty-selection';
    empty.textContent = 'Pick your start holds and maybe some feet. Or pick nothing and generate from scratch.';
    container.append(empty);
  }
  for (const [id, role] of [...selected].sort((a, b) => a[0] - b[0])) {
    const chip = document.createElement('button');
    chip.className = 'hold-chip';
    chip.style.setProperty('--color', COLORS[role]);
    chip.setAttribute('aria-label', `Remove ${ROLE_NAMES[role]} hold ${id}`);
    chip.disabled = busy;
    const dot = document.createElement('i');
    const name = document.createElement('span');
    name.textContent = `${ROLE_NAMES[role]} · ${id}`;
    const cross = document.createElement('span');
    cross.textContent = '×';
    chip.append(dot, name, cross);
    chip.onclick = () => { selected.delete(id); resetResult(); updateSelections(); updateBoard(); };
    container.append(chip);
  }
}

function resetResult() {
  generated = [];
  lastResult = null;
  $('result-actions').hidden = true;
  $('error').hidden = true;
  $('generation-status').textContent = 'Your selected holds stay in every completed climb.';
}

function pickHold(event) {
  if (busy || !manifest || tab !== 'climb') return;
  const id = Number(event.currentTarget.id.slice(5));
  if (selected.get(id) === role) selected.delete(id);
  else {
    const next = new Map(selected).set(id, role);
    try {
      validateRequest({ grade: 0, angle: 0, temperature: 1, seed: 0,
        selected: [...next].map(([id, role]) => ({ id, role })) }, manifest);
    } catch (error) {
      showError(error.message);
      return;
    }
    selected.set(id, role);
  }
  resetResult();
  updateSelections();
  updateBoard();
}

async function draw(manifest) {
  board = await createBoard($('svg-climb'), manifest, pickHold, showError);
  updateBoard();
  setBusy(busy);
  renderShow();
  renderSnake();
}

function setTab(name) {
  if (name === tab || busy) return;
  if (tab === 'show') stopShow();
  if (tab === 'snake') stopSnake();
  tab = name;
  $('controls').dataset.tab = name;
  document.querySelectorAll('[role=tab]').forEach(button => {
    const active = button.dataset.tab === name;
    button.setAttribute('aria-selected', String(active));
    button.tabIndex = active ? 0 : -1;
    $(button.getAttribute('aria-controls')).hidden = !active;
  });
  $('board-scroll').classList.toggle('show-mode', name !== 'climb');
  if (name === 'climb') updateBoard();
  else board?.update([], new Map());
  renderShow();
  renderSnake();
}

function renderShow() {
  const running = Boolean(show);
  $('show-start').disabled = !board;
  $('show-start-label').textContent = !board ? 'Loading board…' : running ? 'Stop light show' : 'Start light show';
  $('pattern-description').textContent = PATTERNS[pattern].description;
  $('show-speed-value').textContent = `${$('show-speed').value} fps`;
  if (!running) {
    $('show-status').textContent = ble?.connected ? `Ready to play on ${bleState.deviceName}.`
      : 'Preview only until a board is connected.';
  }
  renderBoardLink();
}

async function runShow() {
  const run = show = { frame: 0, sending: null };
  renderShow();
  while (show === run) {
    const started = performance.now();
    const holds = showFrame(pattern, manifest.holds, run.frame++);
    board.update(holds, new Map(holds.map(hold => [hold.id, hold.role])));
    if (ble?.phase === 'connected') {
      run.sending = ble.send(holds, { message: `Light show · ${PATTERNS[pattern].name}` });
      try {
        await run.sending;
      } catch {
        // A dropped board leaves the preview running; any other failure stops the show.
        // KilterBluetooth already reports why in its status.
        if (show === run && ble.phase !== 'disconnected') { stopShow({ clear: false }); break; }
      }
    }
    if (show !== run) break;
    $('show-status').textContent = `${ble?.connected ? 'Playing on your wall' : 'Previewing on screen'} · frame ${run.frame} · ${holds.length} holds`;
    const interval = 1000 / Number($('show-speed').value);
    await new Promise(resolve => setTimeout(resolve, Math.max(0, interval - (performance.now() - started))));
  }
}

async function stopShow({ clear = true } = {}) {
  const run = show;
  if (!run) return;
  show = null;
  renderShow();
  if (tab === 'show') board?.update([], new Map());
  await run.sending?.catch(() => {});
  if (clear && ble?.phase === 'connected') {
    try { await ble.send([], { message: 'Light show stopped. Board LEDs cleared.' }); } catch { /* Reported in status. */ }
  }
}

function renderSnake() {
  const running = Boolean(snake?.running);
  $('snake-start').disabled = !board;
  $('snake-start-label').textContent = !board ? 'Loading board…' : running ? 'Stop game' : snake ? 'Play again' : 'Start snake';
  $('snake-score').textContent = snake?.game.score ?? 0;
  $('snake-best').textContent = snakeBest;
  $('snake-speed-value').textContent = `${$('snake-speed').value} moves/s`;
  renderBoardLink();
}

function drawSnake(run) {
  const holds = snakeFrame(run.game);
  board.update(holds, new Map(holds.map(hold => [hold.id, hold.role])));
  return holds;
}

// Each move waits for the wall, so the board never falls behind the game.
async function sendSnake(run, holds, message) {
  if (ble?.phase !== 'connected') return true;
  run.sending = ble.send(holds, { message });
  try {
    await run.sending;
    return true;
  } catch {
    // A dropped board leaves the game running on screen; KilterBluetooth reports any failure.
    return snake !== run || ble.phase === 'disconnected';
  }
}

async function runSnake(firstTurn) {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  // The new run takes over at once, so further key presses steer it instead of restarting.
  const previous = snake;
  const run = snake = { game: newGame(snakeGrid(manifest.holds), values[0]), running: true, sending: null };
  if (firstTurn) turn(run.game, firstTurn);
  $('snake-status').textContent = 'Go! Steer with W A S D.';
  renderSnake();
  await previous?.sending?.catch(() => {});
  if (snake !== run) return;
  let moved = performance.now();
  if (!await sendSnake(run, drawSnake(run), 'Snake · playing')) { stopSnake({ clear: false }); return; }
  while (snake === run) {
    const interval = 1000 / Number($('snake-speed').value);
    await new Promise(resolve => setTimeout(resolve, Math.max(0, interval - (performance.now() - moved))));
    if (snake !== run) return;
    moved = performance.now();
    const { game } = run;
    step(game);
    const holds = drawSnake(run);
    if (game.score > snakeBest) {
      snakeBest = game.score;
      try { localStorage.setItem('snake-best', snakeBest); } catch { /* Storage can be blocked. */ }
    }
    if (game.over) {
      run.running = false;
      $('snake-status').textContent = `${game.won ? 'You filled the board!' : 'Game over.'} Score ${game.score}. Press a W A S D key to play again.`;
    }
    renderSnake();
    const sent = await sendSnake(run, holds, game.over ? `Snake · game over, score ${game.score}` : 'Snake · playing');
    if (snake !== run || game.over) return;
    if (!sent) { stopSnake({ clear: false }); return; }
  }
}

async function stopSnake({ clear = true } = {}) {
  const run = snake;
  if (!run) return;
  const wasRunning = run.running;
  snake = null;
  if (wasRunning) $('snake-status').textContent = `Stopped. Score ${run.game.score}.`;
  renderSnake();
  if (tab === 'snake' && clear) board?.update([], new Map());
  await run.sending?.catch(() => {});
  if (clear && ble?.phase === 'connected') {
    try { await ble.send([], { message: 'Snake stopped. Board LEDs cleared.' }); } catch { /* Reported in status. */ }
  }
}

document.querySelectorAll('[data-role]').forEach(button => button.onclick = () => {
  role = Number(button.dataset.role);
  document.querySelectorAll('[data-role]').forEach(item => {
    item.classList.toggle('active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
});
$('clear').onclick = () => { selected.clear(); resetResult(); updateSelections(); updateBoard(); };
$('angle').oninput = () => {
  $('angle-value').textContent = `${$('angle').value}°`;
  $('angle-note').hidden = Number($('angle').value) <= 70;
  resetResult(); updateBoard();
};
$('grade').onchange = () => { resetResult(); updateBoard(); };
$('temperature').oninput = () => { $('temperature-value').textContent = Number($('temperature').value).toFixed(2); };
function changeZoom(change) {
  zoom = Math.max(100, Math.min(250, zoom + change));
  $('zoom-value').textContent = `${zoom}%`;
  $('svg-climb').style.width = `${zoom}%`;
  $('board-scroll').classList.toggle('zoomed', zoom !== 100);
}
$('zoom-in').onclick = () => changeZoom(25);
$('zoom-out').onclick = () => changeZoom(-25);

$('ble-connect').onclick = () => {
  // connect() must start synchronously in this click so the browser shows its device picker.
  if (ble.phase === 'disconnected') ble.connect().catch(() => {});
  else ble.disconnect();
};
$('ble-send').onclick = () => sendToBoard();
$('ble-clear').onclick = () => (show ? stopShow() : snake ? stopSnake() : sendToBoard([]));
$('ble-protocol').onchange = () => {
  try { ble.setProtocol($('ble-protocol').value ? Number($('ble-protocol').value) : null); }
  catch (error) { showError(error.message); }
};

const tabs = [...document.querySelectorAll('[role=tab]')];
tabs.forEach((button, index) => {
  button.onclick = () => setTab(button.dataset.tab);
  button.onkeydown = event => {
    // While snake is running, the arrow keys steer instead.
    if ((event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') || snake?.running) return;
    const next = tabs[(index + (event.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
    next.focus();
    setTab(next.dataset.tab);
  };
});
document.querySelectorAll('[data-pattern]').forEach(button => button.onclick = () => {
  pattern = button.dataset.pattern;
  document.querySelectorAll('[data-pattern]').forEach(item => {
    item.classList.toggle('active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
  renderShow();
});
$('show-speed').oninput = () => renderShow();
$('show-start').onclick = () => (show ? stopShow() : runShow());
$('snake-speed').oninput = () => renderSnake();
$('snake-start').onclick = () => (snake?.running ? stopSnake() : runSnake());
document.addEventListener('keydown', event => {
  if (tab !== 'snake' || !board || event.ctrlKey || event.metaKey || event.altKey) return;
  const direction = KEYS[event.key.length === 1 ? event.key.toLowerCase() : event.key];
  if (!direction) return;
  if (snake?.running) {
    event.preventDefault();
    turn(snake.game, direction);
  } else if (event.key.length === 1 && !event.target.closest?.('select, input[type=number]')) {
    // Any W A S D key starts a game, already heading that way.
    event.preventDefault();
    runSnake(direction);
  }
});

$('generate').onclick = () => {
  resetResult();
  if ($('vary-seed').checked) {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    $('seed').value = values[0];
  }
  activeRequest = { grade: Number($('grade').value), angle: Number($('angle').value),
    temperature: Number($('temperature').value), seed: Number($('seed').value),
    selected: [...selected].map(([id, role]) => ({ id, role })) };
  setBusy(true);
  updateBoard();
  setStatus('Generating', 'running');
  $('generation-status').textContent = selected.size ? 'The model is adding holds around your picks…' : 'The model is setting a climb from scratch…';
  worker.postMessage({ type: 'generate', id: ++activeId, request: activeRequest });
};
$('stop').onclick = () => { worker.postMessage({ type: 'cancel' }); $('stop').textContent = 'Stopping…'; };
$('keep-generated').onclick = () => {
  for (const hold of allHolds()) selected.set(hold.id, hold.role);
  resetResult(); updateSelections(); updateBoard();
};
$('export').onclick = () => {
  if (!lastResult) return;
  const geometry = new Map(manifest.holds.map(hold => [hold.id, hold]));
  const output = { schema_version: 1, v_grade: activeRequest.grade, grade_hueco: `V${activeRequest.grade}`,
    angle: activeRequest.angle, normalized_angle: activeRequest.angle / 90,
    seed: activeRequest.seed, temperature: activeRequest.temperature,
    ended_with_eos: lastResult.endedWithEos, selected_holds: activeRequest.selected,
    model: { name: manifest.name, checkpoint_sha256: manifest.checkpoint_sha256 },
    frames: [allHolds().map(hold => ({ hold_token: hold.id, hold_use_type: hold.role,
      hold_x: geometry.get(hold.id).x, hold_y: geometry.get(hold.id).y,
      on_kickboard: Number(geometry.get(hold.id).kickboard) }))],
  };
  const blob = new Blob([JSON.stringify(output, null, 2) + '\n'], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = `kilter-V${output.v_grade}-${output.angle}deg-${output.seed}.json`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

worker.onmessage = ({ data }) => {
  if (data.type === 'ready') {
    ready = true;
    setStatus('Model ready');
    setBusy(false);
  } else if (data.type === 'step' && data.id === activeId) {
    generated = data.holds;
    updateBoard();
    const added = generated.filter(hold => !selected.has(hold.id)).length;
    $('generation-status').textContent = `${added} ${added === 1 ? 'hold' : 'holds'} added · keeping your ${selected.size} picks`;
  } else if (data.type === 'done' && data.id === activeId) {
    generated = data.holds;
    lastResult = data;
    setBusy(false);
    $('stop').textContent = 'Stop generation';
    setStatus('Model ready');
    updateBoard();
    $('result-actions').hidden = !allHolds().length;
    const added = generated.filter(hold => !selected.has(hold.id)).length;
    if ($('ble-auto').checked && ble?.connected && allHolds().length) sendToBoard();
    $('generation-status').textContent = `${data.cancelled ? 'Stopped' : 'Complete'} · ${added} added · ${selected.size} picks preserved · V${activeRequest.grade} / ${activeRequest.angle}°`;
  } else if (data.type === 'error') {
    if (data.id && data.id !== activeId) return;
    showError(data.message);
    setBusy(false);
    setStatus(ready ? 'Model ready' : 'Model unavailable', ready ? 'ready' : '');
  }
};
worker.onerror = event => { showError(`Inference could not start: ${event.message}`); setBusy(false); };

async function setUpBluetooth(manifest) {
  const response = await fetch('./vendor/boardsesh/kilter-original-12x12-leds.json');
  if (!response.ok) throw new Error('The Kilter LED map could not be loaded.');
  const mapping = await response.json();
  validateLedMap(mapping, manifest.holds);
  ble = new KilterBluetooth(mapping, { onChange: state => {
    bleState = state;
    if (show || snake?.running) renderBoardLink(); else { renderShow(); renderSnake(); }
  } });
  ble.notify();
}

fetch('./model/manifest.json').then(response => {
  if (!response.ok) throw new Error('Model metadata could not be loaded.');
  return response.json();
}).then(async value => {
  manifest = value;
  $('model-info').textContent = `${manifest.name} · ${manifest.parameters.toLocaleString()} parameters. Trained on Kilter Original 12 × 12 climbs to complete a climb from nothing, from start holds plus some feet, or from any picks, conditioned on numeric V grade and angle / 90. All inference runs on your device.`;
  await Promise.all([draw(manifest), setUpBluetooth(manifest)]);
}).catch(error => { showError(error.message); });
