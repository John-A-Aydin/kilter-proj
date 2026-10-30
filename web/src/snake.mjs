// Snake on the bolt-on holds. Set 1 is a full 17 × 19 lattice, so each hold is one
// cell, and each frame is a list of {id, role} holds sent like any other climb.
import { seededRandom } from './decoding.mjs';

// Well under the 135 same-color holds an API v2 controller can light.
export const MAX_SNAKE_HOLDS = 120;
export const DIRECTIONS = { up: [0, 1], down: [0, -1], left: [-1, 0], right: [1, 0] };
export const KEYS = { w: 'up', a: 'left', s: 'down', d: 'right',
  ArrowUp: 'up', ArrowLeft: 'left', ArrowDown: 'down', ArrowRight: 'right' };
const OPPOSITE = { up: 'down', down: 'up', left: 'right', right: 'left' };
// Head green, body cyan, food magenta: the start, hand, and finish role colors.
const HEAD = 2, BODY = 0, FOOD = 3;

export function snakeGrid(holds) {
  const lattice = holds.filter(hold => hold.set === 1);
  const key = value => Math.round(value * 1e6);
  const xs = [...new Set(lattice.map(hold => key(hold.x)))].sort((a, b) => a - b);
  const ys = [...new Set(lattice.map(hold => key(hold.y)))].sort((a, b) => a - b);
  const ids = new Array(xs.length * ys.length);
  for (const hold of lattice) ids[xs.indexOf(key(hold.x)) + ys.indexOf(key(hold.y)) * xs.length] = hold.id;
  if (lattice.length !== ids.length || ids.includes(undefined)) {
    throw new Error('The bolt-on holds do not form a full grid.');
  }
  return { columns: xs.length, rows: ys.length, ids };
}

export function newGame(grid, seed) {
  const x = Math.floor(grid.columns / 2), y = Math.floor(grid.rows / 3);
  const game = { grid, random: seededRandom(seed), direction: 'up', queue: [], score: 0,
    snake: [0, 1, 2].map(i => x + (y - i) * grid.columns), food: null, over: false, won: false };
  placeFood(game);
  return game;
}

// Two queued turns let a quick double tap (e.g. up, then left) land on consecutive moves.
export function turn(game, direction) {
  const last = game.queue.at(-1) ?? game.direction;
  if (game.over || game.queue.length >= 2 || direction === last || direction === OPPOSITE[last]) return false;
  game.queue.push(direction);
  return true;
}

export function step(game) {
  if (game.over) return game;
  if (game.queue.length) game.direction = game.queue.shift();
  const { columns, rows } = game.grid;
  const [dx, dy] = DIRECTIONS[game.direction];
  const x = game.snake[0] % columns + dx, y = Math.floor(game.snake[0] / columns) + dy;
  const head = x + y * columns;
  const eating = head === game.food;
  // The tail moves out of the way unless the snake is growing.
  const body = eating ? game.snake : game.snake.slice(0, -1);
  if (x < 0 || x >= columns || y < 0 || y >= rows || body.includes(head)) {
    game.over = true;
    return game;
  }
  game.snake = [head, ...body];
  if (eating) { game.score++; placeFood(game); }
  return game;
}

function placeFood(game) {
  const taken = new Set(game.snake);
  const free = game.grid.ids.map((_, cell) => cell).filter(cell => !taken.has(cell));
  game.food = free.length ? free[Math.floor(game.random() * free.length)] : null;
  if (!free.length) game.over = game.won = true;
}

// A crashed snake turns magenta. Past the power budget, the tail end goes dark.
export function snakeFrame(game) {
  const { ids } = game.grid;
  const crashed = game.over && !game.won;
  const holds = game.snake.map((cell, i) => ({ id: ids[cell], role: crashed ? FOOD : i ? BODY : HEAD }));
  if (game.food !== null && !game.over) holds.unshift({ id: ids[game.food], role: FOOD });
  return holds.slice(0, MAX_SNAKE_HOLDS).sort((a, b) => a.id - b.id);
}
