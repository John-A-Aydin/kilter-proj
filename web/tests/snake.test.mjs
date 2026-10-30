import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { MAX_SNAKE_HOLDS, newGame, snakeFrame, snakeGrid, step, turn } from '../src/snake.mjs';
// The browser modules import the compiled Boardsesh protocol from ../vendor/, which only exists in dist/.
import { encodeClimb } from '../dist/src/bluetooth.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const mapping = JSON.parse(readFileSync(new URL('../public/vendor/boardsesh/kilter-original-12x12-leds.json', import.meta.url)));
const grid = snakeGrid(manifest.holds);
const geometry = new Map(manifest.holds.map(hold => [hold.id, hold]));

test('the bolt-on holds form a 17 × 19 grid in board order', () => {
  assert.equal(grid.columns, 17);
  assert.equal(grid.rows, 19);
  assert.equal(new Set(grid.ids).size, 323);
  for (let cell = 0; cell < grid.ids.length; cell++) {
    const hold = geometry.get(grid.ids[cell]);
    if (cell % grid.columns) assert.ok(hold.x > geometry.get(grid.ids[cell - 1]).x);
    if (cell >= grid.columns) assert.ok(hold.y > geometry.get(grid.ids[cell - grid.columns]).y);
  }
});

test('the snake moves, turns, grows on food, and ignores reversing', () => {
  const game = newGame(grid, 7);
  const head = game.snake[0];
  assert.equal(turn(game, 'down'), false);
  step(game);
  assert.equal(game.snake[0], head + grid.columns);
  assert.equal(game.snake.length, 3);
  assert.ok(turn(game, 'left'));
  assert.ok(turn(game, 'down'));
  assert.equal(turn(game, 'right'), false);
  step(game); step(game);
  assert.equal(game.snake[0], head + grid.columns - 1 - grid.columns);
  game.food = game.snake[0] - grid.columns;
  step(game);
  assert.equal(game.score, 1);
  assert.equal(game.snake.length, 4);
  assert.ok(game.food !== null && !game.snake.includes(game.food));
});

test('hitting the edge or the tail ends the game', () => {
  const wall = newGame(grid, 1);
  while (!wall.over) step(wall);
  assert.equal(Math.floor(wall.snake[0] / grid.columns), grid.rows - 1);
  // A hook heading up at (6, 2): the cell above holds its tail, which moves out of the way in time.
  const tail = newGame(grid, 2);
  tail.food = 0;
  tail.snake = [40, 23, 24, 41, 58, 57];
  step(tail);
  assert.equal(tail.over, false);
  // One segment longer, the tail is still there when the head arrives.
  tail.snake = [40, 23, 24, 41, 58, 57, 56];
  step(tail);
  assert.equal(tail.over, true);
  assert.ok(snakeFrame(tail).every(hold => hold.role === 3));
});

test('every frame is an encodable climb, even with a snake past the power budget', () => {
  const game = newGame(grid, 3);
  const frame = snakeFrame(game);
  assert.deepEqual(frame.map(hold => hold.role).sort(), [0, 0, 2, 3]);
  game.snake = grid.ids.map((_, cell) => cell).slice(0, 200);
  game.food = 250;
  const long = snakeFrame(game);
  assert.equal(long.length, MAX_SNAKE_HOLDS);
  assert.ok(long.some(hold => hold.role === 2) && long.some(hold => hold.role === 3));
  for (const holds of [frame, long]) {
    for (const api of [2, 3]) assert.equal(encodeClimb(holds, mapping, api).ledsWritten, holds.length);
  }
});
