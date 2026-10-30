import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
// The browser modules import the compiled Boardsesh protocol from ../vendor/, which only exists in dist/.
import { DEVICE_OPTIONS, KilterBluetooth, climbSignature, encodeClimb, validateLedMap } from '../dist/src/bluetooth.mjs';

const manifest = JSON.parse(readFileSync(new URL('../public/model/manifest.json', import.meta.url)));
const mapping = JSON.parse(readFileSync(new URL('../public/vendor/boardsesh/kilter-original-12x12-leds.json', import.meta.url)));
// Model roles are hand, foot, start, finish; these are Kilter's LED colors after protocol quantization.
const V3_COLORS = [0b00011111, 0b11110100, 0b00011100, 0b11100011];

// Unwrap Aurora frames ([1, length, checksum, 2, ...body, 3]) into their command bodies.
function decodeFrames(bytes) {
  const bodies = [];
  for (let i = 0; i < bytes.length;) {
    assert.equal(bytes[i], 1);
    const length = bytes[i + 1];
    const body = [...bytes.slice(i + 4, i + 4 + length)];
    assert.equal(bytes[i + 3], 2);
    assert.equal(bytes[i + 4 + length], 3);
    assert.equal(bytes[i + 2], body.reduce((sum, value) => (sum + value) & 255, 0) ^ 255);
    bodies.push(body);
    i += length + 5;
  }
  return bodies;
}

function decodeV3(bytes) {
  const leds = new Map();
  for (const body of decodeFrames(bytes)) {
    for (let i = 1; i < body.length; i += 3) leds.set(body[i] | body[i + 1] << 8, body[i + 2]);
  }
  return leds;
}

function fakeBluetooth(name) {
  const writes = [];
  const listeners = new Map();
  const characteristic = {
    properties: { writeWithoutResponse: true },
    async writeValueWithoutResponse(value) { writes.push({ value: [...value], withResponse: false }); },
    async writeValueWithResponse(value) { writes.push({ value: [...value], withResponse: true }); },
  };
  const device = {
    id: 'fake', name,
    gatt: {
      connected: false,
      async connect() { this.connected = true; return this; },
      disconnect() { this.connected = false; },
      async getPrimaryService() { return { getCharacteristic: async () => characteristic }; },
    },
    addEventListener: (type, listener) => listeners.set(type, listener),
    removeEventListener: type => listeners.delete(type),
  };
  const bluetooth = { requested: [], async requestDevice(options) { this.requested.push(options); return device; } };
  return { bluetooth, device, writes, drop: () => { device.gatt.connected = false; listeners.get('gattserverdisconnected')?.(); } };
}

const climb = manifest.holds.slice(100, 112).map((hold, i) => ({ id: hold.id, role: i % 4 }));

test('the LED map covers every Kilter Original 12x12 model hold exactly once', () => {
  validateLedMap(mapping, manifest.holds);
  assert.throws(() => validateLedMap({ ...mapping, size_id: 7 }, manifest.holds), /Original 12×12/);
  const [first, second] = manifest.holds;
  const swapped = { ...mapping, positions: { ...mapping.positions, [first.id]: mapping.positions[second.id] } };
  assert.throws(() => validateLedMap(swapped, manifest.holds), /476/);
});

test('API v3 packets light each hold at its LED position in its role color', () => {
  const { packet, ledsWritten } = encodeClimb(climb, mapping, 3);
  assert.equal(ledsWritten, climb.length);
  const leds = decodeV3(packet);
  assert.equal(leds.size, climb.length);
  for (const hold of climb) assert.equal(leds.get(mapping.positions[hold.id]), V3_COLORS[hold.role], `hold ${hold.id}`);
});

test('API v2 packets use 10-bit positions and full-brightness role colors', () => {
  const leds = new Map();
  for (const body of decodeFrames(encodeClimb(climb, mapping, 2).packet)) {
    assert.ok([77, 78, 79, 80].includes(body[0]));
    for (let i = 1; i < body.length; i += 2) leds.set(body[i] | (body[i + 1] & 3) << 8, body[i + 1] >> 2);
  }
  const colors = [0b001111, 0b111000, 0b001100, 0b110011];
  for (const hold of climb) assert.equal(leds.get(mapping.positions[hold.id]), colors[hold.role], `hold ${hold.id}`);
});

test('large climbs span multiple first/middle/last frames and an empty climb clears the board', () => {
  const big = manifest.holds.slice(0, 120).map(hold => ({ id: hold.id, role: 0 }));
  const bodies = decodeFrames(encodeClimb(big, mapping, 3).packet);
  assert.deepEqual(bodies.map(body => body[0]), [82, 83]);
  assert.equal(decodeV3(encodeClimb(big, mapping, 3).packet).size, 120);
  assert.deepEqual(decodeFrames(encodeClimb([], mapping, 3).packet), [[84]]);
  assert.deepEqual(decodeFrames(encodeClimb([], mapping, 2).packet), [[80]]);
});

test('unknown, duplicate, or recolored holds are rejected before anything is sent', () => {
  assert.throws(() => encodeClimb([{ id: 1, role: 0 }], mapping, 3), /unknown hold/);
  assert.throws(() => encodeClimb([climb[0], climb[0]], mapping, 3), /duplicate/);
  assert.throws(() => encodeClimb([{ ...climb[0], role: 4 }], mapping, 3), /invalid role/);
});

test('connects, sends 20-byte chunks, and reports the climb as on the wall', async () => {
  const fake = fakeBluetooth('Kilter Board#751737@3');
  const states = [];
  const ble = new KilterBluetooth(mapping, { bluetooth: fake.bluetooth, onChange: state => states.push(state) });
  await ble.connect();
  assert.deepEqual(fake.bluetooth.requested, [DEVICE_OPTIONS]);
  assert.equal(ble.apiLevel, 3);
  await ble.send(climb);
  assert.ok(fake.writes.every(write => write.value.length <= 20 && !write.withResponse));
  const leds = decodeV3(Uint8Array.from(fake.writes.flatMap(write => write.value)));
  for (const hold of climb) assert.equal(leds.get(mapping.positions[hold.id]), V3_COLORS[hold.role]);
  assert.equal(states.at(-1).phase, 'connected');
  assert.equal(states.at(-1).lastSent, climbSignature([...climb].reverse()));
  assert.match(states.at(-1).message, /12 holds/);
});

test('bare-name Kilter boxes default to API v2 with acknowledged writes, unless overridden', async () => {
  const fake = fakeBluetooth('Kilter Board');
  const ble = new KilterBluetooth(mapping, { bluetooth: fake.bluetooth });
  await ble.connect();
  assert.equal(ble.apiLevel, 2);
  await ble.send(climb.slice(0, 2));
  assert.ok(fake.writes.every(write => write.withResponse));
  assert.equal(decodeFrames(fake.writes.flatMap(write => write.value))[0][0], 80);
  ble.setProtocol(3);
  fake.writes.length = 0;
  await ble.send(climb.slice(0, 2));
  assert.equal(decodeFrames(fake.writes.flatMap(write => write.value))[0][0], 84);
});

test('rejects other Aurora boards and recovers cleanly when the board drops', async () => {
  const tension = fakeBluetooth('Tension Board#1@3');
  const other = new KilterBluetooth(mapping, { bluetooth: tension.bluetooth });
  await assert.rejects(other.connect(), /Select a Kilter board/);
  assert.equal(other.phase, 'disconnected');

  const fake = fakeBluetooth('Kilter Board#751737@3');
  const ble = new KilterBluetooth(mapping, { bluetooth: fake.bluetooth });
  await ble.connect();
  fake.drop();
  assert.equal(ble.phase, 'disconnected');
  assert.match(ble.message, /disconnected/);
  await assert.rejects(ble.send(climb), /Connect to the board/);
  await ble.connect();
  assert.equal(ble.connected, true);
});

test('an encoding failure is reported in the connection status', async () => {
  const fake = fakeBluetooth('Kilter Board#751737@3');
  let state;
  const ble = new KilterBluetooth(mapping, { bluetooth: fake.bluetooth, onChange: value => { state = value; } });
  await ble.connect();
  await assert.rejects(ble.send([{ id: 1, role: 0 }]));
  assert.match(state.message, /unknown hold/);
  assert.equal(state.phase, 'connected');
  assert.equal(fake.writes.length, 0);
});
