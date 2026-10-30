import { getAuroraBluetoothPacket, parseApiLevel, parseBoardTypeFromDeviceName, isKilterBuiltBox }
  from '../vendor/boardsesh/bluetooth/aurora.mjs';
import { AURORA_ADVERTISED_SERVICE_UUID, UART_SERVICE_UUID, splitMessages }
  from '../vendor/boardsesh/bluetooth/transport.mjs';
import { getUartCharacteristic, writeCharacteristicSeries }
  from '../vendor/boardsesh/bluetooth/web-transport.mjs';
import { framesForHolds } from './board-config.mjs';

export const DEVICE_OPTIONS = {
  filters: [{ services: [AURORA_ADVERTISED_SERVICE_UUID] }, { namePrefix: 'Kilter' }],
  optionalServices: [UART_SERVICE_UUID],
};

export function validateLedMap(mapping, holds) {
  if (mapping.board !== 'kilter' || mapping.layout_id !== 1 || mapping.size_id !== 10
      || mapping.set_ids?.join(',') !== '1,20' || holds.length !== 476) {
    throw new Error('Bluetooth requires the Kilter Original 12×12 + kickboard layout.');
  }
  const positions = holds.map(hold => mapping.positions[hold.id]);
  if (positions.some(position => !Number.isInteger(position) || position < 0 || position > 1023)
      || new Set(positions).size !== 476 || Object.keys(mapping.positions).length !== 476) {
    throw new Error('The controller LED map does not match all 476 model holds.');
  }
}

export function encodeClimb(holds, mapping, apiLevel) {
  const seen = new Set();
  for (const hold of holds) {
    if (!Number.isInteger(hold.id) || !Number.isInteger(hold.role) || hold.role < 0 || hold.role > 3
        || !Object.hasOwn(mapping.positions, hold.id) || seen.has(hold.id)) {
      throw new Error('The climb contains an unknown hold, duplicate hold, or invalid role.');
    }
    seen.add(hold.id);
  }
  const result = getAuroraBluetoothPacket(framesForHolds(holds), mapping.positions, 'kilter', apiLevel);
  if (result.skippedPositionCount || result.skippedRoleCount || result.ledsWritten !== holds.length) {
    throw new Error('Some holds could not be mapped to the board. Nothing was sent.');
  }
  if (result.allLedsDark || result.ledsWithColour !== result.ledsWritten) {
    throw new Error('This climb exceeds the controller’s LED power budget. Reduce the number of holds.');
  }
  return result;
}

export function climbSignature(holds) {
  return JSON.stringify(holds.map(({ id, role }) => [id, role]).sort((a, b) => a[0] - b[0]));
}

function connectionMessage(error) {
  if (error.name === 'NotFoundError') return 'No board selected. Click Connect to try again.';
  if (error.name === 'SecurityError' || error.name === 'NotAllowedError') {
    return 'Bluetooth permission was denied. Open this page on localhost or HTTPS and allow Bluetooth access.';
  }
  if (error.name === 'NetworkError') return 'Could not reach the board. Check Bluetooth and disconnect other board apps, then reconnect.';
  return error.message || 'The Bluetooth operation failed. Reconnect and try again.';
}

export class KilterBluetooth {
  constructor(mapping, { bluetooth = globalThis.navigator?.bluetooth, onChange = () => {},
    secureContext = globalThis.isSecureContext ?? true } = {}) {
    this.mapping = mapping;
    this.bluetooth = bluetooth;
    this.secureContext = secureContext;
    this.onChange = onChange;
    this.phase = 'disconnected';
    this.epoch = 0;
    this.protocolOverride = null;
    this.lastSent = null;
    this.message = 'Connect to put the displayed climb on your board.';
  }

  get apiLevel() { return this.protocolOverride ?? parseApiLevel(this.device?.name); }
  get connected() { return Boolean(this.device?.gatt?.connected && this.characteristic); }
  get available() { return this.secureContext && Boolean(this.bluetooth?.requestDevice); }
  notify() {
    this.onChange({ phase: this.phase, connected: this.connected, available: this.available,
      deviceName: this.device?.name || 'Kilter board', apiLevel: this.apiLevel,
      message: this.message, lastSent: this.lastSent, error: this.error });
  }
  setProtocol(value) {
    if (![null, 2, 3].includes(value)) throw new Error('Choose Auto, API v2, or API v3.');
    if (this.phase === 'sending' || this.phase === 'connecting') throw new Error('Wait for the Bluetooth operation to finish.');
    this.protocolOverride = value;
    this.notify();
  }

  async connect() {
    if (this.phase !== 'disconnected') return;
    if (!this.available) throw new Error('Web Bluetooth requires a supported browser on localhost or HTTPS.');
    const epoch = ++this.epoch;
    this.phase = 'connecting'; this.error = null;
    this.message = 'Choose your Kilter board in the Bluetooth picker.'; this.notify();
    try {
      // Keep requestDevice before any await so the browser sees the user's click.
      const device = await this.bluetooth.requestDevice(DEVICE_OPTIONS);
      if (epoch !== this.epoch) return;
      const boardType = parseBoardTypeFromDeviceName(device.name);
      if (boardType && boardType !== 'kilter') throw new Error('Select a Kilter board. This app uses the Original 12×12 LED map.');
      this.device = device;
      this.onDisconnect = () => {
        this.disconnect(); this.error = 'The board disconnected. Connect again to send a climb.';
        this.message = this.error; this.notify();
      };
      device.addEventListener('gattserverdisconnected', this.onDisconnect);
      this.abort = new AbortController();
      const characteristic = await getUartCharacteristic(device);
      if (epoch !== this.epoch) { device.gatt?.disconnect(); return; }
      if (!characteristic || !device.gatt?.connected) throw new Error('The board does not expose the expected LED controller service.');
      this.characteristic = characteristic;
      this.phase = 'connected'; this.lastSent = null;
      this.message = `Connected to ${device.name || 'Kilter board'}. Ready to send a climb.`;
      this.notify();
    } catch (error) {
      if (epoch !== this.epoch) return;
      this.disconnect(); this.error = connectionMessage(error); this.message = this.error; this.notify();
      throw error;
    }
  }

  disconnect() {
    ++this.epoch;
    this.abort?.abort();
    const device = this.device;
    if (device && this.onDisconnect) device.removeEventListener('gattserverdisconnected', this.onDisconnect);
    this.device = null; this.characteristic = null; this.onDisconnect = null;
    this.phase = 'disconnected'; this.lastSent = null; this.error = null;
    this.message = 'Disconnected. The board keeps its last LED state.';
    if (device?.gatt?.connected) device.gatt.disconnect();
    this.notify();
  }

  // `message` replaces the status text shown while sending and after success.
  async send(holds, { message } = {}) {
    if (!this.connected || this.phase !== 'connected') throw new Error('Connect to the board and wait for any current send to finish.');
    const snapshot = holds.map(({ id, role }) => ({ id, role }));
    let result;
    try {
      result = encodeClimb(snapshot, this.mapping, this.apiLevel);
    } catch (error) {
      this.error = error.message; this.message = error.message; this.notify();
      throw error;
    }
    const epoch = this.epoch;
    this.phase = 'sending'; this.error = null;
    this.message = message ?? (snapshot.length ? `Sending ${snapshot.length} holds…` : 'Clearing the board LEDs…');
    this.notify();
    try {
      await writeCharacteristicSeries(this.characteristic, splitMessages(result.packet), this.abort.signal, {
        forceWithResponse: isKilterBuiltBox(this.device.name),
        allowUnsupportedWithResponseRetry: true,
      });
      if (epoch !== this.epoch || !this.connected) throw new Error('The board disconnected before the send finished.');
      this.lastSent = climbSignature(snapshot);
      this.message = message ?? (snapshot.length ? `Climb sent · ${snapshot.length} holds` : 'Board LEDs cleared.');
      return result;
    } catch (error) {
      if (epoch === this.epoch) { this.error = connectionMessage(error); this.message = this.error; this.lastSent = null; }
      throw error;
    } finally {
      if (epoch === this.epoch) { this.phase = 'connected'; this.notify(); }
    }
  }
}
