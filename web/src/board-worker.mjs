// Local adapter for Boardsesh's unmodified Rust/WASM Aura renderer.
// RGBA decoding follows packages/web/app/lib/board-render-worker/board-render.worker.ts.
import init, { render_overlay } from '../vendor/boardsesh/board_renderer_wasm.js';

let base;
let pending;
let rendering = false;
const initialized = init().then(() => postMessage({ type: 'ready' }))
  .catch(error => postMessage({ type: 'error', message: error.message }));

onmessage = async ({ data }) => {
  if (data.type === 'init') { base = data.config; return; }
  if (data.type !== 'render') return;
  pending = data;
  if (rendering) return;
  rendering = true;
  try {
    await initialized;
    while (pending) {
      const request = pending;
      pending = null;
      const bytes = render_overlay(JSON.stringify({ ...base, frames: request.frames }));
      const header = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const width = header.getUint32(0, true), height = header.getUint32(4, true);
      const rgba = new Uint8ClampedArray(bytes.slice(8).buffer);
      const canvas = new OffscreenCanvas(width, height);
      canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
      const blob = await canvas.convertToBlob({ type: 'image/png' });
      postMessage({ type: 'rendered', id: request.id, blob });
    }
  } catch (error) {
    postMessage({ type: 'error', message: error.message });
  } finally {
    rendering = false;
  }
};
