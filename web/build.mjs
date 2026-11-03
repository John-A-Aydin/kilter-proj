import { cp, mkdir, rm, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
await import('./vendor-bluetooth.mjs');
await access(path.join(root, 'public/model/model.onnx')).catch(() => {
  throw new Error('Export a checkpoint first: uv run --frozen --extra web python -m kilter.export_web runs/<run>/best.pt');
});
await rm(path.join(root, 'dist'), { recursive: true, force: true });
await mkdir(path.join(root, 'dist/vendor/ort'), { recursive: true });
// Ship the compiled Bluetooth modules, not their TypeScript sources.
await cp(path.join(root, 'public'), path.join(root, 'dist'), { recursive: true, filter: source => !source.endsWith('.ts') });
await cp(path.join(root, 'src'), path.join(root, 'dist/src'), { recursive: true });
await cp(path.join(root, 'index.html'), path.join(root, 'dist/index.html'));
for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await cp(path.join(root, 'node_modules/onnxruntime-web/dist', name), path.join(root, 'dist/vendor/ort', name));
}
console.log('Built dist/ with the trained model and local ONNX Runtime assets.');
