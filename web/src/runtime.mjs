import * as ort from '../vendor/ort/ort.wasm.min.mjs';

export async function loadModel() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
  const response = await fetch(new URL('../model/manifest.json', import.meta.url));
  if (!response.ok) throw new Error('Model metadata could not be loaded.');
  const manifest = await response.json();
  const session = await ort.InferenceSession.create(new URL('../model/model.onnx', import.meta.url).href, {
    executionProviders: ['wasm'], graphOptimizationLevel: 'all',
  });
  return { manifest, async predict(tokens, conditions) {
    const output = await session.run({
      tokens: new ort.Tensor('int64', BigInt64Array.from(tokens, BigInt), [1, tokens.length]),
      conditions: new ort.Tensor('float32', Float32Array.from(conditions), [1, 2]),
    });
    return output.logits.data;
  } };
}
