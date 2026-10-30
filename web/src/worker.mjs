import { loadModel } from './runtime.mjs';
import { completeClimb } from './decoding.mjs';

let active = null;
const model = loadModel();
model.then(({ manifest }) => postMessage({ type: 'ready', manifest }))
  .catch(error => postMessage({ type: 'error', message: `Could not load the model: ${error.message}` }));

onmessage = async ({ data }) => {
  if (data.type === 'cancel') {
    if (active) active.cancelled = true;
    return;
  }
  if (data.type !== 'generate' || active) return;
  const run = { id: data.id, cancelled: false };
  active = run;
  try {
    const { manifest, predict } = await model;
    const result = await completeClimb({ request: data.request, manifest, predict,
      cancelled: () => run.cancelled,
      onStep: async holds => {
        postMessage({ type: 'step', id: run.id, holds });
        await new Promise(resolve => setTimeout(resolve, 35));
      },
    });
    postMessage({ type: 'done', id: run.id, ...result });
  } catch (error) {
    postMessage({ type: 'error', id: run.id, message: error.message });
  } finally {
    active = null;
  }
};
