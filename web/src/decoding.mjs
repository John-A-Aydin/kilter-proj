// Mirrors kilter/sampling.py. The model was trained on BOS <given> SEP <rest> EOS,
// so picks are an exact condition: they go in the given block and the model adds the rest.
export const PAD = 0;
export const BOS = 1;
export const EOS = 2;
export const SEP = 3;
export const ROLE_NAMES = ['Hand', 'Foot', 'Start', 'Finish'];
const START = 2;
const FOOT = 1;
const FINISH = 3;

export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// How to build the model's condition row (kilter/export_web.py condition_schema). Manifests from
// before the schema have grade and angle only.
export function conditionSchema(manifest) {
  return manifest.condition_schema ?? { columns: ['v_grade', 'angle_divided_by_90'], tags: [],
    grade: { min: 0, max: 16, nullable: false, null_value: -1 }, angle: { min: 0, max: 90, divide_by: 90 } };
}

// [grade, angle / 90, one 0/1 per tag in the schema's order], as kilter.sampling.generate builds it.
export function conditionVector(request, manifest) {
  const schema = conditionSchema(manifest);
  const tags = new Set(request.tags ?? []);
  return new Float32Array([request.grade === null ? schema.grade.null_value : request.grade,
    request.angle / schema.angle.divide_by, ...schema.tags.map(tag => Number(tags.has(tag)))]);
}

export function validateRequest(request, manifest) {
  const { grade, angle, temperature, seed, selected } = request;
  if (manifest.format !== 'kilter-v2') throw new Error('This page needs a model exported by kilter.export_web.');
  const schema = conditionSchema(manifest);
  if (grade === null) {
    if (!schema.grade.nullable) throw new Error('This model needs a grade.');
  } else if (!Number.isFinite(grade) || grade < schema.grade.min || grade > schema.grade.max) {
    throw new Error(`Choose a numeric V grade from ${schema.grade.min} to ${schema.grade.max}.`);
  }
  const tags = request.tags ?? [];
  if (!Array.isArray(tags) || new Set(tags).size !== tags.length || tags.some(tag => !schema.tags.includes(tag))) {
    throw new Error(schema.tags.length ? 'Style tags must be distinct tags this model knows.' : 'This model takes no style tags.');
  }
  if (!Number.isFinite(angle) || angle < 0 || angle > 90) throw new Error('Angle must be between 0° and 90°.');
  if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 2) throw new Error('Temperature must be in (0, 2].');
  if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('Seed must be a 32-bit unsigned integer.');
  if (!Array.isArray(selected)) throw new Error('Selected holds must be a list.');
  const holds = new Map(manifest.holds.map((hold, index) => [hold.id, index]));
  const seen = new Set();
  const picks = selected.map(({ id, role }) => {
    if (!holds.has(id) || !Number.isInteger(role) || role < 0 || role > 3 || seen.has(id)) {
      throw new Error('Selected holds must have unique board IDs and valid roles.');
    }
    seen.add(id);
    const index = holds.get(id);
    return { id, role, index, token: 4 + 4 * index + role };
  });
  const starts = picks.filter(pick => pick.role === START).length;
  const finishes = picks.filter(pick => pick.role === FINISH).length;
  if (starts > manifest.max_starts || finishes > manifest.max_finishes) {
    throw new Error(`A climb has at most ${manifest.max_starts} start and ${manifest.max_finishes} finish holds.`);
  }
  if (picks.length + (starts ? 0 : 1) + (finishes ? 0 : 1) > manifest.max_holds) {
    throw new Error(`Climbs have at most ${manifest.max_holds} holds, including a start and a finish.`);
  }
  return picks.sort((a, b) => manifest.rank[a.token] - manifest.rank[b.token]);
}

// Only continuations every training climb satisfies: canonical order (start, hand,
// finish, foot; bottom to top), no repeated hold, and 1-2 start and finish holds.
export function allowedTokens({ manifest, used, lastRank, starts, finishes, count }) {
  const { rank, max_holds: maxHolds, max_starts: maxStarts, max_finishes: maxFinishes } = manifest;
  const needed = (starts === 0) + (finishes === 0);
  const allowed = [];
  if (count < maxHolds) {
    for (let token = 4; token < rank.length; token++) {
      const index = (token - 4) >> 2, role = (token - 4) & 3;
      if (rank[token] <= lastRank || used.has(index)) continue;
      if ((role === START && starts >= maxStarts) || (role === FINISH && finishes >= maxFinishes)) continue;
      if (starts === 0) {
        if (role !== START) continue;
      } else if (finishes === 0) {
        // Feet come after finishes in canonical order, so a finish must come first.
        if (role === FOOT || (maxHolds - count <= needed && role !== FINISH)) continue;
      }
      allowed.push(token);
    }
  }
  if (needed === 0) allowed.push(EOS);
  return allowed;
}

export function sampleToken(logits, allowed, temperature, random) {
  let maximum = -Infinity;
  for (const token of allowed) maximum = Math.max(maximum, logits[token]);
  if (!Number.isFinite(maximum)) throw new Error('The model returned invalid probabilities.');
  const weights = allowed.map(token => Math.exp((logits[token] - maximum) / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  let draw = random() * total;
  for (let i = 0; i < allowed.length; i++) {
    draw -= weights[i];
    if (draw <= 0) return allowed[i];
  }
  return allowed.at(-1);
}

export async function completeClimb({ request, manifest, predict, onStep = () => {}, cancelled = () => false }) {
  const picks = validateRequest(request, manifest);
  const random = seededRandom(request.seed);
  const conditions = conditionVector(request, manifest);
  const tokens = [BOS, ...picks.map(pick => pick.token), SEP];
  const used = new Set(picks.map(pick => pick.index));
  const holds = picks.map(({ id, role }) => ({ id, role, locked: true }));
  let starts = picks.filter(pick => pick.role === START).length;
  let finishes = picks.filter(pick => pick.role === FINISH).length;
  let lastRank = -1;
  for (;;) {
    if (cancelled()) return { holds, cancelled: true, endedWithEos: false };
    const logits = await predict(tokens, conditions);
    if (cancelled()) return { holds, cancelled: true, endedWithEos: false };
    const allowed = allowedTokens({ manifest, used, lastRank, starts, finishes, count: holds.length });
    const token = sampleToken(logits, allowed, request.temperature, random);
    if (token === EOS) return { holds, cancelled: false, endedWithEos: true };
    tokens.push(token);
    const index = (token - 4) >> 2, role = (token - 4) & 3;
    used.add(index);
    starts += role === START;
    finishes += role === FINISH;
    lastRank = manifest.rank[token];
    holds.push({ id: manifest.holds[index].id, role, locked: false });
    await onStep([...holds]);
  }
}
