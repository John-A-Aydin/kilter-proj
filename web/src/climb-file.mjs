// Reads and writes climb JSON: this page's export (schema 2, or the older schema 1), or one row of
// data/training/climbs.jsonl. All of them list holds as frames of { hold_token, hold_use_type }.
//
// An export keeps where the climb came from (`source`) apart from the model loaded in the page when
// it was saved (`viewer_model`). A climb imported from another model or from the catalogue keeps its
// original source when exported again, and a grade or angle the file did not record stays null
// instead of taking the page's current controls.
//
// source.kind is one of:
//   generated      made by this page: model, request (grade, angle, temperature, seed, picks),
//                  ended_with_eos, cancelled, and derived_from (the climb its picks were kept from)
//   catalogue      a dataset row: climb_uuid, name, setter_username, angle, grade_hueco, ...
//   legacy_export  a schema 1 export, which recorded the page's model at export time (recorded_model);
//                  that may not be the model that generated it
//   unknown        holds only
export const EXPORT_SCHEMA_VERSION = 2;
const MAX_GRADE = 16;
const GRADE_BASES = new Set(['requested', 'catalogue', 'recorded']);
const CATALOGUE_FIELDS = ['climb_uuid', 'climb_angle_id', 'name', 'setter_username', 'angle', 'grade_hueco',
  'grade_status', 'ascensionist_count'];

const isObject = value => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const holdsOf = holds => holds.map(({ id, role }) => ({ hold_token: id, hold_use_type: role }));

function parseGrade(value) {
  if (value === null || value === undefined) return null;
  const grade = Number(value);
  if (!Number.isInteger(grade) || grade < 0 || grade > MAX_GRADE) throw new Error(`grade must be V0 to V${MAX_GRADE}.`);
  return grade;
}

function parseAngle(angle) {
  if (angle === null || angle === undefined) return null;
  if (!(Number.isInteger(angle) && angle >= 0 && angle <= 90 && angle % 5 === 0)) {
    throw new Error('angle must be 0° to 90° in 5° steps.');
  }
  return angle;
}

// The climb's own grade and what it means: the grade requested from a model, the catalogue's grade,
// or a grade some file recorded. Null when the file has none.
function gradeOf(data) {
  if (data.schema_version === 2) {
    const grade = parseGrade(data.v_grade);
    if (grade !== null && !GRADE_BASES.has(data.grade_basis)) throw new Error('its grade_basis is not recognised.');
    return { grade, gradeBasis: grade === null ? null : data.grade_basis };
  }
  const grade = parseGrade(data.v_grade ?? /^V(\d+)$/.exec(data.grade_hueco ?? '')?.[1] ?? null);
  const catalogue = data.schema_version === undefined && typeof data.climb_uuid === 'string' && data.v_grade === undefined;
  return { grade, gradeBasis: grade === null ? null : catalogue ? 'catalogue' : 'recorded' };
}

function sourceOf(data, request) {
  if (data.schema_version === 2) {
    if (!isObject(data.source) || typeof data.source.kind !== 'string') throw new Error('its "source" must say what kind of climb it is.');
    return structuredClone(data.source);
  }
  if (data.schema_version === 1) {
    return { kind: 'legacy_export', schema_version: 1, recorded_model: isObject(data.model) ? structuredClone(data.model) : null,
      request, ended_with_eos: data.ended_with_eos ?? null };
  }
  if (data.schema_version !== undefined) throw new Error(`export schema ${data.schema_version} is not supported by this page.`);
  if (typeof data.climb_uuid === 'string') {
    return { kind: 'catalogue', ...Object.fromEntries(CATALOGUE_FIELDS.filter(key => data[key] !== undefined).map(key => [key, data[key]])) };
  }
  return { kind: 'unknown' };
}

// The model that generated a climb, if its source names one.
export function generatorCheckpoint(source) {
  return source?.model?.checkpoint_sha256 ?? source?.recorded_model?.checkpoint_sha256 ?? null;
}

export function parseClimbFile(data, manifest) {
  if (!isObject(data)) throw new Error('the file is not a climb object.');
  const nested = Array.isArray(data.frames?.[0]);
  if (nested && data.frames.length > 1) throw new Error('climbs with more than one frame are not supported.');
  const frame = nested ? data.frames[0] : data.frames;
  if (!Array.isArray(frame) || !frame.length) throw new Error('it has no holds. Expected a "frames" list.');

  const known = new Set(manifest.holds.map(hold => hold.id));
  const seen = new Set();
  const holds = frame.map(entry => {
    const id = Number(entry?.hold_token);
    const role = Number(entry?.hold_use_type);
    if (!known.has(id)) throw new Error(`hold ${entry?.hold_token} is not on the Kilter Original 12×12.`);
    if (!Number.isInteger(role) || role < 0 || role > 3) throw new Error(`hold ${id} has an unknown role ${entry?.hold_use_type}.`);
    if (seen.has(id)) throw new Error(`hold ${id} appears twice.`);
    seen.add(id);
    return { id, role };
  }).sort((a, b) => a.id - b.id);

  // An export also records which holds were picked by hand; they must be in the climb.
  const roles = new Map(holds.map(hold => [hold.id, hold.role]));
  const selected = (data.selected_holds || []).map(({ id, role }) => {
    if (roles.get(id) !== role) throw new Error(`picked hold ${id} is not in the climb with that role.`);
    return { id, role };
  });

  const { grade, gradeBasis } = gradeOf(data);
  const angle = parseAngle(data.angle ?? null);
  const request = data.source?.request ?? {};
  const rawSeed = data.seed ?? request.seed, rawTemperature = data.temperature ?? request.temperature;
  const seed = Number.isInteger(rawSeed) && rawSeed >= 0 && rawSeed <= 4294967295 ? rawSeed : null;
  const temperature = Number.isFinite(rawTemperature) && rawTemperature >= 0.2 && rawTemperature <= 1.5 ? rawTemperature : null;
  const source = sourceOf(data, { grade: data.v_grade ?? null, angle: data.angle ?? null, temperature: data.temperature ?? null,
    seed: data.seed ?? null, selected_holds: selected });
  return { holds, selected, grade, gradeBasis, angle, seed, temperature, source,
    name: data.name || source.name || null, endedWithEos: data.ended_with_eos ?? true, checkpoint: generatorCheckpoint(source) };
}

// The source of a climb this page just generated. `derivedFrom` is the climb whose holds were kept
// as this request's picks, if any.
export function generatedSource({ manifest, request, endedWithEos, cancelled, derivedFrom = null }) {
  return {
    kind: 'generated',
    model: { name: manifest.name, checkpoint_sha256: manifest.checkpoint_sha256, model_sha256: manifest.model_sha256 ?? null,
      data_fingerprint: manifest.data_fingerprint ?? null },
    request: { grade: request.grade, angle: request.angle, temperature: request.temperature, seed: request.seed,
      selected_holds: request.selected.map(({ id, role }) => ({ id, role })) },
    ended_with_eos: endedWithEos, cancelled, derived_from: derivedFrom,
  };
}

// What "Keep these holds" records, so the next generation can say which climb its picks came from.
export function keptClimb(result, holds) {
  return { source: result?.source ?? null, grade: result?.grade ?? null, angle: result?.angle ?? null, frames: [holdsOf(holds)] };
}

// Only a finished climb goes to the wall by itself. A stopped generation or a climb without its
// end can still be sent by hand.
export function shouldAutoSend(result) {
  return Boolean(result) && result.cancelled !== true && result.endedWithEos === true;
}

// result: { source, grade, gradeBasis, angle, endedWithEos, selected }
export function buildClimbExport({ holds, geometry, result, manifest, match = null }) {
  const { grade, angle } = result;
  return {
    schema_version: EXPORT_SCHEMA_VERSION,
    v_grade: grade, grade_hueco: grade === null ? null : `V${grade}`, grade_basis: grade === null ? null : result.gradeBasis,
    angle, normalized_angle: angle === null ? null : angle / 90,
    ended_with_eos: result.endedWithEos,
    selected_holds: result.selected.map(({ id, role }) => ({ id, role })),
    source: structuredClone(result.source),
    viewer_model: { name: manifest.name, checkpoint_sha256: manifest.checkpoint_sha256 },
    closest_real_climb: match && { name: match.name, overlap: match.overlap, shared_holds: match.shared,
      same_role_holds: match.sameRole, holds: holdsOf(match.holds), angles: match.angles },
    frames: [holds.map(hold => ({ hold_token: hold.id, hold_use_type: hold.role,
      hold_x: geometry.get(hold.id).x, hold_y: geometry.get(hold.id).y,
      on_kickboard: Number(geometry.get(hold.id).kickboard) }))],
  };
}

export function exportFileName(output) {
  const seed = output.source?.request?.seed ?? output.source?.climb_uuid?.slice(0, 8) ?? 'climb';
  return `kilter-${output.v_grade === null ? 'Vunknown' : `V${output.v_grade}`}-${output.angle === null ? 'unknown' : output.angle}deg-${seed}.json`;
}
