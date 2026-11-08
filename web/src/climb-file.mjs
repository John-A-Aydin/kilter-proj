// Reads a climb from JSON: this page's own export, or one row of data/training/climbs.jsonl.
// Both list holds as frames of { hold_token, hold_use_type }.
const MAX_GRADE = 16;

export function parseClimbFile(data, manifest) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('the file is not a climb object.');
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

  let grade = data.v_grade ?? /^V(\d+)$/.exec(data.grade_hueco ?? '')?.[1] ?? null;
  if (grade !== null) {
    grade = Number(grade);
    if (!Number.isInteger(grade) || grade < 0 || grade > MAX_GRADE) throw new Error(`grade must be V0 to V${MAX_GRADE}.`);
  }
  const angle = data.angle ?? null;
  if (angle !== null && !(Number.isInteger(angle) && angle >= 0 && angle <= 90 && angle % 5 === 0)) {
    throw new Error('angle must be 0° to 90° in 5° steps.');
  }
  const seed = Number.isInteger(data.seed) && data.seed >= 0 && data.seed <= 4294967295 ? data.seed : null;
  const temperature = Number.isFinite(data.temperature) && data.temperature >= 0.2 && data.temperature <= 1.5 ? data.temperature : null;
  return { holds, selected, grade, angle, seed, temperature, name: data.name || null,
    endedWithEos: data.ended_with_eos ?? true, checkpoint: data.model?.checkpoint_sha256 ?? null };
}
