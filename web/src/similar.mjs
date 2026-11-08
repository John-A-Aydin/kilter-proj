// Closest real climb to a set of holds. kilter/export_web.py writes model/climbs.json and
// model/climbs.bin: every climb in the dataset, most total ascents first.
const TYPES = { uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array };
// Roles as in the model: hand, foot, start, finish. Feet are left out of the comparison.
export const FOOT = 1;
const isFoot = value => (value & 3) === FOOT;

export function loadCatalogue(meta, buffer) {
  if (meta.format !== 'kilter-climbs-v1') throw new Error('Unsupported real-climb catalogue.');
  const arrays = {};
  let offset = 0;
  for (const { name, type, length } of meta.arrays) {
    const Type = TYPES[type];
    if (!Type || offset + length * Type.BYTES_PER_ELEMENT > buffer.byteLength) throw new Error('The real-climb catalogue is corrupt.');
    arrays[name] = new Type(buffer, offset, length);
    offset += length * Type.BYTES_PER_ELEMENT;
  }
  const { holds, hold_counts: holdCounts, angle_counts: angleCounts } = arrays;
  const count = holdCounts?.length;
  if (offset !== buffer.byteLength || count !== meta.climbs || meta.names.length !== count) {
    throw new Error('The real-climb catalogue is corrupt.');
  }
  const holdStart = new Uint32Array(count + 1), angleStart = new Uint32Array(count + 1);
  for (let i = 0; i < count; i++) {
    holdStart[i + 1] = holdStart[i] + holdCounts[i];
    angleStart[i + 1] = angleStart[i] + angleCounts[i];
  }
  // Inverted index over hand holds (start, hand, finish): for each physical hold, the climbs
  // using it as one and the role it has there. handCounts is each climb's number of them.
  let maxId = 0, postings = 0;
  const handCounts = new Uint8Array(count);
  for (let climb = 0; climb < count; climb++) {
    for (let j = holdStart[climb]; j < holdStart[climb + 1]; j++) {
      maxId = Math.max(maxId, holds[j] >> 2);
      if (!isFoot(holds[j])) { handCounts[climb]++; postings++; }
    }
  }
  const postingStart = new Uint32Array(maxId + 2);
  for (const value of holds) if (!isFoot(value)) postingStart[(value >> 2) + 1]++;
  for (let id = 0; id <= maxId; id++) postingStart[id + 1] += postingStart[id];
  const next = postingStart.slice(0, -1);
  const postingClimb = new Uint32Array(postings), postingRole = new Uint8Array(postings);
  for (let climb = 0; climb < count; climb++) {
    for (let j = holdStart[climb]; j < holdStart[climb + 1]; j++) {
      if (isFoot(holds[j])) continue;
      const position = next[holds[j] >> 2]++;
      postingClimb[position] = climb;
      postingRole[position] = holds[j] & 3;
    }
  }
  return { ...arrays, count, names: meta.names, holdStart, angleStart, handCounts, maxId, postingStart, postingClimb, postingRole,
    shared: new Uint8Array(count), sameRole: new Uint8Array(count) };
}

export function climbHolds(catalogue, climb) {
  return [...catalogue.holds.subarray(catalogue.holdStart[climb], catalogue.holdStart[climb + 1])]
    .map(value => ({ id: value >> 2, role: value & 3 }));
}

export function climbAngles(catalogue, climb) {
  const angles = [];
  for (let i = catalogue.angleStart[climb]; i < catalogue.angleStart[climb + 1]; i++) {
    angles.push({ angle: catalogue.angles[i], grade: catalogue.grades[i], ascents: catalogue.ascents[i] });
  }
  return angles.sort((a, b) => a.angle - b.angle);
}

// Most shared hand holds (start, hand, finish) relative to both climbs' hand holds: a Jaccard
// overlap with roles ignored among them. Feet are ignored on both sides, so a hold that is a foot
// in either climb is not shared. Ties go to more hand holds in the same role, then to the climb
// with more ascents. Null if you have no hand holds or no climb shares one.
export function closestClimb(catalogue, holds) {
  const { shared, sameRole, postingStart, postingClimb, postingRole, handCounts } = catalogue;
  if (new Set(holds.map(hold => hold.id)).size !== holds.length) throw new Error('A climb cannot repeat a hold.');
  const hands = holds.filter(hold => hold.role !== FOOT);
  const ids = new Set(hands.map(hold => hold.id));
  const touched = [];
  for (const { id, role } of hands) {
    if (id > catalogue.maxId) continue;
    for (let p = postingStart[id]; p < postingStart[id + 1]; p++) {
      const climb = postingClimb[p];
      if (shared[climb]++ === 0) touched.push(climb);
      if (postingRole[p] === role) sameRole[climb]++;
    }
  }
  let best = -1, bestOverlap = -1;
  for (const climb of touched) {
    const overlap = shared[climb] / (ids.size + handCounts[climb] - shared[climb]);
    if (overlap > bestOverlap || (overlap === bestOverlap
        && (sameRole[climb] > sameRole[best] || (sameRole[climb] === sameRole[best] && climb < best)))) {
      best = climb;
      bestOverlap = overlap;
    }
  }
  const result = best < 0 ? null : {
    index: best, name: catalogue.names[best], overlap: bestOverlap, shared: shared[best], sameRole: sameRole[best],
    size: handCounts[best], yours: ids.size, holds: climbHolds(catalogue, best), angles: climbAngles(catalogue, best),
  };
  for (const climb of touched) { shared[climb] = 0; sameRole[climb] = 0; }
  return result;
}

// The grade at the requested angle, or else at the nearest angle the climb was climbed at.
export function gradeNear(angles, angle) {
  return angles.reduce((best, row) => (Math.abs(row.angle - angle) < Math.abs(best.angle - angle) ? row : best));
}
