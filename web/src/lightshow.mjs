// Light-show frames built from hold coordinates. Each frame is a list of {id, role}
// holds, so it goes through the same Kilter role colors and LED encoder as a climb.
import { seededRandom } from './decoding.mjs';

// Keeps every frame well inside an API v2 controller's power budget.
export const MAX_SHOW_HOLDS = 48;
// Green, cyan, magenta, orange: start, hand, finish, and foot role colors.
const CYCLE = [2, 0, 3, 1];

function band(holds, value, center, halfWidth, role) {
  return holds.filter(hold => Math.abs(value(hold) - center) < halfWidth)
    .map(hold => ({ id: hold.id, role, distance: Math.abs(value(hold) - center) }));
}

export const PATTERNS = {
  wave: {
    name: 'Rising wave',
    description: 'A band of light climbs the wall, changing color on each pass.',
    frame(holds, t) {
      const position = t * 0.05, travel = 1.1;
      return band(holds, hold => hold.y, (position % travel) - 0.05, 0.05, CYCLE[Math.floor(position / travel) % 4]);
    },
  },
  scanner: {
    name: 'Scanner',
    description: 'A column sweeps left and right across the board.',
    frame(holds, t) {
      const position = t * 0.06, sweep = Math.floor(position);
      const phase = position % 1;
      const center = -0.05 + 1.1 * (sweep % 2 ? 1 - phase : phase);
      return band(holds, hold => hold.x, center, 0.04, CYCLE[sweep % 4]);
    },
  },
  ripple: {
    name: 'Ripple',
    description: 'Rings spread out from the middle of the wall.',
    frame(holds, t) {
      const position = t * 0.035, spacing = 0.4;
      const distance = hold => Math.hypot(hold.x - 0.5, hold.y - 0.5);
      return [0, 1].flatMap(ring => {
        const travelled = position + ring * spacing;
        const radius = travelled % (2 * spacing);
        return band(holds, distance, radius, 0.022, CYCLE[Math.floor(travelled / (2 * spacing) * 2 + ring) % 4]);
      });
    },
  },
  sparkle: {
    name: 'Sparkle',
    description: 'Random holds twinkle in every color.',
    frame(holds, t) {
      const random = seededRandom(0x5eed + t);
      const chosen = new Set();
      while (chosen.size < Math.min(24, holds.length)) chosen.add(Math.floor(random() * holds.length));
      return [...chosen].map(index => ({ id: holds[index].id, role: CYCLE[Math.floor(random() * 4)], distance: 0 }));
    },
  },
};

export function showFrame(pattern, holds, t) {
  if (!PATTERNS[pattern]) throw new Error(`Unknown light show pattern: ${pattern}`);
  const unique = new Map();
  for (const hold of PATTERNS[pattern].frame(holds, t).sort((a, b) => a.distance - b.distance)) {
    if (!unique.has(hold.id)) unique.set(hold.id, hold);
  }
  return [...unique.values()].slice(0, MAX_SHOW_HOLDS).map(({ id, role }) => ({ id, role }))
    .sort((a, b) => a.id - b.id);
}
