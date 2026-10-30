// Adapted from Boardsesh's board-details.ts, render-config.ts, and board-look
// settings at revision 15f6a8588585992725d9f9ba181271a0de9165f9 (Apache-2.0).
export const BOARD_WIDTH = 1080;
export const BOARD_HEIGHT = 1170;
export const BOARD_COLORS = ['#4DF5FD', '#FFAA00', '#00FF00', '#FF00FF'];
const ROLE_CODES = [13, 15, 12, 14];
const ROLE_NAMES = ['hand', 'foot', 'starting', 'finish'];

export function buildBoardConfig(manifest, geometry) {
  if (manifest.holds.length !== 476 || manifest.holds.some(hold => !geometry.outlines[hold.id])) {
    throw new Error('The Boardsesh Kilter Original artwork does not match this model vocabulary.');
  }
  const r = BOARD_WIDTH / 144 * 4;
  return {
    board_width: BOARD_WIDTH, board_height: BOARD_HEIGHT, output_width: BOARD_WIDTH,
    frames: '', mirrored: false, thumbnail: false,
    stroke_width_multiplier: 1, shape_size_multiplier: 1,
    holds: manifest.holds.map(hold => ({
      id: hold.id, mirroredHoldId: null,
      cx: hold.x * BOARD_WIDTH, cy: (1 - hold.y) * BOARD_HEIGHT, r,
      outline: geometry.outlines[hold.id],
      ...(geometry.ledInner?.[hold.id] ? { led_inner: geometry.ledInner[hold.id] } : {}),
      ...(geometry.ledBright?.[hold.id] ? { led: geometry.ledBright[hold.id] } : {}),
      ...(geometry.silhouetteLightness?.[hold.id] !== undefined
        ? { silhouette_lightness: geometry.silhouetteLightness[hold.id] } : {}),
    })),
    hold_state_map: Object.fromEntries(ROLE_CODES.map((code, role) => [code,
      { color: BOARD_COLORS[role], role: ROLE_NAMES[role] }])),
    render_mode: 'aura', mark_style: 'glow', glow_falloff: 'soft',
    // Boardsesh's editor omits the veil so unselected holds remain visible.
    glow: { reach_scale: 1, plateau_share: 0.4, disc_opacity: 0,
      small_hold_max_boost: 1.7, spread_fraction: 0.91,
      merge_softness: 0.6, seam_blend_fraction: 0.9, seam_sharpness: 3, fringe_deepen: 0.4 },
    fill: { opacity: 0.55 }, glyphs: 'off',
  };
}

export function framesForHolds(holds) {
  return holds.map(hold => `p${hold.id}r${ROLE_CODES[hold.role]}`).join('');
}

export function outlinePath(hold) {
  return hold.outline.reduce((path, coordinate, i) => {
    if (i % 2) return path;
    const x = hold.cx + coordinate * hold.r;
    const y = hold.cy + hold.outline[i + 1] * hold.r;
    return `${path}${i === 0 ? 'M' : 'L'}${x.toFixed(3)},${y.toFixed(3)}`;
  }, '') + 'Z';
}
