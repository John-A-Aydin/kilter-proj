// Kilter Original subset of Boardsesh's board-constants/src/hold-states.ts.
// Apache-2.0, revision 15f6a8588585992725d9f9ba181271a0de9165f9.
// These are physical LED colors, not Aura's calibrated on-screen colors.
export const HOLD_STATE_MAP = { kilter: {
  12: { name: 'STARTING', color: '#00FF00' },
  13: { name: 'HAND', color: '#00FFFF' },
  14: { name: 'FINISH', color: '#FF00FF' },
  15: { name: 'FOOT', color: '#FFAA00' },
} };
// MoonBoard discovery is unused in this Kilter-only application.
export const MOONBOARD_DEVICE_NAME_PREFIXES = [];
