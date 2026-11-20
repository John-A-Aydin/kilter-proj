// Display names and groups for the style tags a model takes (manifest.condition_schema.tags).
// The manifest decides which tags exist; anything not listed here still shows, under "Other".
export const TAG_LABELS = {
  pinch: 'Pinches', crimp: 'Crimps', jug: 'Jugs', undercling: 'Underclings', gaston: 'Gastons',
  campus: 'Campus', traverse: 'Traverse', dynamic: 'Dynamic', crossover: 'Crossovers', coordination: 'Coordination',
  compression: 'Compression', lock_off: 'Lock-offs', drop_knee: 'Drop knees', heel_hook: 'Heel hooks',
  toe_hook: 'Toe hooks', high_feet: 'High feet', body_tension: 'Body tension',
  reachy: 'Reachy', short_friendly: 'Short-friendly', powerful: 'Powerful', endurance: 'Endurance',
};

const GROUPS = [
  ['Holds', ['pinch', 'crimp', 'jug', 'undercling', 'gaston']],
  ['Moves', ['campus', 'traverse', 'dynamic', 'crossover', 'coordination', 'compression', 'lock_off', 'drop_knee',
    'heel_hook', 'toe_hook', 'high_feet', 'body_tension']],
  ['Climber', ['reachy', 'short_friendly', 'powerful', 'endurance']],
];

// Tags the holds alone barely show (experiments/README.md, tag recognizer AUC near 0.5), so the
// effect of asking for them is hard to see.
export const SUBTLE_TAGS = new Set(['high_feet', 'body_tension']);

export const tagLabel = tag => TAG_LABELS[tag] ?? tag.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase());

// [[group name, tags]] for the model's tags, in display order, leaving out empty groups.
export function tagGroups(tags) {
  const known = new Set(tags);
  const groups = GROUPS.map(([name, members]) => [name, members.filter(tag => known.has(tag))]);
  const listed = new Set(GROUPS.flatMap(([, members]) => members));
  groups.push(['Other', tags.filter(tag => !listed.has(tag))]);
  return groups.filter(([, members]) => members.length);
}

// Requested tags in the model's own column order, so a request reads the same however it was built.
export const orderTags = (tags, schemaTags) => schemaTags.filter(tag => tags.includes(tag));
