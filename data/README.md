# Kilter Original 12×12 catalogue — all recorded angles

The Python scripts in this folder download, convert, and validate the complete
qualifying catalogue from saved public snapshots. Each row is a unique
`(climb_uuid, angle)` pair with its own grade and ascensionist count.
No account credentials are needed.

The rebuilt dataset contains **339,659 climb–angle rows** from **207,175 unique
climbs**, across **15 angles (0° to 70° in 5° steps)**, and **4,662,847 hold
entries**. Grades are reported by the native sources for 337,076 rows, supplied
by the published computed-grade table for 2,556 rows, and explicitly unavailable
for 27 rows. All 27 ungraded rows are retained. Processed files occupy about
1.56 GB (the compact metadata-only `climbs.csv` is about 105 MB).

## Run

From the repository root, with Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run --frozen python data/download.py --manifest data/snapshot-lock.json
uv run --frozen python data/prepare.py
uv run --frozen python data/validate.py
uv run --frozen python -m unittest discover -s data -p 'test_*.py' -v
```

With the existing environment, replace `uv run --frozen python` with
`.venv/bin/python`. Downloading requires network access. Conversion and validation
run entirely offline. Paths default to this folder regardless of your working
directory; each script also supports `--raw-dir`, and conversion/validation support
`--output-dir`. Run any script with `--help` for its arguments.

The lockfile pins the dependency versions. Pillow is declared explicitly because
BoardLib 0.15.1's CLI/API imports it without declaring it as a dependency.

The initial download is about 83 MB for the compressed catalogue plus 17 MB
for its matching grade table, plus BoardLib's Android app download. Retained raw
databases occupy about 451 MB; the all-angle exports are larger than the sources.
Interrupted catalogue downloads resume from checked byte-range chunks.
Downloads are checked for byte size, SQLite integrity, table counts, and SHA-256.

## Sources and selection

1. [`boardlib.db.aurora.download_database`](https://github.com/lemeryfertitta/BoardLib)
   downloads the Kilter app's SQLite database. Its `placements`, `holes`,
   `placement_roles`, `layouts`, `product_sizes`, and hold-set tables provide the
   physical board geometry and role meanings. Its public `climb_stats` also retain
   historical ascent evidence. This run uses BoardLib **0.15.1** and the APK database's
   January 31, 2026 climb/stat synchronization cursors.
2. [Boardsesh's published dataset](https://github.com/boardsesh/boardsesh/blob/main/docs/board-snapshots-dataset.md)
   supplies the newer complete Kilter Original climb catalogue and aggregate ascent
   statistics, plus a matching separate grade artifact. The pinned snapshot was
   built **October 21, 2026, 12:37:33 UTC** and
   contains **385,547 climbs** across all Original board sizes. This alternative
   source is also [linked in BoardLib's issue tracker](https://github.com/lemeryfertitta/BoardLib/issues/80).
   The original Aurora sync host failed TLS negotiation during this run.

The exporter selects **Kilter Board Original**, layout **1**, size **10**,
`12 x 12 with kickboard`. This includes both bolt-on and screw-on sets. Climbs
must be listed, published, visible in the current snapshot, and compatible with
size 10 according to `compatible_size_ids`. Every decoded hold must also exist
inside the board's geometry; a bad row stops the export with its climb UUID.
Climbs that fit a smaller wall are included when they also fit this board.

A qualifying **climb–angle pair** has a positive `ascensionist_count` at that
angle in either saved stats source. All recorded angles are included, including
0°; there is no angle filter. We keep the original requirement for at least one
recorded ascensionist at each exported angle. A setter's preferred angle alone
is not ascent evidence. The SQL unions the pair keys from both sources and joins
statistics on **both UUID and angle**, so a climb with information at 30°, 40°,
and 50° produces three separate rows without cross-joining their statistics.

`climb_angle_id` is `UUID:angle`. The JSONL and climb CSV have one row per pair,
sorted by UUID and then numeric angle. The hold CSV repeats the corresponding
holds at each angle and includes the same ID and angle for an unambiguous join.
The underlying board coordinates and physical hold tokens stay the same across
angles. `setter_angle` remains separate from the row's actual `angle`.

The current snapshot reports zero for some pairs with documented ascents in the
older APK. `ascent_evidence_source` identifies which source qualifies each row.
`ascensionist_count` uses the current positive count when available, otherwise
the historical positive count. Both original counts are also exported. Counts
describe distinct ascensionists as reported by the sources; individual ascent
records are not downloaded. Coverage is complete for these saved snapshots and
filters, rather than a live view of every Kilter service.

## Angle-specific grades and metadata

Each row includes `angle`, `ascensionist_count`, `grade`, `grade_font`,
`grade_hueco`, `grade_index`, `grade_difficulty`, `grade_source`, and `grade_status`.
It also retains name, setter, description, creation time, quality average,
benchmark difficulty, difficulty average, first-ascent username/time, frame count,
frame pace, and original ascent counts. JSONL additionally retains the raw frames
and full converted hold lists.

Grades follow this explicit order, always at the **same angle**:

1. Use the native difficulty from the source that supplies the positive ascent
   count: current snapshot first, otherwise the APK. Prefer display difficulty,
   then benchmark difficulty, then average difficulty.
2. If absent, use a native APK display difficulty at that angle when available.
3. If still absent, use `local_grade` from the matching published
   `board_climb_grades` artifact. These are **Boardsesh-computed estimates**, marked
   `grade_source="boardsesh_computed"` and `grade_status="computed"`.
   `computed_grade_confidence` retains the source's confidence label. Universal
   grades and local estimates are both retained as metadata; only the **local**
   value is mapped through the Kilter grade table.
4. Keep rows with no available grade, as requested: `grade=null` and
   `grade_status="unavailable"` (empty grade cells in CSV). No grade is borrowed
   from another angle or guessed from the climb's name.

`grade` is the readable combined Font/Hueco label from BoardLib's grade mapping,
for example `6b/V4`; the two components are exported separately too.
`grade_difficulty` preserves the numerical value used to derive that label, and
`grade_index` is its rounded native index. The converter calls
`boardlib.api.aurora.difficulty_to_grade`, which uses Python's nearest-even
`round` followed by the APK's `difficulty_grades` lookup. Native
`display_difficulty` is retained separately and may be null when the readable
grade comes from another source. The exact grade table, source hashes, row
counts, per-angle counts, and grade-status counts are recorded in the manifest.

## Hold representation

Each hold in `climbs.jsonl` contains exactly these five fields:

| Field | Meaning |
| --- | --- |
| `hold_use_type` | Integer: hand **0**, foot **1**, start **2**, finish **3**. Source `middle` means hand. |
| `hold_x` | `(raw_x - edge_left) / (edge_right - edge_left)`, in `[0,1]`, increasing left to right. |
| `hold_y` | `(raw_y - edge_bottom) / (edge_top - edge_bottom)`, in `[0,1]`, increasing bottom to top. |
| `on_kickboard` | Integer **1** for a hole whose source name identifies a `KB` row; otherwise **0**. |
| `hold_token` | Unique physical hold token: the stable Original **placement ID**, independent of role. |

The fixed board bounds are **x: 0–144, y: 0–156**, in the source coordinate units.
The y axis includes the 12-unit kickboard; both panels share this one coordinate
system. Coordinates refer to the board surface, without projecting for wall angle.
They are normalized against the board, never against each climb's selected holds.
Kickboard holes lie on source rows y=4 and y=8. Start and finish retain their own
roles even though a climber uses them with their hands.

There are **476** physical holds in the vocabulary. Tokens are integers and are
not contiguous; the same physical hold keeps its token in every climb. Use
`hold_vocabulary.csv` to construct a dense embedding index if needed. No special
padding/start/end tokens are assigned.

Role IDs are resolved by name from BoardLib's database across its products,
because some Original climbs contain other Kilter product role codes. The sole
required role missing from this APK is **31**, BKB foot-only; it is explicitly
mapped to foot according to
[the published Kilter decoder](https://github.com/boardsesh/boardsesh/blob/main/packages/board-constants/src/hold-states.ts).
Unknown roles or placements fail conversion instead of being dropped.

## Files and frames

| Output | Contents |
| --- | --- |
| `processed/climbs.jsonl` | One JSON object per climb–angle pair, with grade, count, metadata, raw frame string, and a `frames` array of hold lists. |
| `processed/climbs.csv` | One metadata row per climb–angle pair, including grade, count, and angle; omits the large raw/decoded frame strings. |
| `processed/holds.csv` | One row per hold per decoded frame: `climb_angle_id,climb_uuid,angle,frame_index,hold_index` followed by the five requested fields. |
| `processed/hold_vocabulary.csv` | All 476 physical holds, their tokens, normalized/raw coordinates, source hole names, kickboard flags, and sets. |
| `processed/manifest.json` | Counts, coordinate convention, role mappings, source dates/URLs/checksums, and script/output SHA-256 hashes. |
| `raw/sources.json` | Download provenance and checksums of all three raw databases, including the grade artifact. |
| `snapshot-lock.json` | Tracked source selection and expected raw database hashes for this run. |

For example, read a route without loading the entire catalogue:

```python
import json

with open("data/processed/climbs.jsonl", encoding="utf-8") as source:
    climb = json.loads(next(source))
    print(climb["climb_angle_id"], climb["angle"], climb["grade"], climb["ascensionist_count"])
    first_frame_holds = climb["frames"][0]
    # Each element has hold_use_type, hold_x, hold_y, on_kickboard, hold_token.
```

Multi-frame routes are included at every qualifying angle. The manifest separates
unique climb counts from climb–angle row counts, including multi-frame counts.
Frame and hold-entry totals include their repetitions across angles. The decoder
follows the
[source format](https://github.com/boardsesh/boardsesh/blob/main/packages/board-constants/src/hold-states.ts):
commas separate frames, a leading quote marks a delta from the previous frame,
an unquoted frame replaces the previous state, and `x<ID>` turns a hold off.
A quote-only frame repeats the previous state. Each exported frame is the full
active hold list, sorted by token. `frame_index` and `hold_index` are zero-based;
hold order is deterministic and does not imply a climbing movement sequence.

`frames_count` counts the actual decoded frames; `source_frames_count` retains
the source's declared count. `frame_count_matches_source` flags any disagreement.
One qualifying route (`F142693E7BF54FBEA3DC936E263DEEF5`, “Happy New Year! 🎉”)
declares 46 frames but contains 45. Its supplied frames remain in the export;
no extra frame is invented. `frame_count_mismatch_rows` counts affected angle rows.

## Reproduce or refresh

Keep all three `raw/` databases and `raw/sources.json` to reproduce this exact result.
The APK's upstream “latest” URL can change, and timestamped snapshot URLs may
eventually expire. The tracked lock checks hashes and fails if downloaded sources
differ; it cannot restore an upstream file that is no longer hosted. Archive the
raw inputs with your experiments. Raw and generated data are ignored by Git to
avoid putting large databases in source control.

Rerunning `prepare.py` against the same sources yields byte-identical data files.
`validate.py` checks every converted hold, CSV/JSON agreement, vocabulary,
checksums, raw table counts, angle-specific grades/counts, and coverage of the
entire qualifying SQL result. It streams source rows in pair order, so duplicate,
missing, extra, or crossed-angle rows fail validation.
Regression tests cover absolute/delta/hold frames, removals, invalid inputs,
stable tokens, download range validation, zero-degree angles, historical-only
angles, overlapping pair deduplication, native/computed/missing grade handling,
and catalogue selection.

To fetch the newest published catalogue without replacing the pinned dataset:

```bash
uv run --frozen python data/download.py --latest --raw-dir data/raw/latest
uv run --frozen python data/prepare.py --raw-dir data/raw/latest --output-dir data/processed/latest
uv run --frozen python data/validate.py --raw-dir data/raw/latest --output-dir data/processed/latest
```

The public snapshot excludes account-owned content; retain setters' attribution
when presenting individual climbs. Changes to upstream availability or schema
cause an explicit error rather than a silently truncated catalogue.
