# Kilter Climb Lab

A static browser interface for the v2 climb transformer ([kilter_v2/](../kilter_v2/README.md)).
Pick your start holds and maybe some feet, or nothing at all. Choose a numeric V
grade and angle, and click **Complete climb**. Solid outlines are your picks; dashed
outlines are model additions. Generation appears one hold at a time and runs in a
Web Worker.

## Local use

From the training repository root:

```bash
uv sync --frozen --extra web
uv run --frozen --extra web python -m kilter_v2.export_web runs/<v2 run>/best.pt
cd web
npm ci
npm run build
npm start
```

Open **http://127.0.0.1:4173**. Node 20+ is supported. All runtime assets, including
the model and WebAssembly, are served locally; inference makes no API calls and
does not send holds to an inference server. A first hosted visit still downloads
the page and model assets and passes through the host's normal access checks.

`web/public/model/` is committed, so a fresh clone can skip the export step. The site
serves `null_rule_16`, seed 43. To use another v2 experiment, export its checkpoint
with the same command, then rebuild. The page refuses models from the old `kilter.export_web`, which used a
different sequence format. Export checks ONNX logits against PyTorch for empty,
starts-only, starts-plus-feet, half-given, and full-length contexts at V0/0°,
V5/40°, and V16/90°. It also records decoding fixtures: random generation states
with the tokens Python's rules allow, which the browser tests replay. The manifest
records the checkpoint and model SHA-256, board geometry, each token's canonical
rank, hold and role limits, condition order, parameter count, and parity error.
Model weights and metadata are exported without optimizer state. The export also
writes `climbs.json` and `climbs.bin` (about 12 MB, 6 MB gzipped): every real climb in
the dataset, with its name, holds, and grade and ascent count at each angle.

## Completion behavior

The model was trained on `BOS <given holds> SEP <remaining holds> EOS`. During
training the given holds were none, the start holds plus a random share of the
feet, or a random subset. Your picks go in the given block, so completion is the
distribution the model learned, p(rest | picks, grade, angle). Nothing is
approximated with barriers, and the model can add holds anywhere on the wall.

Decoding follows the canonical order: start, hand, finish, foot, each bottom to
top. It also masks what no training climb does: repeating a hold, more than two
start or finish holds, or ending without at least one of each. The rules are in
`src/decoding.mjs` and mirror `kilter_v2/sampling.py`. Picks are limited the same
way: at most two start and two finish holds, and at most 40 holds in a climb.
Numeric conditions remain `[V_grade, angle_degrees / 90]`. No claim is made that a
generated climb is physically climbable or matches its requested grade.

**Keep these holds** locks a generated result for the next generation. **Clear
picks** starts over. Uncheck **New seed on each generation** for repeatable
sampling. JSON exports contain all five dataset hold fields, the selected holds,
grade, angle, normalized angle, seed, temperature, EOS status, and checkpoint
identity. Stopping generation preserves your selected holds and labels the
result as incomplete in its export.

## Closest real climb

After each generation, a second board below the first shows the most similar real
climb among the 206,783 single-frame Kilter Original climbs in the dataset (all
splits). Similarity is the hold overlap: shared physical holds divided by the holds
in either climb, with roles ignored. Ties go to the climb with more holds in the
same role, then the one with more ascents. Its shared holds are drawn solid and the
rest dashed. The panel shows the climb's name, its grade at your angle (or at the
nearest angle it was climbed at), and its grade at every angle. A badge marks a
generated climb whose hold set already exists. The lookup runs in the browser with
an inverted index from hold to climbs, built once from `climbs.bin`
(`src/similar.mjs`). The JSON export includes the match as `closest_real_climb`.
The page refuses a catalogue from a different dataset than the model.

## Lighting the climb on a board

**Light up the wall** sends the displayed climb (your picks plus model additions)
to a Kilter Original 12×12 + kickboard over Web Bluetooth. Use Chrome or Edge on
desktop or Android, opened from `http://127.0.0.1:4173` or HTTPS. Firefox and
Safari have no Web Bluetooth. Desktop Linux Chrome also needs
`chrome://flags/#enable-experimental-web-platform-features`. Close the Kilter
app or any other app holding the board's connection first.

Click **Connect board**, choose the board in the browser's picker, then click
**Send climb**. Tick **Send each completed climb** to light every finished
generation automatically. **Turn off LEDs** clears the wall. The pill reads
**On the wall** while the board shows exactly what is on screen. The board keeps
its LEDs lit after you disconnect.

The protocol code is Boardsesh's `packages/shared/ble-protocol` at the same
pinned revision, kept unmodified as TypeScript in `public/vendor/boardsesh/bluetooth/`.
`vendor-bluetooth.mjs` compiles it with the pinned TypeScript `tsc` during
`npm run build`. `src/bluetooth.mjs` adds the Kilter-only connection flow. The API
level comes from the board's Bluetooth name (`…@3` → v3; a bare `Kilter Board`
controller → v2 with acknowledged writes). Override it under **Connection
settings** if a climb lights wrongly. Placement→LED positions come from Boardsesh's
generated LED table and match BoardLib's `placements ⋈ leds` for layout 1, size
10, sets 1 and 20. Model roles map to Kilter role codes 12–15 (start green, hand
cyan, finish magenta, foot orange). A climb with an unknown or duplicate hold, or
one too bright for an API v2 controller's power budget, is refused without
sending anything.

## Snake

The **Snake** tab plays snake on the 17 × 19 grid of bolt-on holds, on screen and
on a connected wall. Steer with W A S D or the arrow keys; pressing a W A S D key
also starts a game heading that way. The head is green, the body cyan, and the
food magenta; a crashed snake turns magenta. Each move waits for the wall to
light, so a connected board can slow high speeds. Frames are capped at 120 holds,
inside an API v2 controller's power budget. Leaving the tab or stopping a game
clears the LEDs. The game logic is in `src/snake.mjs`.

## Existing implementation and licenses

The visualization reuses [Boardsesh](https://github.com/boardsesh/boardsesh),
pinned to commit `15f6a8588585992725d9f9ba181271a0de9165f9` under Apache 2.0:

- Original 12×12 image layers for bolt-on and screw-on sets (`45-1.webp` and
  `46-1.webp`), including the kickboard.
- Traced silhouettes for all 476 placements from its `kilter/1-10` geometry.
- The unmodified Rust/WebAssembly **Aura** renderer, using Boardsesh's editor
  glow settings and role palette: green start, cyan hand, magenta finish,
  orange foot.

The local adapter follows Boardsesh's image layering and coordinate projection.
It uses traced SVG paths for precise mouse/touch and keyboard hit areas, with
solid/dashed outlines for locked/generated holds. A separate rendering worker
calls Boardsesh's actual WASM renderer; pending visual updates are coalesced so
model generation can continue independently. The editor uses no dimming veil,
matching Boardsesh's editing mode.

All artwork, geometry, JS, and WASM files are vendored locally under
`public/vendor/boardsesh/`. `provenance.json` records upstream paths and SHA-256
checksums; the upstream license and adaptation notices are retained. No
Boardsesh API, hosted renderer, or external asset server is used at runtime.

Inference uses the pinned `onnxruntime-web` package, also MIT licensed. Its
required JS/WASM assets and license are bundled during build. `build.mjs` is a
small static asset packager; no web framework or inference service is required.

## Verification

```bash
npm test
npm run build
npm run test:browser
```

Browser tests use Chrome at `CHROME_PATH`, the system Chrome installation when
available, or Playwright's Chromium. On a machine without either browser, install
the test browser with `npx playwright install chromium`. The tests use a local
server and cover real model completion, preservation of arbitrary selected holds
and roles, export, numerical inference parity, cancellation, keyboard use,
mobile overflow, and generation from nothing and from two start holds plus a
foot. Tests also verify the copied assets against their pinned
checksums, all 476 coordinate/outline mappings, Kilter role codes, and rendering
through the actual Boardsesh WASM binary. Decoding unit tests replay the exported
Python decoding states exactly, and use adversarial logits that try to repeat or
recolor picked holds or end without a finish.
Bluetooth tests decode the bytes actually written for API v2 and v3, then check
each hold's LED position and color, chunking, and disconnect handling. The
browser test drives the page against a mocked board. `npm test` builds `dist/`
first because the Bluetooth module imports the compiled protocol from there.
These tests cannot replace a check on a physical board.
Browser tests use port 4174 (override with `PLAYWRIGHT_PORT`), leaving a running
app on port 4173 untouched.

## Local artifacts

The interface is kept local, as requested. No application source or model weights
have been uploaded or published. `.openai/hosting.json` retains an unused private
Site registration; it is not needed to run the app. `dist/` contains the complete
static app. This training repository gitignores generated model exports,
dependencies, build output, and browser-test results.
