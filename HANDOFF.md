# Handoff: wrap up the v2 climb transformer

You are finishing work that was paused while two training runs completed. Read
this whole file first. All commands run from the repository root,
`/home/john/School/kilter-proj`. Use `.venv/bin/python`. Do not commit anything
unless the user asks.

## The goal

Review the original transformer (`kilter/`) for things that make it perform
poorly, then build an improved version in a separate folder (`kilter_v2/`) and
update everything else to use it. The improved model must be able to generate a
climb (a) from nothing, or (b) from the user's start holds and maybe some feet.

## Status

Done and tested:

- `kilter_v2/` package: `config.py`, `data.py`, `model.py`, `sampling.py`,
  `experiment.py`, `quality.py`, `export_web.py`, `__main__.py`,
  `configs/default.toml`, `tests/test_v2.py` (12 tests, passing). `kilter/` is
  untouched and its tests still pass.
- Web app switched to the v2 format. `web/src/decoding.mjs` was rewritten to
  mirror `kilter_v2/sampling.py`. `web/src/app.mjs` validates picks with the same
  rules, and `web/index.html` has new wording. `web/tests/decoding.test.mjs` was
  rewritten, and a v2 browser test was added to `web/tests/interface.spec.mjs`.
  `web/README.md` was updated. All web tests pass: `npm test` gives 20, and
  `npx playwright test` gives 5.
- The site currently serves the default run's epoch-1 checkpoint (exported to
  `web/public/model/`). It works, but it is not the final model.
- `.gitignore` now ignores `data/experiments_v2/`.

Steps 1–4 below were completed on 2026-10-29 on `/home/john`, after rebuilding the
data and retraining all three runs on the GPU (the original `/home/alex` runs were not
available). The large run was chosen then. The website now serves `null_rule_16`, seed 43; results are in
[kilter_v2/README.md](kilter_v2/README.md). Only step 5 remains.

Remaining (originally):

1. Confirm both runs finished: each run directory has a `summary.json`. If not,
   follow [TRAINING.md](TRAINING.md).
2. Evaluate and compare (commands below).
3. Export the better run to the website, rebuild, and run all tests.
4. Write `kilter_v2/README.md` (it does not exist yet), and add a short v2
   section to the root `README.md` pointing to it.
5. Optional, if the user wants it: an ascent-weighting experiment
   (`--set train.ascent_weighting=log --set name=climb_v2_weighted`).

## Runs

| Run | Directory |
| --- | --- |
| Baseline (`kilter/`) | `runs/20261030T004007463275Z_ar_transformer/` |
| v2 default (width 128, 4 layers, 0.9M params) | `runs/20261030T004007514421Z_climb_v2/` |
| v2 large (width 192, 6 layers, 2.9M params), **chosen** | `runs/20261030T004007515833Z_climb_v2_d192/` |

If the user restarted a run, its directory name changed. Use `ls runs/`.

## Commands for the remaining steps

```bash
V2=runs/20261030T004007514421Z_climb_v2          # repeat for the d192 run
BASE=runs/20261030T004007463275Z_ar_transformer/best.pt

# Per-epoch validation history (nats per climb for each given-hold mode).
cat $V2/metrics.jsonl | python3 -c "import sys,json
for l in sys.stdin:
    r=json.loads(l); v=r['validation']; print(r['epoch'], {m: round(v[m]['nll_per_climb'],2) for m in ('scratch','starts','starts_feet')})"

# Choose between runs on VALIDATION. Takes a few minutes per run; writes $V2/quality_validation.json.
.venv/bin/python -m kilter_v2 quality $V2/best.pt --baseline $BASE --samples 20

# Only after choosing: held-out test likelihood for the chosen run (writes best_test.json).
.venv/bin/python -m kilter_v2 evaluate $V2/best.pt --split test
.venv/bin/python -m kilter_v2 quality $V2/best.pt --baseline $BASE --split test --samples 20

# Try generation from the CLI.
.venv/bin/python -m kilter_v2 sample $V2/best.pt --grade 5 --angle 40 --count 3
.venv/bin/python -m kilter_v2 sample $V2/best.pt --grade 5 --angle 40 --given 1176:start --given 1192:start --given 1091:foot

# Put the chosen run on the website.
.venv/bin/python -m kilter_v2.export_web <chosen run>/best.pt
cd web && npm test && npx playwright test && cd ..

# Python tests.
.venv/bin/python -m unittest discover -s kilter_v2/tests -t .
.venv/bin/python -m unittest discover -s tests
```

`npm test` rebuilds `web/dist/`, so the site picks up the new export. To run
the site: `cd web && npm start`, then open http://127.0.0.1:4173.

Choose the run with the lower validation `selection_nll` (the mean of the three
modes). Check that its sample-quality numbers are not worse. Run test-split
evaluation only for the chosen run, after choosing.

## What was wrong with the baseline (`kilter/`), and the v2 fix

Measured on the baseline checkpoint above:

1. **Hold order is meaningless for the goal.** The baseline sorts holds by board
   ID, so start holds plus feet can't be given as a prefix. The old web app used
   approximate barrier decoding instead. When given a real climb's start holds
   plus every other foot, it kept adding start holds: only 17% of its completions
   had 1–2 starts (the 23-sample smoke test). *v2:* trains on
   `BOS <given> SEP <rest> EOS`, with a given set redrawn each epoch: 30% none,
   45% all starts plus a random share of the feet, 25% a random subset. Loss
   covers only the rest and EOS, so p(rest | given, grade, angle) is learned
   exactly. The rest is in canonical order: start, hand, finish, foot, each
   bottom to top then left to right (`Board.rank` in `kilter_v2/data.py`).
2. **No geometry.** The baseline learned 1,904 independent hold×role embeddings.
   *v2:* a token embedding is the hold embedding, plus an MLP of the hold's
   features (x, y, kickboard, screw-on set, Fourier features), plus a role
   embedding. The output weights are tied to this composed table, with a vocab
   bias.
3. **Underfitting.** The baseline has 154k parameters (width 48, 2 layers),
   sized by an arbitrary "0.5 parameters per training row" rule. Its training
   loss stayed above validation loss and was still falling at epoch 10. *v2:*
   0.9M and 2.9M parameter runs; learning rate 1e-3, betas (0.9, 0.98), 3%
   warmup, cosine decay to 5%.
4. **Weak grade conditioning.** The baseline passes the raw V grade through a
   single linear layer. Its samples average 12.1 holds at V2 and 11.4 at V11,
   while real climbs go from 14.0 to 10.8. *v2:* the grade (divided by 16) and
   angle (divided by 90) go through sinusoidal features and an MLP, added to
   every position. The interface is still `[V_grade, angle/90]`.
5. **No decoding rules.** Only 71% of the baseline's unconstrained samples were
   structurally valid (240 samples): 20% repeated a hold, and 9% had the wrong
   number of starts. *v2:* `allowed_tokens` masks only what no training climb
   does: breaking canonical order, repeating a hold, more than 2 starts or
   finishes, or ending without a start and a finish. `quality` also reports v2
   unconstrained, so we can show what the model learned on its own.
6. **Data noise and leakage.** v2 drops multi-frame climbs (1,370 rows),
   climbs with 0 or more than 2 starts (50) or finishes (6), and climbs with more
   than 40 holds (10). It also drops 816 training rows whose exact hold set
   appears in validation or test, so 1,033 duplicate hold sets under different
   UUIDs can't leak. The split uses the same UUID hash and seed as the baseline,
   so held-out rows match it: validation 16,936, test 16,897, train 303,547.
   42% of rows have a single ascent; `train.ascent_weighting=log` is
   implemented, but the runs above use `none`.
7. **Evaluation.** The baseline reported only per-token loss, which depends on
   the ordering. *v2* reports nats per climb. The from-nothing ("scratch") value
   is the full probability of a climb's hold/role set, which is comparable with
   the baseline whatever the ordering. It also reports conditional likelihood
   (given starts, and given starts plus feet), and sample checks against real
   held-out climbs: validity, holds and feet by grade, finish height, largest
   vertical gap between hand holds, novelty, and distinctness.

## Numbers so far (validation)

- Baseline, scratch: **50.64 nats/climb** (`quality.baseline_nll`; consistent
  with its 3.725 token loss × 13.65 tokens/climb).
- v2 default, after epoch 1 of 8: scratch **53.18**, starts **46.19**, starts
  plus feet **36.30**. Expect scratch to fall well below the baseline by the end.
  If it doesn't, investigate before claiming an improvement.
- Real climbs, average holds by grade: V0 15.4, V2 14.0, V5 12.1, V8 11.4,
  V11 10.8. Feet: V0 5.9, V5 4.1, V11 3.5.

## What to put in `kilter_v2/README.md`

Keep it the same tone as the root `README.md` (plain, factual, with commands).
Include:

- What v2 is and why (the findings above, condensed).
- Sequence format, canonical order, given-set mixture, model, and decoding rules.
- Commands: prepare, train, resume, evaluate, sample (with `--given`), quality,
  export, tests.
- Results table: baseline vs the chosen v2 run, on validation and test. Rows:
  scratch nats/climb; v2-only conditional nats/climb; sample validity (v2
  constrained, v2 unconstrained, baseline); starts-plus-feet completion validity
  (v2 vs baseline barrier); holds V2→V11 slope against real; novelty.
- Honest caveats. Climbability isn't verified. The decoding rules guarantee
  structural validity, so lead with the unconstrained numbers when saying what
  the model learned. The given-set mixture affects what "given starts" implies
  (the model assumes no more starts after it is given both).

## Gotchas

- `python -I` breaks `kilter_v2` imports from the repo root; run with
  `.venv/bin/python -m ...` from the root.
- `kilter_v2/quality.py` imports `kilter.experiment` only for the baseline comparison.
- The web page rejects the old `kilter.export_web` manifests (`format` must be
  `kilter-v2`). `web/public/model/` is gitignored, so the deployed model exists
  only on this machine.
- `npm test` runs `node build.mjs` first. That compiles the vendored Boardsesh
  Bluetooth TypeScript and copies `public/` into `dist/`.
- The Bluetooth feature (`web/src/bluetooth.mjs`, "Light up the wall" card) is
  unrelated to v2. Don't change it, but keep its tests passing.
