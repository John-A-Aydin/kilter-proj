# Climb transformer

[`kilter/`](../kilter/) is the climb generator. It can generate a climb **from nothing**,
or **complete one from given holds**, such as your start holds plus some feet. It
conditions on the same inputs as the [baseline](baseline.md): numeric V grade and
`angle / 90`. The baseline is kept unchanged in [`kilter/baseline/`](../kilter/baseline/),
so the two can be compared on the same held-out climbs. The [web interface](../web/README.md)
uses this model.

This model was built as a second version of the baseline, so run names (`climb_v2`,
`climb_v2_d192`, …) and result keys (`v2_scratch`, …) say `v2`.

## Why it replaced the baseline

These problems were measured on the baseline checkpoint:

| Baseline problem | Change |
| --- | --- |
| Holds are sorted by board ID, so start holds plus feet can't be a prefix. The old web app used barrier decoding instead. Given a real climb's starts plus every other foot, only 11–17% of its completions had 1–2 starts. | Trains on `BOS <given> SEP <rest> EOS`, with a given set redrawn every epoch. Loss covers only the rest and EOS, so p(rest \| given, grade, angle) is learned directly. |
| No geometry: 1,904 independent hold×role embeddings. | Token embedding = hold embedding + MLP(hold features) + role embedding. Output weights are tied to this composed table, plus a vocabulary bias. |
| Underfit: 154k parameters, with training loss above validation and still falling at epoch 10. | 0.9M- and 2.9M-parameter models, learning rate 1e-3, betas (0.9, 0.98), 3% warmup, cosine decay to 5%. |
| Grade enters through one linear layer. | Grade/16 and angle/90 go through sinusoidal features and an MLP, added at every position. |
| Raw sampling: 20% of samples repeated a hold, and 9% had the wrong number of starts. | Decoding masks only what no training climb does (see below). Unconstrained sampling is still reported. |
| Noisy rows, and duplicate hold sets under different UUIDs across splits. | Filtering and leakage removal (see below). |
| Per-token loss only, which depends on the ordering. | Nats per climb for each given-hold mode, plus sample checks against real held-out climbs. |

## Data

`prepare` reads the committed `data/training/climbs.jsonl.gz` and caches encoded arrays under
`data/cache/climb/`, keyed by the content hash of the source, the vocabulary,
`data.py`, and the data settings. Rows are dropped if they are multi-frame (1,370),
have no V grade (27), have 0 or more than 2 starts (50) or finishes (6), or have more
than 40 holds (10). Then 816 training rows whose exact hold set also appears in
validation or test are removed. The split uses the same UUID hash and seed as the
baseline, so held-out climbs match: **train 303,547, validation 16,936, test 16,897**.

## Sequence format

Every hold/role token is listed in a **canonical order**: start, hand, finish, foot,
and within each role bottom to top, then left to right (`Board.rank` in
[`kilter/data.py`](../kilter/data.py)). A climb becomes

```
BOS <given holds, canonical order> SEP <remaining holds, canonical order> EOS
```

Positions restart at `SEP`, and a segment embedding marks the given and rest
blocks. For each training row and epoch, the given set is drawn from a mixture:

| Probability | Given set |
| ---: | --- |
| 30% | Nothing (generation from scratch) |
| 45% | All start holds, plus each foot with a random per-row probability |
| 25% | A random subset of all holds |

Change it with `train.p_scratch`, `train.p_starts`, and `train.p_random`.

## Model

Pre-LayerNorm causal decoder: GELU feed-forward of width `4 × d_model`, dropout 0.1,
learned positions, and PyTorch scaled dot-product attention. Each hold's features
are x, y, a kickboard flag, a screw-on-set flag, and 6 Fourier frequencies of x and
y. With `model.hold_labels = "data/hold_labels.csv"`, each hold also gets its
hand-labeled type (one-hot over 7 types) and grip directions (multi-hot over 16
compass points), read by [kilter/hold_labels.py](../kilter/hold_labels.py). Only the
`type`, `grip` and `on_kickboard` columns are read, and `on_kickboard` is only checked
against the vocabulary's kickboard flag. The run saves the labels it trained with as
`hold_labels.json`. The vocabulary has 4 special tokens (`PAD BOS EOS SEP`) plus 476 holds × 4 roles,
1,908 in total.

| Run | Width | Layers | Heads | Parameters |
| --- | ---: | ---: | ---: | ---: |
| default ([configs/climb.toml](../configs/climb.toml)) | 128 | 4 | 4 | 902,004 |
| large | 192 | 6 | 6 | 2,856,180 |

Both ran for 8 epochs, with batch size 256 and AdamW weight decay 0.01. Checkpoints
are selected by `selection_nll`, the mean of validation nats per climb across the
three evaluation modes below.

**Null grade.** `train.p_null_grade` replaces the grade with null (stored as `-1`) on
that share of training rows, redrawn every batch. It needs `model.null_grade = true`,
which adds a null flag to the condition inputs. Angle is always kept.
[configs/climb_null_grade.toml](../configs/climb_null_grade.toml) trains the large model
with a 50% share. To sample with a null grade, leave out `--grade`.

**Rule loss.** `train.rule_loss` masks illegal tokens during training, so the loss only
ranks legal ones. A model trained this way needs the decoding rules, so its rules-off
samples mean nothing. The model the website serves, `null_rule_16`, uses it; see
[experiments](../experiments/README.md).

## Decoding rules

`sampling.allowed_tokens` (mirrored in `web/src/decoding.mjs`) masks continuations
that no training climb has:

- breaking canonical order, or repeating a hold;
- more than 2 start or 2 finish holds;
- feet before a finish has been placed;
- `EOS` before at least one start and one finish;
- more than 40 holds.

Everything else is left to the model. These rules guarantee structural validity, so
the unconstrained numbers below are the honest measure of what the model learned.

## Commands

Run from the repository root. Use `uv run --frozen python` or `.venv/bin/python`.
Don't use `python -I`: it breaks the package import from the root.

```bash
# Encode and cache the dataset (training does this automatically too).
.venv/bin/python -m kilter prepare

# Train the default model, or the large one.
.venv/bin/python -m kilter train
.venv/bin/python -m kilter train --set name=climb_v2_d192 --set model.d_model=192 \
  --set model.n_layers=6 --set model.n_heads=6

# Resume from the last completed epoch (uses the run's saved config).
.venv/bin/python -m kilter train --resume runs/<run>/last.pt

# Nats per climb for each given-hold mode. Writes best_<split>.json.
.venv/bin/python -m kilter evaluate runs/<run>/best.pt --split validation

# Generate from nothing, or from given holds (HOLD:ROLE, with role hand|foot|start|finish).
.venv/bin/python -m kilter sample runs/<run>/best.pt --grade 5 --angle 40 --count 3
.venv/bin/python -m kilter sample runs/<run>/best.pt --grade 5 --angle 40 \
  --given 1176:start --given 1192:start --given 1091:foot

# Sample-quality report against the baseline. Writes quality_<split>.json.
.venv/bin/python -m kilter quality runs/<run>/best.pt \
  --baseline runs/<baseline run>/best.pt --samples 20

# Completed-run summaries as CSV.
.venv/bin/python -m kilter compare runs

# Put a checkpoint and the real-climb catalogue on the website, then rebuild and test it.
.venv/bin/python -m kilter.export_web runs/<run>/best.pt
cd web && npm test && npx playwright test
```

`sample` also accepts `--temperature`, `--top-k`, `--seed`, and `--unconstrained`.
Other settings can be overridden with `--set section.key=value`; unknown keys are
rejected. `train.ascent_weighting=log` weights each row by `min(log2(1 + ascents), 6)`.
42% of rows have a single ascent. The runs below use `none`.

Training uses `torch.use_deterministic_algorithms(True)`. On CUDA, that needs
`export CUBLAS_WORKSPACE_CONFIG=:4096:8` before `train`. The lockfile installs
CPU PyTorch; the results below come from `torch==2.15.0+cu132` installed into
`.venv` on an RTX 4060 Ti, where the large run took about 9 minutes. GPU results
are not bit-identical to CPU results.

## Evaluation modes

| Mode | Given | Scored |
| --- | --- | --- |
| `scratch` | nothing | the whole climb |
| `starts` | all start holds | everything else |
| `starts_feet` | start holds plus every other foot | everything else |

`scratch` nats per climb is the full log-probability of a climb's hold/role set, so
it can be compared with the baseline whatever the ordering. The baseline's value is
its sequence NLL over `BOS <holds by ID> EOS`.

The `quality` report samples 20 held-out climbs for each of V2, V5, V8, V11 and 20°,
40°, 50° (207 validation and 211 test climbs, since some cells have fewer). It
compares them with the model from scratch (with and without decoding rules), the
baseline from scratch, and completions of each climb's starts plus every other foot
by the model and by the baseline's barrier decoder. A sample is **valid** when it has
no repeated hold, 1–2 starts, 1–2 finishes, its highest start is below its lowest
finish, and it has at least one foot.

## First results

These are single-seed runs of the default and large models against the baseline.
Three-seed results and later changes are in [experiments](../experiments/README.md).

The baseline was retrained from the same config and data on the same machine.
Validation loss was 3.726 (originally 3.725) and test loss 3.748 (originally 3.745).
The large run was chosen on validation `selection_nll` (37.86, vs 38.55 for the
default). Its sample checks were equal or better, so test metrics were computed
only for it.

### Likelihood (nats per climb, lower is better)

| | Validation | Test |
| --- | ---: | ---: |
| Baseline, scratch | 50.66 | 50.54 |
| Default, scratch | 46.16 | — |
| **Large, scratch** | **45.44** | **45.34** |
| Large, given starts | 38.25 | 38.10 |
| Large, given starts + every other foot | 29.88 | 29.76 |

The baseline can't score the two conditional modes.

### Samples

| | Validation | Test |
| --- | ---: | ---: |
| Real held-out climbs, valid | 92.8% | 96.7% |
| Model from scratch, with rules, valid | 95.7% | 96.2% |
| **Model from scratch, unconstrained, valid** | **91.3%** | **87.7%** |
| Model unconstrained, repeated a hold | 6.3% | 7.6% |
| Baseline from scratch, valid | 90.3% | 89.1% |
| Model completion of starts + feet, valid | 98.1% | 98.1% |
| Baseline barrier completion of starts + feet, valid | 9.7% | 15.6% |
| Baseline barrier completion, 1–2 starts | 11.1% | 16.6% |
| Baseline barrier completion, mean holds | 23.7–25.9 | 22.4–25.9 |
| Mean holds V2 → V11 drop: real | 2.07 | 2.05 |
| … model with rules | 2.45 | 2.32 |
| … model unconstrained | 1.35 | 0.65 |
| … baseline | 1.47 | 1.70 |
| Model samples not in training, all distinct | 100% | 100% |

Real climbs are not 100% valid under these checks. Some have no foot holds, or a
start at or above a finish.

The baseline sampler in `quality.py` is the old web decoder: it only allows
increasing hold IDs, so it can't repeat a hold. That is why its validity is higher
than the 71% measured earlier with the raw `kilter.baseline sample` sampler (240 samples).

## Caveats

- **Climbability isn't verified.** The checks are structural, plus summary
  statistics such as hold count, feet, finish height, and the largest vertical gap
  between hand holds. Nobody has climbed these.
- **Lead with the unconstrained numbers** when saying what the model learned. With
  the decoding rules, structural validity is guaranteed by construction. Without
  them, the model still repeats a hold in about 7% of samples. On the test split, its
  validity is slightly below the baseline's ID-ordered sampler.
- **The grade effect on climb length is weak and noisy.** Each grade has only about
  50 samples. The unconstrained model shows a smaller V2→V11 drop than real climbs
  (0.65 on test), and every sampler produces slightly fewer holds than real climbs at
  low grades. The likelihood gain is solid; grade fidelity is not established.
- **"Given starts" means "these are all the starts."** In training, the starts mode
  always gives every start hold. A user who gives one start is out of distribution
  for that mode, though the 25% random-subset mode partly covers it.
- The baseline barrier completions are poor mostly because they are a decoding
  workaround on a model that never learned conditional completion, not because of
  the baseline's size.
