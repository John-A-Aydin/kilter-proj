# Climb transformer v2

`kilter_v2/` is a rebuilt climb generator. It can generate a climb **from nothing**,
or **complete one from given holds**, such as your start holds plus some feet. It
conditions on the same inputs as the baseline in [`kilter/`](../kilter/): numeric V grade and
`angle / 90`. The baseline is left unchanged, so the two can be compared on the same
held-out climbs. The [web interface](../web/README.md) uses v2.

## Why a v2

These problems were measured on the baseline checkpoint:

| Baseline problem | v2 change |
| --- | --- |
| Holds are sorted by board ID, so start holds plus feet can't be a prefix. The old web app used barrier decoding instead. Given a real climb's starts plus every other foot, only 11–17% of its completions had 1–2 starts. | Trains on `BOS <given> SEP <rest> EOS`, with a given set redrawn every epoch. Loss covers only the rest and EOS, so p(rest \| given, grade, angle) is learned directly. |
| No geometry: 1,904 independent hold×role embeddings. | Token embedding = hold embedding + MLP(hold features) + role embedding. Output weights are tied to this composed table, plus a vocabulary bias. |
| Underfit: 154k parameters, with training loss above validation and still falling at epoch 10. | 0.9M- and 2.9M-parameter models, learning rate 1e-3, betas (0.9, 0.98), 3% warmup, cosine decay to 5%. |
| Grade enters through one linear layer. | Grade/16 and angle/90 go through sinusoidal features and an MLP, added at every position. |
| Raw sampling: 20% of samples repeated a hold, and 9% had the wrong number of starts. | Decoding masks only what no training climb does (see below). Unconstrained sampling is still reported. |
| Noisy rows, and duplicate hold sets under different UUIDs across splits. | Filtering and leakage removal (see below). |
| Per-token loss only, which depends on the ordering. | Nats per climb for each given-hold mode, plus sample checks against real held-out climbs. |

## Data

`prepare` reads `data/processed/climbs.jsonl` and caches encoded arrays under
`data/experiments_v2/`, keyed by the content hash of the source, the vocabulary,
`data.py`, and the data settings. Rows are dropped if they are multi-frame (1,370),
have no V grade (27), have 0 or more than 2 starts (50) or finishes (6), or have more
than 40 holds (10). Then 816 training rows whose exact hold set also appears in
validation or test are removed. The split uses the same UUID hash and seed as the
baseline, so held-out climbs match: **train 303,547, validation 16,936, test 16,897**.

## Sequence format

Every hold/role token is listed in a **canonical order**: start, hand, finish, foot,
and within each role bottom to top, then left to right (`Board.rank` in `data.py`).
A climb becomes

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
y. The vocabulary has 4 special tokens (`PAD BOS EOS SEP`) plus 476 holds × 4 roles,
1,908 in total.

| Run | Width | Layers | Heads | Parameters |
| --- | ---: | ---: | ---: | ---: |
| default ([configs/default.toml](configs/default.toml)) | 128 | 4 | 4 | 902,004 |
| large | 192 | 6 | 6 | 2,856,180 |

Both ran for 8 epochs, with batch size 256 and AdamW weight decay 0.01. Checkpoints
are selected by `selection_nll`, the mean of validation nats per climb across the
three evaluation modes below.

## Null grade experiment

`train.p_null_grade` replaces the grade with null (stored as `-1`) on that share of
training rows, redrawn every batch. It needs `model.null_grade = true`, which adds
a null flag to the condition inputs. Angle is always kept. To train it, run
`python -m kilter_v2 train --config kilter_v2/configs/null_grade.toml`. To sample
with a null grade, leave out `--grade`.

All runs use the large model and 8 epochs. Test nats per climb, mean over seeds 42,
43 and 44 (standard deviations are 0.03 to 0.07):

| Null share | Grade given | Scratch | Starts | Starts + feet | Mean |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0% (`climb_v2_d192`) | real | 45.31 | 38.06 | 29.72 | 37.70 |
| 50% (`climb_v2_null_grade`) | real | 45.37 | 38.10 | 29.76 | 37.74 |
| 50% (`climb_v2_null_grade`) | null | 45.98 | 38.57 | 30.16 | 38.23 |
| 100% (`climb_v2_no_grade`) | null | 45.96 | 38.55 | 30.13 | 38.21 |

The 50% model almost matches each specialist in its own mode, so one model can do
both. Knowing the grade is worth only about 0.5 nats per climb.

## Three-seed results

`.venv/bin/python experiments/run_seeds.py` trains the baseline, v2 default, v2 large,
50% null, and always-null models on seeds 42, 43 and 44, reusing finished runs. It
scores each on validation and test, and samples 50 climbs per grade and angle cell.
The data split stays fixed at `split_seed = 42`.

[`results/`](../results/) keeps each run's config, metrics, evaluations and log, plus
`all_metrics.csv`, `summary.csv`, `epochs.csv`, `samples_by_grade.csv` and
`checkpoints.json`. Checkpoints stay in the git-ignored `runs/` folder, and
`checkpoints.json` records their checksums. The single-seed tables under Results
below use 20 samples per cell.

## Improving the 50% null model

The first three-seed runs always picked their last epoch, so the learning rate ran
out before the model did. About 10% of samples with rules off repeated a hold, and
42% of training rows have a single ascent. Each change below tests one of those
findings, from `climb_v2_null_grade`:

| Model | Change |
| --- | --- |
| `null_16_epochs` | `train.epochs = 16` |
| `null_d256` | width 256, 8 layers, 8 heads (6.6M parameters) |
| `null_ascent_weighted` | `train.ascent_weighting = "log"` |
| `null_rule_loss` | `train.rule_loss = true` |
| `null_rule_16` (chosen) | rule loss and 16 epochs |
| `null_rule_16_d256` | rule loss, 16 epochs, width 256, 8 layers, 8 heads |

`train.rule_loss` masks illegal tokens during training, so the loss only ranks
legal ones. These models are scored with the same mask (`nll_rules`), which is how
all of them are compared here. A model trained this way needs the decoding rules,
so its rules-off samples mean nothing. Mean nats per climb over seeds 42, 43 and 44,
with rules applied:

| Model | Validation mean | Test scratch | Test starts | Test starts + feet | Test mean | Rules-on valid (test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `climb_v2_null_grade` | 37.60 | 45.06 | 37.83 | 29.50 | 37.46 | 96.1% |
| `null_ascent_weighted` | 38.14 | 45.67 | 38.41 | 30.00 | 38.03 | 96.9% |
| `null_d256` | 37.44 | 44.90 | 37.68 | 29.34 | 37.31 | 95.9% |
| `null_16_epochs` | 37.37 | 44.83 | 37.66 | 29.26 | 37.25 | 94.1% |
| `null_rule_loss` | 37.36 | 44.75 | 37.54 | 29.24 | 37.18 | 95.3% |
| `null_rule_16_d256` | 37.30 | 44.77 | 37.58 | 29.16 | 37.17 | 95.9% |
| `null_rule_16` | 37.14 | 44.57 | 37.43 | 29.01 | 37.01 | 94.0% |

`null_rule_16` was chosen on validation, and the website serves its best validation
seed, 43. It scores 0.46 nats better than `climb_v2_null_grade` on test, but about 2
points fewer of its samples pass the validity checks. Most of its runs stopped
improving between epochs 12 and 15, so more epochs alone are unlikely to help much.
`null_rule_16_d256` stopped early at 11 or 12 epochs, with its best at 9 or 10. It
fits worse than the smaller model, so the extra size stops helping once training is
long enough.

## Hold representations

`.venv/bin/python experiments/hold_embeddings.py` asks what `null_rule_16` learned about
holds in each role, writing to [`results/hold_embeddings/`](../results/hold_embeddings/).
Token embeddings are `hold + geometry + role`, so a projection of the table separates the
roles for any model. Every measure is therefore repeated on an untrained copy. Means
over three seeds:

| Measure | Untrained | Trained |
| --- | ---: | ---: |
| Role recognized on unseen holds, position removed | 86% | 78% |
| Same hold in different roles, hidden-state cosine | 0.86 | 0.46 |
| Same role, wall distance vs similarity (Spearman) | -0.21 | -0.65 |
| Hold and role preferences vs real usage (Spearman) | 0.23 | 0.89 |
| Foot share explained by the hold vector (held-out R²) | 0.30 | 0.94 |

Role separation is not learned. What training adds is that a hold's representation
comes to depend on its role, nearby holds grow similar, and foot use is stored in the
hold vectors, mostly through the learned geometry network. The foot share comes from
the training climbs, so this shows what the model stored, not that it generalizes.

## Grading as a downstream task

`.venv/bin/python experiments/grade_climbs.py` predicts a climb's V grade from its holds
and angle, writing to [`results/grading/`](../results/grading/). The model is frozen and
read with a null grade, so the label cannot leak in. Settings are chosen on validation.
Test MSE over 16,897 rows, mean of three seeds (grade variance 6.66):

| Method | MSE | RMSE (grades) | Within 1 grade | MSE, 5+ ascents |
| --- | ---: | ---: | ---: | ---: |
| Training mean at that angle | 5.04 | 2.24 | 34.1% | 6.68 |
| Bag of holds, ridge | 2.86 | 1.69 | 47.7% | 2.86 |
| Untrained model, MLP head | 2.39 | 1.55 | 52.6% | 2.06 |
| Bag of holds, MLP head (main baseline) | 2.03 | 1.42 | 57.3% | 1.61 |
| `null_rule_16`, ridge | 1.85 | 1.36 | 58.9% | 1.54 |
| `null_rule_16`, Bayes over grades, no head | 1.83 | 1.35 | 60.4% | 1.45 |
| `null_rule_16`, MLP head | 1.70 | 1.30 | 62.4% | 1.32 |

The Bayes grader uses p(climb | grade, angle) from the generative model with a prior
per angle. Errors are largest at both ends of the scale, where labels are noisier or rarer.

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
.venv/bin/python -m kilter_v2 prepare

# Train the default model, or the large one.
.venv/bin/python -m kilter_v2 train
.venv/bin/python -m kilter_v2 train --set name=climb_v2_d192 --set model.d_model=192 \
  --set model.n_layers=6 --set model.n_heads=6

# Resume from the last completed epoch (uses the run's saved config).
.venv/bin/python -m kilter_v2 train --resume runs/<run>/last.pt

# Nats per climb for each given-hold mode. Writes best_<split>.json.
.venv/bin/python -m kilter_v2 evaluate runs/<run>/best.pt --split validation

# Generate from nothing, or from given holds (HOLD:ROLE, with role hand|foot|start|finish).
.venv/bin/python -m kilter_v2 sample runs/<run>/best.pt --grade 5 --angle 40 --count 3
.venv/bin/python -m kilter_v2 sample runs/<run>/best.pt --grade 5 --angle 40 \
  --given 1176:start --given 1192:start --given 1091:foot

# Sample-quality report against the baseline. Writes quality_<split>.json.
.venv/bin/python -m kilter_v2 quality runs/<run>/best.pt \
  --baseline runs/<baseline run>/best.pt --samples 20

# Completed-run summaries as CSV.
.venv/bin/python -m kilter_v2 compare runs

# Put a checkpoint and the real-climb catalogue on the website, then rebuild and test it.
.venv/bin/python -m kilter_v2.export_web runs/<run>/best.pt
cd web && npm test && npx playwright test

# Tests.
.venv/bin/python -m unittest discover -s kilter_v2/tests -t .
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
compares them with v2 from scratch (with and without decoding rules), the baseline
from scratch, and completions of each climb's starts plus every other foot by v2 and
by the baseline's barrier decoder. A sample is **valid** when it has no repeated
hold, 1–2 starts, 1–2 finishes, its highest start is below its lowest finish, and it
has at least one foot.

## Results

The baseline was retrained from the same config and data on the same machine.
Validation loss was 3.726 (originally 3.725) and test loss 3.748 (originally 3.745).
The large v2 run was chosen on validation `selection_nll` (37.86, vs 38.55 for the
default). Its sample checks were equal or better, so test metrics were computed
only for it.

### Likelihood (nats per climb, lower is better)

| | Validation | Test |
| --- | ---: | ---: |
| Baseline, scratch | 50.66 | 50.54 |
| v2 default, scratch | 46.16 | — |
| **v2 large, scratch** | **45.44** | **45.34** |
| v2 large, given starts | 38.25 | 38.10 |
| v2 large, given starts + every other foot | 29.88 | 29.76 |

The baseline can't score the two conditional modes.

### Samples

| | Validation | Test |
| --- | ---: | ---: |
| Real held-out climbs, valid | 92.8% | 96.7% |
| v2 from scratch, with rules, valid | 95.7% | 96.2% |
| **v2 from scratch, unconstrained, valid** | **91.3%** | **87.7%** |
| v2 unconstrained, repeated a hold | 6.3% | 7.6% |
| Baseline from scratch, valid | 90.3% | 89.1% |
| v2 completion of starts + feet, valid | 98.1% | 98.1% |
| Baseline barrier completion of starts + feet, valid | 9.7% | 15.6% |
| Baseline barrier completion, 1–2 starts | 11.1% | 16.6% |
| Baseline barrier completion, mean holds | 23.7–25.9 | 22.4–25.9 |
| Mean holds V2 → V11 drop: real | 2.07 | 2.05 |
| … v2 with rules | 2.45 | 2.32 |
| … v2 unconstrained | 1.35 | 0.65 |
| … baseline | 1.47 | 1.70 |
| v2 samples not in training, all distinct | 100% | 100% |

Real climbs are not 100% valid under these checks. Some have no foot holds, or a
start at or above a finish.

The baseline sampler in `quality.py` is the old web decoder: it only allows
increasing hold IDs, so it can't repeat a hold. That is why its validity is higher
than the 71% measured earlier with the raw `kilter sample` sampler (240 samples).

## Caveats

- **Climbability isn't verified.** The checks are structural, plus summary
  statistics such as hold count, feet, finish height, and the largest vertical gap
  between hand holds. Nobody has climbed these.
- **Lead with the unconstrained numbers** when saying what the model learned. With
  the decoding rules, structural validity is guaranteed by construction. Without
  them, v2 still repeats a hold in about 7% of samples. On the test split, its
  validity is slightly below the baseline's ID-ordered sampler.
- **The grade effect on climb length is weak and noisy.** Each grade has only about
  50 samples. Unconstrained v2 shows a smaller V2→V11 drop than real climbs (0.65 on
  test), and every sampler produces slightly fewer holds than real climbs at low
  grades. The likelihood gain is solid; grade fidelity is not established.
- **"Given starts" means "these are all the starts."** In training, the starts mode
  always gives every start hold. A user who gives one start is out of distribution
  for that mode, though the 25% random-subset mode partly covers it.
- The baseline barrier completions are poor mostly because they are a decoding
  workaround on a model that never learned conditional completion, not because of
  the baseline's size.
