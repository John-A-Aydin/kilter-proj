# Experiments

Each experiment is one script in [`experiments/`](.). It writes its results to
`results/<script name>/`, which is committed. Checkpoints stay in the git-ignored
`runs/` folder; [`results/compare_models/checkpoints.json`](../results/compare_models/checkpoints.json)
records which run each result came from and its checksum. Run scripts from the
repository root.

| Script | Question | Results |
| --- | --- | --- |
| [`compare_models.py`](compare_models.py) | How do the baseline and each model variant compare over three seeds? | [`results/compare_models/`](../results/compare_models/) |
| [`hold_embeddings.py`](hold_embeddings.py) | What does the chosen model learn about holds in each role? | [`results/hold_embeddings/`](../results/hold_embeddings/) |
| [`grade_climbs.py`](grade_climbs.py) | Does the model help predict a climb's grade? | [`results/grade_climbs/`](../results/grade_climbs/) |

To add an experiment, add a script to `experiments/` with a docstring that says what it measures
and how to run it, write its outputs to `results/<script name>/`, and add a row above
and a section below. New model settings go in [`configs/`](../configs/) or as
`--set` overrides, so every experiment trains through the same `python -m kilter train`.

## Model comparison

```bash
.venv/bin/python experiments/compare_models.py
```

Trains the baseline, the default and large models, the null-grade models, and the
variants below on seeds 42, 43 and 44, reusing finished runs. It scores each on
validation and test, and samples 50 climbs per grade and angle cell. The data split
stays fixed at `split_seed = 42`. Outputs are `all_metrics.csv` (every number for
every run), `summary.csv` (mean, standard deviation, minimum and maximum over seeds),
`epochs.csv`, `samples_by_grade.csv` and `checkpoints.json`.

The baseline's three-seed means are 3.721 validation and 3.740 test loss per token,
and 50.59 validation and 50.45 test nats per climb.

Real climbs get shorter as they get harder. From V2 to V11, real held-out climbs
lose 2.2 holds on average. The large model's samples lose 2.2 with decoding rules
and 1.9 without, and the baseline's lose 0.9. Each grade has about 50 samples per seed:

<!-- chart: holds-by-grade -->

### Null grade

All runs use the large model and 8 epochs (see [null grade](../docs/model.md#model)).
Test nats per climb, mean over seeds 42, 43 and 44 (standard deviations are 0.03 to 0.07):

| Null share | Grade given | Scratch | Starts | Starts + feet | Mean |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0% (`climb_v2_d192`) | real | 45.31 | 38.06 | 29.72 | 37.70 |
| 50% (`climb_v2_null_grade`) | real | 45.37 | 38.10 | 29.76 | 37.74 |
| 50% (`climb_v2_null_grade`) | null | 45.98 | 38.57 | 30.16 | 38.23 |
| 100% (`climb_v2_no_grade`) | null | 45.96 | 38.55 | 30.13 | 38.21 |

The 50% model almost matches each specialist in its own mode, so one model can do
both. Knowing the grade is worth only about 0.5 nats per climb.

### Improving the 50% null model

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

Rule-loss models are scored with the same mask (`nll_rules`), which is how all of them
are compared here. Mean nats per climb over seeds 42, 43 and 44, with rules applied:

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

```bash
.venv/bin/python experiments/hold_embeddings.py
```

Asks what `null_rule_16` learned about holds in each role. Token embeddings are
`hold + geometry + role`, so a projection of the table separates the roles for any
model. Every measure is therefore repeated on an untrained copy. Means over three seeds:

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

```bash
.venv/bin/python experiments/grade_climbs.py
```

Predicts a climb's V grade from its holds and angle. The model is frozen and read with
a null grade, so the label cannot leak in. Settings are chosen on validation. Test MSE
over 16,897 rows, mean of three seeds (grade variance 6.66):

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

<!-- chart: grading-error -->
