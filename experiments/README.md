# Experiments

Each experiment is one script in [`experiments/`](.). It writes its results to
`results/<script name>/`, which is committed. Checkpoints stay in the git-ignored
`runs/` folder; [`results/compare_models/checkpoints.json`](../results/compare_models/checkpoints.json)
records which run each result came from and its checksum. Run scripts from the
repository root.

| Script | Question | Results |
| --- | --- | --- |
| [`compare_models.py`](compare_models.py) | How do the baseline and each model variant compare over three seeds, on validation? | [`results/model_selection/`](../results/model_selection/), and the historical [`results/compare_models/`](../results/compare_models/) |
| [`final_test.py`](final_test.py) | How do the models chosen on validation score on test? Opens a test set once, on record. | `results/final_test/` |
| [`hold_embeddings.py`](hold_embeddings.py) | What does the chosen model learn about holds in each role? | [`results/hold_embeddings/`](../results/hold_embeddings/) |
| [`tag_conditioning.py`](tag_conditioning.py) | Does conditioning on style tags from climb names and descriptions help the model, and does turning a tag on change what it sets? | [`results/tag_conditioning/`](../results/tag_conditioning/) |
| [`hold_label_probes.py`](hold_label_probes.py) | Do the chosen model's hold representations already encode the hand-labeled hold types and grip directions? | [`results/hold_label_probes/`](../results/hold_label_probes/) |
| [`grade_climbs.py`](grade_climbs.py) | Does the model help predict a climb's grade? | [`results/grade_climbs/`](../results/grade_climbs/) |
| [`bert_grader.py`](bert_grader.py) | How well does a BERT-style encoder trained only for grading do, on the same split? | [`results/bert_grader/`](../results/bert_grader/) |
| [`finetune_grader.py`](finetune_grader.py) | How much does freezing the model cost for grading, against fine-tuning it end to end? | [`results/finetune_grader/`](../results/finetune_grader/) |
| [`lightgbm_grader.py`](lightgbm_grader.py) | How well do gradient-boosted trees on the bag of holds plus geometry features grade? | [`results/lightgbm_grader/`](../results/lightgbm_grader/) |

To add an experiment, add a script to `experiments/` with a docstring that says what it measures
and how to run it, write its outputs to `results/<script name>/`, and add a row above
and a section below. New model settings go in [`configs/`](../configs/) or as
`--set` overrides, so every experiment trains through the same `python -m kilter train`.

## Model comparison

```bash
# Which runs would be reused or trained, and why.
.venv/bin/python experiments/compare_models.py --status
# Train what is missing and score on validation, on the historical or the family split.
.venv/bin/python experiments/compare_models.py --models climb_v2_null_grade null_rule_16
.venv/bin/python experiments/compare_models.py --split-manifest data/splits/family_v1.json --skip ar_transformer
# After choosing on validation: write down why, then open test once.
.venv/bin/python experiments/final_test.py --split-manifest data/splits/family_v1.json \
  --models null_rule_16 --criteria "Lowest mean validation selection_nll_rules over seeds 42-44"
```

`compare_models.py` trains the baseline, the default and large models, the null-grade
models, and the variants below on seeds 42, 43 and 44. It scores each on **validation
only**, and samples 50 climbs per grade and angle cell. `--models`, `--skip`, and
`--seeds` choose what to run. `--set section.key=value` applies to every model, for
example a few batches to check the pipeline; it changes the identity, so such runs are
never reused for real ones. The baseline supports only the historical split.

**Reuse is by identity** ([`kilter/identity.py`](../kilter/identity.py)). Training writes
`identity.json` into each run: the resolved config, a hash of each training source file,
the dataset fingerprint (input contents, preprocessing code, filters, and split), the
label and tag files, and the device type. A run is reused only when that identity
matches the request exactly. Each evaluation file records an `"evaluation"` identity: the
checkpoint hash, evaluator version and code, split, dataset, and every sampling setting.
It is rerun when that identity does not match. `--status` lists what differs. Runs made
before identities existed have none, so they are never reused.

**Test is a separate, recorded step.** `final_test.py` never trains. Every requested
run must already be scored on validation. Before it scores anything, it appends the
time, commit, run identities, and your `--criteria` to
`results/final_test/<split>/access.jsonl`, and it refuses to open a split's test set
again without `--reopen`. The family split's test set has not been opened.

Outputs go to `results/model_selection/<split>/` and `results/final_test/<split>/`:
`all_metrics.csv` (every number for every run), `summary.csv` (mean, standard deviation,
minimum and maximum over seeds), `epochs.csv`, `samples_by_grade.csv`, `runs.json` (run
and evaluation identities), and `sweep.json`. On a family split, `likelihood_distant`,
`likelihood_related`, and `likelihood_near_duplicate` give the same numbers per
[similarity band](../data/README.md#splits).

### Historical results

Everything below and in [`results/compare_models/`](../results/compare_models/) was
made before this workflow existed. It used the historical split, quality evaluator
version 1, and scored validation and test together. Variants were then chosen with test
results available, so treat its test numbers as **development evidence**, not a sealed
benchmark. Version 1's `valid` is the conjunction now called `legacy_valid_v1`. It
requires starts below finishes and at least one foot, which the decoding rules do not
guarantee, so campus-style climbs fail it. Its `novel` counted rows removed from
training (split 255) as training climbs ([details](../docs/model.md#evaluation-modes)).

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

"Rules-on valid" is version 1's `valid` (`legacy_valid_v1`), not schema validity, which the rules guarantee.

| Model | Validation mean | Test scratch | Test starts | Test starts + feet | Test mean | Rules-on valid (test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `climb_v2_null_grade` | 37.60 | 45.06 | 37.83 | 29.50 | 37.46 | 96.1% |
| `null_ascent_weighted` | 38.14 | 45.67 | 38.41 | 30.00 | 38.03 | 96.9% |
| `null_d256` | 37.44 | 44.90 | 37.68 | 29.34 | 37.31 | 95.9% |
| `null_16_epochs` | 37.37 | 44.83 | 37.66 | 29.26 | 37.25 | 94.1% |
| `null_rule_loss` | 37.36 | 44.75 | 37.54 | 29.24 | 37.18 | 95.3% |
| `null_rule_16_d256` | 37.30 | 44.77 | 37.58 | 29.16 | 37.17 | 95.9% |
| `null_rule_16` | 37.14 | 44.58 | 37.43 | 29.01 | 37.01 | 94.0% |

`null_rule_16` was chosen on validation, and the website served its best validation
seed, 43, until it moved to the same recipe with [style tags](#style-tags-from-climb-names-and-descriptions)
(`null_rule_16_tags`, also seed 43). It scores 0.46 nats better than `climb_v2_null_grade` on test, but about 2
points fewer of its samples pass the validity checks. Most of its runs stopped
improving between epochs 12 and 15, so more epochs alone are unlikely to help much.
`null_rule_16_d256` stopped early at 11 or 12 epochs, with its best at 9 or 10. It
fits worse than the smaller model, so the extra size stops helping once training is
long enough.

### Hand-labeled hold features

Every hold was labeled by hand in the website's hold labeler
([`data/hold_labels.csv`](../data/hold_labels.csv)): its type and one grip direction.
For most holds the direction is the way you pull. For pinches it is 90° to the
pulling force, either way. `null_rule_16_labels` is `null_rule_16` with those labels added to
each hold's features (`model.hold_labels = "data/hold_labels.csv"`), and nothing else
changed. Only the `type`, `grip` and `on_kickboard` columns are read. Type becomes a
one-hot over 7 types and grip a multi-hot over 16 compass directions, so a hold's
features grow from 28 to 51 columns. `on_kickboard` was already a feature, so it is
only checked against the vocabulary. All 476 holds have a type and exactly one direction:
195 edges, 170 foot chips, 80 jugs, 22 pinches and 9 slopers.

Until 2026-11-26, 53 holds had several directions (every pinch among them, usually two
opposite sides). Each now has one. The old runs also predate the relabeling of holds
1283 and 1298 from edge to pinch. Both versions were trained and are shown below, with
`null_rule_16` retrained under the current code as the control. It reproduced the
historical runs to every digit. The model choice is made on validation, so the table is
validation. Test scratch comes from the [style tag experiment](#tags-make-real-climbs-more-predictable),
which scores the historical test split (development numbers). Mean over seeds 42, 43 and 44,
nats per climb, with rules applied:

| Model | Grip labels | Validation mean | Scratch | Starts | Starts + feet | Rules-on valid | Test scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `null_rule_16` | none | 37.14 | 44.70 | 37.60 | 29.14 | 95.0% | 44.58 |
| `null_rule_16_labels` | several directions (old) | 37.12 | 44.67 | 37.57 | 29.12 | 94.3% | 44.56 |
| `null_rule_16_labels` | one direction (2026-11-26) | 37.13 | 44.68 | 37.58 | 29.13 | 94.2% | 44.56 |

The labels make no measurable difference, with either grip scheme. The validation mean
improves by 0.01 nats per climb (0.03 with the old labels). Seed for seed, it is 0.01 and
0.03 worse on seeds 42 and 43 and 0.08 better on seed 44, whose `null_rule_16` run is the
weakest of the three. The spread between seeds is 0.02 to 0.05. Validity is 0.8 points
lower, and its sign changes from seed to seed. Best epochs are the same, 12 to 15. A likely reason is that every hold
already has its own learned embedding, trained on about 300,000 climb rows, so anything
fixed about a hold, such as its type or grip, can already be learned from how it is used.
Labels like these would matter more for holds with few climbs, or for a board
the model has not seen. The labels model was not served. The next section
tests the explanation directly: [`null_rule_16`'s hold representations already
encode most of the labels](#hold-labels-in-the-learned-representations).

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

## Hold labels in the learned representations

```bash
.venv/bin/python experiments/hold_label_probes.py
```

Adding the hand labels as features did not improve `null_rule_16`
([above](#hand-labeled-hold-features)). This experiment asks whether that is because
the model already knows them. `null_rule_16` was trained on climbs only and never saw
[`data/hold_labels.csv`](../data/hold_labels.csv). For each of its hold
representations, a simple model called a *probe* is fit to predict the labels from
the representation, and then scored on holds it was not fit on. If a probe beats
chance and the controls below, the representation contains that label.

The probes are deliberately linear and small, so a high score means the label can be
read straight out of the representation, not computed by a powerful probe. All three
seeds (42, 43 and 44) are probed, and every number is their mean.

### What is predicted

| Task | Holds | Target | Score |
| --- | ---: | --- | --- |
| Hand-hold type | 306 | edge, jug, pinch or sloper | balanced accuracy, chance 25% |
| All five types | 476 | the four above or foot chip | balanced accuracy, chance 20% |
| Grip direction | 476 | each hold's one labeled direction | mean angle between predicted and labeled direction |

Balanced accuracy is the mean over types of the share of that type predicted
correctly (its recall), so each type counts equally: the 9 slopers weigh as much as
the 195 edges, and always guessing "edge" scores exactly chance. Hand-hold type is
the main test. A foot chip is easy to find without learning anything, because 153 of
the 170 are screw-on holds and the screw-on flag is one of the model's position inputs.

The angle between two directions is `|((p − l + 180) mod 360) − 180|` degrees, so it
runs from 0° (exact) to 180° (opposite). Every hold has exactly one labeled direction.
For edges, jugs, slopers and foot chips it is the direction you pull. For the 22 pinches
it is 90° to the pulling force, either way, so a pinch's direction is really an axis: 0°
and 180° mean the same pinch. The error above does not know that, so a pinch predicted
exactly backwards scores 180° when it is right. Before 2026-11-26, holds with several
directions (53 of them, every pinch among them) were left out of this task, and it had 423 holds.

### What is probed

Each representation is one vector per hold. The first three are controls: they
contain position or simple counts, but nothing the model learned.

| Representation | Width | What it is |
| --- | ---: | --- |
| Position | 28 | The model's fixed input features for the hold: x, y, kickboard flag, screw-on flag, and sin/cos of x and y at 6 frequencies. |
| Usage counts | 8 | How many training climbs use the hold as hand, foot, start and finish: `log(1 + count)` for each role, and each role's share of the hold's uses. |
| Untrained model | 192 | The hold vector below, computed by the same network re-initialized to random weights, as at the start of training. It is a random function of position. |
| Hold embedding | 192 | The learned per-hold vector, `hold_embedding[h]`. It has no position input, so anything it encodes was learned from the climbs. |
| Hold vector | 192 | `hold_embedding[h] + geometry(position features of h)`, where `geometry` is the learned two-layer network (linear, GELU, linear). Every token for the hold is this vector plus a role vector, so this is the model's whole static picture of the hold. |
| In-context state | 192 | The model's final hidden state (after the last layer norm) at each place the hold appears, over 60,000 training climbs read from scratch with their real grade. Each sequence position's mean state is subtracted first, since starts come first and feet last. The states are averaged per hold over all its roles. |

### How the probes work

[How the hold-type probes work](https://claude.ai/artifact/SYvidXFYzoMhqfXR6BkRdw) walks
through the pipeline below in more depth, with a diagram of where each representation is
read from the transformer, every tensor shape, and how the holds are split for fitting
and scoring.

1. **Prepare the features.** Each representation is centered. The 192-wide ones are
   cut to their top 20 principal components (by singular value decomposition, which
   uses no labels). Position and usage counts keep every column: a projection would
   drop low-variance columns such as the screw-on flag. Every column is then scaled
   to unit variance.
2. **Type probe: multinomial logistic regression.** One weight vector per type and a
   bias. The probabilities are `softmax(z · W + b)`, and the fit minimizes
   cross-entropy plus an L2 penalty `λ · ‖W‖² / n`. Each type's examples are weighted
   by `n / (types × count of that type)`, so rare types are not ignored. It is fit with
   L-BFGS (up to 200 iterations, strong Wolfe line search) in double precision.
3. **Grip probe: ridge regression onto a unit vector.** The labeled direction `θ` is
   turned into `(cos θ, sin θ)`, so 350° and 10° are close. Ridge regression predicts
   both numbers with the closed form `W = (ZᵀZ + λI)⁻¹ Zᵀ (Y − mean Y)`, and the
   predicted direction is `atan2` of the two predictions.
4. **Score on held-out holds: cross-validation.** The holds are split into 5 folds.
   Each fold is predicted by a probe fit on the other 4. Type folds are stratified, so
   every fold has the same mix of types. The whole 5-fold run is repeated 5 times with
   new random folds, and the scores are averaged.
5. **Choose the penalty without the test fold: nested cross-validation.** For every
   outer fold, `λ` is picked from 0.01, 0.1, 1, 10 and 100 by a 4-fold
   cross-validation on that fold's training holds only. The pick maximizes balanced
   accuracy for type and minimizes mean angle error for grip. The held-out fold never
   influences its own probe.
6. **Nulls.** The hold-vector probe is refit 20 times per seed on labels shuffled
   across holds, which shows what a probe scores when there is nothing to find. For
   grip there is also a constant guess: always predict the labels' average direction.

### Results

Mean ± standard deviation over the three seeds:

| Representation | Hand-hold type | All five types | Grip error | Grip within 45° |
| --- | ---: | ---: | ---: | ---: |
| Position | 21.6 ± 0.6% | 36.7 ± 0.8% | 44.1 ± 0.3° | 60.7% |
| Usage counts | 53.9 ± 0.2% | 60.5 ± 1.0% | 41.6 ± 0.1° | 59.8% |
| Untrained model | 21.7 ± 5.5% | 21.4 ± 4.1% | 43.9 ± 0.2° | 60.9% |
| Hold embedding | 74.8 ± 3.5% | 72.4 ± 1.2% | 30.4 ± 0.7° | 77.7% |
| Hold vector | 77.9 ± 0.9% | 81.7 ± 0.6% | 34.4 ± 0.6° | 71.6% |
| In-context state | 51.4 ± 6.8% | 59.0 ± 4.1% | 35.8 ± 0.9° | 70.3% |
| Chance | 25% | 20% | | |
| Shuffled labels, mean (95th percentile) | 25.0% (32.2%) | 20.4% (26.3%) | 43.8° | |
| Constant average direction | | | 41.9° | |

These results use the labels of 2026-11-26, with one grip direction for every hold. No
type changed, but the type scores still move a little: each seed's probes share one random
generator, and the larger grip task draws from it, so every representation after position
gets new folds. Those moves are within the spread between seeds. The grip columns changed
because 53 holds joined the task; the old scores, on 423 holds, were 30.6° for the hold
embedding and 40.0° for usage counts.

<!-- chart: probe-scores -->

**Type is encoded.** From position alone a probe cannot tell the hand-hold types apart
(22%, chance is 25%), and the untrained network does no better. The learned hold
embedding and hold vector reach 75% and 78%, far above the shuffled-label null. They
also beat usage counts (54%), the strongest control. So the model stores more than
how often each hold is used in each role, likely which holds are used together, at
what grades and angles, and in what order.

The hold embedding, which has no position input, does about as well as the full hold
vector. The information sits in what each hold learned for itself, not in the
geometry network. The in-context state is weaker and varies more between seeds: after
the position average is removed, it mixes the hold with whatever surrounds it.

On the boards below, each hold on the right is colored by the probe's prediction made
while that hold was held out, for seed 43 (the `null_rule_16` seed the website served before style tags). Ringed holds
are wrong. Switch to grip to compare labeled and predicted directions.

<!-- chart: probe-boards -->

Almost every pinch is recovered (99% recall), against 58% from usage counts. Edges and jugs
are recovered 76% and 70% of the time and are mostly mistaken for each other. Slopers
(67%) are the hardest for the model, though with 9 of them that number is noisy.
Holds 1283 and 1298 were labeled edge until they were checked again. Before that, the
hold-vector, in-context and usage probes had already predicted pinch for both, which was
scored as wrong. They are now labeled pinch.
Usage counts find slopers well (77%), probably because they are used in their own way.

<!-- chart: probe-confusion -->

The projection below is the hold vector's top two principal components, found
without labels. The first one partly tracks how often a hold is used as a foot. In
two dimensions the types overlap, so this view shows the shape of the space, while
the probes, using 20 dimensions, are the test.

<!-- chart: probe-projection -->

**Grip direction is only partly encoded.** A third of the holds (154 of 476) are labeled
straight up, and always guessing the average direction is off by 41.9° on average. The
hold embedding is off by 30.4°, against 41.6° for usage counts, so some of the
direction is in there, much less clearly than type. Pinches are scored as if their
direction were a pull direction (see [what is predicted](#what-is-predicted)), so this
slightly understates what the representations hold.

<!-- chart: probe-grip -->

### Caveats

- The labels are one person's, made while looking at the board artwork. The model
  never sees images, so whatever it knows about type came from the climbs.
- A linear probe finds information that is linearly readable. A score near chance
  does not prove the information is absent, only that it is not stored that simply.
- Slopers (9) and pinches (22) are few, so their per-type scores move by several
  points between seeds.
- Usage counts and the model both come from the same training climbs, and the labels
  were made after training. Nothing about the labels leaked into the model.

The script writes `summary.json` (every score for every seed, with per-type recall
and confusion matrices) and `holds.csv` (each hold's labels, usage counts, the
projection, and the held-out predictions of seed 43) to
[`results/hold_label_probes/`](../results/hold_label_probes/).

## Style tags from climb names and descriptions

```bash
.venv/bin/python data/export_climb_tags.py      # tags every climb
.venv/bin/python experiments/compare_models.py  # trains null_rule_16_tags
.venv/bin/python experiments/tag_conditioning.py
```

Setters often say how a climb moves in its name or description: "big dyno", "pinchy",
"drop knee to finish". This experiment reads 21 style tags from that text, gives them
to the model next to grade and angle, and asks two questions. Do the tags make real
climbs more predictable? And does turning a tag on change what the model sets the way
real climbs with that tag differ?

Five tags were removed on 2026-11-20: `static`, `cut_loose`, `bump`, `technical` and
`shouldery`. Everything below, including both tag models, was redone with the 21 that remain.

### The tags

[`kilter/climb_tags.py`](../kilter/climb_tags.py) marks a tag when one of its terms
appears as a whole word in the climb's name or setter description and is not negated.
The Kilter data has no style tags of its own, so the text is the only source.

- Ambiguous words (power, cross, finger, throw, pump) count only in descriptions,
  since names are often puns ("Power Nap", "Pinch of Salt").
- A negator (no, not, without, don't, never, sin, sans, kein…) among the 3 words before
  a term cancels it, unless a rule word such as "match" or "feet" sits in between
  ("no matching, just jump" still counts as dynamic).
- Known idioms never count: "pinch of salt", "Achilles heel", "Tension Board",
  "lemon squeezy", "power of".
- `crimp` includes fingery climbs. "Hard for short people" marks a climb `reachy`.

[`data/export_climb_tags.py`](../data/export_climb_tags.py) writes every climb's tags
to `data/training/climb_tags.csv.gz`, and the text behind each tag to
[`results/climb_tags/`](../results/climb_tags/). 16,946 of the 207,175 climbs (8.2%)
mention at least one tag. A tag means the setter mentioned the style; a climb without
it may still have it.

The audit column is 15 random matches per tag read by hand, counting the ones that
really describe the climb. It was done before the last round of idiom fixes, so it is
a floor.

| Tag | Climbs | From description | Audit | What counts |
| --- | ---: | ---: | ---: | --- |
| `dynamic` | 3,535 | 30% | 15/15 | dyno, dynamic, jump, lunge, deadpoint, pogo, double clutch; in descriptions also dino, throw, jeté, salto |
| `campus` | 3,527 | 47% | 15/15 | campus, footless, no feet |
| `pinch` | 2,266 | 20% | 15/15 | pinch, pinchy, pinches; pince, pinza, Zange in descriptions; not "pinch of salt" |
| `crimp` | 1,432 | 28% | 14/15 | crimp, crimpy, edgy, fingery; edges, fingers, réglette, regleta in descriptions |
| `reachy` | 1,345 | 45% | 13/15 | reach, reachy, wingspan, morpho, big moves, tall people, "hard for short people" |
| `jug` | 1,005 | 26% | 13/15 | jug, jugs, juggy; bac, Henkel in descriptions |
| `traverse` | 709 | 22% | 15/15 | traverse, traversing, traversé |
| `body_tension` | 609 | 28% | 12/15 | tension, body tension; not "Tension Board" |
| `crossover` | 488 | 63% | 15/15 | crossover, cross-through, cross to/into; bare "cross" in descriptions |
| `heel_hook` | 481 | 32% | 12/15 | heel, heel hook; talon, talón in descriptions; not "Achilles heel" |
| `undercling` | 420 | 37% | 15/15 | undercling, undies; inversé, Untergriff in descriptions |
| `endurance` | 415 | 35% | 13/15 | endurance, power endurance, 4x4, resistance, stamina; pump in descriptions |
| `short_friendly` | 408 | 41% | 12/15 | shorties, short people, short kings, short friendly, "for the short" |
| `gaston` | 395 | 26% | 12/15 | gaston |
| `compression` | 278 | 14% | 11/15 | compression, compy, compress, squeeze; not "lemon squeezy" |
| `drop_knee` | 247 | 40% | 15/15 | drop knee, dropknee, knee drop, egyptian |
| `toe_hook` | 208 | 52% | 15/15 | toe hook, toehook, toe cam, bicycle |
| `powerful` | 182 | 100% | 14/15 | power, powerful, in descriptions only; not "power endurance" |
| `coordination` | 165 | 34% | 15/15 | coordination, coordo, paddle, run and jump, parkour, comp style, double clutch |
| `lock_off` | 160 | 34% | 15/15 | lock off, lockoff, locky |
| `high_feet` | 160 | 44% | 15/15 | high feet, high foot, high step, rockover |

### How the model uses them

Each climb's tags become a vector of 21 zeros and ones, appended to the grade and angle
features. The model's condition network (linear, GELU, linear) turns grade, angle, the
null-grade flag and the tags into one vector added to every token, so the tags can
change every step of generation. A climb with no tags gets all zeros, the same input
as "tags off". `null_rule_16_tags` is `null_rule_16` with
`model.climb_tags = "data/training/climb_tags.csv.gz"` and nothing else changed,
trained on seeds 42, 43 and 44 by `compare_models.py`. The website serves seed 43, the
lowest validation score of the three (37.02 nats, against 37.07 and 37.08), and lets you
pick tags when generating.

### Tags make real climbs more predictable

Nats per held-out test climb, generated from nothing with the decoding rules and its
real grade, mean of three seeds. "All off" scores the same model with every tag set to
zero. The historical split's test set is used here as before; it was already open during
development, so these are development numbers (see [historical results](#historical-results)).
The tag models were retrained with the 21 tags, and `null_rule_16_tags_labels` again with the
hold labels of 2026-11-26; `null_rule_16` is the historical run.

| Model | Tags given | Tagged climbs (1,415) | Untagged (15,482) | All (16,897) |
| --- | --- | ---: | ---: | ---: |
| `null_rule_16` | none | 42.61 | 44.75 | 44.58 |
| `null_rule_16_tags` | true tags | 41.53 | 44.74 | 44.47 |
| `null_rule_16_tags` | all off | 42.87 | 44.74 | 44.58 |
| `null_rule_16_tags_labels` | true tags | 41.60 | 44.73 | 44.47 |
| `null_rule_16_tags_labels` | all off | 43.01 | 44.73 | 44.59 |

On the 1,415 test climbs that have a tag, giving the model their tags saves 1.35 nats
per climb, and the tagged model beats `null_rule_16` by 1.09. On untagged climbs the
two are about equal (44.74 against 44.75). With every tag turned off, tagged climbs score worse than under
`null_rule_16` (42.87 against 42.61), because the model has learned that no tags means
a typical untagged climb. The hand-labeled hold features add nothing on top of the tags.
With the five tags removed, the tagged climbs are fewer (1,415, from 1,484) and the gain
is the same (1.35 nats, from 1.34).

On validation, every sample from `null_rule_16_tags` is schema-valid, and 96.7% pass the
version 1 checks (`legacy_valid_v1`, which also needs a foot-only hold), against 95.0% for
`null_rule_16`. `tag_conditioning.py` records `schema_valid`, `starts_below_finishes`
and `explicit_feet_present` for each tag separately. That matters most for `campus`,
whose climbs legitimately have no feet.

The gain depends on the tag. Campus (3.8 nats), traverse (2.0) and pinch (1.5) help
most. Tags with fewer than 20 test climbs are left out, and ones with few more give noisy
numbers, including the slightly negative ones for `jug` and `heel_hook`.

<!-- chart: tags-gain -->

### Turning a tag on changes what the model sets

For each tag, 200 grade and angle pairs are drawn from real climbs with the tag. At
each pair the model samples one climb with only that tag on and one with every tag off,
so the two sets differ only in the tag. This is repeated on all three seeds. The
generated climbs are then measured two ways.

1. **A tag recognizer.** One logistic regression per tag on the bag of a climb's
   (hold, role) tokens, trained on the 186,039 training climbs with each tag's examples
   up-weighted to balance them. On the 10,344 test climbs it finds campus (ROC AUC
   0.94), pinch (0.91), traverse (0.90) and undercling (0.87) well. It cannot find
   high feet or body tension (AUC 0.43 and 0.53): those describe how a climb is done,
   which the holds alone do not show.
2. **Plain statistics.** Holds, feet, hand holds, the mean and longest move between
   hand holds ordered by height, the width and height the hand holds span (cm, from the
   board's 12 ft width), and the share of hand holds you labeled pinch, jug, edge or
   sloper, or whose grip points down (undercling) or sideways (sidepull). Each is
   compared with how real climbs with the tag differ from real climbs without it in the
   same grade and angle cell, in standard deviations of the statistic.
   The grip shares read each hold's labeled direction as the way you pull. Since
   2026-11-26 a pinch's direction is 90° to the pull instead, so a pinch you pull
   sideways is labeled about 0° and no longer counts as a sidepull. With that change,
   real `pinch` climbs stand out for sidepulls by 0.38 standard deviations, down from
   1.16, and for underclings by −0.33, from +0.23. No other tag's grip share moved by
   more than 0.23.

<!-- chart: tags-steering -->

Tags the holds can show steer strongly. Turning on `campus` cuts feet from 3.8 to 0.3
per climb, `traverse` widens the hand holds from 141 to 231 cm, and `pinch` raises the
pinch share of hand holds from 7% to 45% (shares use the current hold labels). Across all
21 tags and 13 statistics, the generated changes follow the real differences with a
correlation of 0.84 (0.84 for `null_rule_16_tags_labels` too).

<!-- chart: tags-scatter -->

### Limits

- Tags are sparse. A missing tag does not mean the climb lacks the style, so "all tags
  off" means "a typical untagged climb", not "no dynamic moves".
- Some effects are weaker than in real climbs. Real endurance climbs are 1.7 standard
  deviations longer, but turning the tag on adds only 0.2. The undercling share rises
  about half as much as in real climbs (1.0 standard deviations against 2.1).
- Movement tags such as high feet and body tension cannot be checked from holds alone.
  The model may follow them, but neither test can tell.
- Turning a tag on costs some version 1 validity: 91.9% of samples pass with one tag on,
  against 96.9% with all tags off. Every sample stays schema-valid. Most of the drop is
  `campus` climbs without foot holds, which is what the tag asks for, and most of the rest
  is `traverse` climbs whose starts are not below their finishes.
- The tags come from free text with simple rules, so each carries some noise.

`summary.json` in [`results/tag_conditioning/`](../results/tag_conditioning/) has every
number for every seed, and `examples.json` has generated climbs for a few tags.

## Grading as a downstream task

```bash
.venv/bin/python experiments/grade_climbs.py
```

Predicts a climb's V grade from its holds and angle. The model is frozen and read with
a null grade, so the label cannot leak in. The ridge penalty, MLP epoch and Bayes temperature
are chosen on validation MSE. Three stronger baselines are trained end to end for grading
alone, on the same rows and split, by their own scripts (see
[Models trained for grading](#models-trained-for-grading)): a BERT-style encoder, the generator
fine-tuned, and LightGBM. `grade_climbs.py` reads their test predictions and scores them with
the rest. Every method is scored on four test subsets: every test row,
and the rows whose climb has **more than 5, 10 or 25 ascents** at that angle, where more
climbers agree on the grade. The subsets are nested, and the better-climbed ones have a wider
spread of grades.

| Test subset | Rows | Climbs | Grade variance |
| --- | ---: | ---: | ---: |
| All test rows | 16,897 | 10,344 | 6.66 |
| More than 5 ascents | 3,736 | 2,275 | 8.24 |
| More than 10 ascents | 2,345 | 1,382 | 8.60 |
| More than 25 ascents | 1,247 | 697 | 9.15 |

RMSE (root mean squared error) and MAE (mean absolute error) are in grades; each is the mean
over three seeds of that seed's value. RMSE weighs large misses more heavily than MAE.

**RMSE** (grades, lower is better)

| Method | All | Over 5 | Over 10 | Over 25 |
| --- | ---: | ---: | ---: | ---: |
| Training mean at that angle | 2.24 | 2.60 | 2.70 | 2.83 |
| Bag of holds, ridge | 1.69 | 1.69 | 1.71 | 1.72 |
| Untrained model, MLP head | 1.55 | 1.42 | 1.40 | 1.35 |
| Bag of holds, MLP head (main baseline) | 1.42 | 1.25 | 1.23 | 1.18 |
| `null_rule_16`, ridge | 1.36 | 1.23 | 1.21 | 1.18 |
| `null_rule_16`, Bayes over grades, no head | 1.35 | 1.19 | 1.18 | 1.14 |
| `null_rule_16`, MLP head | 1.30 | 1.12 | 1.10 | 1.05 |
| LightGBM, bag of holds + geometry | 1.29 | 1.11 | 1.08 | 1.05 |
| BERT-style encoder, trained for grading | 1.26 | 1.13 | 1.11 | 1.07 |
| `null_rule_16`, fine-tuned for grading | 1.25 | 1.08 | 1.05 | 1.01 |

**MAE** (grades, lower is better)

| Method | All | Over 5 | Over 10 | Over 25 | Within 1 grade, all |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training mean at that angle | 1.78 | 2.13 | 2.22 | 2.32 | 34.1% |
| Bag of holds, ridge | 1.30 | 1.32 | 1.33 | 1.34 | 47.7% |
| Untrained model, MLP head | 1.18 | 1.09 | 1.08 | 1.04 | 52.6% |
| Bag of holds, MLP head (main baseline) | 1.07 | 0.97 | 0.95 | 0.92 | 57.3% |
| `null_rule_16`, ridge | 1.02 | 0.96 | 0.95 | 0.93 | 58.9% |
| `null_rule_16`, Bayes over grades, no head | 1.00 | 0.93 | 0.93 | 0.91 | 60.4% |
| `null_rule_16`, MLP head | 0.96 | 0.87 | 0.85 | 0.82 | 62.4% |
| LightGBM, bag of holds + geometry | 0.95 | 0.85 | 0.83 | 0.82 | 62.6% |
| BERT-style encoder, trained for grading | 0.93 | 0.89 | 0.88 | 0.87 | 63.4% |
| `null_rule_16`, fine-tuned for grading | 0.92 | 0.84 | 0.83 | 0.82 | 65.0% |

The heatmap shows each method's predictions against the true grade, with predictions
rounded to the nearest grade. Each row is one true grade, and its cells show where that
grade's test climbs were predicted. Choose a method and a test subset above it. For
`null_rule_16` with an MLP head, the rounded prediction is exactly right for 35% of all test
climbs and within one grade for 80% (37% and 84% with more than 5 ascents), against 32%
and 76% for the bag of holds. These count rounded predictions, so they are higher than the
table's "within 1 grade", which needs the unrounded prediction within one grade. The training mean's prediction depends only on the angle,
so every row falls in the same band, V2 to V7, whatever the true grade.

<!-- chart: grading-confusion -->

Among the methods that read the frozen model, `null_rule_16` with an MLP head is best in every
subset, on both measures. Fine-tuning the same model for grading does better still, in every
subset. The BERT-style encoder and LightGBM come close to the frozen model; they are compared
in [Models trained for grading](#models-trained-for-grading). Every learned
method except the bag of holds with ridge does better on the better-climbed subsets than on
all test climbs, though their grades vary more. The bag of holds with ridge does slightly
worse there, and so does the training mean, which can only guess the average. Ridge and Bayes
on `null_rule_16` beat the bag of holds with an MLP head on all test climbs, but their lead
shrinks as the ascent count rises. Earlier versions of this table used climbs with 5 or more ascents (4,324 rows); the
numbers on all test climbs are unchanged.

### Statistical significance

Each comparison below is the difference in RMSE or MAE between two methods on the same test
rows: method minus reference, as a mean over seeds, so negative means the method is better.

- The interval is a 95% bootstrap interval over test climbs. Climbs are resampled with
  replacement (10,000 times), and all angles of a climb are drawn together, since they are not
  independent.
- The *p*-value is from a two-sided paired permutation test at the climb level. If the two
  methods were equally good, each climb's error difference would be equally likely to be
  positive or negative, so the test flips their signs at random (100,000 times). For MAE it uses
  absolute errors; for RMSE, squared errors (equal RMSE means equal MSE).
- The *p*-values are adjusted with Holm's method for the sixteen comparisons in each subset and
  measure (the nine below and the seven in
  [Models trained for grading](#models-trained-for-grading)). A cell shows *p* only when it is
  0.001 or more; every other cell has *p* < 0.001. The smallest value 100,000 permutations can
  show is 0.00001, or 0.00016 after adjustment.

**Difference in RMSE**, method minus reference (negative: the method is better)

| Method | Reference | All | Over 5 | Over 10 | Over 25 |
| --- | --- | ---: | ---: | ---: | ---: |
| `null_rule_16`, MLP head | bag of holds, MLP head | −0.12 (−0.14 to −0.10) | −0.13 (−0.16 to −0.10) | −0.14 (−0.17 to −0.10) | −0.13 (−0.18 to −0.09) |
| `null_rule_16`, Bayes | bag of holds, MLP head | −0.07 (−0.09 to −0.05) | −0.07 (−0.10 to −0.04) | −0.05 (−0.09 to −0.02), *p* = 0.02 | −0.04 (−0.09 to +0.01), *p* = 0.35 |
| `null_rule_16`, ridge | bag of holds, MLP head | −0.07 (−0.08 to −0.05) | −0.03 (−0.06 to 0.00), *p* = 0.17 | −0.03 (−0.06 to +0.01), *p* = 0.46 | 0.00 (−0.05 to +0.05), *p* = 1.00 |
| Untrained, MLP head | bag of holds, MLP head | +0.12 (+0.10 to +0.14) | +0.17 (+0.14 to +0.20) | +0.17 (+0.13 to +0.20) | +0.17 (+0.12 to +0.21) |
| Bag of holds, ridge | bag of holds, MLP head | +0.27 (+0.24 to +0.29) | +0.43 (+0.39 to +0.49) | +0.48 (+0.41 to +0.54) | +0.54 (+0.45 to +0.64) |
| Training mean | bag of holds, MLP head | +0.82 (+0.78 to +0.86) | +1.35 (+1.26 to +1.44) | +1.47 (+1.35 to +1.58) | +1.65 (+1.49 to +1.81) |
| `null_rule_16`, MLP head | `null_rule_16`, Bayes | −0.05 (−0.06 to −0.04) | −0.06 (−0.09 to −0.04) | −0.08 (−0.12 to −0.05) | −0.09 (−0.14 to −0.05) |
| `null_rule_16`, MLP head | `null_rule_16`, ridge | −0.06 (−0.07 to −0.04) | −0.10 (−0.13 to −0.08) | −0.11 (−0.15 to −0.08) | −0.13 (−0.18 to −0.09) |
| `null_rule_16`, MLP head | untrained, MLP head | −0.24 (−0.26 to −0.23) | −0.30 (−0.33 to −0.27) | −0.30 (−0.34 to −0.27) | −0.30 (−0.35 to −0.26) |

**Difference in MAE**, method minus reference (negative: the method is better)

| Method | Reference | All | Over 5 | Over 10 | Over 25 |
| --- | --- | ---: | ---: | ---: | ---: |
| `null_rule_16`, MLP head | bag of holds, MLP head | −0.11 (−0.12 to −0.10) | −0.10 (−0.12 to −0.08) | −0.10 (−0.13 to −0.08) | −0.10 (−0.14 to −0.07) |
| `null_rule_16`, Bayes | bag of holds, MLP head | −0.06 (−0.07 to −0.05) | −0.04 (−0.06 to −0.01), *p* = 0.005 | −0.02 (−0.05 to +0.01), *p* = 0.35 | −0.01 (−0.05 to +0.03), *p* = 1.00 |
| `null_rule_16`, ridge | bag of holds, MLP head | −0.04 (−0.06 to −0.03) | −0.01 (−0.03 to +0.02), *p* = 0.50 | 0.00 (−0.03 to +0.02), *p* = 0.79 | +0.01 (−0.03 to +0.05), *p* = 1.00 |
| Untrained, MLP head | bag of holds, MLP head | +0.11 (+0.10 to +0.12) | +0.13 (+0.10 to +0.15) | +0.13 (+0.10 to +0.16) | +0.11 (+0.08 to +0.15) |
| Bag of holds, ridge | bag of holds, MLP head | +0.24 (+0.22 to +0.25) | +0.35 (+0.31 to +0.39) | +0.38 (+0.33 to +0.43) | +0.41 (+0.34 to +0.49) |
| Training mean | bag of holds, MLP head | +0.72 (+0.69 to +0.75) | +1.17 (+1.09 to +1.25) | +1.27 (+1.17 to +1.38) | +1.40 (+1.24 to +1.56) |
| `null_rule_16`, MLP head | `null_rule_16`, Bayes | −0.04 (−0.05 to −0.03) | −0.06 (−0.08 to −0.04) | −0.08 (−0.10 to −0.05) | −0.09 (−0.13 to −0.06) |
| `null_rule_16`, MLP head | `null_rule_16`, ridge | −0.06 (−0.07 to −0.05) | −0.09 (−0.11 to −0.07) | −0.10 (−0.12 to −0.08) | −0.11 (−0.14 to −0.08) |
| `null_rule_16`, MLP head | untrained, MLP head | −0.22 (−0.23 to −0.20) | −0.23 (−0.25 to −0.21) | −0.23 (−0.26 to −0.20) | −0.22 (−0.26 to −0.18) |

The MLP head on `null_rule_16` beats the main baseline by about 0.1 grades in every subset,
on both measures, and all three seeds agree (`per_seed` in `summary.json`). It also beats
every other way of reading the same model, and the untrained copy by 0.2 to 0.3 grades, so
the gain comes from pretraining, not from the architecture. Ridge and Bayes on `null_rule_16`
beat the baseline significantly on all test climbs, but not on well-climbed ones. Ridge's lead
is not significant from 5 ascents on, on either measure. Bayes's lead is not significant from
10 ascents on for MAE, or with more than 25 ascents for RMSE.

These intervals and tests account for which climbs happen to be in the test set. They don't
account for training randomness: the three seeds are averaged, not resampled. They also
assume the test set played no part in choosing the methods. That holds for the heads (chosen
on validation), but the historical split's test set was seen while the model was developed.

The Bayes grader uses p(climb | grade, angle) from the generative model with a prior
per angle. Errors are largest at both ends of the scale, where labels are noisier or rarer.

<!-- chart: grading-error -->

### Models trained for grading

```bash
.venv/bin/python experiments/bert_grader.py       # sweep on validation, then the chosen one on three seeds
.venv/bin/python experiments/finetune_grader.py
.venv/bin/python experiments/lightgbm_grader.py   # needs the experiments extra: uv pip install --python .venv/bin/python -e ".[experiments]"
.venv/bin/python experiments/grade_climbs.py      # then score every method on test
```

The heads above read a frozen generator. These three baselines are trained for grading only,
to show how much that leaves behind:

- **BERT-style encoder.** A bidirectional transformer over the climb's holds as a set, with no
  positions, a [CLS] token and the angle added at every position. Each hold's input is its
  identity, its role, and an MLP of its geometry features, the same inputs the generator sees
  (no hand-labeled hold types, since `null_rule_16` has none). It is trained from scratch, with
  BERT's masked-hold pretraining on the training rows first in the chosen configuration.
- **`null_rule_16`, fine-tuned.** The generator, read exactly as the frozen probe reads it
  (scratch-mode sequence, null grade, mean and last hidden states plus the angle), with a grade
  head, updating every weight. Seed *s* starts from the generator trained on seed *s*. The
  generator learns at a tenth of the head's rate.
- **LightGBM.** Gradient-boosted trees on the angle, 1,904 bag-of-holds columns (one per hold
  and role) and 38 geometry features: counts per role, spans and means of hand and foot
  positions, and the largest, second-largest and mean gaps between consecutive hands. This is
  the usual strong tabular baseline for a bag of holds.

All three use the same 303,547 training and 16,936 validation rows as the generator. Each
script checks that its rows are byte-identical to the generator's training cache. Every
configuration tried is listed in the script and in its `results/<script>/sweep.csv`, in the
order it was added. Each later round was chosen after reading the earlier round's validation
scores, never test. The configuration with the lowest validation MSE on seed 42 is trained on
seeds 43 and 44, and each script writes test predictions once, for `grade_climbs.py` to score.
The neural models keep the epoch with the lowest validation MSE, using weights averaged over
training (EMA), and LightGBM stops early on validation.

| Method | Configurations tried | Chosen | Parameters | Validation MSE, three seeds |
| --- | ---: | --- | ---: | --- |
| BERT-style encoder | 24 | d384, 8 layers, 50 epochs of masked-hold pretraining, 12 epochs of grading with a distribution loss | 15.0M | 1.635 to 1.650 |
| `null_rule_16`, fine-tuned | 11 | 10 epochs, distribution loss, generator learning rate 1e-4, head 1e-3 | 3.2M | 1.596 to 1.605 |
| LightGBM | 15 | 63 leaves, learning rate 0.01, 13,000 to 19,000 trees | | 1.728 to 1.734 |
| `null_rule_16`, MLP head (frozen) | | | | 1.754 to 1.762 |

The "distribution loss" predicts a distribution over the 17 grades, trained with
cross-entropy against the true grade smoothed by a Gaussian (σ = 0.75 grades), and predicts its
mean. It beat plain MSE on a standardized grade for both neural models. The BERT search was
the largest, since it is the baseline built to be as strong as possible. Its gains came mostly
from pretraining: 1.72 on validation without it and 1.64 with it. Differences under about 0.005
in validation MSE are within seed noise.

**What the sweeps found** (validation MSE, seed 42; every row is in `results/<script>/sweep.csv`):

| Method | Step | Validation MSE |
| --- | --- | ---: |
| BERT-style encoder | Start: d256, 6 layers, MSE loss, 30 epochs | 1.767 |
| | Learning rate ×2 or ×½, 60 epochs, d192 with 4 layers | 1.754 to 1.775 |
| | Dropout 0.2; d384 with 8 layers | 1.861; 1.799 |
| | Distribution loss (12 or 30 epochs; the best epoch was 12 or 13 either way) | 1.717 to 1.719 |
| | + lower learning rate, weight decay 0.2, or σ = 1 (one at a time) | 1.704 to 1.707 |
| | + weighting rows by log ascents | 1.743 |
| | + 20 epochs of masked-hold pretraining | 1.668 |
| | + 50 epochs of masked-hold pretraining | 1.646 |
| | + 50 epochs of pretraining, d384 with 8 layers (**chosen**) | 1.641 |
| `null_rule_16`, fine-tuned | Start: MSE loss, generator learning rate 1e-4, head 1e-3, 10 epochs | 1.613 |
| | Generator learning rate 3e-5 or 3e-4 | 1.629 |
| | Train the head alone for 2 epochs first; 20 epochs | 1.622; 1.611 |
| | Distribution loss | 1.597 |
| | + 15 epochs, σ = 1, head learning rate 3e-4, or weight decay 0.05 (**chosen**) | 1.596 to 1.601 |
| | + weighting rows by log ascents | 1.606 |
| LightGBM | Start: 255 leaves, learning rate 0.05 | 1.768 |
| | 1,023 leaves; more data per leaf, stronger L2, or fewer features per tree | 1.763 to 1.796 |
| | 63, 31 or 15 leaves, and 63 leaves with more data per leaf or fewer features | 1.748 to 1.762 |
| | Learning rate 0.02 or 0.01, 15 to 63 leaves (**chosen**: 63 leaves, 0.01) | 1.733 to 1.745 |

Three things carried across methods. The distribution loss beat MSE for both neural models.
Weighting rows by ascents hurt both, perhaps because validation weighs every row equally. And
BERT overfits quickly: with 30 or 60 epochs on offer, most runs without pretraining peaked
between epochs 7 and 14, and the larger model was worse until it was pretrained. The fine-tuned run
chosen is tied with the plain distribution-loss run (1.5965 against 1.5971), well within seed
noise of about 0.005.

**Compute**, on one RTX 4060 Ti (LightGBM on 12 CPU threads):

| Method | Whole sweep | Chosen configuration, per seed |
| --- | ---: | ---: |
| BERT-style encoder | 3.4 hours (24 runs) | 28 minutes, most of it pretraining |
| `null_rule_16`, fine-tuned | 65 minutes (11 runs) | 5 minutes |
| LightGBM | 22 minutes (15 runs) | 4 minutes |

The test RMSE and MAE are in the tables above. These comparisons use the same tests as the
section above, and are adjusted together with its nine comparisons.

**Difference in RMSE**, method minus reference (negative: the method is better)

| Method | Reference | All | Over 5 | Over 10 | Over 25 |
| --- | --- | ---: | ---: | ---: | ---: |
| Fine-tuned `null_rule_16` | bag of holds, MLP head | −0.17 (−0.19 to −0.15) | −0.17 (−0.20 to −0.15) | −0.18 (−0.22 to −0.15) | −0.17 (−0.21 to −0.12) |
| BERT-style encoder | bag of holds, MLP head | −0.16 (−0.18 to −0.14) | −0.12 (−0.15 to −0.09) | −0.13 (−0.17 to −0.09) | −0.11 (−0.16 to −0.06) |
| LightGBM | bag of holds, MLP head | −0.13 (−0.15 to −0.10) | −0.14 (−0.17 to −0.12) | −0.15 (−0.19 to −0.12) | −0.13 (−0.17 to −0.09) |
| `null_rule_16`, MLP head | fine-tuned `null_rule_16` | +0.05 (+0.04 to +0.06) | +0.04 (+0.03 to +0.06) | +0.05 (+0.03 to +0.06) | +0.03 (+0.01 to +0.06), *p* = 0.06 |
| `null_rule_16`, MLP head | BERT-style encoder | +0.04 (+0.02 to +0.05) | −0.01 (−0.03 to +0.01), *p* = 0.53 | −0.01 (−0.03 to +0.02), *p* = 0.57 | −0.03 (−0.06 to +0.01), *p* = 0.53 |
| `null_rule_16`, MLP head | LightGBM | +0.01 (−0.01 to +0.02), *p* = 0.24 | +0.01 (−0.01 to +0.03), *p* = 0.34 | +0.02 (−0.01 to +0.04), *p* = 0.46 | 0.00 (−0.04 to +0.03), *p* = 1.00 |
| Fine-tuned `null_rule_16` | BERT-style encoder | −0.01 (−0.02 to −0.01) | −0.05 (−0.06 to −0.04) | −0.05 (−0.07 to −0.04) | −0.06 (−0.08 to −0.03) |

**Difference in MAE**, method minus reference (negative: the method is better)

| Method | Reference | All | Over 5 | Over 10 | Over 25 |
| --- | --- | ---: | ---: | ---: | ---: |
| Fine-tuned `null_rule_16` | bag of holds, MLP head | −0.15 (−0.16 to −0.14) | −0.12 (−0.15 to −0.10) | −0.12 (−0.15 to −0.09) | −0.10 (−0.14 to −0.06) |
| BERT-style encoder | bag of holds, MLP head | −0.13 (−0.14 to −0.12) | −0.08 (−0.11 to −0.06) | −0.08 (−0.11 to −0.05) | −0.05 (−0.09 to −0.01), *p* = 0.09 |
| LightGBM | bag of holds, MLP head | −0.11 (−0.12 to −0.10) | −0.11 (−0.13 to −0.09) | −0.12 (−0.15 to −0.09) | −0.10 (−0.14 to −0.07) |
| `null_rule_16`, MLP head | fine-tuned `null_rule_16` | +0.04 (+0.04 to +0.05) | +0.02 (+0.01 to +0.04), *p* = 0.001 | +0.02 (0.00 to +0.04), *p* = 0.08 | 0.00 (−0.03 to +0.02), *p* = 1.00 |
| `null_rule_16`, MLP head | BERT-style encoder | +0.02 (+0.02 to +0.03) | −0.02 (−0.04 to 0.00), *p* = 0.12 | −0.03 (−0.05 to 0.00), *p* = 0.12 | −0.05 (−0.09 to −0.02), *p* = 0.004 |
| `null_rule_16`, MLP head | LightGBM | +0.01 (0.00 to +0.01), *p* = 0.12 | +0.01 (0.00 to +0.03), *p* = 0.31 | +0.02 (0.00 to +0.04), *p* = 0.35 | 0.00 (−0.03 to +0.03), *p* = 1.00 |
| Fine-tuned `null_rule_16` | BERT-style encoder | −0.02 (−0.02 to −0.01) | −0.04 (−0.05 to −0.03) | −0.05 (−0.06 to −0.03) | −0.05 (−0.07 to −0.03) |

**Freezing costs about 0.05 grades of RMSE.** Fine-tuning `null_rule_16` beats the frozen
probe in every subset on RMSE, by 0.03 to 0.05; the gap with more than 25 ascents is not
significant (*p* = 0.06). On MAE the gap is 0.04 on all test climbs, smaller with more
ascents, and zero with more than 25. The fine-tuned model is the best method in the table in
every subset, on both measures.

**Starting from the generator beats a model built for grading.** The BERT-style encoder has
five times the parameters and had the largest search. It beats the frozen probe on all test
climbs, by 0.04 RMSE and 0.02 MAE, but not on well-climbed ones. From 5 ascents on, the RMSE
differences are not significant, and with more than 25 ascents the frozen probe has the lower
MAE, by 0.05 (*p* = 0.004). The fine-tuned generator beats it in every subset. The encoder
predicts low on average (bias −0.23 grades on all test climbs and −0.45 with more than 25
ascents, against +0.02 and −0.22 for the frozen probe). It also has fewer well-climbed
predictions within one grade (63% against 68%). The error chart by true grade shows where: the encoder
has the lowest error of the four plotted methods from V0 to V4, where most test rows are, and
falls behind the frozen probe at every grade from V6 up.

**Hand-made geometry gets as far as the frozen probe.** LightGBM's gap to the frozen probe is
not significant in any subset, on either measure. Its trees spend about 24% of their gain on
the angle, 17% on the bag-of-holds columns, and the rest on the geometry features, most of
all the distances between consecutive hands. The generator learned features worth as much
without being given them. Hold-by-hold features alone, the bag of holds with an MLP head, are
0.12 to 0.15 grades of RMSE behind both, depending on the subset.

The caveats above apply here too. The seeds are averaged, not resampled, and the comparisons
assume each method was chosen on validation, which holds for all three.
