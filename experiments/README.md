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
| [`tag_conditioning.py`](tag_conditioning.py) | Does conditioning on style tags from climb names and descriptions help the model, and does turning a tag on change what it sets? | [`results/tag_conditioning/`](../results/tag_conditioning/) |
| [`hold_label_probes.py`](hold_label_probes.py) | Do the chosen model's hold representations already encode the hand-labeled hold types and grip directions? | [`results/hold_label_probes/`](../results/hold_label_probes/) |
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
| `null_rule_16` | 37.14 | 44.58 | 37.43 | 29.01 | 37.01 | 94.0% |

`null_rule_16` was chosen on validation, and the website serves its best validation
seed, 43. It scores 0.46 nats better than `climb_v2_null_grade` on test, but about 2
points fewer of its samples pass the validity checks. Most of its runs stopped
improving between epochs 12 and 15, so more epochs alone are unlikely to help much.
`null_rule_16_d256` stopped early at 11 or 12 epochs, with its best at 9 or 10. It
fits worse than the smaller model, so the extra size stops helping once training is
long enough.

### Hand-labeled hold features

Every hold was labeled by hand in the website's hold labeler
([`data/hold_labels.csv`](../data/hold_labels.csv)): its type and every direction it
can be pulled on. `null_rule_16_labels` is `null_rule_16` with those labels added to
each hold's features (`model.hold_labels = "data/hold_labels.csv"`), and nothing else
changed. Only the `type`, `grip` and `on_kickboard` columns are read. Type becomes a
one-hot over 7 types and grip a multi-hot over 16 compass directions, so a hold's
features grow from 28 to 51 columns. `on_kickboard` was already a feature, so it is
only checked against the vocabulary. All 476 holds have both a type and a grip: 197
edges, 170 foot chips, 80 jugs, 20 pinches and 9 slopers, and 423 holds pull one way.

Mean over seeds 42, 43 and 44, with rules applied:

| Model | Validation mean | Test scratch | Test starts | Test starts + feet | Test mean | Rules-on valid (test) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `null_rule_16` | 37.14 | 44.58 | 37.43 | 29.01 | 37.01 | 94.0% |
| `null_rule_16_labels` | 37.12 | 44.56 | 37.43 | 29.00 | 37.00 | 93.4% |

The labels make no measurable difference. Test mean improves by 0.01 nats per climb.
Seed for seed, the gain is 0.00 to 0.03, smaller than the spread between seeds (standard
deviation 0.02 to 0.04). Validity is 0.6 points lower, and its sign changes from seed
to seed. Best epochs are the same, 12 to 15. A likely reason is that every hold
already has its own learned embedding, trained on about 300,000 climb rows, so anything
fixed about a hold, such as its type or grip, can already be learned from how it is used.
Labels like these would matter more for holds with few climbs, or for a board
the model has not seen. The website still serves `null_rule_16`. The next section
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
| Grip direction | 423 | the one labeled direction, for holds with exactly one | mean angle between predicted and labeled direction |

Balanced accuracy is the mean over types of the share of that type predicted
correctly (its recall), so each type counts equally: the 9 slopers weigh as much as
the 197 edges, and always guessing "edge" scores exactly chance. Hand-hold type is
the main test. A foot chip is easy to find without learning anything, because 153 of
the 170 are screw-on holds and the screw-on flag is one of the model's position inputs.

The angle between two directions is `|((p − l + 180) mod 360) − 180|` degrees, so it
runs from 0° (exact) to 180° (opposite). Pinches all have two or more labeled
directions, so they are not in the grip task.

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
| Position | 24.2 ± 0.7% | 38.0 ± 1.5% | 42.9 ± 0.2° | 61.9% |
| Usage counts | 52.0 ± 2.1% | 59.7 ± 1.0% | 39.7 ± 0.2° | 61.4% |
| Untrained model | 22.8 ± 5.2% | 21.4 ± 2.5% | 43.1 ± 0.2° | 62.1% |
| Hold embedding | 75.3 ± 1.1% | 72.3 ± 0.8% | 30.6 ± 0.6° | 78.2% |
| Hold vector | 76.7 ± 0.8% | 81.5 ± 0.6% | 34.1 ± 0.8° | 72.5% |
| In-context state | 50.0 ± 5.4% | 59.4 ± 5.6% | 33.9 ± 0.3° | 73.7% |
| Chance | 25% | 20% | | |
| Shuffled labels, mean (95th percentile) | 24.9% (30.8%) | 19.8% (25.4%) | 42.9° | |
| Constant average direction | | | 41.1° | |

<!-- chart: probe-scores -->

**Type is encoded.** From position alone a probe cannot tell the hand-hold types apart
(24%, chance is 25%), and the untrained network does no better. The learned hold
embedding and hold vector reach 75% and 77%, far above the shuffled-label null. They
also beat usage counts (52%), the strongest control. So the model stores more than
how often each hold is used in each role, likely which holds are used together, at
what grades and angles, and in what order.

The hold embedding, which has no position input, does about as well as the full hold
vector. The information sits in what each hold learned for itself, not in the
geometry network. The in-context state is weaker and varies more between seeds: after
the position average is removed, it mixes the hold with whatever surrounds it.

On the boards below, each hold on the right is colored by the probe's prediction made
while that hold was held out, for seed 43, the seed the website serves. Ringed holds
are wrong. Switch to grip to compare labeled and predicted directions.

<!-- chart: probe-boards -->

Almost every pinch is recovered (98% recall), against 49% from usage counts. Edges and jugs
are recovered 76% and 69% of the time and are mostly mistaken for each other. Slopers
(64%) are the hardest for the model, though with 9 of them that number is noisy.
Usage counts find slopers well (76%), probably because they are used in their own way.

<!-- chart: probe-confusion -->

The projection below is the hold vector's top two principal components, found
without labels. The first one partly tracks how often a hold is used as a foot. In
two dimensions the types overlap, so this view shows the shape of the space, while
the probes, using 20 dimensions, are the test.

<!-- chart: probe-projection -->

**Grip direction is only partly encoded.** A third of the single-direction holds pull
straight up, and always guessing the average direction is off by 41.1° on average. The
hold embedding is off by 30.6°, against 39.7° for usage counts, so some of the
direction is in there, much less clearly than type.

<!-- chart: probe-grip -->

### Caveats

- The labels are one person's, made while looking at the board artwork. The model
  never sees images, so whatever it knows about type came from the climbs.
- A linear probe finds information that is linearly readable. A score near chance
  does not prove the information is absent, only that it is not stored that simply.
- Slopers (9) and pinches (20) are few, so their per-type scores move by several
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
"drop knee to finish". This experiment reads 26 style tags from that text, gives them
to the model next to grade and angle, and asks two questions. Do the tags make real
climbs more predictable? And does turning a tag on change what the model sets the way
real climbs with that tag differ?

### The tags

[`kilter/climb_tags.py`](../kilter/climb_tags.py) marks a tag when one of its terms
appears as a whole word in the climb's name or setter description and is not negated.
The Kilter data has no style tags of its own, so the text is the only source.

- Ambiguous words (power, swing, bump, cross, shoulder, finger) count only in
  descriptions, since names are often puns ("Power Nap", "Swing State").
- A negator (no, not, without, don't, never, sin, sans, kein…) among the 3 words before
  a term cancels it, unless a rule word such as "match" or "feet" sits in between
  ("no matching, just jump" still counts as dynamic). A negated dynamic term ("no
  dyno", "no jumping") marks the climb `static` instead.
- Known idioms never count: "pinch of salt", "Achilles heel", "Tension Board",
  "lemon squeezy", "power of".
- `crimp` includes fingery climbs. "Hard for short people" marks a climb `reachy`.

[`data/export_climb_tags.py`](../data/export_climb_tags.py) writes every climb's tags
to `data/training/climb_tags.csv.gz`, and the text behind each tag to
[`results/climb_tags/`](../results/climb_tags/). 17,686 of the 207,175 climbs (8.5%)
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
| `cut_loose` | 292 | 56% | 15/15 | cut loose, cut feet, barn door; swing in descriptions |
| `compression` | 278 | 14% | 11/15 | compression, compy, compress, squeeze; not "lemon squeezy" |
| `static` | 262 | 39% | 13/15 | static, or a negated dynamic term such as "no dyno" |
| `drop_knee` | 247 | 40% | 15/15 | drop knee, dropknee, knee drop, egyptian |
| `shouldery` | 211 | 90% | 12/15 | shouldery; shoulder(s) in descriptions |
| `toe_hook` | 208 | 52% | 15/15 | toe hook, toehook, toe cam, bicycle |
| `powerful` | 182 | 100% | 14/15 | power, powerful, in descriptions only; not "power endurance" |
| `coordination` | 165 | 34% | 15/15 | coordination, coordo, paddle, run and jump, parkour, comp style, double clutch |
| `technical` | 163 | 58% | 13/15 | technical, techy, body position, smear; balance in descriptions |
| `lock_off` | 160 | 34% | 15/15 | lock off, lockoff, locky |
| `high_feet` | 160 | 44% | 15/15 | high feet, high foot, high step, rockover |
| `bump` | 103 | 100% | 13/15 | bump, in descriptions only |

### How the model uses them

Each climb's tags become a vector of 26 zeros and ones, appended to the grade and angle
features. The model's condition network (linear, GELU, linear) turns grade, angle, the
null-grade flag and the tags into one vector added to every token, so the tags can
change every step of generation. A climb with no tags gets all zeros, the same input
as "tags off". `null_rule_16_tags` is `null_rule_16` with
`model.climb_tags = "data/training/climb_tags.csv.gz"` and nothing else changed,
trained on seeds 42, 43 and 44 by `compare_models.py`.

### Tags make real climbs more predictable

Nats per held-out test climb, generated from nothing with the decoding rules and its
real grade, mean of three seeds. "All off" scores the same model with every tag set to
zero. "Samples valid" is the share of climbs sampled with all tags off that pass the
validity checks.

| Model | Tags given | Tagged climbs (1,484) | Untagged (15,413) | All (16,897) | Samples valid |
| --- | --- | ---: | ---: | ---: | ---: |
| `null_rule_16` | none | 42.68 | 44.76 | 44.58 | 94.0% |
| `null_rule_16_tags` | true tags | 41.69 | 44.72 | 44.46 | 96.3% |
| `null_rule_16_tags` | all off | 43.03 | 44.72 | 44.57 |  |

On the 1,484 test climbs that have a tag, giving the model their tags saves 1.34 nats
per climb, and the tagged model beats `null_rule_16` by 0.99. On untagged climbs the
two are about equal (44.72 against 44.76). With every tag turned off, tagged climbs score worse than under
`null_rule_16` (43.03 against 42.68), because the model has learned that no tags means
a typical untagged climb. Its samples are also slightly more often valid.

The gain depends on the tag. Campus (3.8 nats), traverse (2.4) and pinch (1.5) help
most. Tags with fewer than 20 test climbs give noisy numbers, including the negative
ones for `cut_loose` and `static`.

<!-- chart: tags-gain -->

### Turning a tag on changes what the model sets

For each tag, 200 grade and angle pairs are drawn from real climbs with the tag. At
each pair the model samples one climb with only that tag on and one with every tag off,
so the two sets differ only in the tag. This is repeated on all three seeds. The
generated climbs are then measured two ways.

1. **A tag recognizer.** One logistic regression per tag on the bag of a climb's
   (hold, role) tokens, trained on the 186,039 training climbs with each tag's examples
   up-weighted to balance them. On the 10,344 test climbs it finds campus (ROC AUC
   0.93), pinch (0.91), traverse (0.90) and undercling (0.87) well. It cannot find
   static, bump, high feet, technical or body tension (AUC 0.43 to 0.53): those
   describe how a climb is done, which the holds alone do not show.
2. **Plain statistics.** Holds, feet, hand holds, the mean and longest move between
   hand holds ordered by height, the width and height the hand holds span (cm, from the
   board's 12 ft width), and the share of hand holds you labeled pinch, jug, edge or
   sloper, or whose grip points down (undercling) or sideways (sidepull). Each is
   compared with how real climbs with the tag differ from real climbs without it in the
   same grade and angle cell, in standard deviations of the statistic.

<!-- chart: tags-steering -->

Tags the holds can show steer strongly. Turning on `campus` cuts feet from 3.9 to 0.3
per climb, `traverse` widens the hand holds from 141 to 238 cm, and `pinch` raises the
pinch share of hand holds from 6% to 41%. Across all 26 tags and 13 statistics, the
generated changes follow the real differences with a correlation of 0.83.

<!-- chart: tags-scatter -->

### Limits

- Tags are sparse. A missing tag does not mean the climb lacks the style, so "all tags
  off" means "a typical untagged climb", not "no dynamic moves".
- Some effects are weaker than in real climbs. Real endurance climbs are 1.7 standard
  deviations longer, but turning the tag on adds only 0.2. The undercling share rises
  about half as much as in real climbs.
- Movement tags such as static, bump, high feet, technical and body tension cannot be
  checked from holds alone. The model may follow them, but neither test can tell.
- Turning a tag on costs a little validity: 93.1% of samples pass the checks with one
  tag on, against 97.0% with all tags off.
- The tags come from free text with simple rules, so each carries some noise.

`summary.json` in [`results/tag_conditioning/`](../results/tag_conditioning/) has every
number for every seed, and `examples.json` has generated climbs for a few tags.

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
