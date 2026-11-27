# Rerunning the hold-label experiments with the 2026-11-26 labels

On 2026-11-26 every hold in [`data/hold_labels.csv`](../data/hold_labels.csv) was given
exactly one grip direction. This report reruns every experiment that reads that file and
lists what changed. Old numbers come from the last commit (`878382e`) or from the
historical results in [`compare_models/`](compare_models/). New numbers come from the
files this rerun wrote.

## Summary

- **The labels still don't help the model.** With one direction per hold,
  `null_rule_16_labels` scores 37.13 nats per climb on validation, against 37.12 with the
  old labels and 37.14 without labels. All three are within the spread between seeds.
  Adding labels to the tag model (`null_rule_16_tags_labels`) also still adds nothing:
  37.07 against 37.06 for tags alone.
- **The probe conclusions hold.** The model's hold representations still encode hold
  type (hold vector 77.9%, up from 76.3%, chance 25%). Grip direction is still only
  partly encoded: the hold embedding is 30.4° off (was 30.6°), against 41.9° for a
  constant guess (was 41.1°). The grip task now covers all 476 holds instead of 423.
- **One number moved for a real reason: the `pinch` tag's grip shares.** Real climbs
  tagged `pinch` used to stand out for sidepulls (+1.16 standard deviations). Now they
  barely do (+0.38). The cause is the new pinch rule, not the model (see
  [caveats](#caveats)).
- Everything else the write-up quotes is unchanged or rounds to the same value.

## What changed in the labels

| | Before | After |
| --- | ---: | ---: |
| Holds with one grip direction | 423 | 476 |
| Holds with several directions | 53 (45 with 2, 4 with 3, 2 with 4, 2 with 7) | 0 |
| Holds labeled straight up (0°) | 141 | 154 |
| Holds with a downward direction (135°–225°) | 43 | 36 |
| Holds with a sideways direction (67.5°–112.5° or 247.5°–292.5°) | 123 | 97 |

No type changed. The 53 re-labeled holds were 22 pinches, 21 edges, 8 jugs and 2 foot
chips.

- **Edges and jugs:** the direction is still the way you pull. Holds that had
  neighbouring directions got roughly the middle one, for example 45° and 315° became 0°.
- **Pinches:** the direction now means **90° to the pulling force, either way**. A pinch
  that had two opposite directions (90° and 270°) is now labeled 0°.
- 28 of the 53 kept one of their old directions; the other 25 got a new one.

## What was rerun

| Experiment | What was rerun | Why |
| --- | --- | --- |
| [Hand-labeled hold features](../experiments/README.md#hand-labeled-hold-features) | `null_rule_16_labels` trained again on seeds 42–44 and scored on validation | Its features come from the label file |
| (control for the row above) | `null_rule_16` trained again on seeds 42–44 | The old comparison came from runs made before the current code. The retrained runs reproduced the historical ones to every digit, so the comparison below is fair. |
| Style tags + labels | `null_rule_16_tags_labels` trained again on seeds 42–44 and scored on validation | Its features come from the label file |
| [Style tags](../experiments/README.md#style-tags-from-climb-names-and-descriptions) (`tag_conditioning.py`) | The whole script | It uses both label models, and its undercling and sidepull shares read the grip labels |
| [Hold-label probes](../experiments/README.md#hold-labels-in-the-learned-representations) (`hold_label_probes.py`) | The whole script | It predicts the labels |

`null_rule_16_tags` doesn't read the label file, so its existing runs and scores were
reused. Its numbers are identical to before, which also shows the pipeline reproduces.
The hold-embedding and grading experiments don't use the labels and weren't rerun. The
model the website serves (`null_rule_16_tags`) is unchanged.

`tag_conditioning.py` used to take its `null_rule_16_labels` reference from the historical
runs, which were trained on the old labels. It now finds that model by identity, as it
already did for the tag models, so it uses the new runs.

## Results

### Hand-labeled hold features

Validation, nats per climb with rules applied, mean over seeds 42, 43 and 44. "Mean" is
the selection score (`selection_nll_rules`). Test scratch comes from `tag_conditioning.py`
and uses the historical test split, so it is a development number.

| Model | Grip labels | Mean | Scratch | Starts | Starts + feet | Rules-on valid | Test scratch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `null_rule_16` | none | 37.144 | 44.696 | 37.598 | 29.138 | 95.0% | 44.575 |
| `null_rule_16_labels` | old (several directions) | 37.118 | 44.667 | 37.569 | 29.119 | 94.3% | 44.559 |
| `null_rule_16_labels` | new (one direction) | 37.130 | 44.681 | 37.584 | 29.125 | 94.2% | 44.560 |

Per seed, labels minus no labels, on the validation mean (negative means the labels helped):

| Seed | Old labels | New labels |
| ---: | ---: | ---: |
| 42 | −0.008 | +0.010 |
| 43 | +0.004 | +0.032 |
| 44 | −0.073 | −0.083 |

The standard deviation over seeds is 0.054 without labels and 0.015 with the new labels.
Best epochs are 15, 13 and 12 for every model. On held-out test climbs that have a tag,
the new labels model is slightly worse than the old one (42.641 against 42.567). On
untagged climbs it is slightly better (44.735 against 44.741).

**What changed:** the small gain from labels got smaller (0.014 nats, from 0.026). It
comes entirely from seed 44, where the run without labels is the weakest of the three.
Either way the labels make no measurable difference.

### Style tags with hold labels (`null_rule_16_tags_labels`)

Validation, mean over three seeds:

| Metric | `null_rule_16_tags` | Tags + old labels | Tags + new labels |
| --- | ---: | ---: | ---: |
| Selection score (nats) | 37.058 | 37.057 | 37.065 |
| Selection score, grade hidden | 37.560 | 37.556 | 37.559 |
| Scratch | 44.574 | 44.573 | 44.574 |
| Starts | 37.510 | 37.502 | 37.512 |
| Starts + feet | 29.089 | 29.096 | 29.110 |
| Rules-on valid (`legacy_valid_v1`) | 96.7% | 96.1% | 97.3% |
| Holds lost from V2 to V11 | 2.73 | 2.68 | 2.23 |

Per-seed selection scores with the new labels are 37.072, 37.042 and 37.082 (old 37.072,
37.033 and 37.066). Validity per seed is 96.5%, 98.3% and 97.2% (old 97.6%, 95.6% and
95.0%). Validity went up 1.2 points, but its standard deviation over seeds is 0.9 and the
old seeds already ranged from 95.0% to 97.6%, so this is noise. "Holds lost from V2 to
V11" has a standard deviation over seeds of 0.54, so its change is noise too.

### Style tags: likelihood on test climbs

Nats per held-out test climb (historical split), generated from nothing with rules and
the climb's real grade. Only rows that could change are shown. `null_rule_16` and
`null_rule_16_tags` are identical to before.

| Model | Tags given | Tagged (1,415) | Untagged (15,482) | All (16,897) |
| --- | --- | ---: | ---: | ---: |
| `null_rule_16_labels`, old → new | none | 42.567 → 42.641 | 44.741 → 44.735 | 44.559 → 44.560 |
| `null_rule_16_tags_labels`, old → new | true tags | 41.585 → 41.599 | 44.726 → 44.730 | 44.463 → 44.468 |
| `null_rule_16_tags_labels`, old → new | all off | 42.964 → 43.007 | 44.726 → 44.730 | 44.578 → 44.586 |

On tagged climbs, giving the tags still saves 1.35 nats with `null_rule_16_tags`. For the
labels version it saves 1.41, up from 1.38. The README's rounded table changed in the
`null_rule_16_tags_labels` rows only.

### Style tags: steering

The climbs generated by `null_rule_16_tags` are identical to before. Only the two
statistics that read grip labels (undercling share and sidepull share) changed, for
generated and real climbs alike. Every other statistic is unchanged, as are the
recognizer's AUCs and validity.

| | Before | After |
| --- | ---: | ---: |
| Correlation of generated and real changes, `null_rule_16_tags` | 0.843 | 0.836 |
| Same, `null_rule_16_tags_labels` | 0.837 | 0.838 |
| Real `pinch` climbs, sidepull share (standard deviations above untagged) | +1.16 | +0.38 |
| Real `pinch` climbs, undercling share | +0.23 | −0.33 |
| Real `undercling` climbs, undercling share | +1.99 | +2.08 |
| `undercling` tag turned on, undercling share (`null_rule_16_tags`) | +1.02 | +1.01 |
| Undercling share of hand holds, real climbs | 9.2% | 6.9% |
| Sidepull share of hand holds, real climbs | 30.1% | 22.2% |

No other tag's grip share moved by more than 0.23 standard deviations, either in real
climbs or in what turning the tag on does. The README's statements still hold: the
correlation still rounds to 0.84, and the undercling share still rises about half as much
as in real climbs (1.0 against 2.1). Validity with one tag on and with all off is
91.9% and 96.9%, as before (92.0% and 97.0% for the labels model).

### Hold-label probes

Mean over seeds 42–44, old → new. The figure in brackets is the new standard deviation
over seeds.

| Representation | Hand-hold type | All five types | Grip error | Grip within 45° |
| --- | ---: | ---: | ---: | ---: |
| Position | 21.6 → 21.6% (0.6) | 36.7 → 36.7% (0.8) | 43.2 → 44.1° (0.3) | 62.1 → 60.7% |
| Usage counts | 53.6 → 53.9% (0.2) | 61.1 → 60.5% (1.0) | 40.0 → 41.6° (0.1) | 61.2 → 59.8% |
| Untrained model | 22.6 → 21.7% (5.5) | 21.5 → 21.4% (4.1) | 43.1 → 43.9° (0.2) | 62.0 → 60.9% |
| Hold embedding | 74.8 → 74.8% (3.5) | 73.1 → 72.4% (1.2) | 30.6 → 30.4° (0.7) | 78.2 → 77.7% |
| Hold vector | 76.3 → 77.9% (0.9) | 82.6 → 81.7% (0.6) | 34.1 → 34.4° (0.6) | 72.2 → 71.6% |
| In-context state | 51.3 → 51.4% (6.8) | 60.2 → 59.0% (4.1) | 33.7 → 35.8° (0.9) | 73.9 → 70.3% |
| Shuffled labels, mean (95th percentile) | 25.0% (33.5%) → 25.0% (32.2%) | 20.8% (28.5%) → 20.4% (26.3%) | 42.9 → 43.8° | |
| Constant average direction | | | 41.1 → 41.9° | |

Hold-vector recall per hand-hold type, old → new: pinch 98 → 99%, edge 76 → 76%, jug
69 → 70%, sloper 61 → 67%. For usage counts: pinch 60 → 58%, sloper 76 → 77%.

- **Type scores.** No type changed, so these moves don't come from the labels. Each
  seed's probes share one random generator, and the larger grip task takes more draws
  from it, so every representation after position gets new cross-validation folds.
  Position runs first, and its type scores are identical. The largest move is the hold
  vector's +1.7 points on hand-hold type, about two standard deviations over seeds. Sloper
  recall moved 6 points, but it rests on 9 holds.
- **Grip scores.** These are a different task now: 476 holds instead of 423, including
  all 22 pinches. Every control got 0.8–1.6° worse and the constant guess got 0.8°
  worse, so the task is a little harder. The hold embedding stayed at about 30°, and its
  lead over usage counts grew from 9.4° to 11.2°. The in-context state lost the most
  (+2.1°, and 3.6 points fewer within 45°).
- **Reproducibility check.** Rerunning the probes with the old label file reproduced the
  committed results. The only differences were floating-point noise in the GPU-computed
  in-context state (largest difference 0.0004 in a probability; no prediction changed).
  So every change above comes from the labels.

## Caveats

- **Pinch directions are axes, and two experiments read them as pull directions.**
  - `tag_conditioning.py` counts a hold as an undercling or sidepull from its labeled
    direction. A sideways pinch is now labeled about 0°, so it no longer counts as a
    sidepull, while a pinch labeled about 90° does. This explains the `pinch` rows in
    the steering table.
  - `hold_label_probes.py` scores grip error on a 0–180° scale. For a pinch, 0° and 180°
    mean the same thing, so a probe that predicts a pinch exactly backwards is scored as
    maximally wrong. The 22 pinches therefore make the grip error look slightly worse
    than it is.
  - Neither script was changed. Fixing them would mean leaving pinches out of the grip
    shares (or counting a pinch as sideways when its direction is near 0° or 180°), and
    scoring pinches on a 0–90° scale.
- The old `null_rule_16_labels` runs were trained before holds 1283 and 1298 were
  relabeled from edge to pinch (2026-11-20). The old-to-new comparison for that model
  therefore covers those two type changes as well as the grip changes.
- Test numbers here come from the historical split, which was already open during
  development. They are development evidence, not a sealed benchmark. The family split's
  test set was not opened.

## Files changed

- Results: [`model_selection/historical_uuid/`](model_selection/historical_uuid/) (now with
  `null_rule_16` and `null_rule_16_labels`), [`tag_conditioning/`](tag_conditioning/) and
  [`hold_label_probes/`](hold_label_probes/).
- Write-up and website: [`experiments/README.md`](../experiments/README.md) (label features,
  probes and style tags sections), which builds the Experiments page. Also the probe
  board note in `web/src/probes.mjs`, and two tests that hard-coded 423 single-direction
  holds.
- Code: `experiments/tag_conditioning.py` (the labels reference is found by identity) and
  the `hold_label_probes.py` docstring.
