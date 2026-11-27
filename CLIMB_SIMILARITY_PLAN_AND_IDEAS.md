# Climb similarity: plan and ideas

Draft, 2026-11-22. Goal: given a climb (real or generated), find the climbs most like it,
using the pretrained climb transformer, and show that it beats simple baselines on
benchmarks that don't just reward hold overlap.

## 1. What "similar" should mean

"Similar" means several things, and each needs its own benchmark:

| Notion | Example question | What decides it |
| --- | --- | --- |
| **Duplicate / near duplicate** | "Has this already been set?" | Shared holds. Hand Jaccard already does this well. |
| **Variant** | "Is this the sit start, the easy version or the extension of X?" | Mostly shared holds, plus a few changes |
| **Same movement / style** | "Something that climbs like X, on different holds" | Hold types, distances, body positions. Hold overlap can be zero. |
| **Same difficulty feel** | "Something like X, at my grade" | Grade at an angle, plus style |

The pretrained model only makes sense for the last two. For duplicates, hand Jaccard is
already the right tool and should stay (see [PROJECT_REVIEW.md](PROJECT_REVIEW.md), "Movement/style
retrieval"). The research question is:

> Do the pretrained model's representations find climbs that are similar in style or
> difficulty, beyond what hold overlap and hand-made features find?

## 2. What already exists

- **Website closest-climb lookup** ([web/src/similar.mjs](web/src/similar.mjs)): hand-hold
  Jaccard over all 206,783 climbs, with an inverted index. Feet are ignored.
- **Family split** ([kilter/splits.py](kilter/splits.py), `data/splits/family_v1`): climbs that
  share a UUID, an exact hold set or a hand set are always in the same partition. Every held-out
  climb records `nearest_train_hand_jaccard`, and results can be reported per band
  (`distant` < 0.5 ≤ `related` < 0.75 ≤ `near_duplicate`).
- **Evidence that the frozen model encodes useful climb information**
  ([experiments/grade_climbs.py](experiments/grade_climbs.py)): mean and last hidden states, read
  with a null grade, predict grade with RMSE 1.30, against 1.42 for bag-of-holds with the same MLP
  head and 1.55 for an untrained copy. The same pooling code is a natural starting embedding.
  Models trained for grading alone set a ceiling for reference: fine-tuning the generator gets
  1.25, a BERT-style encoder 1.26 and LightGBM on hand-made geometry 1.29
  ([experiments/README.md](experiments/README.md#models-trained-for-grading)). LightGBM's
  geometry features are a ready-made start for baseline 5 below.
- **Labels that don't come from hold overlap**: 21 style tags from names and descriptions
  ([kilter/climb_tags.py](kilter/climb_tags.py)), hand-labelled hold type and grip direction
  ([data/hold_labels.csv](data/hold_labels.csv)), grade per angle, ascent counts, setter, and name.

## 3. Benchmarks (ground truth)

There are no human similarity labels, so the plan uses several proxy benchmarks. For each one it
matters whether it is **independent of hold overlap**. A benchmark built from shared holds will
always favour Jaccard, so a learned method can't show anything on it.

Queries come from **validation** climbs, and the reference corpus is **training climbs only**
(the review asks for this for any retrieval benchmark). The `family_v1` test split stays sealed
until a final run through [experiments/final_test.py](experiments/final_test.py).

### B1. Named variants (overlap-dependent, but realistic)

Same setter, with a name of the form `<base> <suffix>` where `<base>` is another of their climbs.
A quick count finds **1,659 same-setter pairs**. They come in two kinds:

| Suffix | Pairs | Median hand Jaccard | Meaning |
| --- | ---: | ---: | --- |
| `sit`, `stand`, `easy`, `hard`, `extension`, `direct`, `low`, `high` | 131 | 0.67 | True variants |
| `2`, `3`, `ii`, `iii`, `v2`, `part 2` | 1,467 | 0.03 | Sequels: same setter and idea, mostly different holds |
| Other (`left`, `right`, `mirror`, `v5`, …) | 61 | 0.11 | Mixed; check by hand |

- **True variants**: retrieve the base climb from the variant; report Recall@{1,10} and MRR.
  Jaccard should do well here. The model must not do *worse*.
- **Sequels**: a weak but overlap-independent signal (same setter, same theme). Report as
  Recall@100 or AUC against random same-angle climbs, since many sequels won't be very similar.
- The family split may put a variant and its base in the same partition, so pairs where both
  are in training are fine as long as the query is not in the corpus. Build the pair list once
  and commit it (`data/similarity/variant_pairs.csv`), with the regex and a hand-checked sample of 50.

### B2. Style tag agreement (overlap-independent, noisy)

For a query with tag *t*, the share of its top-k neighbours that also have *t*, divided by the
base rate of *t* (**lift@k**). Only positive tags count (a missing tag doesn't mean the climb
lacks the style). Use the tags with enough support: campus, dynamic, pinch, crimp, reachy, jug,
traverse, body tension, heel hook. Report per tag and macro-averaged. **Restrict to neighbours
with hand Jaccard < 0.5 to the query** as a second view, so the score can't come from near copies.

Caveat: tags read from names (e.g. "pinch") also correlate with the holds used, so some tags can
be found from hold types alone. That is why the hold-type feature baseline (section 4) matters.

### B3. Difficulty consistency (overlap-independent)

kNN grade prediction: predict a query's grade at an angle as the mean grade of its k nearest
training climbs *at that angle*. Report RMSE and compare with
[results/grade_climbs/](results/grade_climbs/). This measures whether the neighbourhoods are
coherent in difficulty, not whether the method is a good grader. Also report the neighbours'
grade spread. For scale: the frozen probe gets RMSE 1.30 on all test climbs, the best trained
grader 1.25, the bag-of-holds MLP 1.42 and the training mean at that angle 2.24.

Because the model is read with a null grade, there is no leak. The embedding must never be computed
with the true grade in the condition, or B3 becomes trivial.

### B4. Synthetic perturbations (controlled)

Take a validation climb and make a modified copy. The copy should retrieve its source at rank 1
against the training corpus plus the source:

| Perturbation | What it tests |
| --- | --- |
| Drop 1–2 hand holds | Robustness (Jaccard handles this already) |
| Swap a hand hold for the nearest hold **of the same labelled type** | "Same movement, different hold": Jaccard falls, a good embedding shouldn't |
| Swap a hand hold for a nearby hold of a **different** type | Should hurt more than a same-type swap |
| Replace all feet | Feet-insensitivity (hand Jaccard ignores feet, so this is a check on the model) |
| Left–right mirror (341 of 476 holds have a hold at the mirrored position, 219 with the same type) | Approximate movement-preserving transform. Kilter Original holds are not true mirrors, so this is a soft test. |

The key measure is the **gap between same-type and different-type swaps**: a method that only
counts shared holds can't tell them apart.

### B5. Human judgements (the real target, small)

Blinded triplets: an anchor and two candidates, "Which climbs more like the anchor?" or "neither".
Candidates come from two different methods' top-5, shown in random order. 100–200 triplets from
3–5 climbers (the people in [kevin_notes.md](kevin_notes.md)) is enough to separate methods that
differ clearly. Report the win rate per method pair and inter-rater agreement (Fleiss' κ).
Do this after B1–B4, comparing only the 2–3 best methods. A "Which is more similar?" page in
`web/` would make it easy to collect, and should store raw answers with rater IDs.

### Diagnostic (not a benchmark)

For each pair of methods, top-10 overlap and Spearman correlation of similarity scores. If the
model's neighbours are 90% the same as Jaccard's, it adds little. Also show the
Jaccard distribution of each method's top-10.

## 4. Baselines

All baselines use the same query set and the same training-only corpus, and are scored by the same
code. Ordered roughly by strength:

| # | Baseline | Representation / score | Why include it |
| --- | --- | --- | --- |
| 0 | **Random** | Random training climbs, same angle | Floor; gives base rates for lift |
| 1 | **Hand Jaccard** | Shared hand holds / union (what the website uses) | Main overlap baseline |
| 2 | **Full Jaccard + IDF** | All holds and roles, each hold weighted by log inverse frequency | Rare shared holds are stronger evidence than the common jugs everyone uses |
| 3 | **Soft (spatial) Jaccard** | Holds within radius *r* count as partially matched (Gaussian kernel on x, y) | Handles "moved the hold one slot over" with no learning |
| 4 | **Point-set distance** | Chamfer or optimal-transport (EMD) distance between hand-hold sets in (x, y), optionally with type as an extra coordinate | Geometry-only similarity; a strong non-learned baseline |
| 5 | **Hand-made climb features** | Hold count, height and width span, start/finish heights, mean and max vertical gap between successive hands (sorted by y), lateral spread, hold type histogram, grip direction histogram, foot count, kickboard use, angle; standardised, cosine | Tests whether the model learned more than obvious summary statistics. Very important for B2 and B3. |
| 6 | **Hold co-occurrence embeddings** | PPMI matrix of holds co-occurring in training climbs, SVD to 64 dims (or word2vec over climbs), mean-pooled per climb | A cheap "learned from the catalogue" method without a transformer |
| 7 | **Bag-of-holds PCA / autoencoder** | PCA (or small autoencoder) of the multi-hot token vector used in `grade_climbs.py` | Same input as the bag baseline there |
| 8 | **Untrained transformer** | Same architecture, random weights, same pooling | Separates what pretraining adds from what the architecture and hold features give for free (the same control `grade_climbs.py` and `hold_embeddings.py` use) |

Baselines 1, 3 and 5 are the ones the model really has to beat. If baseline 5 matches the model on
B2 and B3, the conclusion is that the model mostly learned summary statistics.

## 5. Main approach: the pretrained model

Use `null_rule_16` (the model `grade_climbs.py` uses) and `null_rule_16_tags` (the one the site
serves), with three seeds where it's cheap. Retrain on `family_v1` first if those checkpoints are on the
historical split (see the P0 notes: historical runs aren't reused).

### 5a. Frozen embeddings (start here)

Run the climb in `scratch` mode with a **null grade** and a fixed angle, and pool the hidden states.
Options to compare on validation:

- **Pooling**: mean over hold positions, the last position (EOS: it has seen the whole climb), and
  their concatenation (what `grade_climbs.representations` already returns). The decoder is
  causal and holds are in canonical order (starts first, feet last), so mean pooling is weighted
  towards early holds. Also try mean over **hand positions only**, so feet don't dominate.
- **Layer**: final norm vs each intermediate block. Middle layers often transfer better for
  similarity than the last layer, which is tuned for predicting the next token.
- **Angle**: fix it at 40° for every climb (angle-invariant similarity), or use the climb's own
  angle (angle-aware). Try both.
- **Post-processing**: transformer embeddings are usually anisotropic (one direction
  dominates cosine). Try mean-centring, PCA whitening, and removing the top 1–3 principal
  components, all fitted on training climbs only.
- **Static token table**: mean of `token_table()` rows for the climb's tokens. This needs no forward
  pass, which tells us whether the context (the hidden states) adds anything over hold embeddings.

Retrieval is exact cosine top-k over ~190k training climbs. That's a 190k × 192 matrix, so plain
matrix multiplication on CPU is fine and no ANN index is needed.

### 5b. Model-native similarity (uses the model as a model, not just a feature extractor)

1. **Predicted-hold profile.** For a climb, give a random subset of its holds and record the
   model's distribution over the next hold. Average over several masks to get a ~1,900-dim "which
   holds fit this climb" vector, and compare climbs with Jensen–Shannon divergence. Two climbs that
   use neighbouring or interchangeable holds get similar profiles, which handles same-type swaps
   (B4) with no extra training.
2. **Cross-completion likelihood (reranker only, O(k) per query).** For a candidate B and query A,
   score how plausible B's holds are when A's starts (or A's hands) are given, minus B's own
   `starts`-mode NLL, and symmetrise. This is expensive, so use it only to rerank the top 50 from 5a.
3. **Grade-shifted retrieval.** "Like this, but harder": compare embeddings computed with the
   query's angle, and filter or rerank candidates by grade. Optionally test whether the
   difference between embeddings at grade *g* and *g + 2* is a consistent direction. That's a
   product feature as much as a research question.

### 5c. Fine-tuning for similarity (only if 5a/5b leave a gap)

Contrastive fine-tuning (InfoNCE) of the pretrained encoder plus a small projection head:

- **Positives that come from the data**: climbs in the same family (same hand set, different
  feet), true named variants from B1's **training** pairs, and the same climb at a different angle
  (only if the embedding is angle-aware; otherwise those are identical).
- **Augmentations**: drop holds, swap a hold for the nearest same-type hold, resample feet.
- **Hard negatives**: climbs with high Jaccard but different tags or a grade more than 3 apart.
- **Control**: the same objective trained from a random initialisation, to show what pretraining adds
  (the same comparison as the untrained baseline). Keep the swap and drop rules **out of** B4, or
  B4 will just measure the augmentations. If 5c uses swaps, B4 should use a perturbation it never
  saw (e.g. the mirror).

## 6. Protocol

- One script, `experiments/climb_similarity.py`, following the
  [experiments README](experiments/README.md) conventions. Outputs go in `results/climb_similarity/`.
  Every method implements `embed(rows) -> matrix` or `score(query, candidates) -> vector`, so
  benchmarks are written once.
- Queries: validation climbs (one row per climb at its most-ascended angle, unless the benchmark
  is per angle). Corpus: training climbs. Report B1–B4 overall and **per similarity band**
  (`distant`, `related`, `near_duplicate`), since most of the interesting cases are in `distant`.
- Hyperparameters (layer, pooling, whitening, *r* for soft Jaccard, k) are chosen on a split of the
  validation queries, and the reported numbers come from the other part. Test is opened once at the end.
- Three seeds for model-based methods; bootstrap 95% intervals over queries (as `grade_climbs.py` does).
- Save each method's top-20 per query, so B5 triplets and the website can be built from them later.

### Expected outcome per benchmark

| Benchmark | Jaccard | Features (5) | Model, hoped for |
| --- | --- | --- | --- |
| B1 true variants | strong | weak | close to Jaccard |
| B1 sequels | near random | some | better than both |
| B2 tag lift | low on `distant` | medium | best |
| B3 kNN grade RMSE | medium | medium | best (frozen grader already beats bag) |
| B4 same-type swap gap | ~0 | small | clear gap |

If the model doesn't win on B2, B3 and the B4 gap, the honest conclusion is that hold overlap plus
features is enough, and the website lookup should stay as it is.

## 7. Website ideas (after the benchmarks)

- Show "Similar style" (model) next to "Closest real climb" (Jaccard), clearly labelled: one is
  about holds, the other is learned.
- "Like this, but V(n±1)" and "like this, at another angle" filters.
- Embeddings for every climb, precomputed and quantised to int8 (206k × 192 ≈ 40 MB, or ~10 MB at
  64 dims after PCA). Brute-force cosine in the browser is fast enough. Generated climbs need the
  model's forward pass, which the site already runs.
- Diversity check for generation: flag a generated climb whose model-nearest neighbour is very
  close, even when Jaccard says it is novel.
- The triplet-labelling page for B5.
- As the review says, hide neighbour suggestions during any blinded generation assessment.

## 8. Risks and pitfalls

- **Circular benchmarks.** Anything defined by shared holds (families, near-duplicate bands, B1
  variants) favours Jaccard. Don't present wins on those as evidence of "movement similarity".
- **Popularity bias.** Embeddings may cluster common holds and popular climbs. Check the top-k
  ascent-count distribution, and the "hubness" (how often each climb appears in other climbs' top-10).
- **Tag noise.** Tags are setters' words, positive-only and name-biased. Use lift, not accuracy.
- **Grade leak.** Always embed with a null grade for B3. A grade-conditioned embedding is fine for
  the product but not for the benchmark.
- **Angle.** The same holds at 20° and 50° are very different climbs. Decide per benchmark whether
  similarity is angle-aware, and say so.
- **Mirror is approximate** on the Kilter Original: only 219 of 476 holds have a same-type
  mirror.
- **Sealed test.** Don't evaluate on `family_v1` test while choosing methods.

## 9. Suggested order

1. Build the benchmark data: variant pairs (with a hand-checked sample), tag sets, perturbation
   generator. Commit the pair list. *(~1 day)*
2. Baselines 0–5 and the benchmark harness, with results on validation. *(~1–2 days)*
3. Frozen model embeddings (5a), all pooling/layer/whitening options, plus the untrained control.
   *(~1 day)*
4. Diagnostics: how much the model's neighbours overlap with Jaccard's, per band. Decide whether 5b or 5c is worth it.
5. Predicted-hold profile (5b.1). Cross-completion reranking only if 5a looks promising.
6. Small human triplet study for the best 2–3 methods.
7. Write up in `experiments/README.md`, open test once, and add to the website if it wins.

## 10. Open questions

- Which product question matters most: "same style on different holds" or "what to climb next"?
  That decides whether B2 or B3 is the primary metric.
- Should similarity be angle-aware by default?
- Feet: the website ignores them. Should the learned similarity also ignore them (hand-only pooling),
  or are feet part of style (heel hooks, body tension)?
- Does the raw Kilter data have per-user ascent logs? If so, "climbers who sent A also sent B"
  co-occurrence would be the best overlap-independent ground truth there is. The committed data
  only has ascent counts.
