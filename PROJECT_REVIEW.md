# Climbing-board foundation model: repository review and development plan

Reviewed November 19, 2026. Base commit: `3224d95d22767ad6f1691ec26ded22e73d9986ad`, including the working tree as found. Existing uncommitted website changes were reviewed and left intact. This review changes only this document.

## 1. Assessment and recommended direction

**Keep the current generator and build a better measurement and feedback loop around it.** This repository already demonstrates that a small transformer can learn useful structure from the Kilter catalogue. Conditional completion is designed sensibly, generation works in the browser, and frozen representations improve grade prediction. Those are substantial reasons to continue with this approach.

**The largest missing ingredient is evidence about actual climbing.** The training objective predicts catalogue holds. The generation evaluation mostly checks serialization, role counts, simple geometric summaries, and exact novelty. These measure useful properties, but they cannot establish enjoyable movement, appropriate difficulty, sensible beta, or suitability for a particular climber. A larger model trained on the same data does not supply those missing labels.

My recommended near-term product is a **Kilter setting assistant**: generate several candidates, complete or edit partial climbs, estimate difficulty with uncertainty, find related climbs, and record what people actually tried. This makes the existing model useful while collecting the evidence needed for better models.

For the longer-term foundation-model goal, define success through transfer: one pretrained backbone should improve several tasks under controlled adaptation budgets. The established foundation-model framing emphasizes broad pretraining and adaptation across tasks; parameter count alone is insufficient. A useful domain foundation model can be small, but its claimed breadth needs evidence. [Foundation-model report](https://arxiv.org/abs/2108.07258).

The current evidence supports **a board-specific generative model with useful in-domain representations**. Cross-board transfer, personal success prediction, and movement understanding remain research goals.

### Priority order

Here, P0 means “fix before relying on the next experimental comparison”; P1 means “next development cycle”; P2 means “after the measurement loop is working.” Effort is relative, assuming familiarity with this repository.

| Priority | Action | Why it matters | Relative effort |
| --- | --- | --- | --- |
| P0 | Correct validity/novelty reporting and version evaluation outputs | Current headline proxies have misleading semantics and one incorrect training-membership check | Small |
| P0 | Use immutable experiment identities and explicit validation/test stages | Existing reuse can silently accept stale runs/results; the sweep routinely exposes test results | Small–medium |
| P0 | Add a versioned split manifest and a family-separated benchmark | Exact UUID separation misses related hold-set variants | Medium |
| P0 | Preserve import/export provenance and prevent cancelled auto-send | Feedback data would otherwise acquire incorrect generator labels; cancellation currently triggers auto-send | Small |
| P1 | Run a blinded, on-board quality pilot and add feedback export | This directly measures the project’s main goal | Medium, plus climbing sessions |
| P1 | Evaluate a small candidate pool with transparent filtering and reranking | Likely faster route to better setting assistance than another capacity increase; benefit must be measured | Medium |
| P1 | Make tag/null conditions work through the complete export/runtime path | Promising style experiments cannot currently be served by the existing exporter | Medium |
| P1 | Benchmark grading, editing, and retrieval with strong controls | Establishes whether the shared model earns its downstream role | Medium |
| P1 | Test a role-dependent output head and more realistic edit masks | Two small, targeted modeling opportunities | Medium |
| P2 | Add richer hold geometry, new boards, and real movement observations | Necessary for broader transfer and beta tasks, but considerably more data work | Large |

If resources are limited, finish the P0 work and the human pilot before expanding the model family.

## 2. What the repository currently does

| Component | Current behavior | Assessment |
| --- | --- | --- |
| Raw acquisition: `data/download.py` | Downloads pinned public snapshot/grade artifacts and board geometry, checks hashes and database integrity | Good reproducibility foundation |
| Conversion: `data/prepare.py`, `data/validate.py` | Decodes frames; joins grades and ascents by UUID **and angle**; validates complete source coverage and geometry | Careful handling of important data pitfalls |
| Compact catalogue: `data/export_training.py` | Commits hold/role sets, grade, angle, ascents, name, setter, and vocabulary | Convenient, but drops metadata needed for quality, uncertainty, and temporal analysis |
| Encoding: `kilter/data.py` | Filters to a single-frame family; hashes UUIDs into splits; removes exact train/held-out hold-role duplicates; canonicalizes holds | Sensible baseline split, incomplete family separation |
| Core model: `kilter/model.py` | Causal decoder with shared hold, geometry, and role embeddings; numeric grade/angle features; optional null grade, hold labels, and text-derived tags | A practical small model with one notable output-head restriction |
| Training: `kilter/experiment.py` | Redraws given subsets; predicts remaining holds/EOS; AdamW, warmup/cosine schedule, clipping, validation selection, source snapshots | Generally sound; run reuse and CUDA resume need attention |
| Generation: `kilter/sampling.py` | Samples remaining holds while preserving picks, using optional legality constraints | Good fit for interactive setting; no movement or quality verifier |
| Experiments: `experiments/` | Three training seeds, capacity/training ablations, hold probes, style steering, grade prediction | Useful evidence, with several evaluation-scope limitations |
| Browser: `web/` | Local ONNX/WASM inference in a worker, board rendering, nearest real climb, imports/exports, labels, Bluetooth | A strong base for collecting actual setting feedback |
| Wireframes: `wireframe/` | LED registration, camera recovery, pose lifting/contact heuristics, synthetic evaluation | Separate prototype; real-video accuracy is unestablished |

### The learned distribution

There are 476 physical holds, four hold roles, and four special tokens: 1,908 vocabulary entries. The core sequence is:

```text
BOS [given holds] SEP [remaining holds] EOS
```

Each block has a deterministic order: starts, hands, finishes, feet; within a role, bottom to top and then left to right. Positions restart at `SEP`. The loss ignores predictions inside the given block and scores the remaining holds plus EOS. See [`build_example`](kilter/data.py#L270) and [`forward`](kilter/model.py#L97).

The default training mixture is 30% generation from scratch, 45% all starts plus random feet, and 25% arbitrary subsets. This is a meaningful improvement over forcing partial climbs into a scratch-only model’s prefix. Keep the direct completion objective.

The website serves `null_rule_16`, seed 43: width 192, six layers, six heads, approximately **2.856 million parameters**, maximum 40 holds, 50% null-grade training, and legality-masked training loss. It differs from the default quick-start configuration, which trains the smaller, eight-epoch, unmasked `climb_v2` model. Add an explicit served-model config so reproducing the deployed recipe does not require reading experiment overrides.

**The canonical sequence is a representation of a hold set. It is not the climber’s movement sequence.** Every beta-related feature and claim should preserve that distinction.

### Evidence already worth keeping

These numbers come from committed results, rather than new training during this review. Comparisons within each row use the reported matching metric; raw and legality-masked likelihoods are different quantities.

| Evidence | Recorded result | What it supports |
| --- | --- | --- |
| Large generator versus original baseline, scratch raw NLL | 45.31 versus 50.45 nats/climb, three-seed means | Better catalogue modeling on the evaluated held-out family |
| Chosen `null_rule_16`, legality-masked test likelihood | 44.58 scratch; 37.43 starts; 29.01 starts + feet; mean 37.01 | A useful reference for future models using the same support/rules |
| Frozen backbone + grade MLP versus bag-of-holds MLP | MSE 1.697 versus 2.028; roughly 16.3% lower MSE | Representations improve this in-domain downstream task |
| Frozen backbone + grade MLP, absolute error scale | RMSE 1.303 grades; 62.4% within one grade | Useful signal, with substantial remaining uncertainty |
| Hold-vector hand-type probe | Balanced accuracy 76.7%, versus usage counts 52.0% and position 24.2% | Hold identity representations contain meaningful attributes |
| Hold-label features added to generator | Mean test masked NLL about 37.00 versus 37.01 | Little additional benefit on this familiar board under this recipe |
| Style-tag model, true tags on tagged test rows | Scratch masked NLL 41.69 versus 42.68 for the untagged model | Some text-derived style information is useful |

Sources: [`experiments/README.md`](experiments/README.md), [`model comparisons`](results/compare_models/summary.csv), [`grading results`](results/grade_climbs/summary.json), [`hold probes`](results/hold_label_probes/summary.json), and [`tag results`](results/tag_conditioning/summary.json).

Do not interpret these as proof of physical climbability, personal grade accuracy, or transfer to unfamiliar holds. Also, the original baseline is much smaller and differs in ordering, training schedule, filtering, and completion objective. Its defeat does not identify which individual change helped.

## 3. Confirmed issues to address

The issues below are grounded in code inspection, corpus measurements, or a reproduced failure. Predicted effects of proposed model changes are identified separately later.

### 3.1 The reported validity metric conflates several different properties

[`climb_checks`](kilter/sampling.py#L175) defines `valid` as all of:

- No repeated physical hold.
- One or two starts and one or two finishes.
- Every start below every finish.
- At least one explicit foot-only hold.

A legitimate campus climb or climb using hand holds for feet can fail the last condition. Meanwhile, a collection of unreachable hands can satisfy every condition. The “largest hand gap” summary considers only sorted vertical coordinates, omitting horizontal separation and actual beta.

The decoder guarantees a narrower set of properties: unique holds, canonical order, role-count limits, required start/finish before EOS, and the hold budget. It does **not** guarantee the complete reported `valid` predicate. Statements in [`docs/model.md`](docs/model.md) claiming guaranteed structural validity need that qualification.

**Change:** report separate fields for `schema_valid`, `starts_below_finishes`, `explicit_feet_present`, `constraint_satisfied`, and any experimental reach/plausibility score. Keep foot count as a descriptive/style feature. Reserve “climbed successfully” and “enjoyable” for observed feedback. A conventional upward climb can have additional task-specific checks without making those universal rules.

The distinction affects model selection: the chosen model improves masked NLL, but its current test validity proxy is about **94.0%**, versus **96.1%** for the eight-epoch null-grade model. Neither metric alone establishes the better setting model.

### 3.2 Exact novelty includes climbs explicitly removed from training

[`quality.py:109`](kilter/quality.py#L109) builds `train_signatures` from split `0` **and** split `255`. Split `255` contains rows removed from training because their exact signature appears in held-out data.

**Change:** training novelty should use `splits == 0`. Separately report novelty against the full catalogue and against the excluded-duplicate pool. The present check can classify a held-out climb as seen in training even though its corresponding rows were removed. This does not imply the reported generated samples were copied; it makes the reference set incorrect.

### 3.3 Split protection is incomplete across related climbs

The existing UUID split correctly keeps all angles of a UUID together. Train/held-out exact **hold-role** signatures are disjoint after filtering. However, duplicate grouping is applied by removing training rows after UUID splitting; it does not assign entire families to one partition. See [`prepare`](kilter/data.py#L194).

Direct measurements from the committed encoded dataset, `data/experiments_v2/082387aa517d72cb83ff/dataset.npz`:

| Relationship | Measured overlap |
| --- | --- |
| Exact full hold-role signatures shared by validation and test | 7 signatures, affecting 16 validation rows and 22 test rows |
| Test physical hold sets also present in training, ignoring roles | 39/16,897 rows, 13 UUIDs, about 0.23% |
| Test hand sets also present in training, ignoring foot holds and differences among hand/start/finish roles | 673/16,897 rows, 240 UUIDs, about 3.98% |
| Equivalent hand-set overlap for validation | 706/16,936 rows, 262 UUIDs, about 4.17% |

Foot variants and changed start/finish designations can be legitimate different climbs. The last two rows are **relatedness measurements**, not proof that those examples must all be discarded. They do establish a need to measure robustness to family separation before claiming strong novelty or broad generalization.

**Change:** preserve the historical split for comparison; introduce a new versioned family split. Group exact duplicates before splitting, and add a stricter robustness partition using hand-set equality and a defined near-duplicate threshold. Check graph component sizes before using connected components: transitive similarity can otherwise create giant families. Report performance by nearest-training similarity as well as overall.

### 3.4 Experiment reuse can silently mix incompatible runs

[`finished_runs`](experiments/compare_models.py#L73) keys runs only by `(name, seed)`. The sweep reuses them without checking whether the currently requested config, implementation, or dataset matches. Evaluation files are usually reused based on existence, so changes to requested sample count, evaluator code, baseline, or generation settings may also be ignored. The code additionally assumes `runs/logs` already exists and hardcodes CUDA for evaluation at line 93.

**Change:** define a run identity from resolved configuration, source revision/content, dataset/split fingerprint, label/tag hashes, and seed. Define a separate evaluation identity including checkpoint hash, evaluator version, rules, prompt set, seeds, temperature/top-k, and sample budget. Refuse mismatches and create required directories. Add a device argument with a working CPU default/auto selection.

Copying source into a run, saving hashes, rejecting unknown config keys, and atomic checkpoint writes are all good existing practices. Extend that rigor to orchestration and result reuse.

### 3.5 The experiment workflow routinely exposes the test set

[`compare_models.py`](experiments/compare_models.py#L56) evaluates every variant on validation and test. The experiment history describes subsequent variants motivated by earlier results. That workflow makes test feedback available during ongoing development, even if architecture choices were intended to use validation alone.

This is a methodological risk, not an accusation that a particular choice used test performance.

**Change:** split the commands into development evaluation and an explicit final-test action. Treat current test results as historical development evidence. Create a new sealed benchmark, ideally using a later snapshot plus family/UUID separation, and record the decision criteria before opening its results. For temporal tests, assign every version/angle/family consistently; a new ascent at a new angle is not necessarily a new climb.

### 3.6 Tag-conditioned models cannot use the existing browser export path

The model accepts 26 additional tag columns when configured for style tags. The exporter constructs two-column grade/angle examples and fixtures at [`export_web.py:133`](kilter/export_web.py#L133). Browser decoding constructs two columns at [`decoding.mjs:93`](web/src/decoding.mjs#L93), and the runtime fixes the tensor shape to `[1, 2]` at [`runtime.mjs:15`](web/src/runtime.mjs#L15).

A minimal in-memory tagged model fails with exporter-shaped conditions due to the input-width mismatch. The successful tag experiments therefore do not currently imply deployable tag controls.

**Change:** serialize a condition schema containing names, order, defaults, nullable fields, supported ranges, and tag vocabulary. Build Python examples, ONNX fixtures, browser tensors, validation, and controls from that schema. Include null-grade and tag cases in parity checks. The current export parity grid tests numeric grades, but does not test null grade.

Also verify dataset/vocabulary geometry against the checkpoint **before** publishing the model/catalogue bundle, and stage the complete export atomically. The current catalogue export prepares current source data separately from the saved checkpoint metadata.

### 3.7 Imported climbs acquire the wrong provenance when exported again

The parser retains an imported checkpoint identity, but application result state keeps only EOS and the filename. Export then stamps the currently loaded model’s name/hash. See [`app.mjs:493`](web/src/app.mjs#L493), [`app.mjs:543`](web/src/app.mjs#L543), and [`climb-file.mjs`](web/src/climb-file.mjs).

Consequently, a climb imported from another model—or from the real catalogue—can later appear to have been generated by the current model. That would contaminate future quality and preference data.

**Change:** separate `source` from `viewer_model`. Preserve original climb UUID, setter/name where available, generating checkpoint, original request, observed/unknown grade, and subsequent edits. Missing imported grade/angle must remain unknown in provenance rather than silently becoming observed metadata from the current controls. Version the export schema.

### 3.8 Cancellation still triggers automatic board transmission

The `done` handler at [`app.mjs:570`](web/src/app.mjs#L570) auto-sends any nonempty result without checking `cancelled` or `endedWithEos`, despite presenting auto-send as completed-climb behavior.

**Change:** auto-send only when `!data.cancelled && data.endedWithEos`; allow deliberate manual sending of partial climbs. Add a regression test covering cancellation with auto-send enabled.

### 3.9 CUDA RNG state is missing from resume checkpoints

[`experiment.py`](kilter/experiment.py#L292) saves Python, NumPy, CPU Torch, loader, and collator states, but omits CUDA generator states. CUDA dropout will therefore not continue identically after resume, even when deterministic algorithms are enabled.

**Change:** save/restore all CUDA RNG states when applicable. Verify uninterrupted versus resumed training with a small training fixture on each supported backend. The existing CPU-oriented reproducibility work should remain.

### 3.10 Probe preprocessing uses held-out hold features

[`hold_label_probes.py:254`](experiments/hold_label_probes.py#L254) fits centering, PCA, and scaling on all holds before outer/inner cross-validation. This is unsupervised transductive preprocessing, not leakage of type labels into model pretraining, but it is weaker than a strictly inductive probe pipeline. The constant grip baseline also uses all target directions.

**Change:** fit the entire transformation inside each training fold, including inner folds for penalty selection; fit the constant direction on that fold’s training targets. Reuse identical folds across competing representations and use an explicit intercept/fold centering for ridge. Rerun the results before strengthening claims. The large type-probe advantage is encouraging, but its corrected magnitude needs measurement.

## 4. Improve the data and benchmark before scaling

### Corpus size and imbalance

The compact source has **339,659 climb-angle rows** from **207,175 UUIDs**. The current model family retains 338,196 rows before duplicate removal. After excluding 816 training rows for exact held-out duplication:

| Split | Rows | UUIDs |
| --- | ---: | ---: |
| Train | 303,547 | 186,039 |
| Validation | 16,936 | 10,174 |
| Test | 16,897 | 10,344 |

These are not 303,547 independent training climbs: multiple angles repeat the same hold set with different conditions. Both row-weighted and UUID-weighted views are useful.

Further measured properties of training rows:

- About **41.95% have one reported ascensionist**.
- **40° accounts for 31.75%** of rows.
- Only **1,390 rows are V12+**, and **47 are V14+**.
- Data covers 0–70° in 5° increments; support is much thinner in some grade-angle combinations.

The website already notes angles above 70° are outside training coverage. Improve this to a manifest-driven grade-angle support matrix, since “within the angle range” can still be a sparse or unsupported grade-angle request. Unify validation limits: the browser decoder currently permits grades beyond the UI/model’s usual V0–V16 range. Offer extrapolation deliberately and evaluate it separately.

### Preserve useful metadata without bloating every training row

[`data/export_training.py`](data/export_training.py#L23) drops descriptions, creation time, detailed grade provenance, continuous/native difficulty, computed-grade confidence, and quality averages. Many are already present in the full processed catalogue. Export a compact versioned sidecar keyed by UUID/angle containing the fields required for:

- Temporal and setter-held-out tests.
- Native versus computed grade sensitivity analysis.
- Grade uncertainty and consensus strata.
- Quality experiments and text-label auditing.
- Board-size compatibility and future active-area controls.

Keep original continuous/native difficulty alongside Hueco labels. Mapping to V grades loses resolution; evaluate both where meaningful. Exclude or separately score computed-grade targets in the main grading benchmark so an upstream model’s estimates are not treated as equivalent to independently reported grades. There are only about 2,556 computed-grade rows in the complete export, so measure the effect before making it a major project.

The source includes climbs compatible with smaller Original board crops on the 12×12 layout. That is physically legitimate because the placements match. Preserve compatibility/active-area information if the product will support those crops. Do not infer the original setting board solely from a climb’s bounding box; a centrally located climb need not have been set on a smaller wall.

### Treat consensus, popularity, and enjoyment separately

At least one reported ascent provides evidence somebody climbed the problem at that angle. It supplies limited information about agreement on difficulty and almost no direct information about how enjoyable a generated variation will be.

The ascent-weighted variant already worsened catalogue likelihood. That does not prove ascent information is useless; it shows that one global popularity weighting was not better under this objective. Older, easier, visible climbs have more opportunities for ascents.

Recommended experiments are modest: cap/normalize repeated-angle influence, try UUID-balanced batches, and use ascent/grade-source information to model confidence rather than treating popularity as the definition of quality. A broad pretraining set followed by a small, carefully labeled quality adaptation set is a reasonable hypothesis to test. Do not discard every one-ascent climb by default: that would remove substantial diversity and still would not create enjoyment labels.

### Establish a benchmark suite with explicit purposes

| Partition | Question |
| --- | --- |
| Historical UUID split | Can results be compared with the repository’s existing experiments? |
| Exact/family-separated split | Can the model handle climbs beyond close catalogue variants? |
| Setter-held-out split | Does it generalize across setting styles and habits? |
| Temporal/future snapshot split | Does it generalize to later catalogue additions? |
| Sparse grade-angle slices | What happens outside dense training support? |
| Future held-out board/setup | Does the representation transfer across physical hold inventories? |

Store row/family membership in a committed split manifest with checksums, construction rules, and coverage statistics. Keep metadata transformations and hyperparameter choices inside training/validation boundaries. Report both deployment-weighted aggregates and macro averages over sufficiently supported grade-angle cells; include sample counts everywhere.

## 5. Measure and improve actual generated climbs

### Replace the single quality score with a scorecard

| Dimension | Appropriate evidence |
| --- | --- |
| Schema and locked-hold preservation | Exact deterministic checks; expect 100% on accepted requests |
| Difficulty control | Independent grade estimator as a screen; blind climber grades as the main evidence |
| Climbability | Attempts by climbers with appropriate ability; record why an attempt stopped |
| Movement quality | Blinded ratings/preferences after attempts, including flow, awkwardness, memorable movement, and crux |
| Diversity | Pairwise similarity distributions, style/hold-use coverage, repeated-output rate |
| Novelty | Exact and nearest-neighbor similarity to training and full catalogue, with hands and feet separated |
| Editing usefulness | Accepted replacements, edit count, task success, time to an accepted climb |
| Operational quality | Load time, generation latency, cancellation, export integrity, physical-board behavior |

Hold count versus grade is a distributional diagnostic, not a grade-control test. A model can match that trend with the wrong hold difficulty and impossible moves. Also evaluate the same real climb across its recorded angles: measure angle-specific grade error and consistency, without imposing universal monotonicity on every hold configuration.

Current quality sampling covers only V2, V5, V8, V11 at 20°, 40°, 50°. The three-seed test reports use **482 samples per method/seed**, rather than a complete 600-sample grid, because they cap samples at the available real rows in each cell. For example, the test set has no V11 at 20° and only 11 V8 rows at 20°.

[`quality.py`](kilter/quality.py#L128) then pools angles into grade summaries. This means the reported grade/hold-count trend is partly confounded by uneven angle composition. Preserve condition cells, use a fixed evaluation prompt manifest, repeat generation seeds, and standardize angle mixtures for grade comparisons. From-scratch generation does not need its sample count capped by available real prompts; sample it on a prespecified grid and disclose real-reference uncertainty separately.

Training-seed standard deviation is useful but is not a confidence interval over climbers, prompts, catalogue families, or generated samples. Use paired comparisons and resampling at the relevant unit: UUID/family for catalogue tasks, prompt and seed for generation, climber/session and climb for on-board ratings. The grading experiment’s UUID-clustered bootstrap is a good starting pattern.

### Start a blinded on-board pilot

Use the existing Bluetooth site to make this practical. A manageable pilot could use **12 supported grade/angle prompts and three candidate sources**, giving 36 climbs: the current model, one proposed improvement, and de-identified human-set catalogue references. This is a protocol/debugging pilot, not a powered final comparison.

Have at least two suitable climbers assess each climb where feasible. Randomize presentation/order, hide model identity and nearest-catalogue panels during judgment, and ask for perceived grade before revealing the requested grade. Record recognition of existing climbs. Spread attempts across sessions and record fatigue/order so the last candidate is not systematically disadvantaged.

Collect:

- Attempted or visual impression only; completion, attempts, and failure location/reason.
- Perceived grade and confidence; whether the person can normally climb around that level.
- Enjoyment, flow/awkwardness, crux location, and whether they would recommend or repeat it.
- Any holds added, removed, or re-roled to improve it, with the original preserved.
- Optional height/reach and intended/observed beta, if personal accessibility is a target.
- A pairwise preference when candidates answer the same setting request.

Include ordinary climbs with no explicit feet and relevant style-specific examples. “Did not send” alone is not an impossibility label: ability, attempts, fatigue, and beta matter.

Before the larger study, choose the primary outcome and minimum useful improvement. Use pilot variability to plan sample size. A preference advantage should have an interval that supports the claimed improvement; 36 climbs alone should not support a broad success claim.

### Turn the website into a feedback instrument

Add a local-first **Tried this climb** form and export. Store original generation, final edited climb, all model/rules/schema hashes, request conditions, seed/temperature, candidate-pool size, ranking policy, session identifier, and the observations above. Separate generated target grade, predicted grade, and user-observed grade.

Keep explicit feedback distinct from ordinary browsing or cancellation. Retain ratings for rejected candidates where possible; recording only the chosen candidate creates selection bias. An exported feedback file is sufficient initially; a backend is not required to begin collecting reliable data. Any later account synchronization should be an explicit product choice.

### Evaluate candidate generation and reranking

The current system generally displays one stochastic result. Test generating a modest pool, for example 8 candidates, then selecting a diverse shortlist of 3. Compare pool sizes against latency on target devices before committing to browser-side cost.

Apply, in order:

1. Exact schema/locked-hold and board-availability checks.
2. Optional, explicitly requested constraints such as hold budget, excluded holds, active crop, or campus style.
3. A calibrated grade screen and a learned quality/preference score once labeled data exists.
4. A diversity-aware shortlist using similarity to both other candidates and catalogue neighbors.

Use likelihood as a modeling diagnostic or plausibility feature, not the sole quality ranker. Raw total likelihood can favor short/conventional solutions. Maximizing novelty alone can reward nonsense. A grade critic derived from the same generator can be useful operationally but provides circular evidence if used to declare grade fidelity; retain independent and human checks.

Include strong simple setting baselines: retrieve an existing training climb matching the request, and retrieve then make a small hold substitution using co-occurrence or geometry. Track their limited novelty explicitly. If they are preferred to generated climbs, they remain useful product options and expose what the generator still needs to learn. Keep candidate count and selection assistance comparable when judging competing generators.

Log acceptance rate, rejected cases, and total latency. Always evaluate the full displayed pipeline: a strong selected sample can hide an expensive or unreliable proposal distribution.

### Add reach and movement checks gradually

A useful first geometric feature is a graph of hand/foot candidates using physical distances, relative orientation, wall angle, and grip labels. Use metric coordinates: normalized x and y span different physical lengths (144 versus 156 source units), so treating normalized Euclidean distance as centimetres is incorrect. Model the kickboard separately when physical reach/contact geometry depends on it.

Use graph connectivity, large hand separations, poor foot support, and finish access as **features or review flags**. A universal maximum reach cutoff will reject legitimate dynamic moves, traverses, and smaller/larger climbers differently. Height-sorted hands are not a verified path. A future stance/contact planner should allow multiple betas and be calibrated against actual attempts before it controls generation.

## 6. Modeling changes worth testing

### First ablation: make output hold preference depend on role

The tied output table is additive:

```text
u_i = learned_hold_i + geometry(features_i)
token(i, r) = u_i + role_r
logit(i, r | h) = h · u_i + h · role_r + bias(i, r)
```

This follows directly from [`token_table`](kilter/model.py#L91) and the final projection at line 116. At a fixed hidden state, the contextual preference between two holds is shared across roles, apart from static hold-role biases. The head cannot independently decide that a geometry is a particularly good hand candidate versus foot candidate in that same context.

This is a **proven expressivity restriction**, but its practical importance is a hypothesis. Canonical role ordering may make it less limiting because the hidden context changes between role phases.

Test one simple alternative at a time: a low-rank hold×role residual in the output embeddings, role-specific output projections, or an independent token output table. Keep shared input embeddings initially. Match or disclose parameter/compute changes and compare completion likelihood, calibrated difficulty, candidate diversity, and blind climb judgments. Do not begin with several simultaneous architecture changes.

For attribution, add a few matched controls around the current architecture: flat hold-role embeddings versus composed embeddings; hold identity versus geometry versus both; and scratch-only versus mixed completion training. Use the same filtered data, split, width/depth or disclosed parameter budget, optimizer schedule, and training-token budget. These comparisons can determine whether improvements come from geometry, parameter sharing, or task mixture; the historical 154k-parameter baseline cannot isolate them.

### Match the completion masks to real editing tasks

The existing mixture is good for the original starts-plus-feet interface. Broader setting assistance should explicitly train/evaluate:

- One selected start when a second start may still be needed.
- Fixed start and finish, with the middle missing.
- A missing local section or crux region.
- One removed hand or foot, for substitution ranking.
- An almost complete climb requiring one small edit.
- Feet-only edits with hands fixed, and hand-only edits with feet fixed.
- Full given climbs, with EOS as the correct continuation.

The “all starts” training mode carries an implicit signal that those are **all** starts; arbitrary user picks do not necessarily carry that meaning. Represent the distinction explicitly if the interface offers it. Likewise, forbidden holds, unavailable board areas, or exact hold budgets need supported request semantics and matching training/evaluation; they cannot be assumed from a generic given set.

Do not require exact reconstruction as the only completion metric. Multiple different completions can be good. Use held-out target likelihood/ranking for offline diagnostics and accepted viable edits for practical usefulness. Add stress cases with contradictory or physically awkward locked picks: preserve the picks, reject formally unsatisfiable requests clearly, and allow uncertainty about physical feasibility rather than promising every arbitrary selection can become a good climb.

### Test better whole-climb representations

The current causal backbone’s final state is trained to predict EOS, and a mean of causal states mixes different amounts of prefix context. Existing pooling works well enough to beat the bag baseline, but it need not be optimal for grading or retrieval.

Start with learned attention pooling over frozen states. Then test a bidirectional encoder or a prefix architecture where given holds attend to one another and the generated block remains causal. A set encoder plus autoregressive decoder is another reasonable comparison. Set attention provides an established approach to learning interactions without imposing a semantic order on inputs; its suitability here still needs an ablation. [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html).

Masked/denoising objectives provide useful precedents for representation learning and transfer. Apply them as controlled alternatives to the existing objective, rather than assuming language-model results carry over automatically. [BERT](https://arxiv.org/abs/1810.04805), [T5](https://arxiv.org/abs/1910.10683).

A full-board per-hold state representation—off/hand/foot/start/finish—is also a useful non-generative baseline and possible denoising formulation. It handles unavailable/forbidden holds naturally, but independent predictions alone would miss interactions and cardinality constraints. Compare it before adopting a much more complex discrete diffusion system.

### Use geometry where it can test a real hypothesis

Current absolute Fourier features encode placement, kickboard, and screw-on status. They do not describe the hold’s three-dimensional surface or angle-dependent grip affordance. Sensible ablations include relative x/y displacement in attention, physical units/panel geometry, and hold features interacting with wall angle.

The near-zero gain from hand labels on this board should not end the hold-geometry direction. Every familiar hold already has its own well-trained identity embedding. Labels/images/meshes should be tested on low-data holds and held-out boards, where that shortcut is unavailable. A recent primary example, PointGrade, uses scanned hold point clouds for MoonBoard grading; it motivates an experiment, not an assumed gain for this dataset. [PointGrade](https://arxiv.org/abs/2609.17770).

Do not use naive horizontal mirroring or rotation augmentation: real holds, grip directions, and their physical identities generally do not have that symmetry. An augmented climb must map to actual equivalent holds and preserve the relevant board geometry.

### Improve control before adding capacity

Grade currently enters as a normalized number with Fourier features; it is not an ordinal target distribution. Test an auxiliary calibrated ordinal grade head and a modest balanced grade-angle sampling scheme, reporting any loss on the natural catalogue distribution. Rare extremes need explicit uncertainty and support reporting; oversampling cannot manufacture missing examples.

The 50% null-grade recipe also permits an inexpensive inference experiment: use conditional and null-grade predictions for token-level guidance. For the same prefix and angle, combine **finite, unmasked** logits as:

```text
guided = null_logits + s * (grade_logits - null_logits)
```

Then apply legality masks, temperature, and sampling. Here `s=1` is ordinary conditional generation; larger values strengthen the conditional difference. Do not subtract already masked `-inf` values. Sweep a small range on validation and measure difficulty, diversity, validity, and roughly doubled inference cost. This follows the autoregressive guidance idea; it does not guarantee the requested grade or exact global conditional sampling. [Autoregressive classifier-free guidance](https://arxiv.org/abs/2306.17806).

Keep the deployed rule-loss model constrained. Illegal logits are intentionally untrained under that loss, so raw sampling/NLL from it are not evidence about its learned unconstrained validity. Compare models using the same legal support when evaluating the constrained pipeline.

The existing 6.6M-parameter, longer-training variant did not improve validation over the chosen smaller model. That result does not establish a universal size ceiling, but it makes evaluation, data, output-head, and control experiments more compelling than another width increase.

## 7. Make downstream usefulness a first-class benchmark

### Difficulty prediction

Preserve the existing bag-of-holds baselines, untrained-model control, validation tuning, null-grade inference, and UUID-clustered bootstrap. Extend with:

- A supervised-from-scratch encoder of comparable size and tuning budget.
- Frozen, pooling-head, partially fine-tuned, and fully fine-tuned pretrained variants.
- Pretraining with no grade at all, versus the current 50% grade-conditioned recipe.
- Label-efficiency curves at 1%, 5%, 10%, and 100% of downstream training UUIDs.
- Ordinal probabilities, calibration, interval coverage/width, and per-grade/angle/ascent/source error.
- Family, setter, and temporal partitions from Section 4.

The current backbone saw training grades during pretraining. Passing null at inference correctly prevents direct target-input leakage, but does not make the experiment entirely self-supervised. For a “few grade labels” claim, control grade information used in pretraining too; alternatively state that the experiment measures adaptation with few **additional** labels.

Use a grade distribution or interval in the product. A point estimate with RMSE around 1.3 grades is useful guidance, but presenting it as a precise difficulty label would conceal material error. Treat the consensus-only subset as a complementary benchmark, not a replacement for all-climb performance.

### Recommended task portfolio

| Task | How to start | Strong comparison / key metric | Missing data |
| --- | --- | --- | --- |
| Partial completion | Existing infilling with a fixed realistic mask suite | Same-budget baseline and target NLL; human viable-completion rate | Human judgments for alternative completions |
| Hold replacement / controlled editing | Remove one hold/region; rank alternatives | Frequency/co-occurrence and supervised set baselines; accepted edit rate and edit distance | Accepted/rejected edit records |
| Difficulty and uncertainty | Existing frozen grader plus ordinal/calibrated heads | Bag MLP and matched supervised encoder; error and calibration | Better disagreement/confidence labels |
| Movement/style retrieval | Export climb embeddings; retrieve neighbors | Hand Jaccard and simple hold-feature baselines; blinded pair relevance/nDCG | Human “similar movement” labels |
| Observable style control | Begin with pinch, traverse, campus | Tag-free generator and conditional ablations; held-out labels and human style adherence | Cleaner positive/negative/unknown annotations |
| Personal recommendation/success | Later, once attempt histories exist | Item/user baselines and temporal ranking; success calibration and satisfaction | Ability, attempts, success, preferences, optional morphology |
| Beta/contact prediction | Separate movement pipeline | Contact order/limb/timing labels with multiple acceptable betas | Real annotated videos/contacts |

Current nearest-neighbor hand-Jaccard retrieval is useful for duplicate detection. Keep it alongside learned retrieval, which should be judged on semantic relevance. Do not call hold overlap “movement similarity” until people validate that connection.

The browser catalogue intentionally includes all splits. That is fine for showing real neighbors in the product. Any retrieval-assisted benchmark or reranker must use a designated training-only reference corpus unless its evaluation explicitly permits transductive access. Hide nearest-neighbor suggestions during blind generation assessment.

### Interpret the hold probes correctly

The type/grip probes hold labels out from the probe, while pretraining has learned embeddings for all 476 physical holds. This is a useful test of unprovided attributes of familiar holds. It is not recognition of a new hold from its shape or transfer to a new layout.

Keep the position, usage, untrained, and shuffled-label controls. Add fold-correct preprocessing, repeated annotation by another climber, and disagreement reporting. Nine slopers cannot support very precise per-class claims. A low-data/unseen-hold feature-encoder experiment should be separate from the familiar-hold probe.

### Treat text-derived tags as weak labels

The repository already handles negation and several ambiguous terms thoughtfully. Still, absent text evidence means **unknown**, not a verified negative. The all-zero tag vector currently also serves as “tags off,” and the model learns the distribution of untagged climbs. Keep this limitation visible in product controls.

Add an observation mask or explicit unknown condition, train with tag dropout, and distinguish positive, negative, and unspecified preferences. Audit precision and recall on randomly sampled tagged **and untagged** climbs. Test tag combinations and contradictory requests, not only one-tag toggles. Measure grade/novelty/diversity alongside steering.

The reported pinch/traverse/campus effects are promising because the selected holds can reveal those properties. Static movement, bumping, tension, and similar beta-dependent tags require actual movement observations. A recognizer that fails to identify such tags cannot certify that the generator follows them.

### Provide a stable representation interface

Add a small documented API for `encode_climb`, `score_climb`, `complete_climb`, and `predict_grade_distribution`. Fix conventions for null grade, tag defaults, pooling, physical board identity, token roles, and schema version. Release a compact inference checkpoint with its config, vocabulary/features, hashes, and model card, without optimizer state. The deployed ONNX file is useful for generation but does not expose the hidden-state API used by downstream scripts; many referenced research checkpoints are currently only git-ignored run paths.

For broader board transfer, move toward candidate scoring from board-local geometry and hold attributes, with optional learned identity residuals. Supply board/setup identity, available-hold masks, and physical dimensions. Demonstrate held-out-board adaptation before making a general climbing-board claim. Additional boards are a substantial data/model project and should follow clear task evidence on the current one.

## 8. Engineering, deployment, and movement research

### Preserve the good deployment choices

Keep worker-based local inference, cancellation, ONNX/PyTorch parity, shared decoding fixtures, pinned board assets, and protocol tests. They make the project easy to try and reduce differences between a research script and the real interface. The served manifest records a maximum ONNX absolute error of approximately `6.2e-6`; this is a recorded export result, while browser parity was also exercised in this review.

Add PR checks for Python/data suites and at least one real-WASM browser completion/parity test. The existing deployment workflow only runs `npm test` before publishing. For new coverage, concentrate on the confirmed gaps: stale-run rejection, split disjointness, import provenance, cancelled auto-send, null/tag parity, and request preservation.

Use content-addressed/versioned model bundles or verify artifact pairing during packaging so a stale manifest, model, or catalogue cannot be combined silently. Recorded checksums are most useful when the build/loading path enforces them. Make model/schema/rules versions available in every feedback export.

Measure download size, cold start, p50/p95 generation time, cancellation latency, and peak memory on a typical phone and laptop. The autoregressive implementation recomputes full prefixes; benchmark cached token tables and KV caching only if those measurements show a bottleneck. A maximum 40-hold context and a small model may make simpler execution preferable. ONNX single-thread behavior is already documented for the current hosting setup.

Keep Snake/light-show code as separate UI features with their existing tests. They are not the current bottleneck for climb quality; new research engineering effort should prioritize feedback, evaluation, and editing.

### Wireframes need real-world validation before model integration

The synthetic pipeline validates registration/lifting under its simulated assumptions. Its noisy pose inputs are derived from ground truth; the synthetic test does not exercise full MediaPipe detection on a real climber. The centimetre errors in [`wireframe/README.md`](wireframe/README.md) should retain that scope.

Next steps, in order:

1. Collect permissioned real board videos across several views, lighting conditions, and climbers. Annotate a small evaluation set of 2D joints and hand/foot contacts first.
2. Evaluate registration success/reprojection error, joint visibility, contact identity, limb assignment, and contact timing separately. Use abstention when observations are ambiguous.
3. Add temporal persistence and bone-length constraints. Per-frame proximity to a lit hold can mistake passing limbs for contacts.
4. Save camera/homography per frame when cameras move; current extraction/tables reuse a top-level camera in ways unsuitable for moving-camera analysis.
5. Pin the pose artifact/hash rather than relying on `latest` with a warning. Represent the kickboard’s separate plane when needed.
6. Sample contact transitions densely enough for fast movements; a fixed 0.25-second interval can miss them.

Reliable contacts could support route-use statistics, unused-hold detection, beta alternatives, and movement retrieval before accurate full-body 3D reconstruction exists. Keep simulated and observed labels visibly distinct. Avoid using the current synthetic wireframes as ground-truth evidence for generated-climb quality.

## 9. Concrete experiment and implementation sequence

Run small validation experiments first, then repeat promising changes over three seeds. Use the sealed test only at the agreed milestones. Record parameter count, training tokens, compute, inference cost, prompt set, and evaluation version with every comparison.

| Experiment | One primary change | Main decision criterion |
| --- | --- | --- |
| E0: corrected reference | Current served recipe with corrected/versioned evaluation and splits | Reproducible reference; exact rule/locked-hold checks; documented proxy failures |
| E1: candidate pipeline | Single result versus 8 proposals → 3 diverse candidates | Blinded usefulness/preference gain at acceptable latency; report all rejection costs |
| E2: editable masks | Add local removal, substitution, nearly/full-given cases | Better accepted completions/edits without scratch regression |
| E3: role-dependent head | Add one small output interaction mechanism | Better same-support likelihood and practical candidate quality; budget disclosed |
| E4: whole-climb pooling | Learned pooling, then a bidirectional/set comparison if warranted | Grade/retrieval transfer versus matched supervised and frozen controls |
| E5: grade control | Guidance sweep or auxiliary ordinal head, separately | Better blind grade fidelity without material diversity/quality loss |
| E6: weak-style interface | Unknown-aware tag conditioning through Python/ONNX/browser | Verified tag/null parity and human adherence for observable styles |
| E7: data confidence | UUID balancing or source/consensus treatment, separately | Robustness/calibration gains across clean and natural-distribution benchmarks |
| E8: new board/hold transfer | Feature-based candidates and held-out setup | Better low-data adaptation than supervised-from-scratch controls |

### Milestone A: trustworthy measurements

Deliver corrected metric names/reference sets, run/evaluation identity checks, explicit development/final-test commands, a served-model config, provenance fixes, cancellation fix, and versioned split/prompt manifests. Add the focused regression checks needed for those changes.

Acceptance: the same requested experiment cannot silently reuse a different config/evaluator; training novelty refers to actual training rows; exact families never cross the new split; exports preserve source identity; cancelled generations do not auto-send; existing completion and parity checks still pass. Historical results remain labeled with their original evaluator version.

### Milestone B: actual climbing evidence

Deliver the feedback schema/form/export, blind pilot, corrected scorecard, and first candidate-pipeline comparison. Use the pilot to choose the larger evaluation’s sample size and main outcome.

Acceptance: every rating can be traced to the exact displayed climb and generation pipeline; attempted results are distinguishable from visual judgments; grade and enjoyment have direct observations; failures and edits are inspectable. Report an inconclusive result as inconclusive.

### Milestone C: useful shared representations

Deliver reproducible grading, editing, and retrieval benchmarks; matched supervised controls; fine-tuning/frozen comparisons; label-efficiency curves; and a reusable checkpoint/API.

Acceptance: improvements survive at least one harder partition and are reported with uncertainty and adaptation budgets. A claimed label-efficiency gain accounts for grade information used in pretraining. A task-specific model that wins should remain an available product option.

### Milestone D: broader climbing model

Only after the earlier results, invest in additional boards, feature-based unseen-hold representations, personal histories, and real contact/beta datasets. Gate each expansion on a concrete downstream task rather than general scale.

### Work to defer

- Large capacity sweeps without a new data/evaluation hypothesis.
- Reinforcement learning or preference optimization before reliable human preference labels and a validated reward exist.
- A full physics simulator or precise 3D movement model as a prerequisite for a useful setting assistant.
- Discrete diffusion as a default replacement before controlled set/encoder baselines.
- Height conditioning without height-linked attempt/quality observations.
- Claims that nearest-neighbor novelty, low NLL, a self-grader, or synthetic pose accuracy proves good climbing.

## 10. Verification and limits of this review

I inspected the Python data/model/training/sampling/export code, baseline, experiment scripts and committed results, browser inference/decoding/import/export/board integration, tests, documentation/configuration, deployment workflow, and wireframe prototype. Corpus overlap/support figures were measured from the committed encoded arrays; the larger model-quality/grade/probe tables were read from committed experiment records.

Checks run during this review:

| Check | Result |
| --- | --- |
| `.venv/bin/python -m unittest discover -s tests` | 33 total; 29 passed, 4 optional wireframe tests skipped because the OpenCV dependency is absent |
| `.venv/bin/python -m unittest discover -s data -p 'test_*.py'` | 16 passed |
| `cd web && npm test` | Build and 9 Node test-file subtests passed |
| `cd web && npm run test:browser` | All 14 Playwright tests passed, including real model/WASM parity, completion, cancellation, imports, rendering, and mocked Bluetooth |
| Tagged model with current two-column exporter-shaped input | Shape mismatch reproduced in memory |
| Exact/family-related overlap and corpus support | Measured as reported above |

Browser testing initially hit the sandbox’s localhost-bind restriction; rerunning with local-server permission succeeded. Tests exercise a mocked board, not physical Bluetooth hardware. No new full training sweep, human climbing study, real-video benchmark, or direct physical-board test was performed. Passing software checks establishes that tested paths work; it does not establish climb quality.

The highest-value next result would be a clean, blinded demonstration that the setting assistant produces climbs people prefer and can use with fewer edits, followed by evidence that the same backbone helps grading, editing, and retrieval on harder held-out groups. This repository already has enough working infrastructure to measure both.
