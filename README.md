# Kilter project

Experiments train an autoregressive transformer to generate hold/role sequences
conditioned on **numeric V grade** and **wall angle divided by 90**. The baseline
uses ordinary next-token cross entropy and automatically sizes the model to
roughly **0.5 trainable parameters per training climb–angle row**.

The [data pipeline](data/README.md) downloads and converts the Kilter Original
12×12 catalogue across all recorded ascent angles. Each climb–angle pair has its
own row with grade, ascensionist count, angle, and other metadata. Hold entries
include normalized coordinates, usage types, kickboard flags, and stable tokens.

See [data/README.md](data/README.md) for reproduction commands, schemas,
source dates, grade/ascent provenance, and validation.

## v2 climb transformer

[`kilter_v2/`](kilter_v2/README.md) is a rebuilt model that generates a climb from
nothing, or completes one from given holds such as your start holds plus some feet.
It trains on `BOS <given> SEP <rest> EOS` in a canonical start/hand/finish/foot
order, uses hold geometry in its embeddings, and is larger (0.9M and 2.9M
parameters). On the same held-out climbs, the chosen 2.9M run scores **45.34 nats
per climb** on test, against **50.54** for this baseline. Completing a real climb's
starts plus every other foot gives a structurally valid climb 98% of the time,
against 16% for the baseline's barrier decoding. The web interface uses v2, now the 2.9M `null_rule_16` model (37.01 nats per climb across
the three modes with decoding rules applied; see the v2 README). The
baseline below is unchanged. See the [v2 README](kilter_v2/README.md) for the
format, commands, full results, and caveats.

```bash
.venv/bin/python -m kilter_v2 train
.venv/bin/python -m kilter_v2 sample runs/<run>/best.pt --grade 5 --angle 40 --given 1176:start --given 1192:start
```

## Interactive climb completion

The [web interface](web/README.md) lets you select holds and roles on the board,
set a V grade and angle, and watch the trained model complete a climb while
preserving your picks. It uses Boardsesh's Kilter Original artwork, traced hold
shapes, and Aura renderer, and runs the checkpoint directly in the browser using
ONNX Runtime Web. The web README
has setup commands, checkpoint export instructions, and decoding details.

## Run experiments

From the repository root, using Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
# Download/prepare the catalogue first if data/processed is absent; see data/README.md.
uv run --frozen python -m kilter prepare
uv run --frozen python -m kilter train --config configs/ar_transformer.toml
```

The lockfile installs CPU PyTorch so a GPU is not required. With the existing
environment, `.venv/bin/python` can replace `uv run --frozen python`. `python main.py`
also exposes the same CLI. The default device is CUDA when available, otherwise
CPU; installing a CUDA build of PyTorch is required to use an NVIDIA GPU.

All settings live in [configs/ar_transformer.toml](configs/ar_transformer.toml).
Copy that file for a new experiment, or use repeated `--set section.key=value`
overrides. Unknown keys fail instead of silently running a different experiment.
For example, compare positional embeddings with the same architecture, split,
and training seed:

```bash
uv run --frozen python -m kilter train --set name=without_positions --set model.positional_embeddings=false
uv run --frozen python -m kilter compare runs > runs/comparison.csv
```

Auto sizing always uses the learned-position baseline as its reference, so this
switch keeps width, depth, and heads fixed. Turning positions off removes only
their parameters. Set `model.d_model` to an explicit multiple of `model.n_heads`
to fix width across other comparisons. For seed sweeps:

```bash
for seed in 42 43 44; do
  uv run --frozen python -m kilter train --set name=seed_$seed --set train.seed=$seed
done
```

Keep `data.split_seed` fixed when comparing training seeds. A quick end-to-end
check can use `--set name=smoke --set train.epochs=1 --set train.max_train_batches=4
--set train.max_eval_batches=2`. Bounded runs are explicitly marked as partial
in metrics and summaries; their parameter budget still uses the full training
split. Each command creates a separate timestamped run without overwriting
previous experiments.

## Baseline and representation

The baseline is a decoder transformer with two pre-LayerNorm blocks, four heads,
GELU feed-forward layers of width `4 × d_model`, learned absolute positional
embeddings, dropout 0.1, a final LayerNorm, and tied input/output token embeddings.
For the current processed dataset, automatic width is **48**, giving **154,512
trainable parameters** against a target of **152,350** from **304,699 training
rows** (0.507 parameters per row). All trainable parameters, including embeddings,
normalizations, biases, and the conditioning projection, count toward the budget;
tied weights count once.

The [first completed baseline](experiments/ar_transformer_baseline.json) trained
for ten epochs on CPU. Epoch 9 gave the best validation cross entropy, **3.725**.
Its held-out test cross entropy is **3.745**, perplexity **42.32**, and token
accuracy **17.12%** across all 16,926 test rows. The checkpoint and three V5/40°
samples are saved under `runs/20261029T174149845596Z_ar_transformer/`.

The training defaults are AdamW, learning rate `3e-4`, betas `(0.9, 0.999)`,
weight decay `0.01`, batch size 256, ten epochs, 5% linear warmup followed by
cosine decay to 10% of the initial learning rate, gradient clipping at 1.0, and
early stopping after three epochs without validation improvement. Training uses
float32. Randomized length buckets reduce padding (`train.length_bucketing=true`);
every training row is still seen once per epoch. The vocabulary projection and
cross entropy run only on non-padding targets. The best checkpoint is selected by validation cross entropy. Loss and
accuracy aggregate by the number of non-padding target tokens, including EOS.
The implementation uses PyTorch's [causal scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)
and [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

Each of the 476 physical holds has four role-specific tokens (hand, foot, start,
finish), plus `PAD`, `BOS`, `EOS`, and `FRAME`: **1,908 vocabulary entries**.
A sequence is `BOS hold/role … FRAME hold/role … EOS`. Within each frame, holds
are sorted by their stable physical token. This is a deterministic serialization
of a climb's holds, **not a recorded movement sequence**. Multi-frame climbs keep
frame boundaries. Coordinates and kickboard identity can be recovered from
`data/processed/hold_vocabulary.csv`; they are not additional model inputs.

The model projects `[v_grade, angle_degrees / 90]` through a learned `Linear(2,
d_model)` layer and adds the result to every token embedding. `V5` becomes the
number `5.0`, **not** the native Font grade index or a categorical embedding.
0°, 45°, and 90° become 0.0, 0.5, and 1.0 respectively. The source currently
contains angles from 0° through 70°, so higher-angle samples are extrapolations.

Inputs and targets are shifted by one token. A causal mask prevents future-token
access; right-padding is ignored in loss and metrics. Sampled climbs are raw
model outputs, with no constraints enforcing unique holds, role consistency, or
physical climbability. `PAD` and `BOS` are excluded during sampling.

## Data splits and cache

Splits use a seeded SHA-256 hash of climb UUID with 90%/5%/5% thresholds. Every
angle of a UUID goes into the same split, regardless of file order. Different
UUIDs with identical holds are not deduplicated. After the baseline filters:

| Split | Climb–angle rows | Distinct climbs |
| --- | ---: | ---: |
| Train | 304,699 | 186,384 |
| Validation | 16,946 | 10,178 |
| Test | 16,926 | 10,355 |

Rows without a numeric V grade are excluded (27 in this snapshot). Computed
grades are included by default; disable `data.include_computed_grades` to exclude
them. Sequences needing more than `data.max_seq_len` input/target pairs are
excluded whole, never truncated. With the default 128-token context this excludes
1,061 rows. Increase the context to include longer climbs; filtering counts and
the resulting parameter target are recalculated automatically.

Preprocessing streams JSONL and stores compressed ragged arrays, row IDs,
conditions, and split memberships under `data/experiments/`. Content hashes of
the entire source, vocabulary, preprocessing code, and preprocessing settings
identify the cache. The encoded cache checksum is verified before reuse.
Changing only model/training settings reuses the cache. All source exports stay
untouched; the cache and training artifacts are gitignored.

## Saved runs, resume, evaluate, and sample

Each `runs/<timestamp>_<name>/` contains:

| File | Contents |
| --- | --- |
| `config.json`, `model.json` | Effective settings, resolved architecture, actual and target parameter counts |
| `dataset.json` | Data fingerprint, tokenizer, conditions, exclusions, split counts, histograms, cache location |
| `environment.json`, `source/` | Software versions, device, Git state, copied source/configs/tests/lockfile and working-tree diff |
| `metrics.jsonl` | Epoch training loss, validation loss/perplexity/accuracy, learning rate, timing |
| `best.pt`, `last.pt` | Best-validation and latest completed-epoch checkpoints |
| `summary.json` | Completed-run results for `compare` |

Checkpoints include model, optimizer, scheduler, epoch/step, early stopping state,
and Python/NumPy/PyTorch/data-loader RNG states. Resume starts at the next epoch
after the last completed checkpoint; interrupted-epoch updates are replayed.
Exact configuration and data fingerprints are checked. Reproducibility is tested
on CPU; results can differ across hardware or PyTorch versions. Checkpoints use
PyTorch serialization and should only be loaded from trusted sources.

```bash
# Replace the path with the actual timestamped run directory.
uv run --frozen python -m kilter train --resume runs/<run>/last.pt

# Evaluate held-out test data after selecting your experiment on validation data.
uv run --frozen python -m kilter evaluate runs/<run>/best.pt --split test

# Numeric V grade and angle in degrees; normalization happens internally.
uv run --frozen python -m kilter sample runs/<run>/best.pt --grade 5 --angle 40 --count 3 --seed 42
```

Resume uses the saved config and rejects `--set` overrides. Start a new run to
change training settings. Test evaluation is explicit rather than automatic
after every training run, so experiment selection can use validation data alone.
Evaluation writes `<checkpoint>_<split>.json`; sampling writes JSON to stdout,
including whether generation stopped at EOS or reached the context limit.

## Verify

```bash
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m unittest discover -s data -p 'test_*.py' -v
```

The experiment suite covers causal isolation, both numeric conditions, shifted
targets/padding, role/frame tokenization, parameter sizing, tiny-batch overfitting,
UUID grouping, cache invalidation, sampling, evaluation, and bit-for-bit agreement
between interrupted/resumed and uninterrupted training.
