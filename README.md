# Kilter climb generation

A transformer that generates climbs for the Kilter Board (Original, 12×12), conditioned
on a numeric V grade and the wall angle. It can generate a climb **from nothing**, or
**complete one from given holds**, such as your start holds plus some feet. A
[web interface](web/README.md) runs the model in the browser and can light the climb up
on a real board over Bluetooth.

On held-out test climbs, the large model scores **45.31 nats per climb** from scratch,
against **50.45** for the [baseline](docs/baseline.md), with no decoding rules (means
over three seeds). Completing a real climb's starts plus every other foot gives a
structurally valid climb 98% of the time, against 11% for the baseline. See
[docs/model.md](docs/model.md) for the model and [experiments/](experiments/README.md)
for every result, including the `null_rule_16` model the website serves.

## Repository layout

| Path | Contents |
| --- | --- |
| [`data/`](data/README.md) | Download, convert, and validate the Kilter catalogue |
| [`kilter/`](kilter/) | The climb transformer: data encoding, model, training, sampling, evaluation, web export |
| [`kilter/baseline/`](kilter/baseline/) | The original transformer, kept as the comparison baseline |
| [`configs/`](configs/) | Training configs (`climb.toml`, `climb_null_grade.toml`, `baseline.toml`) |
| [`experiments/`](experiments/README.md) | One script per experiment |
| [`results/`](results/) | Each experiment's committed outputs, in `results/<script name>/` |
| [`docs/`](docs/) | Write-ups of the [model](docs/model.md) and the [baseline](docs/baseline.md) |
| [`tests/`](tests/) | Python tests for both models |
| [`web/`](web/README.md) | The browser interface |

Generated files stay out of Git: the raw and processed data under `data/`, encoded
datasets under `data/cache/`, and training runs under `runs/`.

## Quick start

From the repository root, using Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
# Download and prepare the catalogue into data/processed/; see data/README.md.

# Train the default model, then generate from it.
uv run --frozen python -m kilter train
uv run --frozen python -m kilter sample runs/<run>/best.pt --grade 5 --angle 40 --count 3
uv run --frozen python -m kilter sample runs/<run>/best.pt --grade 5 --angle 40 \
  --given 1176:start --given 1192:start --given 1091:foot
```

The lockfile installs CPU PyTorch, so a GPU is not required. With the existing
environment, `.venv/bin/python` can replace `uv run --frozen python`. The default device
is CUDA when available, otherwise CPU; installing a CUDA build of PyTorch is required
to use an NVIDIA GPU. Each training run gets its own timestamped folder under `runs/`.
[docs/model.md](docs/model.md#commands) lists every command.

To run the web interface with the committed model:

```bash
cd web && npm ci && npm run build && npm start
```

## Tests

```bash
uv run --frozen python -m unittest discover -s tests
uv run --frozen python -m unittest discover -s data -p 'test_*.py'
cd web && npm test && npx playwright test
```
