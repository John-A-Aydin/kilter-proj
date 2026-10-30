# Finishing the v2 training runs

Two v2 runs were started on 2026-10-29 at about 15:37 local time. Both are 8
epochs on CPU, 4 threads each. Run every command from the repository root
(`/home/alex/School/kilter-proj`).

| Run | Model | Directory | Log | Approx. time per epoch |
| --- | --- | --- | --- | --- |
| default | width 128, 4 layers, 0.9M params | `runs/20261029T193736564212Z_climb_v2/` | `runs/logs/v2_d128.log` | 5 min alone, 9 min alongside the other |
| large | width 192, 6 layers, 2.9M params | `runs/20261029T193821590530Z_climb_v2_d192/` | `runs/logs/v2_d192.log` | 13 min alone, 21 min alongside the other |

## 1. Check whether they are still running

```bash
ps aux | grep "kilter_v2 train" | grep -v grep
tail -n 2 runs/logs/v2_d128.log runs/logs/v2_d192.log
```

A run is finished when its directory contains `summary.json`. If both have one,
skip to step 3.

## 2. If a run was stopped before finishing, resume it

Resume continues from the last completed epoch (`last.pt`), using the run's saved
config. It is fine to run both at once on this 8-core machine.

```bash
nohup .venv/bin/python -m kilter_v2 train --resume runs/20261029T193736564212Z_climb_v2/last.pt \
  >> runs/logs/v2_d128.log 2>&1 &
nohup .venv/bin/python -m kilter_v2 train --resume runs/20261029T193821590530Z_climb_v2_d192/last.pt \
  >> runs/logs/v2_d192.log 2>&1 &
```

If a run directory has no `last.pt` (stopped during its first epoch), delete that
directory and start the run again from scratch:

```bash
# Default model
nohup .venv/bin/python -m kilter_v2 train > runs/logs/v2_d128.log 2>&1 &
# Large model
nohup .venv/bin/python -m kilter_v2 train --set name=climb_v2_d192 --set model.d_model=192 \
  --set model.n_layers=6 --set model.n_heads=6 > runs/logs/v2_d192.log 2>&1 &
```

A new run gets a new timestamped directory, so update the paths in
[HANDOFF.md](HANDOFF.md) to match.

To stop a run early: `pkill -f "kilter_v2 train"`. It loses at most the current epoch.

## 3. When both have `summary.json`

Give [HANDOFF.md](HANDOFF.md) to the agent that wraps up. It evaluates both runs,
picks the better one, puts it on the website, and writes up the results.
