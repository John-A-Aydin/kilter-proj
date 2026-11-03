"""Downstream task: predict a climb's V grade from its holds and angle, scored by MSE.

Trained model, against baselines that see the same climb and angle:

- ``mean``: the training mean grade at that angle.
- ``bag``: which hold is in which role, as a multi-hot vector, with the same heads as the model.
- ``untrained``: the same heads on an untrained copy of the model.
- ``model``: the frozen model, always read with a null grade so the label cannot leak in.
- ``bayes``: no new training. p(climb | grade, angle) from the model, with a prior per angle.

Heads are a ridge regression ("linear") and a small MLP. The ridge penalty, MLP epoch and Bayes
temperature are chosen on validation, and the test split is only read to report.

    .venv/bin/python experiments/grade_climbs.py
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hold_embeddings import reset  # noqa: E402
from kilter.data import PAD, ClimbDataset, Collator, load_arrays, prepare  # noqa: E402
from kilter.experiment import load_run  # noqa: E402
from kilter.model import NULL_GRADE  # noqa: E402
from kilter.sampling import LegalTokens  # noqa: E402

MODEL = "null_rule_16"
SEEDS = (42, 43, 44)
TRAIN, VALIDATION, TEST = 0, 1, 2
GRADES = 17
MIN_ASCENTS = 5
PENALTIES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
BOOTSTRAPS = 2000


def rows_of(arrays, split):
    return np.flatnonzero(arrays["splits"] == split)


def batches(arrays, split, size=1024):
    dataset = ClimbDataset(arrays, split)
    return torch.utils.data.DataLoader(dataset, batch_size=size, collate_fn=Collator(mode="scratch"))


def bag_of_holds(arrays, split, vocab):
    """Multi-hot token vector plus angle, as (rows, vocab - 3) floats on demand."""
    rows = rows_of(arrays, split)
    offsets, tokens = arrays["offsets"], arrays["tokens"]
    bag = torch.zeros(len(rows), vocab - 4, dtype=torch.bool)
    for i, row in enumerate(rows):
        bag[i, tokens[offsets[row]:offsets[row + 1]] - 4] = True
    return bag, torch.from_numpy(arrays["conditions"][rows, 1]).unsqueeze(1)


@torch.no_grad()
def representations(model, arrays, split, device):
    """Mean and final hidden state of every climb, read with a null grade."""
    states = []
    hook = model.norm.register_forward_hook(lambda _, __, output: states.append(output))
    pooled = []
    for x, _, conditions, _ in batches(arrays, split):
        conditions[:, 0] = NULL_GRADE
        x, conditions = x.to(device), conditions.to(device)
        model(x, conditions)
        hidden, mask = states.pop(), x != PAD
        mean = (hidden * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)
        last = hidden[torch.arange(len(x)), mask.sum(1) - 1]
        pooled.append(torch.cat([mean, last, conditions[:, 1:]], dim=1).cpu())
    hook.remove()
    return torch.cat(pooled)


def fit_ridge(train, target, validation, target_validation, test, device, chunk=8192):
    """Closed-form ridge on standardized inputs, with the penalty chosen on validation."""
    mean = train.float().mean(0).to(device)
    deviation = train.float().std(0).clamp(min=1e-3).to(device)
    prepared = lambda x: (x.float().to(device) - mean) / deviation
    width = train.shape[1]
    gram, moment = torch.zeros(width, width, device=device, dtype=torch.float64), torch.zeros(width, device=device, dtype=torch.float64)
    y_mean = target.mean().item()
    for start in range(0, len(train), chunk):
        x = prepared(train[start:start + chunk]).double()
        gram += x.T @ x
        moment += x.T @ (target[start:start + chunk].to(device).double() - y_mean)
    predict = lambda x, w: torch.cat([prepared(x[s:s + chunk]).double() @ w for s in range(0, len(x), chunk)]).cpu().numpy() + y_mean
    best = (np.inf, None, None)
    for penalty in PENALTIES:
        weights = torch.linalg.solve(gram + penalty * torch.eye(width, device=device, dtype=torch.float64), moment)
        score = float(np.mean((predict(validation, weights) - target_validation.numpy()) ** 2))
        if score < best[0]:
            best = (score, penalty, weights)
    return predict(test, best[2]), {"penalty": best[1], "validation_mse": best[0]}


def fit_mlp(train, target, validation, target_validation, test, device, seed, epochs=15, batch=1024):
    """The same small head for every input, keeping the epoch with the best validation MSE."""
    torch.manual_seed(seed)
    mean, deviation = train.float().mean(0).to(device), train.float().std(0).clamp(min=1e-3).to(device)
    prepared = lambda x: (x.float().to(device) - mean) / deviation
    head = nn.Sequential(nn.Linear(train.shape[1], 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, 256), nn.GELU(),
                         nn.Linear(256, 1)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * (len(train) // batch + 1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    y_mean, y_std = target.mean().item(), target.std().item()
    generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def predict(x):
        head.eval()
        return torch.cat([head(prepared(x[s:s + 8192])).squeeze(1) for s in range(0, len(x), 8192)]).cpu().numpy() * y_std + y_mean

    best = (np.inf, 0, None)
    for epoch in range(1, epochs + 1):
        head.train()
        for index in torch.randperm(len(train), generator=generator).split(batch):
            loss = F.mse_loss(head(prepared(train[index])).squeeze(1), (target[index].to(device) - y_mean) / y_std)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        score = float(np.mean((predict(validation) - target_validation.numpy()) ** 2))
        if score < best[0]:
            best = (score, epoch, predict(test))
    return best[2], {"epoch": best[1], "validation_mse": best[0]}


@torch.no_grad()
def grade_likelihoods(model, legal, arrays, split, device):
    """log p(climb | grade, angle) with the decoding rules, shape (rows, grades)."""
    rows = []
    for x, y, conditions, _ in batches(arrays, split, 256):
        x, y, conditions = x.to(device), y.to(device), conditions.to(device)
        allowed, mask = legal(x), y != PAD
        scores = []
        for grade in range(GRADES):
            logits = model(x, torch.stack([torch.full_like(conditions[:, 1], grade), conditions[:, 1]], 1))
            log_probs = logits.masked_fill(~allowed, -torch.inf).log_softmax(-1)
            picked = log_probs.gather(2, y.clamp(min=0).unsqueeze(2)).squeeze(2)
            scores.append(picked.masked_fill(~mask, 0.0).sum(1))
        rows.append(torch.stack(scores, 1).cpu())
    return torch.cat(rows).numpy()


def angle_prior(arrays):
    """log p(grade | angle) from training rows, with one added count per cell."""
    train = rows_of(arrays, TRAIN)
    grades, angles = arrays["conditions"][train, 0].astype(int), np.round(arrays["conditions"][train, 1] * 90).astype(int)
    counts = np.ones((91, GRADES))
    np.add.at(counts, (angles, grades), 1)
    return np.log(counts / counts.sum(1, keepdims=True))


def posterior_mean(likelihood, prior, angles, temperature):
    log_posterior = likelihood / temperature + prior[angles]
    log_posterior -= log_posterior.max(1, keepdims=True)
    posterior = np.exp(log_posterior)
    return (posterior / posterior.sum(1, keepdims=True)) @ np.arange(GRADES)


def score(predicted, truth, ascents):
    consensus = ascents >= MIN_ASCENTS
    mse = lambda keep: float(np.mean((predicted[keep] - truth[keep]) ** 2))
    variance = lambda keep: float(np.var(truth[keep]))
    everything = np.ones(len(truth), dtype=bool)
    return {"mse": mse(everything), "rmse": mse(everything) ** 0.5, "r2": 1 - mse(everything) / variance(everything),
            "within_one_grade": float(np.mean(np.abs(predicted - truth) <= 1)),
            "bias": float(np.mean(predicted - truth)),
            f"mse_{MIN_ASCENTS}_plus_ascents": mse(consensus), f"r2_{MIN_ASCENTS}_plus_ascents": 1 - mse(consensus) / variance(consensus)}


def bootstrap_gap(first, second, truth, climbs, seed=0):
    """95% interval of MSE(first) - MSE(second), resampling climbs so a climb's angles stay together."""
    _, groups = np.unique(climbs, return_inverse=True)
    n = groups.max() + 1
    gap = (first - truth) ** 2 - (second - truth) ** 2
    totals, sizes = np.bincount(groups, gap, n), np.bincount(groups, minlength=n)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), BOOTSTRAPS)
    values = (weights @ totals) / (weights @ sizes)
    return {"mean": float(gap.mean()), "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grade_climbs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device
    checkpoints = json.loads((ROOT / "results" / "compare_models" / "checkpoints.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    first = ROOT / checkpoints[f"{MODEL}_s{SEEDS[0]}"]["run"] / "best.pt"
    model, board, config, _ = load_run(first)
    arrays = load_arrays(prepare(config.data)[0])
    y = {split: torch.from_numpy(arrays["conditions"][rows_of(arrays, split), 0]) for split in (TRAIN, VALIDATION, TEST)}
    test_rows = rows_of(arrays, TEST)
    truth, ascents, climbs = y[TEST].numpy(), arrays["ascents"][test_rows], arrays["uuids"][test_rows]
    angles = {split: np.round(arrays["conditions"][rows_of(arrays, split), 1] * 90).astype(int)
              for split in (VALIDATION, TEST)}
    train_angles = np.round(arrays["conditions"][rows_of(arrays, TRAIN), 1] * 90).astype(int)
    by_angle = np.array([y[TRAIN].numpy()[train_angles == a].mean() if (train_angles == a).any() else y[TRAIN].mean().item()
                         for a in range(91)])
    bag, bag_angle = zip(*(bag_of_holds(arrays, split, board.vocab_size) for split in (TRAIN, VALIDATION, TEST)))
    prior = angle_prior(arrays)

    scores, choices, predictions = {}, {}, {}

    def record(name, seed, predicted, choice=None):
        scores.setdefault(name, []).append(score(predicted, truth, ascents))
        predictions.setdefault(name, []).append(predicted)
        if choice:
            choices.setdefault(name, []).append(choice)

    def heads(name, x, seed):
        record(f"{name}_linear", seed, *fit_ridge(x[0], y[TRAIN], x[1], y[VALIDATION], x[2], device))
        record(f"{name}_mlp", seed, *fit_mlp(x[0], y[TRAIN], x[1], y[VALIDATION], x[2], device, seed))

    joined = [torch.cat([b.to(torch.uint8), (a * 255).to(torch.uint8)], dim=1) for b, a in zip(bag, bag_angle)]
    record("mean_at_angle", 0, by_angle[angles[TEST]])
    for seed in SEEDS:
        checkpoint = ROOT / checkpoints[f"{MODEL}_s{seed}"]["run"] / "best.pt"
        heads("bag", joined, seed)
        model, board, config, _ = load_run(checkpoint, device)
        legal = LegalTokens(board, config.data.max_holds, device)
        heads("model", [representations(model, arrays, split, device) for split in (TRAIN, VALIDATION, TEST)], seed)
        validation_likelihood = grade_likelihoods(model, legal, arrays, VALIDATION, device)
        temperature = min(TEMPERATURES, key=lambda t: np.mean(
            (posterior_mean(validation_likelihood, prior, angles[VALIDATION], t) - y[VALIDATION].numpy()) ** 2))
        test_likelihood = grade_likelihoods(model, legal, arrays, TEST, device)
        record("bayes", seed, posterior_mean(test_likelihood, prior, angles[TEST], temperature), {"temperature": temperature})
        reset(model, seed)
        heads("untrained", [representations(model, arrays, split, device) for split in (TRAIN, VALIDATION, TEST)], seed)
        print(json.dumps({"seed": seed, **{n: v[-1]["mse"] for n, v in scores.items()}}), flush=True)

    summary = {name: {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                               "std": float(np.std([r[metric] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
                      for metric in rows[0]} for name, rows in scores.items()}
    averaged = {name: np.mean(values, axis=0) for name, values in predictions.items()}
    baseline = "bag_mlp"
    gaps = {name: bootstrap_gap(averaged[name], averaged[baseline], truth, climbs)
            for name in ("model_mlp", "model_linear", "bayes", "untrained_mlp")}
    per_seed_gap = {name: [float(np.mean((predictions[name][i] - truth) ** 2) - np.mean((predictions[baseline][i] - truth) ** 2))
                           for i in range(len(SEEDS))] for name in ("model_mlp", "model_linear", "bayes", "untrained_mlp")}
    (args.output / "summary.json").write_text(json.dumps({
        "model": MODEL, "seeds": SEEDS, "test_rows": len(truth), "test_grade_variance": float(np.var(truth)),
        "methods": summary, "per_seed": scores, "chosen_on_validation": choices,
        "mse_gap_to_bag_mlp_seed_averaged_predictions": gaps, "mse_gap_to_bag_mlp_per_seed": per_seed_gap}, indent=2) + "\n")
    with (args.output / "by_grade.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "grade", "rows", "mse", "mean_predicted"])
        for name, values in predictions.items():
            for grade in range(GRADES):
                keep = truth == grade
                if keep.any():
                    errors = np.mean([np.mean((v[keep] - truth[keep]) ** 2) for v in values])
                    writer.writerow([name, grade, int(keep.sum()), errors, float(np.mean([v[keep].mean() for v in values]))])
    print(json.dumps({name: values["mse"] for name, values in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
