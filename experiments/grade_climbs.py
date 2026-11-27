"""Downstream task: predict a climb's V grade from its holds and angle, scored by RMSE and MAE.

Trained model, against baselines that see the same climb and angle:

- ``mean``: the training mean grade at that angle.
- ``bag``: which hold is in which role, as a multi-hot vector, with the same heads as the model.
- ``untrained``: the same heads on an untrained copy of the model.
- ``model``: the frozen model, always read with a null grade so the label cannot leak in.
- ``bayes``: no new training. p(climb | grade, angle) from the model, with a prior per angle.

Three graders are trained by their own scripts on the same rows and split, and read from
results/<script>/test_predictions.npz when that file exists (refused if it was made for different
test rows):

- ``bert``: a BERT-style encoder trained from scratch for grading (experiments/bert_grader.py).
- ``finetuned``: the generator fine-tuned end to end with a grade head (experiments/finetune_grader.py).
- ``lightgbm``: gradient-boosted trees on the bag of holds plus geometry features
  (experiments/lightgbm_grader.py).

Heads are a ridge regression ("linear") and a small MLP. The ridge penalty, MLP epoch and Bayes
temperature are chosen on validation MSE, and the test split is only read to report.

Every score is reported on four test subsets (``SUBSETS``): every row, and rows whose climb has
more than 5, 10 or 25 ascents at that angle, where the grade is a consensus of more climbers.
Scores with no suffix are the whole test set; ``_more_than_<n>_ascents`` marks a subset. RMSE and
MAE (mean absolute error) are in grades, and their mean over seeds is the mean of each seed's value.

Significance (``COMPARISONS``, in summary.json under ``significance``): for each pair of methods
and each subset, the difference in RMSE and in MAE, with

- a 95% interval from a climb-level bootstrap: climbs are resampled, so a climb's angles stay
  together, and the difference is recomputed as a mean over seeds, like the table;
- a two-sided p-value from a climb-level sign-flip permutation test. Under the null hypothesis the
  two methods' errors are exchangeable within a climb, so each climb's summed error difference is
  equally likely to have either sign. For MAE the statistic is the mean absolute-error difference;
  for RMSE it is the mean squared-error difference (equal RMSE means equal MSE);
- the p-values adjusted with Holm's method across every comparison in that subset and metric.

The test climbs are the only source of randomness these account for. The three training seeds
are averaged, not resampled, so ``per_seed`` lists each seed's difference to show whether they agree.

Outputs in results/grade_climbs/: summary.json (every score and test), by_grade.csv (RMSE and MAE
by true grade), and confusion.json: for each method and subset, test rows counted by true grade
(rows) and prediction rounded to the nearest grade within V0-V16 (columns), averaged over seeds.

    .venv/bin/python experiments/grade_climbs.py
    # Keep the test predictions, so the statistics can be recomputed without training again:
    .venv/bin/python experiments/grade_climbs.py --predictions /tmp/grade_predictions.npz
"""
import argparse
import csv
import hashlib
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
# Rows of climbs with more than each of these ascents at their angle form a consensus subset.
ASCENTS_OVER = (5, 10, 25)
# Subset name -> (score suffix, smallest ascent count it excludes, or None for every row).
SUBSETS = {"all": ("", None), **{f"more_than_{n}_ascents": (f"_more_than_{n}_ascents", n) for n in ASCENTS_OVER}}
PENALTIES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TEMPERATURES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
BOOTSTRAPS = 10000
PERMUTATIONS = 100000
MAIN_BASELINE = "bag_mlp"
# (method, reference) pairs tested in every subset: everything against the main baseline, and the
# best model against the other ways of reading the same model.
COMPARISONS = tuple((name, MAIN_BASELINE) for name in
                    ("model_mlp", "model_linear", "bayes", "untrained_mlp", "bag_linear", "mean_at_angle")) + (
    ("model_mlp", "bayes"), ("model_mlp", "model_linear"), ("model_mlp", "untrained_mlp"),
    ("bert", MAIN_BASELINE), ("finetuned", MAIN_BASELINE), ("lightgbm", MAIN_BASELINE),
    # What the frozen generator leaves behind, what freezing it costs, and what starting from it gives.
    ("model_mlp", "bert"), ("model_mlp", "finetuned"), ("finetuned", "bert"), ("model_mlp", "lightgbm"))
# Methods trained by their own scripts: name -> results folder.
EXTERNAL = {"bert": "bert_grader", "finetuned": "finetune_grader", "lightgbm": "lightgbm_grader"}


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


def subset_mask(name, ascents):
    over = SUBSETS[name][1]
    return np.ones(len(ascents), dtype=bool) if over is None else ascents > over


def score(predicted, truth, ascents):
    """RMSE, MAE, R², share within one grade, and bias, on each subset of the test rows."""
    scores = {}
    for name, (suffix, _) in SUBSETS.items():
        keep = subset_mask(name, ascents)
        error = predicted[keep] - truth[keep]
        mse = float(np.mean(error ** 2))
        scores |= {f"rmse{suffix}": mse ** 0.5, f"mae{suffix}": float(np.mean(np.abs(error))), f"r2{suffix}": 1 - mse / float(np.var(truth[keep])),
                   f"within_one_grade{suffix}": float(np.mean(np.abs(error) <= 1)), f"bias{suffix}": float(np.mean(error))}
    return scores


def confusion(predicted, truth):
    """Rows counted by true grade (rows) and rounded, clipped prediction (columns)."""
    rounded = np.clip(np.rint(predicted), 0, GRADES - 1).astype(int)
    matrix = np.zeros((GRADES, GRADES))
    np.add.at(matrix, (truth.astype(int), rounded), 1)
    return matrix


def rmse(predicted, truth):
    return float(np.sqrt(np.mean((predicted - truth) ** 2)))


def mae(predicted, truth):
    return float(np.mean(np.abs(predicted - truth)))


def holm(p_values):
    """Holm-adjusted p-values, in the order given."""
    order = np.argsort(p_values)
    adjusted, running = np.empty(len(p_values)), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def resampled_climbs(n, draws, seed, chunk=500):
    """Bootstrap weights: how many times each of ``n`` climbs is drawn, in chunks of (rows, n)."""
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk):
        rows = min(chunk, draws - start)
        picks = rng.integers(0, n, (rows, n)) + n * np.arange(rows)[:, None]
        yield np.bincount(picks.ravel(), minlength=rows * n).reshape(rows, n).astype(np.float64)


def sign_flips(n, draws, seed, chunk=500):
    rng = np.random.default_rng(seed)
    for start in range(0, draws, chunk):
        yield rng.choice((-1.0, 1.0), (min(chunk, draws - start), n))


def significance_in(predictions, truth, climbs, comparisons):
    """Every comparison on these rows: difference in RMSE and MAE, bootstrap interval, p-value, per seed.

    A method with one seed (the training mean) is paired with every seed of the other. Every
    comparison draws the same bootstrap samples and sign flips, so their results are comparable.
    """
    _, groups = np.unique(climbs, return_inverse=True)
    n = groups.max() + 1
    sizes = np.bincount(groups, minlength=n).astype(np.float64)
    losses = {"rmse": lambda p: (p - truth) ** 2, "mae": lambda p: np.abs(p - truth)}
    finish = {"rmse": np.sqrt, "mae": lambda x: x}
    errors = {"rmse": rmse, "mae": mae}
    # Per method and metric, per-seed summed loss of each climb: (seeds, climbs).
    totals = {(name, metric): np.stack([np.bincount(groups, loss(p), n) for p in predictions[name]])
              for name in {m for pair in comparisons for m in pair} for metric, loss in losses.items()}

    def paired(a, b, metric):
        seeds = max(len(totals[a, metric]), len(totals[b, metric]))
        index = lambda name: np.arange(seeds) % len(totals[name, metric])
        return totals[a, metric][index(a)], totals[b, metric][index(b)]

    draws = {pair: {metric: [] for metric in losses} for pair in comparisons}
    for weights in resampled_climbs(n, BOOTSTRAPS, seed=0):
        count = weights @ sizes
        for a, b in comparisons:
            for metric in losses:
                first, second = paired(a, b, metric)
                gap = finish[metric]((weights @ first.T) / count[:, None]) - finish[metric]((weights @ second.T) / count[:, None])
                draws[a, b][metric].append(gap.mean(1))
    # Each climb's loss difference averaged over seeds; the permutation statistic is linear in it.
    differences = {(pair, metric): np.subtract(*paired(*pair, metric)).mean(0) for pair in comparisons for metric in losses}
    extreme = dict.fromkeys(differences, 0)
    for signs in sign_flips(n, PERMUTATIONS, seed=1):
        for key, difference in differences.items():
            extreme[key] += int((np.abs(signs @ difference) >= abs(difference.sum()) * (1 - 1e-12)).sum())

    result = {}
    for a, b in comparisons:
        seeds = max(len(predictions[a]), len(predictions[b]))
        pairs = [(predictions[a][i % len(predictions[a])], predictions[b][i % len(predictions[b])]) for i in range(seeds)]
        result[f"{a}_vs_{b}"] = {}
        for metric in losses:
            per_seed = [errors[metric](x, truth) - errors[metric](z, truth) for x, z in pairs]
            values = np.concatenate(draws[a, b][metric])
            result[f"{a}_vs_{b}"][metric] = {
                "difference": float(np.mean(per_seed)), "low": float(np.percentile(values, 2.5)),
                "high": float(np.percentile(values, 97.5)),
                "p_value": (extreme[(a, b), metric] + 1) / (PERMUTATIONS + 1), "per_seed": per_seed}
    return result


def significance(predictions, truth, ascents, climbs):
    """Every comparison whose methods were run, in every subset, with Holm-adjusted p-values per subset and metric."""
    comparisons = tuple(pair for pair in COMPARISONS if all(name in predictions for name in pair))
    tests = {}
    for subset in SUBSETS:
        keep = subset_mask(subset, ascents)
        tests[subset] = significance_in({name: [p[keep] for p in values] for name, values in predictions.items()},
                                        truth[keep], climbs[keep], comparisons)
        for metric in ("rmse", "mae"):
            names = list(tests[subset])
            for name, value in zip(names, holm([tests[subset][k][metric]["p_value"] for k in names])):
                tests[subset][name][metric]["p_holm"] = value
        print(f"significance: {subset} done", flush=True)
    return {"rule": "difference = method - reference, in grades, mean over seeds (negative: method is better); "
                    "95% climb-bootstrap interval; two-sided climb-level sign-flip permutation p-value "
                    "(squared-error difference for RMSE), Holm-adjusted within each subset and metric",
            "bootstraps": BOOTSTRAPS, "permutations": PERMUTATIONS, "smallest_p_value": 1 / (PERMUTATIONS + 1),
            "comparisons": [list(pair) for pair in comparisons], "subsets": tests}


def external_predictions(folder, test_row_ids):
    """Per-seed test predictions written by experiments/<folder>.py and its choices, or None if it has not run."""
    path = ROOT / "results" / folder / "test_predictions.npz"
    if not path.exists():
        return None
    saved = np.load(path)
    if str(saved["rows_sha256"]) != hashlib.sha256("\n".join(map(str, test_row_ids)).encode()).hexdigest():
        raise ValueError(f"{path} was made for different test rows; rerun experiments/{folder}.py")
    summary = json.loads((path.parent / "summary.json").read_text())
    choices = [{"config": summary["chosen"]["name"], "epoch": run["best_epoch"], "validation_mse": run["validation"]["mse"]}
               for run in summary["runs"]]
    return [p.astype(np.float64) for p in saved["predictions"]], choices


def train_and_predict(arrays, board, y, angles, train_angles, checkpoints, device):
    """Fit every method on training rows, choose its settings on validation, and predict test rows.

    Returns per-method lists of test predictions (one per seed) and the settings chosen per seed.
    """
    by_angle = np.array([y[TRAIN].numpy()[train_angles == a].mean() if (train_angles == a).any() else y[TRAIN].mean().item()
                         for a in range(91)])
    bag, bag_angle = zip(*(bag_of_holds(arrays, split, board.vocab_size) for split in (TRAIN, VALIDATION, TEST)))
    prior = angle_prior(arrays)
    choices, predictions = {}, {}

    def record(name, predicted, choice=None):
        predictions.setdefault(name, []).append(predicted)
        if choice:
            choices.setdefault(name, []).append(choice)

    def heads(name, x, seed):
        record(f"{name}_linear", *fit_ridge(x[0], y[TRAIN], x[1], y[VALIDATION], x[2], device))
        record(f"{name}_mlp", *fit_mlp(x[0], y[TRAIN], x[1], y[VALIDATION], x[2], device, seed))

    joined = [torch.cat([b.to(torch.uint8), (a * 255).to(torch.uint8)], dim=1) for b, a in zip(bag, bag_angle)]
    record("mean_at_angle", by_angle[angles[TEST]])
    truth = y[TEST].numpy()
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
        record("bayes", posterior_mean(test_likelihood, prior, angles[TEST], temperature), {"temperature": temperature})
        reset(model, seed)
        heads("untrained", [representations(model, arrays, split, device) for split in (TRAIN, VALIDATION, TEST)], seed)
        print(json.dumps({"seed": seed, **{n: rmse(v[-1], truth) for n, v in predictions.items()}}), flush=True)
    return predictions, choices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "grade_climbs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--predictions", type=Path,
                        help="npz of test predictions: read it if it exists instead of training, otherwise write it")
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
    if args.predictions and args.predictions.exists():
        saved = np.load(args.predictions)
        predictions = {name: list(saved[name]) for name in saved.files if name != "choices"}
        choices = json.loads(str(saved["choices"]))
        print(f"Read test predictions from {args.predictions}", flush=True)
    else:
        predictions, choices = train_and_predict(arrays, board, y, angles, train_angles, checkpoints, device)
        if args.predictions:
            np.savez_compressed(args.predictions, choices=json.dumps(choices),
                                **{name: np.stack(values) for name, values in predictions.items()})
    for name, folder in EXTERNAL.items():
        external = external_predictions(folder, arrays["row_ids"][test_rows])
        if external is not None:
            predictions[name], choices[name] = external
    scores = {name: [score(predicted, truth, ascents) for predicted in values] for name, values in predictions.items()}

    summary = {name: {metric: {"mean": float(np.mean([r[metric] for r in rows])),
                               "std": float(np.std([r[metric] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
                      for metric in rows[0]} for name, rows in scores.items()}
    (args.output / "summary.json").write_text(json.dumps({
        "model": MODEL, "seeds": SEEDS, "test_rows": len(truth), "test_grade_variance": float(np.var(truth)),
        "subsets": {name: {"rule": "every test row" if over is None else f"ascents at that angle > {over}",
                           "suffix": suffix, "rows": int(subset_mask(name, ascents).sum()),
                           "climbs": len(set(climbs[subset_mask(name, ascents)])),
                           "grade_variance": float(np.var(truth[subset_mask(name, ascents)]))}
                    for name, (suffix, over) in SUBSETS.items()},
        "methods": summary, "per_seed": scores, "chosen_on_validation": choices,
        "significance": significance(predictions, truth, ascents, climbs)}, indent=2) + "\n")
    (args.output / "confusion.json").write_text(json.dumps({
        "grades": list(range(GRADES)),
        "rule": "rows: true V grade; columns: prediction rounded to the nearest grade and clipped to V0-V16; "
                "counts of test rows, averaged over seeds",
        "subsets": {name: int(subset_mask(name, ascents).sum()) for name in SUBSETS},
        "methods": {name: {subset: np.round(np.mean([confusion(v[subset_mask(subset, ascents)], truth[subset_mask(subset, ascents)])
                                                     for v in values], axis=0), 4).tolist() for subset in SUBSETS}
                    for name, values in predictions.items()},
    }) + "\n")
    with (args.output / "by_grade.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "grade", "rows", "rmse", "mae", "mean_predicted"])
        for name, values in predictions.items():
            for grade in range(GRADES):
                keep = truth == grade
                if keep.any():
                    writer.writerow([name, grade, int(keep.sum()), np.mean([rmse(v[keep], truth[keep]) for v in values]),
                                     np.mean([mae(v[keep], truth[keep]) for v in values]),
                                     float(np.mean([v[keep].mean() for v in values]))])
    print(json.dumps({name: {"rmse": values["rmse"], "mae": values["mae"]} for name, values in summary.items()}, indent=1))


if __name__ == "__main__":
    main()
