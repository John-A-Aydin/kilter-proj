"""Do the hold representations of a model trained without hold labels encode the labels anyway?

``null_rule_16`` never saw data/hold_labels.csv. For each of its hold representations, a linear
*probe* is fit to predict the labels and scored on holds it was not fit on. A probe that beats
chance and the controls shows the representation contains the label. The full write-up, with
results, is the "Hold labels in the learned representations" section of experiments/README.md.

Tasks (all scored on held-out holds):

- type: jug, edge, sloper, pinch or foot chip (476 holds), scored by balanced accuracy, the mean
  over types of per-type recall. Chance is 1 / 5 whatever the class sizes.
- hand_type: the same without foot chips (306 holds, chance 1 / 4). Foot chips are easy: most
  are screw-on holds, and the screw-on flag is a position input.
- grip: the pull direction of the 423 holds labeled with exactly one, scored by the mean angle
  between predicted and labeled directions, |((p - l + 180) mod 360) - 180| degrees.

Representations (one vector per hold, per seed):

- position: the model's 28 fixed input features (x, y, kickboard and screw-on flags, sin and cos
  of x and y at 6 frequencies). Control.
- usage: log(1 + count) of the hold's uses as hand, foot, start and finish in the training
  climbs, and each role's (smoothed) share of its uses. Control.
- untrained: the hold vector of the same network re-initialized to random weights. Control.
- identity: the learned per-hold embedding, ``model.hold.weight``. No position input.
- hold_vector: identity + ``model.geometry(position features)``, the hold's part of every token.
- context: the final hidden state (output of the last LayerNorm) at the hold's token in real
  training climbs read in "scratch" mode, minus the mean state at that sequence position,
  averaged per hold over its roles (hold_embeddings.collect).

Algorithms:

1. Features. Each representation is centered. The 192-wide ones are reduced to their top 20
   principal components by SVD (unsupervised). Position and usage keep every column. Every
   column is scaled to unit variance.
2. Type probe: multinomial logistic regression, softmax(z W + b), minimizing class-weighted
   cross-entropy (weight n / (classes * class count)) plus lambda * ||W||^2 / n, fit by L-BFGS
   with a strong Wolfe line search (at most 200 iterations), float64.
3. Grip probe: ridge regression from z to (cos theta, sin theta), closed form
   W = (Z'Z + lambda I)^-1 Z'(Y - mean Y); the prediction is atan2 of the two outputs.
4. Evaluation: 5-fold cross-validation (stratified by type for the type tasks), repeated 5 times
   with new folds; scores are averaged over repeats, then over seeds 42, 43 and 44.
5. Penalty: lambda in {0.01, 0.1, 1, 10, 100}, picked separately for each outer fold by 4-fold
   cross-validation on that fold's training holds only (best balanced accuracy, or lowest mean
   angle error). The outer test fold never influences its own probe.
6. Nulls: the hold_vector probe refit on labels shuffled across holds (20 times per seed, one
   cross-validation repeat each), and for grip a constant guess of the average labeled direction.

Outputs in results/hold_label_probes/: summary.json (every score per seed, per-type recall,
confusion matrices, nulls, and their means over seeds) and holds.csv (each hold's labels, usage
counts, hold-vector projection, and held-out predictions of seed 43, the seed the website serves).
The experiments page draws its figures from these two files.

    .venv/bin/python experiments/hold_label_probes.py
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from hold_embeddings import collect, reset, role_counts  # noqa: E402
from kilter.data import ROLE_NAMES, load_arrays, prepare  # noqa: E402
from kilter.experiment import load_run  # noqa: E402
from kilter.hold_labels import read_hold_labels  # noqa: E402
from kilter.sampling import LegalTokens  # noqa: E402

MODEL = "null_rule_16"
SEEDS = (42, 43, 44)
PLOT_SEED = 43
TYPES = ("jug", "edge", "sloper", "pinch", "foot")
HAND_TYPES = ("jug", "edge", "sloper", "pinch")
REPRESENTATIONS = ("position", "usage", "untrained", "identity", "hold_vector", "context")
LEARNED = ("untrained", "identity", "hold_vector", "context")
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
PCA_DIMS = 20
FOLDS = 5


def standardize(features, dims=PCA_DIMS):
    """Columns scaled to unit variance; wider features are first cut to their top principal components.

    Narrow features (position, usage) keep every column: a projection would drop low-variance
    columns such as the screw-on flag, which alone separates most foot chips.
    """
    centered = features - features.mean(0)
    if centered.shape[1] <= 2 * PCA_DIMS and dims == PCA_DIMS:
        keep = centered.std(0) > 1e-9
        return (centered[:, keep] / centered[:, keep].std(0)).astype(np.float64)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    keep = min(dims, int((s > 1e-6 * s[0]).sum()))
    z = u[:, :keep] * s[:keep]
    return (z / z.std(0)).astype(np.float64)


def stratified_folds(labels, folds, rng):
    """Fold index per row, each class spread evenly over the folds."""
    fold = np.empty(len(labels), dtype=int)
    for value in np.unique(labels):
        rows = rng.permutation(np.flatnonzero(labels == value))
        fold[rows] = (np.arange(len(rows)) + rng.integers(folds)) % folds
    return fold


def fit_logistic(z, labels, classes, penalty):
    """Class-weighted multinomial logistic regression with an L2 penalty, fit by L-BFGS."""
    x = torch.as_tensor(z)
    y = torch.as_tensor(labels)
    counts = torch.bincount(y, minlength=classes).double()
    weights = len(y) / (classes * counts.clamp(min=1))
    w = torch.zeros(z.shape[1], classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(x @ w + b, y, weight=weights) + penalty * (w ** 2).sum() / len(y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return lambda new: torch.softmax(torch.as_tensor(new) @ w.detach() + b.detach(), -1).numpy()


def fit_ridge(z, targets, penalty):
    """Ridge regression in closed form, intercept unpenalized: W = (Z'Z + penalty I)^-1 Z'(Y - mean Y)."""
    mean = targets.mean(0)
    weights = np.linalg.solve(z.T @ z + penalty * np.eye(z.shape[1]), z.T @ (targets - mean))
    return lambda new: new @ weights + mean


def balanced_accuracy(labels, predicted, classes):
    """Mean over the classes present of the share of that class predicted correctly (recall)."""
    return float(np.mean([(predicted[labels == c] == c).mean() for c in range(classes) if (labels == c).any()]))


def angle_error(a, b):
    """Absolute angle between directions in degrees, 0 to 180."""
    return np.abs((a - b + 180) % 360 - 180)


def direction_of(vectors):
    """Angle in degrees of each (cos, sin) row, in [0, 360)."""
    return np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 360


def unit(angles):
    """Directions in degrees as (cos, sin) rows, so that 350 and 10 degrees are close."""
    radians = np.radians(angles)
    return np.stack([np.cos(radians), np.sin(radians)], 1)


def classify(z, labels, classes, rng, repeats):
    """Out-of-fold class probabilities for each repeat, with the penalty picked by inner folds."""
    runs = []
    for _ in range(repeats):
        fold = stratified_folds(labels, FOLDS, rng)
        probabilities = np.zeros((len(labels), classes))
        for k in range(FOLDS):
            train, test = fold != k, fold == k
            inner = stratified_folds(labels[train], 4, rng)

            def inner_score(penalty):
                hits = []
                for j in range(4):
                    model = fit_logistic(z[train][inner != j], labels[train][inner != j], classes, penalty)
                    hits.append(balanced_accuracy(labels[train][inner == j], model(z[train][inner == j]).argmax(1), classes))
                return np.mean(hits)

            penalty = max(PENALTIES, key=inner_score)
            probabilities[test] = fit_logistic(z[train], labels[train], classes, penalty)(z[test])
        runs.append(probabilities)
    return runs


def regress_direction(z, angles, rng, repeats):
    """Out-of-fold predicted directions for each repeat; the target is the direction's unit vector."""
    targets = unit(angles)
    runs = []
    for _ in range(repeats):
        fold = rng.permutation(len(angles)) % FOLDS
        predicted = np.zeros(len(angles))
        for k in range(FOLDS):
            train, test = fold != k, fold == k
            inner = rng.permutation(int(train.sum())) % 4

            def inner_error(penalty):
                errors = []
                for j in range(4):
                    model = fit_ridge(z[train][inner != j], targets[train][inner != j], penalty)
                    errors.append(angle_error(direction_of(model(z[train][inner == j])), angles[train][inner == j]).mean())
                return np.mean(errors)

            penalty = min(PENALTIES, key=inner_error)
            predicted[test] = direction_of(fit_ridge(z[train], targets[train], penalty)(z[test]))
        runs.append(predicted)
    return runs


def representations(checkpoint, arrays, board, rows, device, seed):
    """Every representation for one seed's checkpoint, plus each hold's context reads and role counts."""
    model, _, config, _ = load_run(checkpoint, device)
    legal = LegalTokens(board, config.data.max_holds, device)
    n = len(board.holds)
    with torch.no_grad():
        identity = model.hold.weight.cpu().numpy()
        hold_vector = identity + model.geometry(model.hold_features).cpu().numpy()
        position = model.hold_features.cpu().numpy()
    _, adjusted, seen, _ = collect(model, arrays, rows, device, legal)
    # A hold's context vector averages its tokens over the roles it was read in.
    seen = seen[4:].reshape(n, len(ROLE_NAMES))
    states = adjusted[4:].reshape(n, len(ROLE_NAMES), -1)
    context = (states * seen[..., None]).sum(1) / np.maximum(seen.sum(1, keepdims=True), 1)
    reset(model, seed)
    with torch.no_grad():
        untrained = (model.hold.weight + model.geometry(model.hold_features)).cpu().numpy()
    counts = role_counts(arrays)[:n].astype(np.float64)
    usage = np.concatenate([np.log1p(counts), (counts + 1) / (counts.sum(1, keepdims=True) + 4)], 1)
    return {"position": position, "usage": usage, "untrained": untrained, "identity": identity,
            "hold_vector": hold_vector, "context": context}, seen.sum(1), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60000, help="Training rows run through the model for context")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats of the 5-fold cross-validation")
    parser.add_argument("--shuffles", type=int, default=20, help="Shuffled-label probes per seed")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "hold_labels.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "hold_label_probes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoints = json.loads((ROOT / "results" / "compare_models" / "checkpoints.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)

    results, per_hold = {}, None
    for seed in SEEDS:
        checkpoint = ROOT / checkpoints[f"{MODEL}_s{seed}"]["run"] / "best.pt"
        _, board, config, _ = load_run(checkpoint)
        labels = read_hold_labels(args.labels, board)
        arrays = load_arrays(prepare(config.data)[0])
        features, contexts, counts = representations(checkpoint, arrays, board, args.rows, args.device, seed)
        hold_labels = [labels["holds"][str(h)] for h in board.holds]
        kind = np.array([TYPES.index(h["type"]) for h in hold_labels])
        hand = np.flatnonzero(kind != TYPES.index("foot"))
        single = np.flatnonzero([len(h["grip"]) == 1 for h in hold_labels])
        grip = np.array([h["grip"][0] for h in hold_labels])[single]
        z = {name: standardize(values) for name, values in features.items()}
        rng = np.random.default_rng(seed)
        seed_results = {"holds": len(board.holds), "hand_holds": len(hand), "single_grip_holds": len(single)}
        predictions = {}
        for name in REPRESENTATIONS:
            type_runs = classify(z[name], kind, len(TYPES), rng, args.repeats)
            hand_runs = classify(z[name][hand], kind[hand], len(HAND_TYPES), rng, args.repeats)
            grip_runs = regress_direction(z[name][single], grip, rng, args.repeats)
            seed_results[name] = {
                "type_balanced_accuracy": float(np.mean([balanced_accuracy(kind, p.argmax(1), len(TYPES)) for p in type_runs])),
                "type_accuracy": float(np.mean([(p.argmax(1) == kind).mean() for p in type_runs])),
                "type_recall": {t: float(np.mean([(p.argmax(1)[kind == c] == c).mean() for p in type_runs]))
                                for c, t in enumerate(TYPES)},
                "type_confusion": np.mean([np.eye(len(TYPES))[kind].T @ np.eye(len(TYPES))[p.argmax(1)] for p in type_runs],
                                          0).round(2).tolist(),
                "hand_type_balanced_accuracy": float(np.mean([balanced_accuracy(kind[hand], p.argmax(1), len(HAND_TYPES))
                                                              for p in hand_runs])),
                "hand_type_recall": {t: float(np.mean([(p.argmax(1)[kind[hand] == c] == c).mean() for p in hand_runs]))
                                     for c, t in enumerate(HAND_TYPES)},
                "grip_mean_error_degrees": float(np.mean([angle_error(p, grip).mean() for p in grip_runs])),
                "grip_within_45_degrees": float(np.mean([(angle_error(p, grip) <= 45).mean() for p in grip_runs])),
            }
            predictions[name] = {"type": np.mean(type_runs, 0), "grip": grip_runs[0]}
            print(json.dumps({"seed": seed, name: {k: v for k, v in seed_results[name].items()
                                                   if not isinstance(v, (dict, list))}}), flush=True)
        # Nulls: the hold_vector probe with labels shuffled across holds, and a constant guess.
        shuffled_type, shuffled_hand, shuffled_grip = [], [], []
        for _ in range(args.shuffles):
            permuted = rng.permutation(kind)
            shuffled_type.append(balanced_accuracy(permuted, classify(z["hold_vector"], permuted, len(TYPES), rng, 1)[0].argmax(1),
                                                   len(TYPES)))
            hand_permuted = rng.permutation(kind[hand])
            shuffled_hand.append(balanced_accuracy(hand_permuted, classify(z["hold_vector"][hand], hand_permuted,
                                                                           len(HAND_TYPES), rng, 1)[0].argmax(1), len(HAND_TYPES)))
            grip_permuted = rng.permutation(grip)
            shuffled_grip.append(float(angle_error(regress_direction(z["hold_vector"][single], grip_permuted, rng, 1)[0],
                                                   grip_permuted).mean()))
        mean_direction = direction_of(unit(grip).mean(0, keepdims=True))[0]
        seed_results["null"] = {
            "type_chance": 1 / len(TYPES), "hand_type_chance": 1 / len(HAND_TYPES),
            "type_shuffled": shuffled_type, "hand_type_shuffled": shuffled_hand, "grip_shuffled": shuffled_grip,
            "grip_constant_mean_direction_error": float(angle_error(mean_direction, grip).mean()),
            "grip_constant_mean_direction": float(mean_direction)}
        results[seed] = seed_results
        print(json.dumps({"seed": seed, "null": {k: (float(np.mean(v)) if isinstance(v, list) else v)
                                                 for k, v in seed_results["null"].items()}}), flush=True)

        if seed == PLOT_SEED:
            vector = z["hold_vector"]
            projection = standardize(features["hold_vector"], 2)
            single_index = {int(i): j for j, i in enumerate(single)}
            per_hold = []
            for i, hold in enumerate(board.holds):
                row = {"hold": hold, "x": float(board.x[i]), "y": float(board.y[i]), "kickboard": int(board.kickboard[i]),
                       "type": TYPES[kind[i]], "grip": ";".join(f"{d:g}" for d in hold_labels[i]["grip"]),
                       **{f"uses_{role}": int(counts[i, r]) for r, role in enumerate(ROLE_NAMES)},
                       "context_reads": int(contexts[i]), "pc1": float(projection[i, 0]), "pc2": float(projection[i, 1])}
                for name in ("position", "usage", "hold_vector", "context"):
                    row[f"{name}_type"] = TYPES[int(predictions[name]["type"][i].argmax())]
                    row[f"{name}_type_probability"] = float(predictions[name]["type"][i, kind[i]])
                    row[f"{name}_grip"] = (float(predictions[name]["grip"][single_index[i]]) if i in single_index else "")
                per_hold.append(row)
            assert vector.shape[0] == len(per_hold)

    names = [*REPRESENTATIONS]
    summary = {"model": MODEL, "seeds": list(SEEDS), "plot_seed": PLOT_SEED, "types": TYPES, "hand_types": HAND_TYPES,
               "representations": names, "repeats": args.repeats, "folds": FOLDS, "pca_dims": PCA_DIMS,
               "penalties": PENALTIES, "rows": args.rows, "labels_sha256": labels["sha256"], "by_seed": results}

    def over_seeds(get):
        values = [get(results[s]) for s in SEEDS]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}

    summary["mean"] = {name: {metric: over_seeds(lambda r: r[name][metric]) for metric in
                              ("type_balanced_accuracy", "hand_type_balanced_accuracy", "grip_mean_error_degrees",
                               "grip_within_45_degrees")} for name in names}
    summary["mean"]["null"] = {
        "type_shuffled": over_seeds(lambda r: float(np.mean(r["null"]["type_shuffled"]))),
        "type_shuffled_95th": float(np.percentile([v for s in SEEDS for v in results[s]["null"]["type_shuffled"]], 95)),
        "hand_type_shuffled": over_seeds(lambda r: float(np.mean(r["null"]["hand_type_shuffled"]))),
        "hand_type_shuffled_95th": float(np.percentile([v for s in SEEDS for v in results[s]["null"]["hand_type_shuffled"]], 95)),
        "grip_shuffled": over_seeds(lambda r: float(np.mean(r["null"]["grip_shuffled"]))),
        "grip_shuffled_5th": float(np.percentile([v for s in SEEDS for v in results[s]["null"]["grip_shuffled"]], 5)),
        "grip_constant_mean_direction_error": results[SEEDS[0]]["null"]["grip_constant_mean_direction_error"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "holds.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_hold[0]))
        writer.writeheader()
        writer.writerows(per_hold)
    print(json.dumps(summary["mean"], indent=1))


if __name__ == "__main__":
    main()
