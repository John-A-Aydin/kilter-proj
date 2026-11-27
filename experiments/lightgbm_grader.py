"""Gradient-boosted trees (LightGBM) grading climbs from the bag of holds plus hand-made geometry features.

A strong non-neural baseline for grade_climbs.py. It sees the same information as the generator
(``null_rule_16``): which hold is used in which role, the holds' board positions and flags, and
the wall angle, on exactly the same rows and split. No hand-labelled hold types.

Features (``FEATURES`` names them all):

- the bag of holds: one binary column per hold and role (1,904), as in grade_climbs.py;
- counts: holds per role, kickboard and screw-on holds;
- hands (start, hand and finish holds): height and width spans, mean and spread, start and
  finish positions, and the gaps between successive hands sorted by height (vertical and
  straight-line: mean, largest, second largest);
- feet: count, lowest, highest and mean height, width span, and how far below the lowest hand
  the highest foot is;
- the angle.

Protocol, as experiments/graders.py describes: every configuration in ``SWEEP`` is trained on seed 42
with early stopping on validation MSE (the criterion every grading method uses), the one with the
lowest validation MSE is retrained on seeds 42, 43 and 44 (seeds change row and feature
subsampling), and only then is the test split predicted. Outputs in results/lightgbm_grader/:
sweep.csv, summary.json (with feature importances), and test_predictions.npz, which
grade_climbs.py reads as the method ``lightgbm``.

    .venv/bin/python experiments/lightgbm_grader.py
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
import time
import warnings

import lightgbm
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from kilter.config import write_json  # noqa: E402
from kilter.data import FINISH, FOOT, HAND, START, Board, load_arrays, prepare  # noqa: E402
from graders import GENERATOR, generator_data_config, rows_sha256, same_rows_as_generator  # noqa: E402

SEARCH_SEED = 42
SEEDS = (42, 43, 44)
TRAIN, VALIDATION, TEST = 0, 1, 2
OUTPUT = ROOT / "results" / "lightgbm_grader"
THREADS = 12


@dataclass(frozen=True)
class TreeConfig:
    name: str
    learning_rate: float = 0.05
    num_leaves: int = 255
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.5
    bagging_fraction: float = 0.8
    lambda_l2: float = 1.0
    max_bin: int = 255
    max_rounds: int = 20000
    early_stopping: int = 300


BASE = TreeConfig("leaves_255")
SWEEP = (
    replace(BASE, name="leaves_63", num_leaves=63),
    BASE,
    replace(BASE, name="leaves_1023", num_leaves=1023, min_data_in_leaf=100),
    replace(BASE, name="leaves_255_min_200", min_data_in_leaf=200),
    replace(BASE, name="leaves_255_features_0.2", feature_fraction=0.2),
    replace(BASE, name="leaves_255_l2_10", lambda_l2=10.0),
    # Round 2, after round 1: the smallest trees (63 leaves) did best, so smaller still, and slower.
    replace(BASE, name="leaves_31", num_leaves=31),
    replace(BASE, name="leaves_15", num_leaves=15),
    replace(BASE, name="leaves_63_lr_0.02", num_leaves=63, learning_rate=0.02),
    replace(BASE, name="leaves_31_lr_0.02", num_leaves=31, learning_rate=0.02),
    replace(BASE, name="leaves_63_min_200", num_leaves=63, min_data_in_leaf=200),
    replace(BASE, name="leaves_63_features_0.2", num_leaves=63, feature_fraction=0.2),
    # Round 3, after round 2: a lower learning rate and sampling fewer features each helped; combine them.
    replace(BASE, name="leaves_63_lr_0.02_features_0.2", num_leaves=63, learning_rate=0.02, feature_fraction=0.2),
    replace(BASE, name="leaves_63_lr_0.01", num_leaves=63, learning_rate=0.01, max_rounds=40000),
    replace(BASE, name="leaves_15_lr_0.02_features_0.2", num_leaves=15, learning_rate=0.02, feature_fraction=0.2,
            max_rounds=40000),
)


# ---------------------------------------------------------------- features

def padded(arrays):
    offsets, flat = arrays["offsets"], arrays["tokens"]
    lengths = np.diff(offsets)
    tokens = np.zeros((len(lengths), int(lengths.max())), dtype=np.int64)
    for i in range(len(lengths)):
        tokens[i, :lengths[i]] = flat[offsets[i]:offsets[i + 1]]
    return tokens


def masked_stats(values, mask, prefix):
    """min, max, span, mean, std of ``values`` where ``mask``; NaN for rows with none."""
    count = mask.sum(1)
    safe = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanmin(safe, 1), np.nanmax(safe, 1)
        mean, std = np.nanmean(safe, 1), np.nanstd(safe, 1)
    low, high = np.where(count > 0, low, np.nan), np.where(count > 0, high, np.nan)
    return {f"{prefix}_min": low, f"{prefix}_max": high, f"{prefix}_span": high - low, f"{prefix}_mean": mean,
            f"{prefix}_std": std}


def gaps(x, y, mask):
    """Gaps between successive masked holds sorted by height: vertical and straight-line, mean and top two."""
    order = np.argsort(np.where(mask, y, np.inf), axis=1, kind="stable")
    ys, xs = np.take_along_axis(y, order, 1), np.take_along_axis(x, order, 1)
    valid = np.take_along_axis(mask, order, 1)
    pair = valid[:, 1:] & valid[:, :-1]
    vertical = np.where(pair, np.diff(ys, axis=1), np.nan)
    straight = np.where(pair, np.hypot(np.diff(xs, axis=1), np.diff(ys, axis=1)), np.nan)
    out = {}
    for name, values in (("vertical", vertical), ("straight", straight)):
        ordered = -np.sort(-np.nan_to_num(values, nan=-1.0), axis=1)
        has = pair.sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"hand_gap_{name}_mean"] = np.nanmean(values, 1)
        out[f"hand_gap_{name}_max"] = np.where(has >= 1, ordered[:, 0], np.nan)
        out[f"hand_gap_{name}_second"] = np.where(has >= 2, ordered[:, 1] if ordered.shape[1] > 1 else np.nan, np.nan)
    return out


def geometry_features(tokens, board):
    real = tokens >= 4
    dense, role = np.divmod(np.where(real, tokens - 4, 0), 4)
    x, y = board.x[dense].astype(np.float64), board.y[dense].astype(np.float64)
    kick, screw = board.kickboard[dense] > 0, board.screw_on[dense] > 0
    is_role = {r: real & (role == r) for r in (START, HAND, FINISH, FOOT)}
    hands = is_role[START] | is_role[HAND] | is_role[FINISH]
    feet = is_role[FOOT]
    columns = {"holds": real.sum(1), "starts": is_role[START].sum(1), "hands": is_role[HAND].sum(1),
               "finishes": is_role[FINISH].sum(1), "feet": feet.sum(1), "kickboard_holds": (real & kick).sum(1),
               "screw_on_holds": (real & screw).sum(1), "screw_on_hand_holds": (hands & screw).sum(1)}
    columns |= masked_stats(y, hands, "hand_y") | masked_stats(x, hands, "hand_x")
    for name, mask in (("start", is_role[START]), ("finish", is_role[FINISH])):
        stats = masked_stats(y, mask, f"{name}_y") | masked_stats(x, mask, f"{name}_x")
        columns |= {k: stats[k] for k in (f"{name}_y_mean", f"{name}_x_mean", f"{name}_x_span")}
    columns["finish_minus_start_y"] = columns["finish_y_mean"] - columns["start_y_mean"]
    columns["finish_minus_start_x_abs"] = np.abs(columns["finish_x_mean"] - columns["start_x_mean"])
    columns["hand_y_per_hand"] = columns["hand_y_span"] / np.maximum(hands.sum(1), 1)
    columns |= gaps(x, y, hands)
    foot = masked_stats(y, feet, "foot_y") | masked_stats(x, feet, "foot_x")
    columns |= {k: foot[k] for k in ("foot_y_min", "foot_y_max", "foot_y_mean", "foot_x_span")}
    columns["lowest_hand_minus_highest_foot"] = columns["hand_y_min"] - columns["foot_y_max"]
    names = list(columns)
    return np.stack([np.asarray(columns[n], dtype=np.float64) for n in names], 1), names


def build(arrays, board):
    """Feature matrix (CSR) for every row, and the column names."""
    tokens = padded(arrays)
    geometry, names = geometry_features(tokens, board)
    angle = arrays["conditions"][:, 1:2].astype(np.float64) * 90
    rows, columns = np.nonzero(tokens >= 4)
    bag = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, tokens[rows, columns] - 4)),
                            shape=(len(tokens), board.vocab_size - 4))
    hold_names = [f"hold_{board.holds[t // 4]}_{('hand', 'foot', 'start', 'finish')[t % 4]}" for t in range(board.vocab_size - 4)]
    matrix = sparse.hstack([sparse.csr_matrix(np.hstack([angle, geometry])), bag], format="csr")
    return matrix, ["angle", *names, *hold_names]


# ---------------------------------------------------------------- training

def parameters(config, seed):
    return {"objective": "regression", "learning_rate": config.learning_rate, "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_data_in_leaf, "feature_fraction": config.feature_fraction,
            "bagging_fraction": config.bagging_fraction, "bagging_freq": 1, "lambda_l2": config.lambda_l2,
            "max_bin": config.max_bin, "seed": seed, "deterministic": True, "force_row_wise": True,
            "num_threads": THREADS, "verbosity": -1}


def scores(predicted, truth):
    error = predicted - truth
    mse = float(np.mean(error ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(error))),
            "within_one_grade": float(np.mean(np.abs(error) <= 1))}


def fit(config, seed, x, y, names):
    started = time.time()
    train = lightgbm.Dataset(x[TRAIN], y[TRAIN], feature_name=names, free_raw_data=False,
                             params={"max_bin": config.max_bin})
    validation = lightgbm.Dataset(x[VALIDATION], y[VALIDATION], reference=train)
    booster = lightgbm.train(parameters(config, seed), train, num_boost_round=config.max_rounds, valid_sets=[validation],
                             callbacks=[lightgbm.early_stopping(config.early_stopping, verbose=False),
                                        lightgbm.log_evaluation(500)])
    predicted = booster.predict(x[VALIDATION], num_iteration=booster.best_iteration)
    return booster, {"rounds": booster.best_iteration, "validation": scores(predicted, y[VALIDATION]),
                     "minutes": round((time.time() - started) / 60, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    data_config = generator_data_config()
    cache_dir, metadata = prepare(data_config)
    arrays = load_arrays(cache_dir)
    same_rows = same_rows_as_generator(arrays)
    board = Board.from_json(metadata["board"])
    matrix, names = build(arrays, board)
    split_rows = {s: np.flatnonzero(arrays["splits"] == s) for s in (TRAIN, VALIDATION, TEST)}
    x = {s: matrix[rows] for s, rows in split_rows.items()}
    y = {s: arrays["conditions"][rows, 0].astype(np.float64) for s, rows in split_rows.items()}
    print(f"{matrix.shape[1]} features ({len(names) - (board.vocab_size - 4)} dense)", flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for config in SWEEP:
        _, result = fit(config, SEARCH_SEED, x, y, names)
        sweep.append({**asdict(config), "rounds": result["rounds"],
                      **{f"validation_{k}": v for k, v in result["validation"].items()}, "minutes": result["minutes"]})
        print(json.dumps(sweep[-1]), flush=True)
    with (OUTPUT / "sweep.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    chosen = min(SWEEP, key=lambda c: next(r for r in sweep if r["name"] == c.name)["validation_mse"])
    print(f"Chosen on validation MSE: {chosen.name}", flush=True)

    runs, predictions, importance = [], [], np.zeros(len(names))
    for seed in SEEDS:
        booster, result = fit(chosen, seed, x, y, names)
        predictions.append(booster.predict(x[TEST], num_iteration=booster.best_iteration))
        importance += booster.feature_importance("gain", iteration=booster.best_iteration)
        runs.append({"seed": seed, "best_epoch": result["rounds"], "rounds": result["rounds"],
                     "validation": result["validation"]})
    test_ids = arrays["row_ids"][split_rows[TEST]]
    np.savez_compressed(OUTPUT / "test_predictions.npz", predictions=np.stack(predictions), seeds=np.asarray(SEEDS),
                        rows_sha256=rows_sha256(test_ids), data_fingerprint=metadata["fingerprint"], config=chosen.name)
    share = importance / importance.sum()
    top = np.argsort(-share)[:40]
    write_json(OUTPUT / "summary.json", {
        "generator": GENERATOR, "data_fingerprint": metadata["fingerprint"], "same_rows_as_generator_training": same_rows,
        "lightgbm": lightgbm.__version__, "features": len(names),
        "selection": "lowest validation MSE on seed 42 among the configurations in sweep.csv",
        "chosen": asdict(chosen), "seeds": list(SEEDS), "runs": runs,
        "validation_mean": {k: float(np.mean([r["validation"][k] for r in runs])) for k in runs[0]["validation"]},
        "test_rows": len(test_ids), "test_rows_sha256": rows_sha256(test_ids),
        "test_note": "Test scores are computed by grade_climbs.py from test_predictions.npz, with every other method.",
        "gain_share_top_40": {names[i]: round(float(share[i]), 5) for i in top},
        "gain_share_bag_of_holds": round(float(share[len(names) - (board.vocab_size - 4):].sum()), 5),
        "sweep": sweep,
    })
    print(json.dumps({"seed_" + str(r["seed"]): r["validation"] for r in runs}, indent=1))


if __name__ == "__main__":
    main()
