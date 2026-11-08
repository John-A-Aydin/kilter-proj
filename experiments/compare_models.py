"""Train and evaluate every model on three seeds, then save the results under results/compare_models/.

Finished runs are reused, so this can be stopped and started again. Checkpoints stay
in the git-ignored runs/ folder; results/compare_models/ keeps everything else and their checksums.

    .venv/bin/python experiments/compare_models.py
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
SEEDS = (42, 43, 44)
CLIMB = "kilter"
NULL = ["--config", "configs/climb_null_grade.toml"]
LARGE = ["--set", "name=climb_v2_d192", "--set", "model.d_model=192", "--set", "model.n_layers=6",
         "--set", "model.n_heads=6"]
MODELS = {
    "ar_transformer": ("kilter.baseline", []),
    "climb_v2": (CLIMB, []),
    "climb_v2_d192": (CLIMB, LARGE),
    "climb_v2_null_grade": (CLIMB, NULL),
    "climb_v2_no_grade": (CLIMB, [*NULL, "--set", "name=climb_v2_no_grade", "--set", "train.p_null_grade=1.0"]),
    # One change each from the 50% null model, chosen from what the runs above showed.
    "null_16_epochs": (CLIMB, [*NULL, "--set", "name=null_16_epochs", "--set", "train.epochs=16"]),
    "null_d256": (CLIMB, [*NULL, "--set", "name=null_d256", "--set", "model.d_model=256", "--set", "model.n_layers=8",
                       "--set", "model.n_heads=8"]),
    "null_ascent_weighted": (CLIMB, [*NULL, "--set", "name=null_ascent_weighted", "--set", "train.ascent_weighting=log"]),
    "null_rule_loss": (CLIMB, [*NULL, "--set", "name=null_rule_loss", "--set", "train.rule_loss=true"]),
    # The changes that helped on validation, together.
    "null_rule_16": (CLIMB, [*NULL, "--set", "name=null_rule_16", "--set", "train.rule_loss=true",
                          "--set", "train.epochs=16"]),
    "null_rule_16_d256": (CLIMB, [*NULL, "--set", "name=null_rule_16_d256", "--set", "train.rule_loss=true",
                               "--set", "train.epochs=16", "--set", "model.d_model=256", "--set", "model.n_layers=8",
                               "--set", "model.n_heads=8"]),
    # The chosen model with each hold's hand-labeled type and grip directions as features.
    "null_rule_16_labels": (CLIMB, [*NULL, "--set", "name=null_rule_16_labels", "--set", "train.rule_loss=true",
                                 "--set", "train.epochs=16", "--set", "model.hold_labels=data/hold_labels.csv"]),
}
SPLITS = ("validation", "test")
RUN_FILES = ("config.json", "dataset.json", "environment.json", "metrics.jsonl", "model.json", "summary.json",
             "best_validation.json", "best_test.json", "quality_validation.json", "quality_test.json",
             "hold_labels.json")
LIKELIHOOD_MODES = ("scratch", "starts", "starts_feet")
SAMPLE_METHODS = ("real", "v2_scratch", "v2_scratch_unconstrained", "v2_given_starts_feet",
                  "v2_scratch_null_grade", "baseline_scratch", "baseline_barrier_given_starts_feet")
SAMPLE_CHECKS = ("valid", "no_repeated_hold", "starts_1_2", "finishes_1_2", "start_below_finish", "has_feet",
                 "novel", "distinct", "holds_drop_V2_to_V11")


def run(command, log):
    print(" ".join(command), flush=True)
    with open(log, "a") as output:
        subprocess.run(command, cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT)


def finished_runs():
    found = {}
    for path in sorted((ROOT / "runs").glob("*/summary.json")):
        config = json.loads((path.parent / "config.json").read_text())
        found[config["name"], config["train"]["seed"]] = path.parent
    return found


def train(name, seed, logs):
    package, options = MODELS[name]
    run([PYTHON, "-m", package, "train", *options, "--set", f"train.seed={seed}"], logs / f"{name}_s{seed}.log")


def evaluate(name, seed, directory, baseline, samples, logs):
    package = MODELS[name][0]
    log = logs / f"{name}_s{seed}.log"
    checkpoint = str(directory.relative_to(ROOT) / "best.pt")
    for split in SPLITS:
        path = directory / f"best_{split}.json"
        if not path.exists() or (package == CLIMB and "selection_nll_rules" not in json.loads(path.read_text())):
            run([PYTHON, "-m", package, "evaluate", checkpoint, "--split", split, "--device", "cuda"], log)
        if package == CLIMB and not (directory / f"quality_{split}.json").exists():
            compare = ["--baseline", str(baseline.relative_to(ROOT) / "best.pt")] if name == "climb_v2_d192" else []
            run([PYTHON, "-m", CLIMB, "quality", checkpoint, "--split", split, "--samples", str(samples), *compare], log)


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def collect(runs, logs, output):
    """Copy every run's records into <output>/runs and write one long table of all their numbers."""
    rows, epochs, by_grade, checkpoints = [], [], [], {}
    shutil.rmtree(output / "runs", ignore_errors=True)
    for (name, seed), directory in sorted(runs.items()):
        target = output / "runs" / f"{name}_s{seed}"
        target.mkdir(parents=True)
        for file in RUN_FILES:
            if (directory / file).exists():
                shutil.copy2(directory / file, target / file)
        if (logs / f"{name}_s{seed}.log").exists():
            shutil.copy2(logs / f"{name}_s{seed}.log", target / "run.log")
        checkpoints[f"{name}_s{seed}"] = {"run": str(directory.relative_to(ROOT)), "best.pt": sha256(directory / "best.pt")}

        for line in (directory / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            validation = record["validation"]
            epochs.append({"model": name, "seed": seed, "epoch": record["epoch"], "train_loss": record["train_loss"],
                           "validation": validation.get("selection_nll", validation.get("loss")),
                           "learning_rate": record["learning_rate"], "seconds": record["seconds"]})

        def add(group, metric, split, value):
            rows.append({"model": name, "seed": seed, "group": group, "metric": metric, "split": split, "value": value})

        summary = json.loads((directory / "summary.json").read_text())
        add("training", "best_epoch", "", summary["best_epoch"])
        add("training", "epochs_completed", "", summary["epochs_completed"])
        add("training", "seconds", "", summary["session_seconds"])
        add("training", "parameters", "", summary["parameters"])
        for split in SPLITS:
            if name == "ar_transformer":
                result = json.loads((directory / f"best_{split}.json").read_text())
                add("baseline_tokens", "loss_per_token", split, result["loss"])
                add("baseline_tokens", "token_accuracy", split, result["token_accuracy"])
                continue
            result = json.loads((directory / f"best_{split}.json").read_text())
            for group, values in (("likelihood", result), ("likelihood_null_grade", result.get("null_grade"))):
                if values:
                    add(group, "selection_nll", split, values["selection_nll"])
                    add(group, "selection_nll_rules", split, values["selection_nll_rules"])
                    for mode in LIKELIHOOD_MODES:
                        add(group, mode, split, values[mode]["nll_per_climb"])
                        add(group, f"{mode}_rules", split, values[mode]["nll_rules"])
                        add(group, f"{mode}_token_accuracy", split, values[mode]["token_accuracy"])
            quality = json.loads((directory / f"quality_{split}.json").read_text())
            for method in SAMPLE_METHODS:
                values = quality["samples"].get(method)
                if values:
                    for check in SAMPLE_CHECKS:
                        add(f"samples_{method}", check, split, values[check])
                    for grade, stats in values["by_grade"].items():
                        by_grade.append({"model": name, "seed": seed, "method": method, "split": split,
                                         "grade": int(grade), **stats})
            scratch = quality["likelihood"].get("baseline_scratch_nll_per_climb")
            if scratch is not None:
                rows.append({"model": "ar_transformer", "seed": seed, "group": "likelihood", "metric": "scratch",
                             "split": split, "value": scratch})
    with (output / "all_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, table in (("epochs.csv", epochs), ("samples_by_grade.csv", by_grade)):
        with (output / filename).open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["group"], row["metric"], row["split"]), []).append(row["value"])
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "group", "metric", "split", "seeds", "mean", "std", "min", "max"])
        for key, values in sorted(groups.items()):
            writer.writerow([*key, len(values), statistics.mean(values),
                             statistics.stdev(values) if len(values) > 1 else "", min(values), max(values)])
    (output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jobs", type=int, default=3)
    parser.add_argument("--evaluate-jobs", type=int, default=6)
    parser.add_argument("--samples", type=int, default=50, help="Samples per grade/angle cell")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "compare_models")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild the results from finished runs")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(MODELS), help="Models to leave out")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logs = ROOT / "runs" / "logs"
    wanted = [(name, seed) for name in MODELS for seed in SEEDS if name not in args.skip]
    done = finished_runs()
    if not args.collect_only:
        with ThreadPoolExecutor(args.train_jobs) as pool:
            for future in [pool.submit(train, *key, logs) for key in wanted if key not in done]:
                future.result()
        done = finished_runs()
    runs = {key: done[key] for key in wanted}
    if not args.collect_only:
        with ThreadPoolExecutor(args.evaluate_jobs) as pool:
            jobs = [pool.submit(evaluate, *key, directory, runs["ar_transformer", key[1]], args.samples, logs)
                    for key, directory in runs.items()]
            for future in jobs:
                future.result()
    args.output.mkdir(parents=True, exist_ok=True)
    collect(runs, logs, args.output)
    print(f"Saved {len(runs)} runs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
