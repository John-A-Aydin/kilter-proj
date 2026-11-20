"""Train the model variants on three seeds and score them on validation, for choosing between them.

Finished work is reused only when its recorded identity matches the request (kilter/identity.py).
A run matches on its resolved config, training source code, dataset and split, hold-label and
style-tag files, seed, and device type. An evaluation matches on its checkpoint, evaluator version
and code, and sampling settings. Anything else is trained or scored again; runs from before
identities existed are never reused. This command never scores the test split: that is
experiments/final_test.py, run once, after the choice is recorded.

Checkpoints stay in the git-ignored runs/ folder. Results go to results/model_selection/<split>/,
where <split> is historical_uuid or the split manifest's name. results/compare_models/ holds the
historical results, made before this workflow, with test scores computed alongside validation.

    .venv/bin/python experiments/compare_models.py --status
    .venv/bin/python experiments/compare_models.py --models climb_v2_null_grade null_rule_16
    .venv/bin/python experiments/compare_models.py --split-manifest data/splits/family_v1.json --models null_rule_16
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kilter.config import load_config, resolve_device  # noqa: E402
from kilter.identity import checkpoint_sha256, differences, read_run_identity, training_identity  # noqa: E402
from kilter.splits import HISTORICAL_SPLIT, read_split_manifest  # noqa: E402

PYTHON = sys.executable
SEEDS = (42, 43, 44)
CLIMB, BASELINE = "kilter", "kilter.baseline"
CLIMB_CONFIG, NULL_CONFIG, BASELINE_CONFIG = "configs/climb.toml", "configs/climb_null_grade.toml", "configs/baseline.toml"
LARGE = ["model.d_model=192", "model.n_layers=6", "model.n_heads=6"]
WIDE = ["model.d_model=256", "model.n_layers=8", "model.n_heads=8"]
RULE_16 = ["train.rule_loss=true", "train.epochs=16"]
TAGS = "model.climb_tags=data/training/climb_tags.csv.gz"
LABELS = "model.hold_labels=data/hold_labels.csv"
# name: (package, config, overrides). Each name is set as the config's name.
MODELS = {
    "ar_transformer": (BASELINE, BASELINE_CONFIG, []),
    "climb_v2": (CLIMB, CLIMB_CONFIG, []),
    "climb_v2_d192": (CLIMB, CLIMB_CONFIG, LARGE),
    "climb_v2_null_grade": (CLIMB, NULL_CONFIG, []),
    "climb_v2_no_grade": (CLIMB, NULL_CONFIG, ["train.p_null_grade=1.0"]),
    # One change each from the 50% null model, chosen from what the runs above showed.
    "null_16_epochs": (CLIMB, NULL_CONFIG, ["train.epochs=16"]),
    "null_d256": (CLIMB, NULL_CONFIG, WIDE),
    "null_ascent_weighted": (CLIMB, NULL_CONFIG, ["train.ascent_weighting=log"]),
    "null_rule_loss": (CLIMB, NULL_CONFIG, ["train.rule_loss=true"]),
    # The changes that helped on validation, together: the served recipe (configs/null_rule_16.toml).
    "null_rule_16": (CLIMB, NULL_CONFIG, RULE_16),
    "null_rule_16_d256": (CLIMB, NULL_CONFIG, [*RULE_16, *WIDE]),
    # The chosen model with each hold's hand-labeled type and grip directions as features.
    "null_rule_16_labels": (CLIMB, NULL_CONFIG, [*RULE_16, LABELS]),
    # The chosen model conditioned on style tags read from climb names and descriptions, without and
    # with the hand-labeled hold features.
    "null_rule_16_tags": (CLIMB, NULL_CONFIG, [*RULE_16, TAGS]),
    "null_rule_16_tags_labels": (CLIMB, NULL_CONFIG, [*RULE_16, TAGS, LABELS]),
}
# The old baseline-against-v2 sample comparison runs on this model only.
BASELINE_COMPARISON = "climb_v2_d192"
RUN_FILES = ("config.json", "identity.json", "dataset.json", "environment.json", "metrics.jsonl", "model.json",
             "summary.json", "hold_labels.json", "climb_tags.json")
LIKELIHOOD_MODES = ("scratch", "starts", "starts_feet")
SAMPLE_METHODS = ("real", "v2_scratch", "v2_scratch_unconstrained", "v2_given_starts_feet",
                  "v2_scratch_null_grade", "baseline_scratch", "baseline_barrier_given_starts_feet")
SAMPLE_CHECKS = ("schema_valid", "constraint_satisfied", "starts_below_finishes", "explicit_feet_present",
                 "no_repeated_hold", "starts_1_2", "finishes_1_2", "within_hold_budget", "ended_with_eos",
                 "legacy_valid_v1", "novel_vs_train", "novel_vs_excluded_duplicates", "novel_vs_all_rows",
                 "novel_hand_set_vs_train", "distinct", "holds_drop_V2_to_V11")


def split_name(split_manifest):
    return read_split_manifest(ROOT / split_manifest, rows=False)["name"] if split_manifest else HISTORICAL_SPLIT


def requested(name, seed, split_manifest, extra=()):
    """The package, train command options, and identity of model ``name`` on ``seed``.

    ``extra`` overrides go to every model, e.g. a few batches for a quick check of the pipeline;
    they change the identity, so such runs are never reused for full ones.
    """
    package, config_path, overrides = MODELS[name]
    overrides = [f"name={name}", *overrides, *extra, f"train.seed={seed}"]
    if split_manifest:
        if package != CLIMB:
            raise SystemExit(f"{name} has no split manifest support; leave it out with --skip {name}")
        overrides.append(f"data.split_manifest={split_manifest}")
    if package == CLIMB:
        config = load_config(ROOT / config_path, overrides)
    else:
        from kilter.baseline.config import load_config as load_baseline_config
        config = load_baseline_config(ROOT / config_path, overrides)
    options = ["--config", config_path, *(item for value in overrides for item in ("--set", value))]
    return package, options, training_identity(package, config)


def recorded_runs():
    """Completed runs that recorded an identity, by run_id (earliest first), and every run's (name, seed)."""
    by_id, by_name = {}, {}
    for path in sorted((ROOT / "runs").glob("*/config.json")):
        directory = path.parent
        config = json.loads(path.read_text())
        identity = read_run_identity(directory)
        complete = (directory / "summary.json").exists()
        by_name.setdefault((config["name"], config["train"]["seed"]), []).append((directory, identity, complete))
        if identity and complete:
            by_id.setdefault(identity["run_id"], directory)
    return by_id, by_name


def explain(key, identity, by_name):
    """Why earlier runs of the same model and seed do not match, one line each."""
    lines = []
    for directory, recorded, complete in by_name.get(key, []):
        where = directory.relative_to(ROOT)
        if recorded is None:
            lines.append(f"{where}: no identity (trained before identities were recorded)")
        elif recorded["run_id"] == identity["run_id"]:
            lines.append(f"{where}: same identity but unfinished; resume with "
                         f"python -m {identity['package']} train --resume {where}/last.pt")
        else:
            changed = differences(identity, recorded)
            lines.append(f"{where}: differs in {', '.join(changed[:6])}{' …' if len(changed) > 6 else ''}")
    return lines


def run(command, log):
    print(" ".join(command), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as output:
        subprocess.run(command, cwd=ROOT, check=True, stdout=output, stderr=subprocess.STDOUT)


def log_path(logs, name, seed, run_id):
    """One log per run identity, so output of different runs of a model and seed never mixes."""
    return logs / f"{name}_s{seed}_{run_id}.log"


def train(name, seed, package, options, identity, logs):
    run([PYTHON, "-m", package, "train", *options], log_path(logs, name, seed, identity["run_id"]))
    directory = recorded_runs()[0].get(identity["run_id"])
    if directory is None:
        raise RuntimeError(f"Training {name} seed {seed} finished, but no run recorded identity "
                           f"{identity['run_id']}; did the code or data change during the sweep?")
    return directory


def evaluation_plan(name, seed, directory, split, runs, samples):
    """(output file, requested identity, command) for each evaluation of one run on ``split``."""
    from kilter.experiment import likelihood_identity
    from kilter.quality import quality_identity
    package = MODELS[name][0]
    checkpoint = directory / "best.pt"
    relative = str(checkpoint.relative_to(ROOT))
    if package == BASELINE:
        from kilter.baseline.experiment import likelihood_identity as baseline_identity
        return [(directory / f"best_{split}.json", baseline_identity(checkpoint, split),
                 [PYTHON, "-m", BASELINE, "evaluate", relative, "--split", split])]
    plan = [(directory / f"best_{split}.json", likelihood_identity(checkpoint, split),
             [PYTHON, "-m", CLIMB, "evaluate", relative, "--split", split])]
    baseline = runs.get(("ar_transformer", seed)) if name == BASELINE_COMPARISON else None
    compare = ["--baseline", str((baseline / "best.pt").relative_to(ROOT))] if baseline else []
    plan.append((directory / f"quality_{split}.json",
                 quality_identity(checkpoint, split, samples, baseline=baseline / "best.pt" if baseline else None),
                 [PYTHON, "-m", CLIMB, "quality", relative, "--split", split, "--samples", str(samples), *compare]))
    return plan


def recorded_evaluation(path):
    return json.loads(path.read_text()).get("evaluation") if path.exists() else None


def evaluate(name, seed, directory, splits, runs, samples, device, logs):
    """Run each evaluation whose saved output does not record exactly the requested identity."""
    for split in splits:
        for path, identity, command in evaluation_plan(name, seed, directory, split, runs, samples):
            if recorded_evaluation(path) == identity:
                continue
            if "evaluate" in command:
                command = [*command, "--device", device]
            run(command, log_path(logs, name, seed, read_run_identity(directory)["run_id"]))
            if recorded_evaluation(path) != identity:
                raise RuntimeError(f"{path} does not record the requested evaluation; did the code change?")


def stale_evaluations(runs, splits, samples):
    """Evaluation files that are missing or record a different identity than requested."""
    return [str(path.relative_to(ROOT)) for (name, seed), directory in sorted(runs.items()) for split in splits
            for path, identity, _ in evaluation_plan(name, seed, directory, split, runs, samples)
            if recorded_evaluation(path) != identity]


def collect(runs, splits, logs, output, description):
    """Copy every run's records into <output>/runs and write one long table of all their numbers."""
    rows, epochs, by_grade, records = [], [], [], {}
    shutil.rmtree(output / "runs", ignore_errors=True)
    for (name, seed), directory in sorted(runs.items()):
        target = output / "runs" / f"{name}_s{seed}"
        target.mkdir(parents=True)
        evaluations = [f"best_{split}.json" for split in splits]
        if MODELS[name][0] == CLIMB:
            evaluations += [f"quality_{split}.json" for split in splits]
        for file in (*RUN_FILES, *evaluations):
            if (directory / file).exists():
                shutil.copy2(directory / file, target / file)
        run_id = read_run_identity(directory)["run_id"]
        if log_path(logs, name, seed, run_id).exists():
            shutil.copy2(log_path(logs, name, seed, run_id), target / "run.log")
        records[f"{name}_s{seed}"] = {
            "run": str(directory.relative_to(ROOT)), "run_id": run_id,
            "best.pt": checkpoint_sha256(directory / "best.pt"),
            "evaluations": {file: json.loads((directory / file).read_text())["evaluation"]["evaluation_id"]
                            for file in evaluations}}

        for line in (directory / "metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            validation = record["validation"]
            epochs.append({"model": name, "seed": seed, "epoch": record["epoch"], "train_loss": record["train_loss"],
                           "validation": validation.get("selection_nll", validation.get("loss")),
                           "learning_rate": record["learning_rate"], "seconds": record["seconds"]})

        def add(group, metric, split, value):
            if value is not None:
                rows.append({"model": name, "seed": seed, "group": group, "metric": metric, "split": split,
                             "value": value})

        def add_likelihood(group, values, split):
            add(group, "selection_nll", split, values["selection_nll"])
            add(group, "selection_nll_rules", split, values["selection_nll_rules"])
            for mode in LIKELIHOOD_MODES:
                add(group, mode, split, values[mode]["nll_per_climb"])
                add(group, f"{mode}_rules", split, values[mode]["nll_rules"])
                add(group, f"{mode}_token_accuracy", split, values[mode]["token_accuracy"])

        summary = json.loads((directory / "summary.json").read_text())
        add("training", "best_epoch", "", summary["best_epoch"])
        add("training", "epochs_completed", "", summary["epochs_completed"])
        add("training", "seconds", "", summary["session_seconds"])
        add("training", "parameters", "", summary["parameters"])
        for split in splits:
            result = json.loads((directory / f"best_{split}.json").read_text())
            if MODELS[name][0] == BASELINE:
                add("baseline_tokens", "loss_per_token", split, result["loss"])
                add("baseline_tokens", "token_accuracy", split, result["token_accuracy"])
                continue
            add_likelihood("likelihood", result, split)
            if result.get("null_grade"):
                add_likelihood("likelihood_null_grade", result["null_grade"], split)
            for band, values in result.get("by_similarity", {}).items():
                add(f"likelihood_{band}", "rows", split, values["rows"])
                add_likelihood(f"likelihood_{band}", values, split)
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
    output.mkdir(parents=True, exist_ok=True)
    for filename, table in (("all_metrics.csv", rows), ("epochs.csv", epochs), ("samples_by_grade.csv", by_grade)):
        if table:
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
    (output / "runs.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (output / "sweep.json").write_text(json.dumps({**description, "splits": list(splits),
                                                   "collected_at": datetime.now(timezone.utc).isoformat()},
                                                  indent=2, sort_keys=True) + "\n")


def parse_models(args):
    names = args.models or list(MODELS)
    unknown = set(names) - set(MODELS)
    if unknown:
        raise SystemExit(f"Unknown models: {sorted(unknown)}")
    return [name for name in names if name not in args.skip]


def resolve(names, seeds, split_manifest, extra=()):
    """Requested (package, options, identity) per (name, seed), and the matching finished runs."""
    wanted = {(name, seed): requested(name, seed, split_manifest, extra) for name in names for seed in seeds}
    by_id, by_name = recorded_runs()
    found = {key: by_id[identity["run_id"]] for key, (_, _, identity) in wanted.items() if identity["run_id"] in by_id}
    return wanted, found, by_name


def print_status(wanted, found, by_name):
    for key, (_, _, identity) in sorted(wanted.items()):
        name, seed = key
        if key in found:
            print(f"{name} seed {seed}: reuse {found[key].relative_to(ROOT)} ({identity['run_id']})")
        else:
            print(f"{name} seed {seed}: train ({identity['run_id']})")
            for line in explain(key, identity, by_name):
                print(f"    {line}")


def common_arguments(parser):
    parser.add_argument("--models", nargs="+", metavar="MODEL", help=f"Models to include (default all): {', '.join(MODELS)}")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(MODELS), help="Models to leave out")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--split-manifest", default="",
                        help="Split manifest path relative to the repository, e.g. data/splits/family_v1.json; "
                             "default is the historical UUID split")
    parser.add_argument("--samples", type=int, default=50, help="Samples per grade/angle cell")
    parser.add_argument("--evaluate-jobs", type=int, default=6)
    parser.add_argument("--device", default="auto", help="Device for likelihood evaluation: cpu, cuda, or auto")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override for every model, e.g. train.max_train_batches=5 for a quick pipeline check")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common_arguments(parser)
    parser.add_argument("--train-jobs", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Default: results/model_selection/<split>/")
    parser.add_argument("--status", action="store_true", help="Show which runs would be reused or trained, and why")
    parser.add_argument("--collect-only", action="store_true", help="Rebuild the results from matching runs only")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    logs = ROOT / "runs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    names = parse_models(args)
    wanted, found, by_name = resolve(names, args.seeds, args.split_manifest, args.set)
    if args.status:
        print_status(wanted, found, by_name)
        return
    missing = [key for key in wanted if key not in found]
    if args.collect_only and missing:
        print_status({key: wanted[key] for key in missing}, found, by_name)
        raise SystemExit("--collect-only needs a matching finished run for every requested model and seed")
    if missing:
        with ThreadPoolExecutor(args.train_jobs) as pool:
            jobs = {key: pool.submit(train, *key, *wanted[key], logs) for key in missing}
            for key, job in jobs.items():
                found[key] = job.result()
    device = resolve_device(args.device)
    if args.collect_only:
        stale = stale_evaluations(found, ("validation",), args.samples)
        if stale:
            raise SystemExit("--collect-only found evaluations to redo:\n  " + "\n  ".join(stale))
    else:
        with ThreadPoolExecutor(args.evaluate_jobs) as pool:
            jobs = [pool.submit(evaluate, *key, directory, ("validation",), found, args.samples, device, logs)
                    for key, directory in found.items()]
            for job in jobs:
                job.result()
    split = split_name(args.split_manifest)
    output = args.output or ROOT / "results" / "model_selection" / split
    collect(found, ("validation",), logs, output, {
        "stage": "model_selection", "split": split, "split_manifest": args.split_manifest or None,
        "models": names, "seeds": args.seeds, "samples_per_cell": args.samples, "overrides": args.set})
    print(f"Saved {len(found)} runs to {output}", flush=True)


if __name__ == "__main__":
    main()
