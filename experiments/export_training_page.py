"""Collect the numbers behind the website's "How it works" page from results/.

Writes web/public/training.json. Every value comes from files under results/, so
rerun this after experiments/run_seeds.py or the downstream experiments change them.

    .venv/bin/python experiments/export_training_page.py
"""
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "web/public/training.json"
CHOSEN = "null_rule_16"
SERVED_SEED = 43
MODES = ("scratch", "starts", "starts_feet")
# Runs that score validation with the decoding rules every epoch, so their curves are comparable.
CURVE_MODELS = ("null_16_epochs", "null_rule_loss", "null_rule_16", "null_rule_16_d256")
ABLATIONS = ("climb_v2_null_grade", "null_ascent_weighted", "null_d256", "null_16_epochs", "null_rule_loss",
             "null_rule_16_d256", "null_rule_16")


def rounded(value, digits=4):
    return None if value is None else round(value, digits)


def summary():
    table = {}
    with (RESULTS / "summary.csv").open() as source:
        for row in csv.DictReader(source):
            table[row["model"], row["group"], row["metric"], row["split"]] = (float(row["mean"]), float(row["std"]))
    return table


def mean_std(table, *key):
    mean, std = table[key]
    return {"mean": rounded(mean), "std": rounded(std)}


def metrics(model, seed):
    with (RESULTS / "runs" / f"{model}_s{seed}" / "metrics.jsonl").open() as source:
        return [json.loads(line) for line in source]


def curves():
    output = {}
    for model in CURVE_MODELS:
        by_epoch = defaultdict(list)
        modes = defaultdict(lambda: defaultdict(list))
        best = []
        for seed in (42, 43, 44):
            records = metrics(model, seed)
            best.append(min(records, key=lambda r: r["validation"]["selection_nll_rules"])["epoch"])
            for record in records:
                validation = record["validation"]
                by_epoch[record["epoch"]].append(validation["selection_nll_rules"])
                for mode in MODES:
                    modes[mode][record["epoch"]].append(validation[mode]["nll_rules"])
        # Average only epochs every seed reached, so early stopping can't bend the mean.
        epochs = sorted(e for e, values in by_epoch.items() if len(values) == 3)
        output[model] = {
            "epochs": epochs,
            "selection_nll_rules": [rounded(statistics.mean(by_epoch[e])) for e in epochs],
            "modes": {mode: [rounded(statistics.mean(modes[mode][e])) for e in epochs] for mode in MODES},
            "best_epochs": best,
            "last_epochs": [len(metrics(model, seed)) for seed in (42, 43, 44)],
        }
    return output


def schedule():
    with (RESULTS / "runs" / f"{CHOSEN}_s{SERVED_SEED}" / "config.json").open() as source:
        config = json.load(source)
    train = config["train"]
    steps_per_epoch = metrics(CHOSEN, SERVED_SEED)[0]["step"]
    total = steps_per_epoch * train["epochs"]
    warmup = round(total * train["warmup_fraction"])

    def factor(step):
        # Same formula as kilter_v2/experiment.py.
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return train["min_lr_ratio"] + (1 - train["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2

    points = sorted({*range(0, total + 1, max(1, total // 160)), warmup, total})
    return {
        "steps_per_epoch": steps_per_epoch, "total_steps": total, "warmup_steps": warmup,
        "peak": train["learning_rate"], "epochs": train["epochs"],
        "points": [[step, rounded(train["learning_rate"] * factor(step), 7)] for step in points],
        "train": {k: train[k] for k in ("batch_size", "learning_rate", "weight_decay", "beta1", "beta2",
                                         "warmup_fraction", "min_lr_ratio", "grad_clip", "patience", "epochs",
                                         "p_scratch", "p_starts", "p_random", "p_null_grade", "rule_loss",
                                         "ascent_weighting")},
        "model": config["model"],
    }


def ablations(table):
    rows = []
    for model in ABLATIONS:
        row = {"model": model,
               "validation": mean_std(table, model, "likelihood", "selection_nll_rules", "validation"),
               "test": mean_std(table, model, "likelihood", "selection_nll_rules", "test")}
        for mode in MODES:
            row[mode] = mean_std(table, model, "likelihood", f"{mode}_rules", "test")
        row["valid"] = mean_std(table, model, "samples_v2_scratch", "valid", "test")
        rows.append(row)
    return rows


def null_grade(table):
    rows = []
    for label, model, group in (("Never null", "climb_v2_d192", "likelihood"),
                                ("50% null, real grade", "climb_v2_null_grade", "likelihood"),
                                ("50% null, null grade", "climb_v2_null_grade", "likelihood_null_grade"),
                                ("Always null", "climb_v2_no_grade", "likelihood")):
        rows.append({"label": label, "model": model,
                     **{mode: mean_std(table, model, group, mode, "test") for mode in MODES}})
    return rows


def headline(table):
    baseline = table["ar_transformer", "likelihood", "scratch", "test"] if (
        "ar_transformer", "likelihood", "scratch", "test") in table else None
    return {
        "baseline_scratch_test": rounded(baseline[0]) if baseline else None,
        "v2_large_scratch_test": mean_std(table, "climb_v2_d192", "likelihood", "scratch", "test"),
        "chosen_test": {mode: mean_std(table, CHOSEN, "likelihood", f"{mode}_rules", "test") for mode in MODES},
    }


def samples_by_grade():
    values = defaultdict(list)
    with (RESULTS / "samples_by_grade.csv").open() as source:
        for row in csv.DictReader(source):
            if row["split"] != "test":
                continue
            if row["model"] == CHOSEN and row["method"] in ("real", "v2_scratch", "v2_given_starts_feet"):
                values[row["method"], int(row["grade"])].append(float(row["holds"]))
            elif row["method"] == "baseline_scratch":
                values["baseline_scratch", int(row["grade"])].append(float(row["holds"]))
    output = defaultdict(dict)
    for (method, grade), items in sorted(values.items()):
        output[method][grade] = rounded(statistics.mean(items), 3)
    return output


def validity(table):
    output = []
    for label, model, group in (("Real held-out climbs", CHOSEN, "samples_real"),
                                ("v2 from nothing, rules on", CHOSEN, "samples_v2_scratch"),
                                ("v2 completing starts + feet", CHOSEN, "samples_v2_given_starts_feet"),
                                ("v2 from nothing, no rules (large, no rule loss)", "climb_v2_d192",
                                 "samples_v2_scratch_unconstrained"),
                                ("Baseline from nothing", "climb_v2_d192", "samples_baseline_scratch"),
                                ("Baseline completing starts + feet", "climb_v2_d192",
                                 "samples_baseline_barrier_given_starts_feet")):
        output.append({"label": label, "model": model, "valid": mean_std(table, model, group, "valid", "test"),
                       "starts_1_2": mean_std(table, model, group, "starts_1_2", "test")})
    return output


def grading():
    with (RESULTS / "grading/summary.json").open() as source:
        data = json.load(source)
    labels = {"mean_at_angle": "Training mean at that angle", "bag_linear": "Bag of holds, ridge",
              "untrained_mlp": "Untrained model, MLP head", "untrained_linear": "Untrained model, ridge", "bag_mlp": "Bag of holds, MLP head",
              "model_linear": "Generator features, ridge", "bayes": "Generator, Bayes over grades",
              "model_mlp": "Generator features, MLP head"}
    rows = [{"method": name, "label": labels.get(name, name), "mse": rounded(m["mse"]["mean"]),
             "mse_std": rounded(m["mse"]["std"]), "within_one": rounded(m["within_one_grade"]["mean"])}
            for name, m in data["methods"].items()]
    return {"variance": rounded(data["test_grade_variance"]), "rows": sorted(rows, key=lambda r: -r["mse"])}


def served_best_epoch():
    with (RESULTS / "runs" / f"{CHOSEN}_s{SERVED_SEED}" / "summary.json").open() as source:
        return json.load(source)["best_epoch"]


def main():
    table = summary()
    page = {
        "chosen": CHOSEN, "served_seed": SERVED_SEED, "served_best_epoch": served_best_epoch(),
        "headline": headline(table), "schedule": schedule(), "curves": curves(),
        "ablations": ablations(table), "null_grade": null_grade(table),
        "samples_by_grade": samples_by_grade(), "validity": validity(table), "grading": grading(),
    }
    OUTPUT.write_text(json.dumps(page, indent=1, allow_nan=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
