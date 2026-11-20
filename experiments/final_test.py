"""Score models already chosen on validation on the test split, once, after recording the choice.

Model selection (experiments/compare_models.py) never looks at test. This command is the only
one in the workflow that does. Before scoring anything, it appends to
results/final_test/<split>/access.jsonl: the time, commit, models, seeds, run identities, and the
decision criteria you give. It refuses to open a split's test set a second time unless you pass
--reopen, so every look at test results stays on record. It never trains: every requested run
must already exist, with its validation evaluation done.

    .venv/bin/python experiments/final_test.py --split-manifest data/splits/family_v1.json \\
        --models null_rule_16 climb_v2_null_grade \\
        --criteria "Chose null_rule_16 for the lowest mean validation selection_nll_rules over seeds 42-44"
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import compare_models as sweep
from compare_models import ROOT, parse_models, resolve, split_name


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sweep.common_arguments(parser)
    parser.add_argument("--criteria", required=True,
                        help="How the models were chosen and what result would change a decision, written "
                             "before seeing test results; '@path' reads it from a file")
    parser.add_argument("--reopen", action="store_true", help="Open a test set that has already been opened")
    parser.add_argument("--output", type=Path, help="Default: results/final_test/<split>/")
    args = parser.parse_args()
    if not args.models:
        parser.error("--models is required: name the models chosen on validation")
    os.chdir(ROOT)
    criteria = Path(args.criteria[1:]).read_text().strip() if args.criteria.startswith("@") else args.criteria.strip()
    if not criteria:
        parser.error("--criteria must not be empty")
    logs = ROOT / "runs" / "logs"
    names = parse_models(args)
    wanted, found, by_name = resolve(names, args.seeds, args.split_manifest, args.set)
    missing = [key for key in wanted if key not in found]
    if missing:
        sweep.print_status({key: wanted[key] for key in missing}, found, by_name)
        raise SystemExit("Every model must be trained and evaluated on validation first (compare_models.py)")
    stale = [path for path in sweep.stale_evaluations(found, ("validation",), args.samples)
             if Path(path).name.startswith("best_")]
    if stale:
        raise SystemExit("Validation likelihood is missing or out of date, so no choice can have been made on it:\n  "
                         + "\n  ".join(stale))

    split = split_name(args.split_manifest)
    output = args.output or ROOT / "results" / "final_test" / split
    output.mkdir(parents=True, exist_ok=True)
    log = output / "access.jsonl"
    previous = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if previous and not args.reopen:
        opened = "\n  ".join(f"{p['opened_at']}: {', '.join(p['models'])}" for p in previous)
        raise SystemExit(f"The {split} test set was already opened:\n  {opened}\nPass --reopen to open it again; "
                         "the earlier openings stay on record.")
    entry = {"opened_at": datetime.now(timezone.utc).isoformat(), "split": split,
             "split_manifest": args.split_manifest or None, "git_commit": git("rev-parse", "HEAD"),
             "git_clean": git("status", "--porcelain") == "", "models": names, "seeds": args.seeds,
             "runs": {f"{name}_s{seed}": wanted[name, seed][2]["run_id"] for name, seed in sorted(wanted)},
             "overrides": args.set, "criteria": criteria, "reopened": bool(previous)}
    with log.open("a") as target:
        target.write(json.dumps(entry, sort_keys=True) + "\n")

    device = sweep.resolve_device(args.device)
    with ThreadPoolExecutor(args.evaluate_jobs) as pool:
        jobs = [pool.submit(sweep.evaluate, *key, directory, ("test",), found, args.samples, device, logs)
                for key, directory in found.items()]
        for job in jobs:
            job.result()
    sweep.collect(found, ("test",), logs, output, {
        "stage": "final_test", "split": split, "split_manifest": args.split_manifest or None,
        "models": names, "seeds": args.seeds, "samples_per_cell": args.samples, "overrides": args.set,
        "criteria": criteria})
    print(f"Saved test results for {len(found)} runs to {output}; this opening is in {log}", flush=True)


if __name__ == "__main__":
    main()
