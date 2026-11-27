"""The generator fine-tuned end to end for grading: what does freezing it cost?

grade_climbs.py reads the frozen generator (``null_rule_16``) with an MLP head. This script trains
the same generator, read the same way (scratch-mode sequence, null grade, the mean and last hidden
states plus the angle), with a grade head, updating every weight. Seed s starts from the generator
trained on seed s. The head learns at ``learning_rate`` and the generator at
``backbone_learning_rate``; ``probe_epochs`` first trains the head alone on the frozen generator.

Comparing it with the frozen probe measures what freezing costs; comparing it with
experiments/bert_grader.py (trained from scratch for grading only) measures what starting from a
generator gives or takes away. Training, selection on validation, and outputs are shared with that
script (experiments/graders.py). Outputs in results/finetune_grader/; grade_climbs.py reads them as
the method ``finetuned``.

    .venv/bin/python experiments/finetune_grader.py              # sweep, then final
    .venv/bin/python experiments/finetune_grader.py --stage sweep
"""
import argparse
from dataclasses import replace

from graders import ROOT, FineTuneConfig, final, load_data, setup, sweep

RUNS = ROOT / "runs" / "finetune_grader"
OUTPUT = ROOT / "results" / "finetune_grader"

BASE = FineTuneConfig("base")
# Every configuration tried, in order. Selection reads validation only.
SWEEP = (
    BASE,
    replace(BASE, name="backbone_lr_3e-5", backbone_learning_rate=3e-5),
    replace(BASE, name="backbone_lr_3e-4", backbone_learning_rate=3e-4),
    replace(BASE, name="probe_2", probe_epochs=2),
    replace(BASE, name="epochs_20", epochs=20),
    replace(BASE, name="ce", loss="ce"),
    # Round 2, after round 1: the distribution head (ce) was best; vary it as the BERT sweep does.
    replace(BASE, name="ce_epochs_15", loss="ce", epochs=15),
    replace(BASE, name="ce_sigma_1", loss="ce", label_sigma=1.0),
    replace(BASE, name="ce_log_ascents", loss="ce", ascent_weighting="log"),
    replace(BASE, name="ce_head_lr_3e-4", loss="ce", learning_rate=3e-4),
    replace(BASE, name="ce_wd_0.05", loss="ce", weight_decay=0.05),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("sweep", "all"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = setup(args.device)
    data = load_data(device)
    chosen, rows = sweep(SWEEP, data, device, RUNS, OUTPUT)
    if args.stage == "all":
        final(chosen, rows, data, device, RUNS, OUTPUT, "null_rule_16 fine-tuned end to end for grading")


if __name__ == "__main__":
    main()
