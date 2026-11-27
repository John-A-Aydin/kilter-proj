"""A BERT-style encoder trained end to end to grade climbs: how much does the frozen generator leave behind?

grade_climbs.py grades climbs with heads on the frozen generator (``null_rule_16``). This script
trains a model whose only job is grading, on exactly the same rows and split, with the same inputs:
each hold's identity, role and geometry features, plus the wall angle. No hand-labelled hold types,
since ``null_rule_16`` has none.

The model (``GradeEncoder`` in experiments/graders.py) is a bidirectional pre-LayerNorm transformer
over the climb's holds as a set, with a [CLS] token and the angle at every position. Weights are
averaged over training (EMA), and the averaged weights are evaluated and kept. Some configurations
start with BERT's masked-hold pretraining on the training rows.

Every configuration tried is in ``SWEEP``, in the order it was added; selection is on validation
MSE only (experiments/graders.py has the protocol). Outputs in results/bert_grader/; grade_climbs.py
reads them as the method ``bert``.

    .venv/bin/python experiments/bert_grader.py              # sweep, then final
    .venv/bin/python experiments/bert_grader.py --stage sweep
"""
import argparse
from dataclasses import replace

from graders import ROOT, BertConfig, final, load_data, setup, sweep

RUNS = ROOT / "runs" / "bert_grader"
OUTPUT = ROOT / "results" / "bert_grader"

BASE = BertConfig("base")
# Round 1: one change at a time from BASE.
ROUND_1 = (
    BASE,
    replace(BASE, name="ce", loss="ce"),
    replace(BASE, name="ce_mse", loss="ce_mse"),
    replace(BASE, name="lr_1e-3", learning_rate=1e-3),
    replace(BASE, name="lr_2.5e-4", learning_rate=2.5e-4),
    replace(BASE, name="dropout_0.2", dropout=0.2),
    replace(BASE, name="d384_l8", d_model=384, n_layers=8),
    replace(BASE, name="d192_l4", d_model=192, n_layers=4, n_heads=6),
    replace(BASE, name="epochs_60", epochs=60),
    replace(BASE, name="mlm_20", pretrain_epochs=20),
)
# Round 2, after round 1: the 17-way distribution head (ce) was best, and every run peaked early
# (epoch 6-19 of 30) and then overfit. So: ce with schedules short enough to finish annealing near
# that point, and the regularizers that might let it train longer.
CE = replace(BASE, loss="ce", epochs=12)
ROUND_2 = (
    replace(CE, name="ce_epochs_12"),
    replace(CE, name="ce_epochs_8", epochs=8),
    replace(CE, name="ce_epochs_12_lr_2.5e-4", learning_rate=2.5e-4),
    replace(CE, name="ce_epochs_12_wd_0.2", weight_decay=0.2),
    replace(CE, name="ce_epochs_12_sigma_1", label_sigma=1.0),
    replace(CE, name="ce_epochs_12_log_ascents", ascent_weighting="log"),
    replace(CE, name="ce_epochs_12_d384_l8", d_model=384, n_layers=8),
    replace(CE, name="ce_epochs_12_mlm_20", pretrain_epochs=20),
)
# Round 3, after round 2: masked-hold pretraining was the largest gain (1.717 -> 1.668 validation MSE),
# and a lower learning rate and more weight decay each helped a little. So: more pretraining, a harder
# masking task, those two settings with it, and the larger model, which may need pretraining to pay off.
MLM = replace(CE, pretrain_epochs=20)
ROUND_3 = (
    replace(MLM, name="ce_epochs_12_mlm_50", pretrain_epochs=50),
    replace(MLM, name="ce_epochs_12_mlm_20_lr_2.5e-4_wd_0.2", learning_rate=2.5e-4, weight_decay=0.2),
    replace(MLM, name="ce_epochs_8_mlm_20", epochs=8),
    replace(MLM, name="ce_epochs_12_mlm_20_mask_0.3", mask_fraction=0.3),
    replace(MLM, name="ce_epochs_12_mlm_50_lr_2.5e-4_wd_0.2", pretrain_epochs=50, learning_rate=2.5e-4, weight_decay=0.2),
    replace(MLM, name="ce_epochs_12_d384_l8_mlm_50", d_model=384, n_layers=8, pretrain_epochs=50),
)
SWEEP = ROUND_1 + ROUND_2 + ROUND_3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("sweep", "all"), default="all")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = setup(args.device)
    data = load_data(device)
    chosen, rows = sweep(SWEEP, data, device, RUNS, OUTPUT)
    if args.stage == "all":
        final(chosen, rows, data, device, RUNS, OUTPUT, "BERT-style encoder trained from scratch for grading")


if __name__ == "__main__":
    main()
