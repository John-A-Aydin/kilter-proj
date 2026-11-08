"""Does conditioning on style tags help the model, and does turning a tag on change what it sets?

Style tags (kilter/climb_tags.py) are read from climb names and setter descriptions and given to
``null_rule_16_tags`` and ``null_rule_16_tags_labels`` (the same model with the hand-labeled hold
features) as a multi-hot vector next to grade and angle. Both are trained by compare_models.py
on seeds 42, 43 and 44. This script measures, for every seed:

1. Likelihood. Nats per climb of each held-out test climb's holds and roles, generated from
   nothing with the decoding rules (the "scratch" score), with its real grade. Tag models are
   scored with the climb's true tags and with every tag turned off; the difference is how much
   the tags tell the model. ``null_rule_16`` and ``null_rule_16_labels`` are the reference.
2. A tag recognizer. One logistic regression per tag on the bag of a climb's (hold, role)
   tokens, trained on the training climbs with each tag's examples up-weighted to balance them,
   and scored by ROC AUC on the test climbs. It shows which tags are visible in the holds at all,
   and scores generated climbs.
3. Steering. For each tag, 200 (grade, angle) pairs are drawn from the real climbs with that tag.
   For each pair the model generates one climb with only that tag on and one with every tag off.
   The two sets differ only in the tag. Each climb is scored by the recognizer and by plain
   statistics: holds, feet, hand holds, move length and span (centimeters, from the board's
   size), and the share of hand holds of each labeled type or grip direction. The change from
   off to on is compared with how real climbs with the tag differ from real climbs without it at
   the same grade and angle.

Outputs go to results/tag_conditioning/: summary.json (every number, per seed and averaged) and
examples.json (generated climbs for a few tags, seed 43, for figures).

    .venv/bin/python experiments/tag_conditioning.py
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kilter.climb_tags import TAGS, read_tag_file  # noqa: E402
from kilter.data import FOOT, PAD, ClimbDataset, Collator, load_arrays, prepare  # noqa: E402
from kilter.experiment import load_dataset, load_run  # noqa: E402
from kilter.hold_labels import read_hold_labels  # noqa: E402
from kilter.sampling import LegalTokens, climb_checks, generate_batch  # noqa: E402

SEEDS = (42, 43, 44)
PLOT_SEED = 43
REFERENCE = ("null_rule_16", "null_rule_16_labels")
TAGGED = ("null_rule_16_tags", "null_rule_16_tags_labels")
SAMPLES = 200
# The Kilter Original 12x12 is 12 feet (366 cm) wide; the board drawing is 1080 x 1170.
WIDTH_CM, HEIGHT_CM = 365.8, 365.8 * 1170 / 1080
EXAMPLE_TAGS = ("campus", "traverse", "pinch", "jug", "crimp", "undercling", "reachy", "short_friendly", "endurance")
STATS = ("holds", "feet", "hand_holds", "mean_move_cm", "max_move_cm", "width_cm", "height_cm",
         "pinch_share", "jug_share", "edge_share", "sloper_share", "undercling_share", "sidepull_share")


@torch.no_grad()
def row_nll(model, arrays, split, device, legal, batch_size=512):
    """Nats per climb for every row of ``split``, from nothing, with the decoding rules applied."""
    dataset = ClimbDataset(arrays, split)
    totals = []
    for x, y, conditions, _ in DataLoader(dataset, batch_size=batch_size, collate_fn=Collator(mode="scratch")):
        x, y, conditions = x.to(device), y.to(device), conditions.to(device)
        mask = y != PAD
        logits = model(x, conditions, output_mask=mask).float().masked_fill(~legal(x)[mask], -torch.inf)
        losses = torch.nn.functional.cross_entropy(logits, y[mask], reduction="none")
        totals.append(torch.zeros(len(x), device=device).index_add_(0, mask.nonzero()[:, 0], losses).cpu())
    return torch.cat(totals).numpy(), dataset.indices


def climb_tokens(arrays, rows):
    offsets = arrays["offsets"]
    return [arrays["tokens"][offsets[i]:offsets[i + 1]] for i in rows]


def first_rows(arrays, splits):
    """One row per climb (its first angle) among rows in ``splits``."""
    seen, rows = set(), []
    for i in np.flatnonzero(np.isin(arrays["splits"], splits)):
        if arrays["uuids"][i] not in seen:
            seen.add(arrays["uuids"][i])
            rows.append(i)
    return np.array(rows)


class Recognizer(torch.nn.Module):
    """Per-tag logistic regression on the bag of a climb's tokens."""

    def __init__(self, vocab):
        super().__init__()
        self.weights = torch.nn.EmbeddingBag(vocab, len(TAGS), mode="sum")
        torch.nn.init.zeros_(self.weights.weight)
        self.bias = torch.nn.Parameter(torch.zeros(len(TAGS)))

    def forward(self, flat, offsets):
        return self.weights(flat, offsets) + self.bias


def bag(climbs, device):
    flat = torch.as_tensor(np.concatenate(climbs), dtype=torch.long, device=device)
    offsets = torch.as_tensor(np.cumsum([0] + [len(c) for c in climbs[:-1]]), dtype=torch.long, device=device)
    return flat, offsets


def train_recognizer(climbs, labels, vocab, device, epochs=30, batch=4096, decay=1e-4, seed=0):
    torch.manual_seed(seed)
    model = Recognizer(vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    y = torch.as_tensor(labels, device=device)
    positive = y.mean(0).clamp(min=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(1 - positive) / positive)
    order = np.arange(len(climbs))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            part = order[start:start + batch]
            logits = model(*bag([climbs[i] for i in part], device))
            loss = loss_fn(logits, y[part]) + decay * (model.weights.weight ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def recognize(model, climbs, device, batch=8192):
    out = [model(*bag(climbs[s:s + batch], device)).cpu() for s in range(0, len(climbs), batch)]
    return torch.cat(out).numpy()


def auc(scores, labels):
    """ROC AUC by ranks (ties averaged)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    positive = labels.astype(bool)
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


class ClimbStats:
    """Plain statistics of a climb from its tokens, using the board geometry and the hold labels."""

    def __init__(self, board, labels):
        self.board = board
        self.x = board.x * WIDTH_CM
        self.y = board.y * HEIGHT_CM
        types = [labels["holds"][str(h)]["type"] for h in board.holds]
        grips = [labels["holds"][str(h)]["grip"] for h in board.holds]
        self.kind = {t: np.array([k == t for k in types]) for t in ("pinch", "jug", "edge", "sloper")}
        self.under = np.array([any(135 <= g <= 225 for g in grip) for grip in grips])
        self.side = np.array([any(67.5 <= g <= 112.5 or 247.5 <= g <= 292.5 for g in grip) for grip in grips])

    def __call__(self, tokens):
        dense = (np.asarray(tokens) - 4) // 4
        roles = (np.asarray(tokens) - 4) % 4
        hands = dense[roles != FOOT]
        order = hands[np.argsort(self.y[hands])]
        moves = np.hypot(np.diff(self.x[order]), np.diff(self.y[order])) if len(order) > 1 else np.array([np.nan])
        stats = {"holds": len(dense), "feet": int((roles == FOOT).sum()), "hand_holds": len(hands),
                 "mean_move_cm": float(np.mean(moves)), "max_move_cm": float(np.max(moves)),
                 "width_cm": float(np.ptp(self.x[hands])) if len(hands) else np.nan,
                 "height_cm": float(np.ptp(self.y[hands])) if len(hands) else np.nan}
        for t, mask in self.kind.items():
            stats[f"{t}_share"] = float(mask[hands].mean()) if len(hands) else np.nan
        stats["undercling_share"] = float(self.under[hands].mean()) if len(hands) else np.nan
        stats["sidepull_share"] = float(self.side[hands].mean()) if len(hands) else np.nan
        return stats


def stat_matrix(stats_fn, climbs):
    rows = [stats_fn(c) for c in climbs]
    return np.array([[r[s] for s in STATS] for r in rows], dtype=np.float64)


def real_differences(arrays, rows, tag_matrix, stats, sd):
    """Per tag: how real climbs with the tag differ from those without, at the same grade and angle,
    in units of each statistic's standard deviation."""
    grade = arrays["conditions"][rows, 0]
    angle = np.round(arrays["conditions"][rows, 1] * 90)
    cell = grade * 1000 + angle
    out = {}
    for t, tag in enumerate(TAGS):
        has = tag_matrix[:, t] > 0
        diffs = []
        for c in np.unique(cell[has]):
            inside = cell == c
            if (inside & ~has).sum() < 20:
                continue
            control = np.nanmean(stats[inside & ~has], 0)
            diffs.append((np.nanmean(stats[inside & has], 0) - control, (inside & has).sum()))
        if diffs:
            weights = np.array([n for _, n in diffs], dtype=float)
            out[tag] = (np.sum([d * w for d, w in diffs], 0) / weights.sum() / sd).tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Generated climbs per tag, with it on and off")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tag_conditioning")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = json.loads((ROOT / "results" / "compare_models" / "checkpoints.json").read_text())
    checkpoint = lambda name, seed: ROOT / checkpoints[f"{name}_s{seed}"]["run"] / "best.pt"

    _, board, base_config, _ = load_run(checkpoint("null_rule_16", 42))
    arrays = load_arrays(prepare(base_config.data)[0])
    tags_by_uuid = read_tag_file(ROOT / "data/training/climb_tags.csv.gz")
    tag_rows = np.stack([tags_by_uuid[u] for u in arrays["uuids"]])
    labels = read_hold_labels(ROOT / "data/hold_labels.csv", board)
    stats_fn = ClimbStats(board, labels)
    summary = {"tags": TAGS, "stats": list(STATS), "seeds": list(SEEDS), "samples": args.samples,
               "tag_counts": {tag: int(tag_rows[first_rows(arrays, [0, 1, 2, 255]), t].sum()) for t, tag in enumerate(TAGS)}}

    # 1. Likelihood of held-out climbs.
    test = np.flatnonzero(arrays["splits"] == 2)
    tagged_test = tag_rows[test]
    likelihood = {}
    for name in (*REFERENCE, *TAGGED):
        for seed in SEEDS:
            model, _, config, _ = load_run(checkpoint(name, seed), device)
            legal = LegalTokens(board, config.data.max_holds, device)
            data = load_dataset(config, checkpoint(name, seed).parent)[2]
            nll, rows = row_nll(model, data, 2, device, legal)
            assert np.array_equal(rows, test)
            entry = {"with_tags": nll}
            if config.model.climb_tags:
                hidden = {**data, "conditions": data["conditions"].copy()}
                hidden["conditions"][:, 2:] = 0
                entry["tags_off"] = row_nll(model, hidden, 2, device, legal)[0]
            likelihood[name, seed] = entry
            print(json.dumps({"likelihood": name, "seed": seed, "mean": float(nll.mean())}), flush=True)

    def subset_means(mask):
        out = {}
        for name in (*REFERENCE, *TAGGED):
            for key in ("with_tags", "tags_off"):
                values = [float(likelihood[name, s][key][mask].mean()) for s in SEEDS if key in likelihood[name, s]]
                if values:
                    out[f"{name}:{key}"] = {"mean": float(np.mean(values)), "values": values}
        return out

    any_tag = tagged_test.sum(1) > 0
    summary["likelihood"] = {"all": {"rows": int(len(test)), **subset_means(np.ones(len(test), bool))},
                             "tagged": {"rows": int(any_tag.sum()), **subset_means(any_tag)},
                             "untagged": {"rows": int((~any_tag).sum()), **subset_means(~any_tag)},
                             "per_tag": {tag: {"rows": int(tagged_test[:, t].sum()), **subset_means(tagged_test[:, t] > 0)}
                                         for t, tag in enumerate(TAGS) if tagged_test[:, t].sum() >= 10}}

    # 2. The tag recognizer, on one row per climb.
    train_rows, test_rows = first_rows(arrays, [0]), first_rows(arrays, [2])
    recognizer = train_recognizer(climb_tokens(arrays, train_rows), tag_rows[train_rows], board.vocab_size, device)
    test_scores = recognize(recognizer, climb_tokens(arrays, test_rows), device)
    summary["recognizer"] = {
        "train_climbs": int(len(train_rows)), "test_climbs": int(len(test_rows)),
        "auc": {tag: auc(test_scores[:, t], tag_rows[test_rows, t]) for t, tag in enumerate(TAGS)},
        "test_positives": {tag: int(tag_rows[test_rows, t].sum()) for t, tag in enumerate(TAGS)},
        # Mean logit of real test climbs with the tag minus those without: the gap to compare with.
        "real_logit_gap": {tag: float(test_scores[tag_rows[test_rows, t] > 0, t].mean() - test_scores[tag_rows[test_rows, t] == 0, t].mean())
                           for t, tag in enumerate(TAGS) if tag_rows[test_rows, t].sum()},
    }
    print(json.dumps({"recognizer_auc": summary["recognizer"]["auc"]}), flush=True)

    # Real reference for the statistics, over every climb (first angle).
    all_rows = first_rows(arrays, [0, 1, 2, 255])
    real_stats = stat_matrix(stats_fn, climb_tokens(arrays, all_rows))
    sd = np.nanstd(real_stats, 0)
    summary["stat_sd"] = sd.tolist()
    summary["stat_mean"] = np.nanmean(real_stats, 0).tolist()
    summary["real_stat_difference"] = real_differences(arrays, all_rows, tag_rows[all_rows], real_stats, sd)

    # 3. Steering: each tag on versus every tag off, at grade/angle pairs drawn from real climbs with the tag.
    steering, examples = {}, {}
    for name in TAGGED:
        for seed in SEEDS:
            model, _, config, _ = load_run(checkpoint(name, seed), device)
            per_tag = {}
            for t, tag in enumerate(TAGS):
                rng = np.random.default_rng(1000 * seed + t)
                pool = np.flatnonzero(tag_rows[:, t] > 0)
                picks = rng.choice(pool, args.samples)
                base = torch.as_tensor(arrays["conditions"][picks, :2], dtype=torch.float32)
                off = torch.cat([base, torch.zeros(len(base), len(TAGS))], 1)
                on = off.clone()
                on[:, 2 + t] = 1
                generator = torch.Generator(device=device).manual_seed(1000 * seed + t)
                climbs = generate_batch(model, board, torch.cat([on, off]).to(device), generator, max_holds=config.data.max_holds)
                climbs_on, climbs_off = climbs[:len(picks)], climbs[len(picks):]
                logits_on = recognize(recognizer, [np.array(c) for c in climbs_on], device)
                logits_off = recognize(recognizer, [np.array(c) for c in climbs_off], device)
                stats_on, stats_off = stat_matrix(stats_fn, climbs_on), stat_matrix(stats_fn, climbs_off)
                y_of = dict(zip(board.holds, board.y))
                valid = lambda cs: float(np.mean([climb_checks(board.decode(c), y_of)[0]["valid"] for c in cs]))
                per_tag[tag] = {
                    "logit_shift": (logits_on.mean(0) - logits_off.mean(0)).tolist(),
                    "probability_on": float((1 / (1 + np.exp(-logits_on[:, t]))).mean()),
                    "probability_off": float((1 / (1 + np.exp(-logits_off[:, t]))).mean()),
                    "stat_difference": ((np.nanmean(stats_on, 0) - np.nanmean(stats_off, 0)) / sd).tolist(),
                    "stat_on": np.nanmean(stats_on, 0).tolist(), "stat_off": np.nanmean(stats_off, 0).tolist(),
                    "valid_on": valid(climbs_on), "valid_off": valid(climbs_off),
                }
                if seed == PLOT_SEED and tag in EXAMPLE_TAGS:
                    examples.setdefault(name, {})[tag] = {
                        "conditions": [{"grade": float(g), "angle": round(float(a) * 90)} for g, a in base[:6].tolist()],
                        "on": [board.decode(c) for c in climbs_on[:6]], "off": [board.decode(c) for c in climbs_off[:6]]}
            steering[name, seed] = per_tag
            print(json.dumps({"steering": name, "seed": seed, "diagonal_logit_shift": {
                tag: round(per_tag[tag]["logit_shift"][t], 2) for t, tag in enumerate(TAGS)}}), flush=True)

    summary["steering"] = {}
    for name in TAGGED:
        averaged = {}
        for tag in TAGS:
            runs = [steering[name, s][tag] for s in SEEDS]
            averaged[tag] = {key: (np.mean([r[key] for r in runs], 0).tolist() if isinstance(runs[0][key], list)
                                   else float(np.mean([r[key] for r in runs]))) for key in runs[0]}
            averaged[tag]["target_logit_shift_by_seed"] = [r["logit_shift"][TAGS.index(tag)] for r in runs]
        real = summary["real_stat_difference"]
        pairs = [(averaged[tag]["stat_difference"][k], real[tag][k]) for tag in TAGS if tag in real for k in range(len(STATS))]
        generated, observed = np.array(pairs).T
        keep = np.isfinite(generated) & np.isfinite(observed)
        summary["steering"][name] = {"per_tag": averaged,
                                     "stat_correlation_with_real": float(np.corrcoef(generated[keep], observed[keep])[0, 1])}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.output / "examples.json").write_text(json.dumps(examples) + "\n")
    print(json.dumps({name: summary["steering"][name]["stat_correlation_with_real"] for name in TAGGED}))


if __name__ == "__main__":
    main()
