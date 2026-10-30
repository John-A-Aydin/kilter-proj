"""What do the hold representations of a trained model encode, and does it depend on the role?

The token table is hold + geometry + role, so one fixed vector separates the roles in it for any
model, trained or not. That makes a projection of the table misleading. This script measures that
effect, then looks where role and hold can actually interact:

- context: the model's hidden state just after reading a token in a real climb, averaged per token.
  Starts come first and feet last, so it is also scored with the mean state of each position removed.
- affinity: a token's mean log probability among legal tokens, over contexts where the next hold
  has that role. Illegal tokens are skipped, since a rule-aware model is never trained on them.
- output_bias: the one parameter each token has of its own.

Every number is also computed for an untrained copy of the model, because hidden states keep a
direct path to the input token. Only the gap between the two is learned.

    .venv/bin/python experiments/hold_embeddings.py
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kilter_v2.data import ROLE_NAMES, ClimbDataset, Collator, load_arrays, prepare  # noqa: E402
from kilter_v2.experiment import load_run  # noqa: E402
from kilter_v2.sampling import LegalTokens  # noqa: E402

MODEL = "null_rule_16"
SEEDS = (42, 43, 44)
PLOT_SEED = 43
ROLES = len(ROLE_NAMES)
MIN_TOKEN_COUNT = 30
MIN_HOLD_COUNT = 100
MIN_LEGAL_CONTEXTS = 20


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return corr(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def pca(values, dims):
    centered = values - values.mean(0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return u[:, :dims] * s[:dims], (s[:dims] ** 2 / (s ** 2).sum()).tolist()


def role_counts(arrays):
    """Training uses of every hold in every role, shape (holds, roles)."""
    tokens = arrays["tokens"]
    offsets = arrays["offsets"]
    train = np.flatnonzero(arrays["splits"] == 0)
    used = np.concatenate([tokens[offsets[i]:offsets[i + 1]] for i in train])
    size = int(used.max()) + 1 - 4
    counts = np.bincount(used - 4, minlength=size)
    return counts.reshape(-1, ROLES)


@torch.no_grad()
def collect(model, arrays, rows, device, legal, seed=0):
    """Mean hidden state per token, raw and with each position's mean removed, and affinity by next role."""
    dataset = ClimbDataset(arrays, 0)
    dataset.indices = np.random.default_rng(seed).permutation(dataset.indices)[:rows]
    loader = DataLoader(dataset, batch_size=512, collate_fn=Collator(mode="scratch"))
    hidden = []
    hook = model.norm.register_forward_hook(lambda _, __, output: hidden.append(output))
    width, max_len = model.config.d_model, model.max_len
    vocab = model.vocab_size
    position_sum = torch.zeros(max_len, width, device=device)
    position_count = torch.zeros(max_len, device=device)
    batches = []
    affinity_sum = torch.zeros(ROLES, vocab, device=device)
    affinity_count = torch.zeros(ROLES, vocab, device=device)
    for x, y, conditions, _ in loader:
        x, y, conditions = x.to(device), y.to(device), conditions.to(device)
        logits = model(x, conditions)
        states = hidden.pop()
        held = x >= 4
        positions = torch.arange(x.shape[1], device=device).expand_as(x)
        position_sum.index_add_(0, positions[held], states[held])
        position_count.index_add_(0, positions[held], torch.ones(int(held.sum()), device=device))
        batches.append((x[held].cpu(), positions[held].cpu(), states[held].cpu()))
        next_hold = y >= 4
        roles = (y[next_hold] - 4) % ROLES
        allowed = legal(x)[next_hold]
        log_probs = logits[next_hold].masked_fill(~allowed, -torch.inf).log_softmax(-1).masked_fill(~allowed, 0.0)
        affinity_sum.index_add_(0, roles, log_probs)
        affinity_count.index_add_(0, roles, allowed.float())
    hook.remove()
    position_mean = (position_sum / position_count.clamp(min=1).unsqueeze(1)).cpu()
    raw = torch.zeros(vocab, width)
    adjusted = torch.zeros(vocab, width)
    count = torch.zeros(vocab)
    for tokens, positions, states in batches:
        raw.index_add_(0, tokens, states)
        adjusted.index_add_(0, tokens, states - position_mean[positions])
        count.index_add_(0, tokens, torch.ones(len(tokens)))
    scale = count.clamp(min=1).unsqueeze(1)
    affinity = (affinity_sum / affinity_count.clamp(min=1)).masked_fill(affinity_count < MIN_LEGAL_CONTEXTS, torch.nan).cpu()
    return (raw / scale).numpy(), (adjusted / scale).numpy(), count.numpy(), affinity.numpy()


def centroid_accuracy(vectors, holds, roles, rng, shuffles=20):
    """Nearest role centroid, with the centroids always fit without the hold being classified."""
    def score(labels):
        correct = 0
        for hold in np.unique(holds):
            test = holds == hold
            centroids = np.stack([vectors[~test & (labels == r)].mean(0) for r in range(ROLES)])
            distance = ((vectors[test, None] - centroids[None]) ** 2).sum(-1)
            correct += (distance.argmin(1) == labels[test]).sum()
        return correct / len(labels)

    return score(roles), float(np.mean([score(rng.permutation(roles)) for _ in range(shuffles)]))


def pair_similarity(vectors, holds, roles, x, y):
    """Cosine between tokens after removing each role's mean, by how the two tokens relate."""
    centered = vectors.copy()
    for role in range(ROLES):
        centered[roles == role] -= centered[roles == role].mean(0)
    unit = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    cosine = unit @ unit.T
    upper = np.triu(np.ones_like(cosine, dtype=bool), 1)
    same_hold, same_role = holds[:, None] == holds[None], roles[:, None] == roles[None]
    wall = np.hypot(x[holds][:, None] - x[holds][None], y[holds][:, None] - y[holds][None])
    same_role_pairs = upper & same_role & ~same_hold
    return {"same_hold_different_role": float(cosine[upper & same_hold].mean()),
            "different_hold_same_role": float(cosine[same_role_pairs].mean()),
            "different_hold_different_role": float(cosine[upper & ~same_hold & ~same_role].mean()),
            "same_role_wall_distance_spearman": spearman(wall[same_role_pairs], cosine[same_role_pairs])}


def ridge_r2(features, target, rng, folds=5, dims=20, penalty=1.0):
    """Held-out R^2 of predicting ``target`` from the top principal components of ``features``."""
    z, _ = pca(features, dims)
    z = (z - z.mean(0)) / z.std(0)
    order, predicted = rng.permutation(len(z)), np.zeros(len(z))
    for fold in np.array_split(order, folds):
        train = np.setdiff1d(order, fold)
        mean = target[train].mean()
        weights = np.linalg.solve(z[train].T @ z[train] + penalty * np.eye(z.shape[1]), z[train].T @ (target[train] - mean))
        predicted[fold] = z[fold] @ weights + mean
    return float(1 - ((target - predicted) ** 2).sum() / ((target - target.mean()) ** 2).sum())


def reset(model, seed):
    """Back to the untrained state, keeping the architecture and hold features."""
    torch.manual_seed(seed)
    model.apply(model._initialize)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.reset_parameters()
    torch.nn.init.zeros_(model.output_bias)


def analyze(checkpoint, arrays, rows, device, seed, untrained=False):
    model, board, config, _ = load_run(checkpoint, device)
    if untrained:
        reset(model, seed)
    legal = LegalTokens(board, config.data.max_holds, device)
    rng = np.random.default_rng(0)
    n_holds = len(board.holds)
    with torch.no_grad():
        table = model.token_table()[4:].cpu().numpy()
        identity = model.hold.weight.cpu().numpy()
        geometry = model.geometry(model.hold_features).cpu().numpy()
        role_vectors = model.role.weight.cpu().numpy()
        bias = model.output_bias[4:].cpu().numpy().reshape(n_holds, ROLES)
    holds_vector = identity + geometry
    grid = table.reshape(n_holds, ROLES, -1)
    forced = grid - holds_vector[:, None] - role_vectors[None]
    role_part = ((role_vectors - role_vectors.mean(0)) ** 2).sum(-1).mean()
    hold_part = ((holds_vector - holds_vector.mean(0)) ** 2).sum(-1).mean()

    counts = role_counts(arrays)[:n_holds]
    raw, adjusted, seen, affinity = collect(model, arrays, rows, device, legal)
    keep = np.flatnonzero(seen[4:] >= MIN_TOKEN_COUNT)
    token_hold, token_role = keep // ROLES, keep % ROLES
    x, y = board.x, board.y

    report = {"token_table": {"largest_departure_from_hold_plus_role": float(np.abs(forced).max()),
                              "role_share_of_table_variance": float(role_part / (role_part + hold_part)),
                              "identity_vs_geometry_norm": float(np.linalg.norm(identity, axis=1).mean()
                                                                 / np.linalg.norm(geometry, axis=1).mean())},
              "tokens_kept": int(len(keep)), "tokens_total": int(len(seen) - 4)}

    table_accuracy, table_chance = centroid_accuracy(table[keep], token_hold, token_role, rng)
    raw_accuracy, raw_chance = centroid_accuracy(raw[4:][keep], token_hold, token_role, rng)
    adjusted_accuracy, adjusted_chance = centroid_accuracy(adjusted[4:][keep], token_hold, token_role, rng)
    report["role_from_vector_unseen_hold"] = {
        "token_table": table_accuracy, "context": raw_accuracy, "context_position_removed": adjusted_accuracy,
        "shuffled_labels": float(np.mean([table_chance, raw_chance, adjusted_chance]))}

    report["pairs_context_position_removed"] = pair_similarity(adjusted[4:][keep], token_hold, token_role, x, y)
    report["pairs_token_table"] = pair_similarity(table[keep], token_hold, token_role, x, y)

    used = counts.sum(1) >= MIN_HOLD_COUNT
    log_counts = np.log1p(counts)
    interaction = log_counts - log_counts.mean(1, keepdims=True) - log_counts.mean(0, keepdims=True) + log_counts.mean()
    bias_interaction = bias - bias.mean(1, keepdims=True) - bias.mean(0, keepdims=True) + bias.mean()
    grid_affinity = affinity[:, 4:].reshape(ROLES, n_holds, ROLES)
    role_affinity = np.stack([grid_affinity[r, :, r] for r in range(ROLES)], axis=1)
    both = used & np.isfinite(role_affinity).all(1)
    usable = role_affinity[both]
    usage_interaction = (log_counts[both] - log_counts[both].mean(1, keepdims=True) - log_counts[both].mean(0, keepdims=True)
                         + log_counts[both].mean())
    affinity_interaction = usable - usable.mean(1, keepdims=True) - usable.mean(0, keepdims=True) + usable.mean()
    report["usage"] = {
        "holds_used": int(used.sum()),
        "holds_with_all_four_affinities": int(both.sum()),
        "bias_interaction_vs_usage_spearman": (spearman(bias_interaction[used].ravel(), interaction[used].ravel())
                                               if bias.any() else None),
        "affinity_interaction_vs_usage_spearman": spearman(affinity_interaction.ravel(), usage_interaction.ravel()),
        "role_correlation_usage": np.corrcoef(log_counts[both].T).round(3).tolist(),
        "role_correlation_model": np.corrcoef(usable.T).round(3).tolist(),
        "role_names": ROLE_NAMES}

    foot_share = (counts[used, 1] + 1) / (counts[used].sum(1) + 2)
    target = np.log(foot_share / (1 - foot_share))
    report["foot_share_held_out_r2"] = {
        "wall_position_only": ridge_r2(np.stack([x, y, x * y, x ** 2, y ** 2], 1)[used], target, rng, dims=5),
        "geometry_features": ridge_r2(model.hold_features.cpu().numpy()[used], target, rng),
        "learned_identity": ridge_r2(identity[used], target, rng),
        "learned_geometry": ridge_r2(geometry[used], target, rng),
        "hold_vector": ridge_r2(holds_vector[used], target, rng),
        "hold_vector_shuffled": ridge_r2(holds_vector[used], rng.permutation(target), rng)}

    plots = {"holds": pca(holds_vector, 2), "table": pca(table[keep], 2), "context": pca(raw[4:][keep], 2),
             "context_position_removed": pca(adjusted[4:][keep], 2)}
    report["pca_variance_share"] = {name: share for name, (_, share) in plots.items()}
    tables = {
        "holds": [{"hold": board.holds[i], "x": float(x[i]), "y": float(y[i]), "kickboard": int(board.kickboard[i]),
                   **{f"uses_{ROLE_NAMES[r]}": int(counts[i, r]) for r in range(ROLES)},
                   "pc1": float(plots["holds"][0][i, 0]), "pc2": float(plots["holds"][0][i, 1])}
                  for i in range(n_holds)],
        "tokens": [{"hold": board.holds[token_hold[j]], "role": ROLE_NAMES[token_role[j]], "seen": int(seen[4:][keep[j]]),
                    **{f"{name}_{axis + 1}": float(plots[name][0][j, axis]) for name in ("table", "context", "context_position_removed")
                       for axis in range(2)}}
                   for j in range(len(keep))]}
    return report, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60000, help="Training rows to run through the model")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "hold_embeddings")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoints = json.loads((ROOT / "results" / "checkpoints.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for seed in SEEDS:
        checkpoint = ROOT / checkpoints[f"{MODEL}_s{seed}"]["run"] / "best.pt"
        _, _, config, _ = load_run(checkpoint)
        arrays = load_arrays(prepare(config.data)[0])
        trained, tables = analyze(checkpoint, arrays, args.rows, args.device, seed)
        untrained, control = analyze(checkpoint, arrays, args.rows, args.device, seed, True)
        reports[seed] = {"trained": trained, "untrained": untrained}
        print(json.dumps({seed: reports[seed]}, indent=1), flush=True)
        if seed == PLOT_SEED:
            for name, rows in {**tables, "holds_untrained": control["holds"], "tokens_untrained": control["tokens"]}.items():
                with (args.output / f"{name}.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps({"model": MODEL, "plot_seed": PLOT_SEED, "rows": args.rows,
                                                          "seeds": reports}, indent=2) + "\n")


if __name__ == "__main__":
    main()
