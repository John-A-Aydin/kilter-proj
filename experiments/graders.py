"""Shared code for the supervised grading baselines: data, models, training, and outputs.

experiments/bert_grader.py and experiments/finetune_grader.py differ only in their model and their
list of configurations; everything that decides what a run produces lives here. A run's identity
hashes this file (not the scripts), with the configuration, seed, data fingerprint, device type,
torch version, and for a fine-tuned generator its checkpoint. A finished run is reused only for the
same identity, so editing a script's configuration list never invalidates the runs already done.
experiments/lightgbm_grader.py uses the same rows and checks from here, with its own training loop.

Data: exactly the rows and split the generator ``null_rule_16`` trained on (its checkpoint's data
config), checked row by row against its training cache when that cache is still on disk.

Protocol: every configuration in a script's sweep is trained on seed 42 and scored on validation.
The one with the lowest validation MSE (the criterion every grading method uses) is trained on
seeds 42, 43 and 44, and only then is the test split predicted. Outputs: results/<script>/
sweep.csv, summary.json, and test_predictions.npz, which grade_climbs.py reads.
"""
import copy
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kilter.config import Config, write_json  # noqa: E402
from kilter.data import BOS, PAD, SEP, Board, load_arrays, prepare  # noqa: E402
from kilter.experiment import load_run  # noqa: E402
from kilter.identity import canonical, checkpoint_sha256, digest, file_sha256  # noqa: E402
from kilter.model import NULL_GRADE  # noqa: E402

GENERATOR = "null_rule_16"
SEARCH_SEED = 42
SEEDS = (42, 43, 44)
TRAIN, VALIDATION, TEST = 0, 1, 2
GRADES = 17
EVAL_BATCH = 4096
LOSSES = ("mse", "ce", "ce_mse")


# ---------------------------------------------------------------- configurations

@dataclass(frozen=True)
class Training:
    """Settings every grader shares. ``loss``: ``mse`` regresses the standardized grade; ``ce`` is a
    17-way distribution on Gaussian-smoothed targets, predicting its mean; ``ce_mse`` adds the squared
    error of that mean. ``ascent_weighting="log"`` weights a row by log2(1 + ascents), capped at 6."""
    name: str
    loss: str = "mse"
    label_sigma: float = 0.75
    epochs: int = 30
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.02
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ascent_weighting: str = "none"

    def validate(self):
        if self.loss not in LOSSES or self.ascent_weighting not in {"none", "log"}:
            raise ValueError(f"{self.name}: loss must be one of {LOSSES}, ascent_weighting none or log")


@dataclass(frozen=True)
class BertConfig(Training):
    """A bidirectional encoder over the climb's holds. ``pretrain_epochs`` adds BERT's masked-hold
    pretraining first (``mask_fraction`` of the holds chosen: 80% hidden, 10% swapped, 10% unchanged)."""
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.1
    coordinate_frequencies: int = 6
    pretrain_epochs: int = 0
    pretrain_learning_rate: float = 1e-3
    mask_fraction: float = 0.15

    def validate(self):
        super().validate()
        if self.d_model % self.n_heads:
            raise ValueError(f"{self.name}: d_model must be a multiple of n_heads")


@dataclass(frozen=True)
class FineTuneConfig(Training):
    """The generator with a grade head, trained end to end. The head learns at ``learning_rate`` and the
    generator at ``backbone_learning_rate``. ``probe_epochs`` trains the head alone first (linear
    probing, then fine-tuning), with the generator frozen."""
    learning_rate: float = 1e-3
    backbone_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    probe_epochs: int = 0


# ---------------------------------------------------------------- data

def generator_run(seed=SEEDS[0]):
    checkpoints = json.loads((ROOT / "results" / "compare_models" / "checkpoints.json").read_text())
    return ROOT / checkpoints[f"{GENERATOR}_s{seed}"]["run"]


def generator_data_config():
    """The data config ``null_rule_16`` trained with, its paths resolved against the repository root.
    The data fingerprint hashes file contents, not paths, so resolving them changes nothing."""
    data = Config.from_dict(json.loads((generator_run() / "config.json").read_text())).data
    absolute = lambda path: str(ROOT / path) if path and not Path(path).is_absolute() else path
    values = {f.name: getattr(data, f.name) for f in fields(data)}
    for key in ("source", "vocabulary", "cache_dir", "split_manifest"):
        values[key] = absolute(values[key])
    return type(data)(**values)


def same_rows_as_generator(arrays):
    """Whether ``arrays`` are exactly the rows the generator trained on, when its cache is still on disk.

    The data fingerprint also hashes the preprocessing code, so it changes when that code changes
    even if the rows do not. This compares the rows themselves. None when the cache is gone.
    """
    cache = Path(json.loads((generator_run() / "dataset.json").read_text())["cache_directory"])
    if not (cache / "dataset.npz").exists():
        return None
    with np.load(cache / "dataset.npz", allow_pickle=False) as trained:
        keys = ("tokens", "offsets", "conditions", "splits", "row_ids", "uuids", "ascents")
        same = all(trained[k].shape == arrays[k].shape and bool((trained[k] == arrays[k]).all()) for k in keys)
    if not same:
        raise ValueError(f"The encoded rows differ from those {GENERATOR} trained on ({cache})")
    return True


def generator_arrays():
    """The generator's encoded arrays, their metadata, and whether they match its training cache."""
    cache_dir, metadata = prepare(generator_data_config())
    arrays = load_arrays(cache_dir)
    return arrays, metadata, same_rows_as_generator(arrays)


def rows_sha256(row_ids):
    return hashlib.sha256("\n".join(map(str, row_ids)).encode()).hexdigest()


def load_data(device):
    """The generator's rows as padded hold tensors on ``device``, with its board."""
    arrays, metadata, same_rows = generator_arrays()
    offsets, flat = arrays["offsets"], arrays["tokens"]
    lengths = np.diff(offsets)
    tokens = np.zeros((len(lengths), int(lengths.max())), dtype=np.int64)
    for i in range(len(lengths)):
        tokens[i, :lengths[i]] = flat[offsets[i]:offsets[i + 1]]
    rows = {split: np.flatnonzero(arrays["splits"] == split) for split in (TRAIN, VALIDATION, TEST)}
    tensor = lambda values: torch.from_numpy(np.ascontiguousarray(values)).to(device)
    train_grades = arrays["conditions"][rows[TRAIN], 0]
    return {
        "tokens": tensor(tokens), "lengths": tensor(lengths), "angle": tensor(arrays["conditions"][:, 1]),
        "grade": tensor(arrays["conditions"][:, 0]), "ascents": tensor(arrays["ascents"].astype(np.float32)),
        "rows": {split: tensor(index) for split, index in rows.items()}, "split_rows": rows,
        "row_ids": arrays["row_ids"], "board": Board.from_json(metadata["board"]),
        "fingerprint": metadata["fingerprint"], "same_rows_as_generator_training": same_rows,
        "statistics": {"mean": float(train_grades.mean()), "std": float(train_grades.std())},
    }


def batches(data, rows, batch_size, generator):
    """One epoch of training batches: shuffled, then grouped by length within pools of 50 batches, as the
    generator's LengthBucketSampler does, so little of each batch is padding. Yields (tokens, rows)."""
    order = rows[torch.randperm(len(rows), generator=generator).to(rows.device)]
    lengths = data["lengths"][order]
    pool = 50 * batch_size
    groups = []
    for start in range(0, len(order), pool):
        groups.extend(order[start:start + pool][torch.argsort(lengths[start:start + pool], stable=True)].split(batch_size))
    for i in torch.randperm(len(groups), generator=generator).tolist():
        index = groups[i]
        yield data["tokens"][index, :int(data["lengths"][index].max())], index


def steps_per_epoch(rows, batch_size):
    pool = 50 * batch_size
    return sum(math.ceil(min(pool, rows - start) / batch_size) for start in range(0, rows, pool))


# ---------------------------------------------------------------- models

class Block(nn.Module):
    """Pre-LayerNorm transformer block with bidirectional attention over the unpadded tokens."""

    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.projection = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, ff * d), nn.GELU(), nn.Linear(ff * d, d), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attend):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=attend,
                                                   dropout_p=self.dropout.p if self.training else 0.0)
        x = x + self.dropout(self.projection(attention.transpose(1, 2).reshape(batch, length, width)))
        return x + self.mlp(self.norm2(x))


def angle_features(angle):
    """angle/90 with sinusoidal features, as the generator's condition_features does for its angle."""
    scaled = angle.unsqueeze(-1) * (torch.pi * 2.0 ** torch.arange(4, device=angle.device))
    return torch.cat([angle.unsqueeze(-1), scaled.sin(), scaled.cos()], dim=-1)


def outputs(config):
    return 1 if config.loss == "mse" else GRADES


def initialize(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


class GradeEncoder(nn.Module):
    """BERT-style encoder: the climb's holds as a set (no positions, so order cannot matter), a [CLS]
    token, and the angle added at every position. Token embeddings are built like the generator's:
    hold embedding + MLP(hold features) + role embedding. The grade is read from [CLS] and the mean
    hold state."""

    def __init__(self, config, board):
        super().__init__()
        d = config.d_model
        self.config = config
        hold_features = board.features(config.coordinate_frequencies)
        self.register_buffer("hold_features", torch.as_tensor(hold_features, dtype=torch.float32), persistent=False)
        n_holds = self.hold_features.shape[0]
        self.hold = nn.Embedding(n_holds, d)
        self.role = nn.Embedding(4, d)
        self.geometry = nn.Sequential(nn.Linear(self.hold_features.shape[1], d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(d))
        self.mask = nn.Parameter(torch.zeros(d))
        self.angle = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(d, config.n_heads, config.ff_multiplier, config.dropout)
                                     for _ in range(config.n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(d, outputs(config)))
        self.mlm_bias = nn.Parameter(torch.zeros(4 * n_holds))
        self.apply(initialize)

    def token_table(self):
        """One embedding per hold/role token, in token-id order minus the 4 specials."""
        holds = self.hold.weight + self.geometry(self.hold_features)
        return (holds.unsqueeze(1) + self.role.weight.unsqueeze(0)).reshape(-1, holds.shape[1])

    def encode(self, tokens, angle, hidden=None):
        """Hidden states of [CLS] + holds, and which hold positions are real. ``hidden`` masks holds (MLM)."""
        real = tokens != PAD
        x = F.embedding((tokens - 4).clamp(min=0), self.token_table())
        if hidden is not None:
            x = torch.where(hidden.unsqueeze(-1), self.mask.to(x.dtype), x)
        x = torch.cat([self.cls.expand(len(tokens), 1, -1), x], dim=1) + self.angle(angle_features(angle)).unsqueeze(1)
        attend = torch.cat([torch.ones_like(real[:, :1]), real], dim=1)[:, None, None, :]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attend)
        return self.norm(x), real

    def forward(self, tokens, angle):
        x, real = self.encode(tokens, angle)
        weights = real.unsqueeze(-1).to(x.dtype)
        mean = (x[:, 1:] * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.head(torch.cat([x[:, 0], mean], dim=-1)).float()

    def mlm_logits(self, tokens, angle, hidden, chosen):
        """Token logits at the ``chosen`` holds, with the ``hidden`` ones replaced by the mask vector."""
        x, _ = self.encode(tokens, angle, hidden)
        return x[:, 1:][chosen] @ self.token_table().T.to(x.dtype) + self.mlm_bias

    def parameter_groups(self, config):
        return [{"params": list(self.parameters()), "lr": config.learning_rate}]


class FineTunedGenerator(nn.Module):
    """The generator read as grade_climbs.py's probe reads it, with a grade head on top, trained end to end.

    Input is the scratch-mode sequence BOS SEP <holds> with a null grade and the climb's angle. The head
    sees the mean and last hidden states (after the final norm) and the angle, as the frozen probe does.
    """

    def __init__(self, config, generator):
        super().__init__()
        self.config, self.generator = config, generator
        d = generator.config.d_model
        self.head = nn.Sequential(nn.LayerNorm(2 * d + 1), nn.Linear(2 * d + 1, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, 256), nn.GELU(), nn.Linear(256, outputs(config)))
        self.head.apply(initialize)
        self._hidden = None
        generator.norm.register_forward_hook(lambda _, __, output: setattr(self, "_hidden", output))

    def forward(self, holds, angle):
        prefix = torch.tensor([BOS, SEP], device=holds.device).expand(len(holds), 2)
        tokens = torch.cat([prefix, holds], dim=1)
        conditions = torch.stack([torch.full_like(angle, NULL_GRADE), angle], dim=1)
        # An all-false output mask skips the vocabulary projection; the hook keeps the hidden states.
        self.generator(tokens, conditions, output_mask=torch.zeros_like(tokens, dtype=torch.bool))
        hidden, real = self._hidden, tokens != PAD
        weights = real.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * weights).sum(1) / weights.sum(1)
        last = hidden[torch.arange(len(tokens), device=tokens.device), real.sum(1) - 1]
        return self.head(torch.cat([mean, last, angle.unsqueeze(-1).to(hidden.dtype)], dim=-1)).float()

    def parameter_groups(self, config):
        return [{"params": list(self.generator.parameters()), "lr": config.backbone_learning_rate},
                {"params": list(self.head.parameters()), "lr": config.learning_rate}]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------- training

def schedule(step, total, config):
    """Linear warmup, then cosine decay to ``min_lr_ratio`` of each group's learning rate."""
    warmup = round(total * config.warmup_fraction)
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2


def optimizer_for(groups, weight_decay):
    """AdamW with weight decay on weight matrices and embeddings, not on biases, norms or single vectors."""
    split = []
    for group in groups:
        split.append({"params": [p for p in group["params"] if p.dim() >= 2], "lr": group["lr"], "weight_decay": weight_decay})
        split.append({"params": [p for p in group["params"] if p.dim() < 2], "lr": group["lr"], "weight_decay": 0.0})
    return torch.optim.AdamW([g for g in split if g["params"]], betas=(0.9, 0.98), fused=torch.cuda.is_available())


def autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def to_grade(output, config, statistics):
    if config.loss == "mse":
        return output.squeeze(-1) * statistics["std"] + statistics["mean"]
    return output.softmax(-1) @ torch.arange(GRADES, device=output.device, dtype=output.dtype)


def grade_loss(output, grade, weights, config, statistics):
    if config.loss == "mse":
        losses = (output.squeeze(-1) - (grade - statistics["mean"]) / statistics["std"]) ** 2
    else:
        levels = torch.arange(GRADES, device=grade.device, dtype=torch.float32)
        target = torch.exp(-0.5 * ((levels - grade.unsqueeze(-1)) / config.label_sigma) ** 2)
        losses = -(target / target.sum(-1, keepdim=True) * output.log_softmax(-1)).sum(-1)
        if config.loss == "ce_mse":
            losses = losses + ((output.softmax(-1) @ levels - grade) / statistics["std"]) ** 2
    return (losses * weights).sum() / weights.sum()


def row_weights(data, index, config):
    if config.ascent_weighting == "log":
        return torch.log2(1 + data["ascents"][index]).clamp(max=6.0)
    return torch.ones(len(index), device=index.device)


@torch.no_grad()
def predict(model, data, split, statistics, device):
    """Predicted grades for every row of ``split``, in row order (batched by length for speed)."""
    model.eval()
    rows = data["rows"][split]
    by_length = torch.argsort(data["lengths"][rows], stable=True)
    out = torch.empty(len(rows), device=rows.device)
    for positions in by_length.split(EVAL_BATCH):
        index = rows[positions]
        with autocast(device):
            output = model(data["tokens"][index, :int(data["lengths"][index].max())], data["angle"][index])
        out[positions] = to_grade(output, model.config, statistics).float()
    return out.cpu().numpy().astype(np.float64)


def scores(predicted, truth):
    error = predicted - truth
    mse = float(np.mean(error ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(error))),
            "within_one_grade": float(np.mean(np.abs(error) <= 1))}


class EMA:
    """Exponential moving average of a model's parameters, evaluated and kept instead of the raw weights."""

    def __init__(self, model, decay):
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        if isinstance(model, FineTunedGenerator):
            # The copy's hook still points at the original; give it its own.
            self.model.generator.norm._forward_hooks.clear()
            self.model.generator.norm.register_forward_hook(lambda _, __, output: setattr(self.model, "_hidden", output))
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(list(self.model.parameters()), list(model.parameters()), 1 - self.decay)


def mlm_pretrain(model, data, config, seed, device, log):
    """BERT masked-hold pretraining on training rows, updating ``model`` in place."""
    train_rows = data["rows"][TRAIN]
    total = steps_per_epoch(len(train_rows), config.batch_size) * config.pretrain_epochs
    optimizer = optimizer_for([{"params": list(model.parameters()), "lr": config.pretrain_learning_rate}],
                              config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, total, config))
    order = torch.Generator().manual_seed(seed + 1000)
    noise = torch.Generator(device=device).manual_seed(seed + 2000)
    n_tokens = data["board"].vocab_size - 4
    for epoch in range(1, config.pretrain_epochs + 1):
        model.train()
        started, losses, correct, count = time.time(), [], 0, 0
        for tokens, index in batches(data, train_rows, config.batch_size, order):
            real = tokens != PAD
            draw = torch.rand(tokens.shape, device=device, generator=noise)
            chosen = real & (draw < config.mask_fraction)
            # Every climb has at least one chosen hold: its lowest draw.
            chosen[torch.arange(len(tokens), device=device), draw.masked_fill(~real, 2.0).argmin(1)] = True
            kind = torch.rand(tokens.shape, device=device, generator=noise)
            hidden = chosen & (kind < 0.8)
            swapped = chosen & (kind >= 0.8) & (kind < 0.9)
            inputs = torch.where(swapped, torch.randint(4, 4 + n_tokens, tokens.shape, device=device, generator=noise), tokens)
            with autocast(device):
                # As in BERT, every chosen hold is predicted: hidden, swapped for a random one, or unchanged.
                logits = model.mlm_logits(inputs, data["angle"][index], hidden, chosen)
            target = tokens[chosen] - 4
            loss = F.cross_entropy(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach())
            correct += int((logits.argmax(-1) == target).sum())
            count += len(target)
        log({"phase": "pretrain", "epoch": epoch, "mlm_loss": float(torch.stack(losses).mean()),
             "mlm_accuracy": correct / count, "seconds": round(time.time() - started, 1)})


def train_grade(model, data, config, seed, device, log, epochs, groups, run_dir, identity, phase="grade"):
    """Train ``model`` on grade for ``epochs``; keep the EMA weights with the lowest validation MSE."""
    statistics = data["statistics"]
    train_rows = data["rows"][TRAIN]
    truth = data["grade"][data["rows"][VALIDATION]].cpu().numpy().astype(np.float64)
    total = steps_per_epoch(len(train_rows), config.batch_size) * epochs
    optimizer = optimizer_for(groups, config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: schedule(s, total, config))
    parameters = [p for g in groups for p in g["params"]]
    ema = EMA(model, config.ema_decay)
    order = torch.Generator().manual_seed(seed + (0 if phase == "grade" else 3000))
    best = {"mse": math.inf}
    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.time(), []
        for tokens, index in batches(data, train_rows, config.batch_size, order):
            with autocast(device):
                output = model(tokens, data["angle"][index])
            loss = grade_loss(output, data["grade"][index], row_weights(data, index, config), config, statistics)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
            optimizer.step()
            scheduler.step()
            ema.update(model)
            losses.append(loss.detach())
        validation = scores(predict(ema.model, data, VALIDATION, statistics, device), truth)
        log({"phase": phase, "epoch": epoch, "train_loss": float(torch.stack(losses).mean()),
             **{f"validation_{k}": v for k, v in validation.items()},
             "lr": [round(v, 8) for v in scheduler.get_last_lr()], "seconds": round(time.time() - started, 1)})
        if phase == "grade" and validation["mse"] < best["mse"]:
            best = {**validation, "epoch": epoch}
            torch.save({"model": ema.model.state_dict(), "config": asdict(config), "seed": seed,
                        "statistics": statistics, "identity": identity}, run_dir / "best.pt")
    if phase != "grade":
        # Hand the averaged weights on, so the next phase starts from what was evaluated.
        model.load_state_dict(ema.model.state_dict())
    return best


def build(config, data, seed, device):
    """The model for ``config``; a fine-tuned generator starts from the generator checkpoint of ``seed``."""
    if isinstance(config, BertConfig):
        return GradeEncoder(config, data["board"]).to(device), None
    checkpoint = generator_run(seed) / "best.pt"
    generator, _, _, _ = load_run(checkpoint, device)
    return FineTunedGenerator(config, generator).to(device), checkpoint


def run_identity(config, seed, data, device, checkpoint):
    identity = canonical({"kind": type(config).__name__, "config": asdict(config), "seed": seed,
                          "data_fingerprint": data["fingerprint"], "source_sha256": file_sha256(__file__),
                          "device_type": device.type, "torch": torch.__version__,
                          "generator_checkpoint_sha256": checkpoint_sha256(checkpoint) if checkpoint else None})
    return {"run_id": digest(identity)[:16], **identity}


def train_run(config, seed, data, device, runs):
    """Train one configuration on one seed, or reuse a finished run with the same identity."""
    config.validate()
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, checkpoint = build(config, data, seed, device)
    identity = run_identity(config, seed, data, device, checkpoint)
    run_dir = runs / f"{config.name}_s{seed}_{identity['run_id']}"
    if (run_dir / "summary.json").exists():
        print(f"Reusing {run_dir}", flush=True)
        return run_dir, json.loads((run_dir / "summary.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "identity.json", identity)
    started = time.time()
    with (run_dir / "metrics.jsonl").open("w") as metrics:
        def log(entry):
            metrics.write(json.dumps(entry) + "\n")
            metrics.flush()
            print(json.dumps({"run": config.name, "seed": seed, **entry}), flush=True)

        if isinstance(config, BertConfig):
            if config.pretrain_epochs:
                mlm_pretrain(model, data, config, seed, device, log)
        elif config.probe_epochs:
            for p in model.generator.parameters():
                p.requires_grad_(False)
            train_grade(model, data, config, seed, device, log, config.probe_epochs,
                        [{"params": list(model.head.parameters()), "lr": config.learning_rate}], run_dir, identity, "probe")
            for p in model.generator.parameters():
                p.requires_grad_(True)
        best = train_grade(model, data, config, seed, device, log, config.epochs, model.parameter_groups(config),
                           run_dir, identity)
    summary = {"config": asdict(config), "seed": seed, "run_id": identity["run_id"], "parameters": parameter_count(model),
               "best_epoch": best.pop("epoch"), "validation": best, "minutes": round((time.time() - started) / 60, 2)}
    write_json(run_dir / "summary.json", summary)
    return run_dir, summary


def load_best(run_dir, data, device):
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    kind = BertConfig if state["identity"]["kind"] == "BertConfig" else FineTuneConfig
    config = kind(**state["config"])
    model, _ = build(config, data, state["seed"], device)
    model.load_state_dict(state["model"])
    return model.eval(), state["statistics"]


# ---------------------------------------------------------------- stages

def sweep(configs, data, device, runs, output):
    """Every configuration on the search seed; returns the one with the lowest validation MSE and the table."""
    rows = []
    for config in configs:
        _, summary = train_run(config, SEARCH_SEED, data, device, runs)
        rows.append({**asdict(config), "parameters": summary["parameters"], "best_epoch": summary["best_epoch"],
                     **{f"validation_{k}": v for k, v in summary["validation"].items()}, "minutes": summary["minutes"]})
    output.mkdir(parents=True, exist_ok=True)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "sweep.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    chosen = min(configs, key=lambda c: next(r for r in rows if r["name"] == c.name)["validation_mse"])
    print(f"Chosen on validation MSE: {chosen.name}", flush=True)
    return chosen, rows


def final(chosen, sweep_rows, data, device, runs, output, description):
    """Train the chosen configuration on every seed, then predict the test split once per seed."""
    run_records, predictions = [], []
    for seed in SEEDS:
        run_dir, summary = train_run(chosen, seed, data, device, runs)
        model, statistics = load_best(run_dir, data, device)
        predictions.append(predict(model, data, TEST, statistics, device))
        run_records.append({"seed": seed, "run": str(run_dir.relative_to(ROOT)), "run_id": summary["run_id"],
                            "best_epoch": summary["best_epoch"], "parameters": summary["parameters"],
                            "validation": summary["validation"]})
    test_ids = data["row_ids"][data["split_rows"][TEST]]
    np.savez_compressed(output / "test_predictions.npz", predictions=np.stack(predictions), seeds=np.asarray(SEEDS),
                        rows_sha256=rows_sha256(test_ids), data_fingerprint=data["fingerprint"], config=chosen.name)
    write_json(output / "summary.json", {
        "description": description, "generator": GENERATOR, "data_fingerprint": data["fingerprint"],
        "same_rows_as_generator_training": data["same_rows_as_generator_training"],
        "selection": "lowest validation MSE on seed 42 among the configurations in sweep.csv",
        "chosen": asdict(chosen), "seeds": list(SEEDS), "runs": run_records,
        "validation_mean": {k: float(np.mean([r["validation"][k] for r in run_records])) for k in run_records[0]["validation"]},
        "test_rows": len(test_ids), "test_rows_sha256": rows_sha256(test_ids),
        "test_note": "Test scores are computed by grade_climbs.py from test_predictions.npz, with every other method.",
        "sweep": sweep_rows,
    })
    print(json.dumps({"seed_" + str(r["seed"]): r["validation"] for r in run_records}, indent=1))


def setup(device):
    import os
    from kilter.config import resolve_device
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(resolve_device(device))
