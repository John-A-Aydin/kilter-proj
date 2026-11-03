"""Training, checkpointing, and likelihood evaluation. Checkpoints are trusted local files."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .config import Config, ModelConfig, write_json
from .data import EVAL_MODES, PAD, Board, ClimbDataset, Collator, load_arrays, prepare
from .model import NULL_GRADE, ClimbTransformer, parameter_count, weighted_token_loss
from .sampling import LegalTokens


def setup(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


class LengthBucketSampler(Sampler):
    """Shuffle each epoch, group nearby lengths within random pools, then shuffle batches."""

    def __init__(self, dataset, batch_size, generator):
        self.lengths = np.diff(dataset.arrays["offsets"])[dataset.indices]
        self.batch_size = batch_size
        self.generator = generator

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        permutation = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        batches = []
        pool_size = 50 * self.batch_size
        for start in range(0, len(permutation), pool_size):
            pool = sorted(permutation[start:start + pool_size], key=lambda i: self.lengths[i])
            batches.extend(pool[j:j + self.batch_size] for j in range(0, len(pool), self.batch_size))
        for i in torch.randperm(len(batches), generator=self.generator).tolist():
            yield batches[i]


def build_model(model_config, metadata, data_config):
    board = Board.from_json(metadata["board"])
    return ClimbTransformer(model_config, board.features(model_config.coordinate_frequencies),
                            data_config.max_holds + 2), board


def eval_loader(arrays, split, mode, batch_size):
    return DataLoader(ClimbDataset(arrays, split), batch_size=batch_size, shuffle=False,
                      collate_fn=Collator(mode=mode))


@torch.no_grad()
def evaluate(model, arrays, split, device, batch_size=512, max_batches=0, modes=EVAL_MODES, null_grade=False,
             legal=None):
    """Per-climb negative log-likelihood (nats) of the non-given holds and EOS, for each given mode.

    ``scratch`` is the full joint probability of a climb's holds and roles, comparable to any
    model that assigns one sequence per climb, including the baseline.
    ``null_grade`` hides every row's grade from the model. With ``legal``, each mode also gets
    ``nll_rules``: the same score when the decoding rules remove illegal tokens first.
    """
    model.eval()
    results = {}
    for mode in modes:
        total, rule_total, tokens, rows, correct = 0.0, 0.0, 0, 0, 0
        for i, (x, y, conditions, _) in enumerate(eval_loader(arrays, split, mode, batch_size)):
            if max_batches and i >= max_batches:
                break
            if null_grade:
                conditions[:, 0] = NULL_GRADE
            x, y, conditions = x.to(device), y.to(device), conditions.to(device)
            mask = y != PAD
            logits = model(x, conditions, output_mask=mask)
            losses = torch.nn.functional.cross_entropy(logits, y[mask], reduction="sum")
            total += losses.item()
            if legal is not None:
                ruled = logits.masked_fill(~legal(x)[mask], -torch.inf)
                rule_total += torch.nn.functional.cross_entropy(ruled, y[mask], reduction="sum").item()
            correct += (logits.argmax(-1) == y[mask]).sum().item()
            tokens += mask.sum().item()
            rows += len(x)
        results[mode] = {"nll_per_climb": total / rows, "loss_per_token": total / tokens,
                         "token_accuracy": correct / tokens, "rows": rows}
        if legal is not None:
            results[mode]["nll_rules"] = rule_total / rows
    results["selection_nll"] = float(np.mean([results[m]["nll_per_climb"] for m in modes]))
    if legal is not None:
        results["selection_nll_rules"] = float(np.mean([results[m]["nll_rules"] for m in modes]))
    return results


def evaluate_grades(model, arrays, split, device, p_null_grade, max_batches=0, legal=None):
    """Score with real grades, unless the model never saw one, and with null grades if it trained on them."""
    results = evaluate(model, arrays, split, device, max_batches=max_batches, null_grade=p_null_grade == 1,
                       legal=legal)
    if 0 < p_null_grade < 1:
        results["null_grade"] = evaluate(model, arrays, split, device, max_batches=max_batches, null_grade=True,
                                         legal=legal)
    return results


def checkpoint_save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def git_output(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def snapshot_source(run_dir):
    root = Path(__file__).resolve().parent
    shutil.copytree(root, run_dir / "source" / "kilter",
                    ignore=shutil.ignore_patterns("__pycache__"))


def train(config, resume=None):
    device = setup(config.train)
    cache_dir, metadata = prepare(config.data)
    arrays = load_arrays(cache_dir)
    model, board = build_model(config.model, metadata, config.data)
    model.to(device)
    legal = LegalTokens(board, config.data.max_holds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                                  betas=(config.train.beta1, config.train.beta2),
                                  weight_decay=config.train.weight_decay)
    generator = torch.Generator().manual_seed(config.train.seed)
    t = config.train
    collator = Collator(probabilities=(t.p_scratch, t.p_starts, t.p_random), weighting=t.ascent_weighting,
                        seed=t.seed)
    train_dataset = ClimbDataset(arrays, 0)
    train_loader = DataLoader(train_dataset, batch_sampler=LengthBucketSampler(train_dataset, t.batch_size, generator),
                              collate_fn=collator)
    batches = min(len(train_loader), t.max_train_batches or len(train_loader))
    total_steps = batches * t.epochs
    warmup_steps = round(total_steps * t.warmup_fraction)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return t.min_lr_ratio + (1 - t.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start_epoch, global_step, best_loss, stale_epochs = 0, 0, float("inf"), 0
    if resume:
        resume = Path(resume).resolve()
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != config.to_dict() or state["data_fingerprint"] != metadata["fingerprint"]:
            raise ValueError("Resume requires exactly the saved configuration and dataset")
        run_dir = resume.parent
        if (run_dir / "summary.json").exists():
            raise ValueError("Run already completed; use a new run for a new experiment")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, global_step = state["epoch"], state["global_step"]
        best_loss, stale_epochs = state["best_loss"], state["stale_epochs"]
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"])
        generator.set_state(state["loader_state"])
        collator.rng.bit_generator.state = state["collator_state"]
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            kept = [line for line in metrics_path.read_text().splitlines() if json.loads(line)["epoch"] <= start_epoch]
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = Path(config.output_dir) / f"{timestamp}_{config.name}"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config.to_dict())
        write_json(run_dir / "dataset.json", {**metadata, "cache_directory": str(cache_dir.resolve())})
        write_json(run_dir / "model.json", {**asdict(config.model), "parameters": parameter_count(model),
                                           "max_len": model.max_len, "vocab_size": model.vocab_size})
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device), "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output("status", "--short"), "started_at": timestamp,
        })
        snapshot_source(run_dir)
    print(json.dumps({"run": str(run_dir), "device": str(device), "parameters": parameter_count(model),
                      "splits": metadata["splits"], "excluded": metadata["excluded"]}), flush=True)
    session_start = time.perf_counter()
    completed_epochs = start_epoch
    end_epoch = start_epoch if stale_epochs >= t.patience else t.epochs
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, weight_sum = 0.0, 0.0
        for i, (x, y, conditions, weights) in enumerate(train_loader):
            if i >= batches:
                break
            if t.p_null_grade:
                conditions[torch.rand(len(x)) < t.p_null_grade, 0] = NULL_GRADE
            x, y, conditions, weights = x.to(device), y.to(device), conditions.to(device), weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, conditions, output_mask=y != PAD)
            if t.rule_loss:
                logits = logits.masked_fill(~legal(x)[y != PAD], -torch.inf)
            loss, _ = weighted_token_loss(logits, y, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            weight_sum += 1
            global_step += 1
            if (i + 1) % t.log_every == 0:
                print(json.dumps({"epoch": epoch + 1, "batch": i + 1, "batches": batches,
                                  "train_loss": loss_sum / weight_sum,
                                  "seconds": round(time.perf_counter() - epoch_start, 1)}), flush=True)
        validation = evaluate_grades(model, arrays, 1, device, t.p_null_grade, t.max_eval_batches, legal)
        score = validation["selection_nll_rules" if t.rule_loss else "selection_nll"]
        improved = score < best_loss
        if improved:
            best_loss, stale_epochs = score, 0
        else:
            stale_epochs += 1
        record = {"epoch": epoch + 1, "step": global_step, "train_loss": loss_sum / max(1, weight_sum),
                  "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - epoch_start, "partial_training_epoch": batches < len(train_loader)}
        with (run_dir / "metrics.jsonl").open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        state = {
            "config": config.to_dict(), "data_fingerprint": metadata["fingerprint"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "global_step": global_step, "best_loss": best_loss, "stale_epochs": stale_epochs,
            "random_state": random.getstate(), "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(), "loader_state": generator.get_state(),
            "collator_state": collator.rng.bit_generator.state,
        }
        if improved:
            checkpoint_save(run_dir / "best.pt", state)
        checkpoint_save(run_dir / "last.pt", state)
        completed_epochs = epoch + 1
        print(json.dumps(record), flush=True)
        if stale_epochs >= t.patience:
            break
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "name": config.name, "run": str(run_dir), "parameters": parameter_count(model),
        "best_epoch": best["epoch"], "best_validation_selection_nll": best["best_loss"],
        "epochs_completed": completed_epochs, "steps": global_step,
        "session_seconds": time.perf_counter() - session_start, "data_fingerprint": metadata["fingerprint"],
        "partial_training_epochs": batches < len(train_loader),
        "partial_validation": bool(t.max_eval_batches),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return run_dir


def load_run(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = Config.from_dict(state["config"])
    metadata = json.loads((checkpoint.parent / "dataset.json").read_text())
    model, board = build_model(ModelConfig(**state["config"]["model"]), metadata, config.data)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, board, config, metadata


def evaluate_run(checkpoint, split="test", device="cpu"):
    model, board, config, metadata = load_run(checkpoint, device)
    torch.set_num_threads(config.train.threads)
    cache_dir, current = prepare(config.data)
    if current["fingerprint"] != metadata["fingerprint"]:
        raise ValueError("Dataset changed since training")
    result = {"split": split, "checkpoint": str(checkpoint),
              **evaluate_grades(model, load_arrays(cache_dir), {"validation": 1, "test": 2}[split], device,
                                config.train.p_null_grade,
                                legal=LegalTokens(board, config.data.max_holds, device))}
    write_json(Path(checkpoint).parent / f"{Path(checkpoint).stem}_{split}.json", result)
    print(json.dumps(result), flush=True)
    return result
