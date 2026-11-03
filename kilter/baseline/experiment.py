"""Training, checkpointing, evaluation, and sampling. Checkpoints are trusted local files."""
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
from .data import BOS, EOS, PAD, ClimbDataset, Tokenizer, collate, load_arrays, prepare
from .model import AutoregressiveTransformer, parameter_count, resolve_model, token_loss


def setup(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
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


def make_loader(arrays, split, config, generator=None):
    dataset = ClimbDataset(arrays, split)
    common = dict(generator=generator, num_workers=config.num_workers, collate_fn=collate)
    if split == 0 and config.length_bucketing:
        return DataLoader(dataset, batch_sampler=LengthBucketSampler(dataset, config.batch_size, generator), **common)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=split == 0, **common)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    total_loss, correct, count, rows = 0.0, 0, 0, 0
    for i, (x, y, conditions) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y, conditions = x.to(device), y.to(device), conditions.to(device)
        mask = y != PAD
        logits = model(x, conditions, output_mask=mask)
        targets = y[mask]
        total_loss += token_loss(logits, targets, reduction="sum").item()
        correct += (logits.argmax(-1) == targets).sum().item()
        count += mask.sum().item()
        rows += len(x)
    if not count:
        raise ValueError("Evaluation split has no target tokens")
    loss = total_loss / count
    return {"loss": loss, "perplexity": math.exp(min(loss, 700)),
            "token_accuracy": correct / count, "tokens": count, "rows": rows,
            "partial": bool(max_batches and rows < len(loader.dataset))}


def checkpoint_save(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def git_output(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def snapshot_source(run_dir):
    root = Path(__file__).resolve().parents[2]
    destination = run_dir / "source"
    destination.mkdir()
    for name in ("kilter", "configs", "tests"):
        shutil.copytree(root / name, destination / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        shutil.copy2(root / name, destination / name)
    (destination / "working-tree.diff").write_text(git_output("diff", "HEAD"))


def train(config, resume=None):
    device = setup(config.train)
    cache_dir, metadata = prepare(config.data)
    arrays = load_arrays(cache_dir)
    resolved, target = resolve_model(config.model, metadata["vocab_size"], config.data.max_seq_len,
                                     metadata["splits"]["train"]["rows"])
    model = AutoregressiveTransformer(resolved, metadata["vocab_size"], config.data.max_seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate,
                                  betas=(config.train.beta1, config.train.beta2),
                                  weight_decay=config.train.weight_decay)
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = make_loader(arrays, 0, config.train, generator)
    validation_loader = make_loader(arrays, 1, config.train)
    batches = min(len(train_loader), config.train.max_train_batches or len(train_loader))
    total_steps = batches * config.train.epochs
    warmup_steps = round(total_steps * config.train.warmup_fraction)

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return config.train.min_lr_ratio + (1 - config.train.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2

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
        if device.type == "cuda" and state["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_state"])
        # Remove metrics newer than the last successfully saved epoch.
        metrics_path = run_dir / "metrics.jsonl"
        if metrics_path.exists():
            kept = [line for line in metrics_path.read_text().splitlines()
                    if json.loads(line)["epoch"] <= start_epoch]
            metrics_path.write_text("\n".join(kept) + ("\n" if kept else ""))
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = Path(config.output_dir) / f"{timestamp}_{config.name}"
        run_dir.mkdir(parents=True)
        write_json(run_dir / "config.json", config.to_dict())
        write_json(run_dir / "dataset.json", {**metadata, "cache_directory": str(cache_dir.resolve())})
        write_json(run_dir / "model.json", {**asdict(resolved), "parameters": parameter_count(model),
                                           "target_parameters": target,
                                           "parameters_per_training_row": parameter_count(model) / metadata["splits"]["train"]["rows"]})
        write_json(run_dir / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "device": str(device), "git_commit": git_output("rev-parse", "HEAD"),
            "git_status": git_output("status", "--short"), "started_at": timestamp,
        })
        snapshot_source(run_dir)
    print(json.dumps({"run": str(run_dir), "device": str(device), "parameters": parameter_count(model),
                      "target_parameters": target, "model": asdict(resolved), "splits": metadata["splits"]}), flush=True)
    session_start = time.perf_counter()
    completed_epochs = start_epoch
    end_epoch = start_epoch if stale_epochs >= config.train.patience else config.train.epochs
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, tokens_seen = 0.0, 0
        for i, (x, y, conditions) in enumerate(train_loader):
            if i >= batches:
                break
            x, y, conditions = x.to(device), y.to(device), conditions.to(device)
            optimizer.zero_grad(set_to_none=True)
            mask = y != PAD
            logits = model(x, conditions, output_mask=mask)
            loss = token_loss(logits, y[mask])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            n = mask.sum().item()
            loss_sum += loss.item() * n
            tokens_seen += n
            global_step += 1
            if (i + 1) % config.train.log_every == 0:
                print(json.dumps({"epoch": epoch + 1, "batch": i + 1, "batches": batches,
                                  "train_loss": loss_sum / tokens_seen,
                                  "seconds": round(time.perf_counter() - epoch_start, 1)}), flush=True)
        validation = evaluate(model, validation_loader, device, config.train.max_eval_batches)
        improved = validation["loss"] < best_loss
        if improved:
            best_loss, stale_epochs = validation["loss"], 0
        else:
            stale_epochs += 1
        record = {"epoch": epoch + 1, "step": global_step, "train_loss": loss_sum / tokens_seen,
                  "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - epoch_start,
                  "partial_training_epoch": batches < len(train_loader)}
        with (run_dir / "metrics.jsonl").open("a") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")
        state = {
            "config": config.to_dict(), "model_config": asdict(resolved), "data_fingerprint": metadata["fingerprint"],
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "global_step": global_step, "best_loss": best_loss, "stale_epochs": stale_epochs,
            "random_state": random.getstate(), "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(), "loader_state": generator.get_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        if improved:
            checkpoint_save(run_dir / "best.pt", state)
        checkpoint_save(run_dir / "last.pt", state)
        completed_epochs = epoch + 1
        print(json.dumps(record), flush=True)
        if stale_epochs >= config.train.patience:
            break
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {
        "name": config.name, "run": str(run_dir), "parameters": parameter_count(model),
        "target_parameters": target, "best_epoch": best["epoch"], "best_validation_loss": best["best_loss"],
        "epochs_completed": completed_epochs, "steps": global_step,
        "session_seconds": time.perf_counter() - session_start,
        "data_fingerprint": metadata["fingerprint"],
        "partial_training_epochs": batches < len(train_loader),
        "partial_validation": bool(config.train.max_eval_batches and config.train.max_eval_batches < len(validation_loader)),
    }
    # Test is deliberately opt-in after model selection, to avoid tuning against it.
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return run_dir


def load_run(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = Config.from_dict(state["config"])
    metadata = json.loads((checkpoint.parent / "dataset.json").read_text())
    model = AutoregressiveTransformer(ModelConfig(**state["model_config"]), metadata["vocab_size"], config.data.max_seq_len)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, config, metadata


def evaluate_run(checkpoint, split="test", device="cpu"):
    model, config, metadata = load_run(checkpoint, device)
    torch.set_num_threads(config.train.threads)
    cache_dir, current = prepare(config.data)
    if current["fingerprint"] != metadata["fingerprint"]:
        raise ValueError("Dataset changed since training")
    loader = make_loader(load_arrays(cache_dir), {"validation": 1, "test": 2}[split], config.train)
    result = {"split": split, "checkpoint": str(checkpoint), **evaluate(model, loader, device)}
    write_json(Path(checkpoint).parent / f"{Path(checkpoint).stem}_{split}.json", result)
    print(json.dumps(result), flush=True)
    return result


@torch.no_grad()
def sample(checkpoint, grade, angle, count=1, temperature=1.0, top_k=0, seed=42, device="cpu"):
    if grade < 0 or not 0 <= angle <= 90 or temperature <= 0 or count < 1 or top_k < 0:
        raise ValueError("Invalid sample grade, angle, count, temperature, or top_k")
    model, config, metadata = load_run(checkpoint, device)
    torch.set_num_threads(config.train.threads)
    torch.manual_seed(seed)
    tokenizer = Tokenizer(metadata["holds"])
    conditions = torch.tensor([[grade, angle / 90.0]], device=device, dtype=torch.float32)
    results = []
    for _ in range(count):
        tokens = [BOS]
        for _ in range(config.data.max_seq_len):
            logits = model(torch.tensor([tokens], device=device), conditions)[0, -1] / temperature
            # Special tokens PAD/BOS cannot occur inside a generated climb.
            logits[PAD] = logits[BOS] = -float("inf")
            if top_k:
                threshold = logits.topk(min(top_k, logits.numel())).values[-1]
                logits[logits < threshold] = -float("inf")
            token = torch.multinomial(logits.softmax(-1), 1).item()
            tokens.append(token)
            if token == EOS:
                break
        results.append({"v_grade": grade, "angle": angle, "normalized_angle": angle / 90.0,
                        "ended_with_eos": tokens[-1] == EOS, "frames": tokenizer.decode(tokens)})
    return results
