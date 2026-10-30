"""Sample-quality and likelihood report for a v2 checkpoint, optionally against the kilter/ baseline.

All methods are scored on the same held-out rows and grade/angle cells:

* likelihood: nats per climb for a climb's full hold/role set ("scratch"), which is
  comparable between the baseline's hold-ID order and v2's canonical order;
* from nothing: v2 with and without decoding rules, and the raw baseline sampler;
* from given holds: each held-out climb's start holds plus every other foot, completed by
  v2 (exact conditional) and by the baseline's barrier decoding (the old web method);
* the real held-out climbs themselves, as the reference for every statistic.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import write_json
from .data import BOS, EOS, PAD, START, given_mask, load_arrays, prepare
from .experiment import evaluate_grades, load_run
from .sampling import climb_checks, generate

GRADES = (2, 5, 8, 11)
ANGLES = (20, 40, 50)


def signature(tokens):
    return hashlib.sha1(np.asarray(sorted(int(t) for t in tokens), dtype=np.int32).tobytes()).hexdigest()[:16]


@torch.no_grad()
def baseline_nll(model, rows, batch_size=512):
    """Nats per climb under the kilter/ model: BOS, tokens sorted by hold ID, EOS."""
    total = 0.0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        sequences = [[BOS, *sorted(int(t) for t in tokens), EOS] for tokens, _ in chunk]
        length = max(len(s) for s in sequences) - 1
        x = torch.full((len(chunk), length), PAD)
        y = torch.full((len(chunk), length), PAD)
        for i, s in enumerate(sequences):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])
            y[i, :len(s) - 1] = torch.tensor(s[1:])
        conditions = torch.tensor(np.stack([c for _, c in chunk]))
        mask = y != PAD
        logits = model(x, conditions, output_mask=mask)
        total += torch.nn.functional.cross_entropy(logits, y[mask], reduction="sum").item()
    return total / len(rows)


@torch.no_grad()
def baseline_sample(model, board, grade, angle, required, generator, temperature=1.0, max_holds=32):
    """The kilter/ web decoder: hold-ID order, required holds are barriers that cannot be skipped."""
    conditions = torch.tensor([[grade, angle / 90]])
    required = sorted(required)
    tokens, holds, last, cursor = [BOS], [], -1, 0
    max_holds = max(max_holds, len(required))
    while True:
        logits = model(torch.tensor([tokens]), conditions)[0, -1] / temperature
        allowed = torch.zeros_like(logits, dtype=torch.bool)
        next_token = required[cursor] if cursor < len(required) else None
        remaining = len(required) - cursor
        if len(holds) >= max_holds or max_holds - len(holds) <= remaining:
            allowed[next_token if next_token is not None else EOS] = True
        else:
            upper = board.split(next_token)[0] if next_token is not None else len(board.holds)
            allowed[4 + 4 * (last + 1):4 + 4 * upper] = True
            allowed[next_token if next_token is not None else EOS] = True
        logits[~allowed] = -float("inf")
        token = torch.multinomial(logits.softmax(-1), 1, generator=generator).item()
        if token == EOS:
            return holds, True
        tokens.append(token)
        holds.append(token)
        last = board.split(token)[0]
        cursor += next_token == token


def summarize(climbs, y_of, train_signatures, board):
    """climbs: list of (grade, token list). Fractions of passing checks plus mean statistics by grade."""
    checks, by_grade, signatures = defaultdict(int), defaultdict(lambda: defaultdict(list)), []
    for grade, tokens in climbs:
        holds = board.decode(tokens)
        passed, stats = climb_checks(holds, y_of)
        for key, value in passed.items():
            checks[key] += value
        for key, value in stats.items():
            by_grade[grade][key].append(value)
        signatures.append(signature(tokens))
    n = len(climbs)
    result = {key: checks[key] / n for key in checks}
    result["novel"] = float(np.mean([s not in train_signatures for s in signatures]))
    result["distinct"] = len(set(signatures)) / n
    result["by_grade"] = {str(g): {k: float(np.nanmean(v)) for k, v in stats.items()} for g, stats in sorted(by_grade.items())}
    holds = [result["by_grade"][str(g)]["holds"] for g in GRADES if str(g) in result["by_grade"]]
    result["holds_drop_V2_to_V11"] = holds[0] - holds[-1] if len(holds) > 1 else float("nan")
    result["count"] = n
    return result


def quality_report(checkpoint, baseline=None, split="validation", samples=20, output=None, seed=7):
    torch.set_num_threads(2)
    model, board, config, metadata = load_run(checkpoint)
    arrays = load_arrays(prepare(config.data)[0])
    split_id = {"validation": 1, "test": 2}[split]
    y_of = {h["id"]: h["y"] for h in metadata["board"]}
    train_signatures = set(arrays["signatures"][(arrays["splits"] == 0) | (arrays["splits"] == 255)])
    offsets = arrays["offsets"]
    held = np.flatnonzero(arrays["splits"] == split_id)
    rows = [(arrays["tokens"][offsets[i]:offsets[i + 1]], arrays["conditions"][i]) for i in held]
    base = None
    if baseline:
        from kilter.experiment import load_run as load_baseline
        base = load_baseline(baseline)[0]
    report = {"checkpoint": str(checkpoint), "baseline": str(baseline) if baseline else None, "split": split,
              "samples_per_cell": samples, "grades": GRADES, "angles": ANGLES}
    report["likelihood"] = {"v2": evaluate_grades(model, arrays, split_id, "cpu", config.train.p_null_grade)}
    if base is not None:
        report["likelihood"]["baseline_scratch_nll_per_climb"] = baseline_nll(base, rows)

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    methods = defaultdict(list)
    # A model that always trained on null grades has never seen a real one.
    null_only = config.train.p_null_grade == 1
    for grade in GRADES:
        for angle in ANGLES:
            cell = [i for i, (_, c) in enumerate(rows) if c[0] == grade and round(c[1] * 90) == angle]
            if not cell:
                continue
            chosen = rng.choice(cell, size=min(samples, len(cell)), replace=False)
            for index in chosen:
                tokens = rows[index][0]
                methods["real"].append((grade, list(tokens)))
                mask = given_mask(tokens, "starts_feet")
                given = [(board.holds[board.split(t)[0]], board.split(t)[1]) for t in tokens[mask]]
                completed = generate(model, board, None if null_only else grade, angle, given, generator=generator,
                                     max_holds=config.data.max_holds)
                methods["v2_given_starts_feet"].append((grade, board.encode(completed["holds"])))
                if base is not None:
                    holds, _ = baseline_sample(base, board, grade, angle, [int(t) for t in tokens[mask]], generator)
                    methods["baseline_barrier_given_starts_feet"].append((grade, holds))
            for _ in range(len(chosen)):
                for name, constrained in (("v2_scratch", True), ("v2_scratch_unconstrained", False)):
                    result = generate(model, board, None if null_only else grade, angle, generator=generator,
                                      constrained=constrained, max_holds=config.data.max_holds)
                    methods[name].append((grade, [board.token(board.dense[h["hold_token"]], h["hold_use_type"])
                                                  for h in result["holds"]]))
                if 0 < config.train.p_null_grade < 1:
                    result = generate(model, board, None, angle, generator=generator, max_holds=config.data.max_holds)
                    methods["v2_scratch_null_grade"].append((grade, [board.token(board.dense[h["hold_token"]], h["hold_use_type"])
                                                                     for h in result["holds"]]))
                if base is not None:
                    holds, _ = baseline_sample(base, board, grade, angle, [], generator, max_holds=128)
                    methods["baseline_scratch"].append((grade, holds))
    report["samples"] = {name: summarize(climbs, y_of, train_signatures, board) for name, climbs in methods.items()}
    # Given holds always come from the real climb, so novelty there measures the completion.
    output = Path(output) if output else Path(checkpoint).parent / f"quality_{split}.json"
    write_json(output, report)
    print(json.dumps({name: {k: v for k, v in s.items() if k != "by_grade"} for name, s in report["samples"].items()},
                     indent=2))
    print(json.dumps(report["likelihood"], indent=2))
    return report


__all__ = ["quality_report", "baseline_nll", "baseline_sample", "START"]
