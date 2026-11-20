"""Sample-quality and likelihood report for a checkpoint, optionally against the baseline.

All methods are scored on the same held-out rows and grade/angle cells:

* likelihood: nats per climb for a climb's full hold/role set ("scratch"), which is
  comparable between the baseline's hold-ID order and v2's canonical order;
* from nothing: v2 with and without decoding rules, and the raw baseline sampler;
* from given holds: each held-out climb's start holds plus every other foot, completed by
  v2 (exact conditional) and by the baseline's barrier decoding (the old web method);
* the real held-out climbs themselves, as the reference for every statistic.

This is evaluator version 2 (``QUALITY_EVALUATOR_VERSION``). Every report records its
evaluation identity (kilter/identity.py) and the exact prompt rows it used. Version 1 reports,
which have no ``"evaluation"`` key (all of results/compare_models/), differ in two ways:

* v1 ``valid`` was one conjunction: no repeated hold, 1-2 starts, 1-2 finishes, every start
  below every finish, and at least one foot. Version 2 reports those separately (see
  ``climb_checks``). ``legacy_valid_v1`` keeps the old conjunction, only for comparison with
  historical tables.
* v1 ``novel`` counted rows removed from training for duplicating a held-out climb (split 255)
  as training climbs. ``novel_vs_train`` uses training rows only; the other reference sets are
  reported separately.

None of these checks says that a climb is climbable, enjoyable, or graded as requested.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .config import Config, write_json
from .data import (BOS, EOS, EXCLUDED, FINISH, FOOT, MAX_FINISHES, MAX_STARTS, PAD, START, data_specification,
                   given_mask)
from .experiment import EVAL_MODES, SPLIT_IDS, evaluate_grades, load_dataset, load_run
from .identity import checkpoint_sha256, evaluation_identity
from .sampling import generate
from .splits import hand_sets

QUALITY_EVALUATOR_VERSION = 2
GRADES = (2, 5, 8, 11)
ANGLES = (20, 40, 50)
PROMPT_SEED = 7
PROMPT_MODE = "starts_feet"
TEMPERATURE = 1.0
TOP_K = 0
CHECKS = ("no_repeated_hold", "starts_1_2", "finishes_1_2", "within_hold_budget", "schema_valid",
          "constraint_satisfied", "starts_below_finishes", "explicit_feet_present", "legacy_valid_v1")
NOVELTY = ("novel_vs_train", "novel_vs_excluded_duplicates", "novel_vs_all_rows", "novel_hand_set_vs_train")


def signature(tokens):
    return hashlib.sha1(np.asarray(sorted(int(t) for t in tokens), dtype=np.int32).tobytes()).hexdigest()[:16]


def climb_checks(holds, y_of, given=(), max_holds=None):
    """Separate properties of one climb (a list of hold dicts), and descriptive statistics.

    * ``schema_valid``: no repeated hold, 1-2 starts, 1-2 finishes, and within the hold budget.
      The decoding rules guarantee these for every climb that ends with EOS.
    * ``constraint_satisfied``: every given (hold_id, role) pair is in the climb with that role.
    * ``starts_below_finishes``: every start is lower on the wall than every finish. Most
      climbs, but not a rule: the decoder does not enforce it.
    * ``explicit_feet_present``: at least one foot-only hold. Descriptive, not validity: campus
      climbs and climbs that use hand holds for feet are legitimate.
    * ``legacy_valid_v1``: v1's ``valid`` conjunction, for comparison with historical results only.

    ``largest_vertical_hand_gap`` sorts hand holds by height alone. It ignores horizontal
    distance and is not a reach or movement measure.
    """
    ids = [h["hold_token"] for h in holds]
    by_role = {r: sorted(y_of[h["hold_token"]] for h in holds if h["hold_use_type"] == r) for r in range(4)}
    present = {(h["hold_token"], h["hold_use_type"]) for h in holds}
    route = sorted(y_of[h["hold_token"]] for h in holds if h["hold_use_type"] != FOOT)
    checks = {
        "no_repeated_hold": len(set(ids)) == len(ids),
        "starts_1_2": 1 <= len(by_role[START]) <= MAX_STARTS,
        "finishes_1_2": 1 <= len(by_role[FINISH]) <= MAX_FINISHES,
        "within_hold_budget": max_holds is None or len(holds) <= max_holds,
    }
    checks["schema_valid"] = all(checks.values())
    checks["constraint_satisfied"] = all((hold, role) in present for hold, role in given)
    checks["starts_below_finishes"] = (bool(by_role[START] and by_role[FINISH])
                                       and by_role[START][-1] < by_role[FINISH][0])
    checks["explicit_feet_present"] = bool(by_role[FOOT])
    checks["legacy_valid_v1"] = (checks["no_repeated_hold"] and checks["starts_1_2"] and checks["finishes_1_2"]
                                 and checks["starts_below_finishes"] and checks["explicit_feet_present"])
    return checks, {"holds": len(holds), "feet": len(by_role[FOOT]),
                    "finish_height": by_role[FINISH][-1] if by_role[FINISH] else float("nan"),
                    "largest_vertical_hand_gap": float(np.diff(route).max()) if len(route) > 1 else float("nan")}


def reference_sets(arrays):
    """Exact hold-and-role signatures and hand sets of the catalogue rows that novelty is measured against."""
    splits, signatures = arrays["splits"], arrays["signatures"]
    hands = hand_sets(arrays)
    return {"train": set(signatures[splits == 0]), "excluded_duplicates": set(signatures[splits == EXCLUDED]),
            "all_rows": set(signatures), "train_hand_sets": {h for h, s in zip(hands, splits) if s == 0}}


def hand_set(tokens):
    return tuple(sorted({(int(t) - 4) // 4 for t in tokens if (int(t) - 4) % 4 != FOOT}))


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


def _means(groups):
    """Mean of each statistic per group, with its count; null where every value is undefined."""
    output = {}
    for key, stats in sorted(groups.items()):
        output[str(key)] = {"count": len(stats["holds"])}
        for name, values in stats.items():
            finite = [v for v in values if not np.isnan(v)]
            output[str(key)][name] = float(np.mean(finite)) if finite else None
    return output


def summarize(climbs, references, board, y_of, max_holds):
    """climbs: dicts with grade, angle, tokens, given [(hold_id, role)], and ended (with EOS).

    Shares of climbs passing each check and each novelty test, plus mean statistics by grade and
    by grade/angle cell. Pooled grade means mix angles in whatever proportion the cells have.
    """
    checks, by_grade, by_cell, signatures, hands = (defaultdict(int), defaultdict(lambda: defaultdict(list)),
                                                    defaultdict(lambda: defaultdict(list)), [], [])
    for climb in climbs:
        passed, stats = climb_checks(board.decode(climb["tokens"]), y_of, climb["given"], max_holds)
        for key, value in passed.items():
            checks[key] += value
        for key, value in stats.items():
            by_grade[climb["grade"]][key].append(value)
            by_cell[f"V{climb['grade']}@{climb['angle']}"][key].append(value)
        signatures.append(signature(climb["tokens"]))
        hands.append(hand_set(climb["tokens"]))
    n = len(climbs)
    result = {key: checks[key] / n for key in CHECKS}
    result["ended_with_eos"] = float(np.mean([c["ended"] for c in climbs]))
    result["novel_vs_train"] = float(np.mean([s not in references["train"] for s in signatures]))
    result["novel_vs_excluded_duplicates"] = float(np.mean([s not in references["excluded_duplicates"]
                                                            for s in signatures]))
    result["novel_vs_all_rows"] = float(np.mean([s not in references["all_rows"] for s in signatures]))
    result["novel_hand_set_vs_train"] = float(np.mean([h not in references["train_hand_sets"] for h in hands]))
    result["distinct"] = len(set(signatures)) / n
    result["by_grade"] = _means(by_grade)
    result["by_cell"] = _means(by_cell)
    holds = [result["by_grade"][str(g)]["holds"] for g in GRADES if str(g) in result["by_grade"]]
    result["holds_drop_V2_to_V11"] = holds[0] - holds[-1] if len(holds) > 1 else None
    result["count"] = n
    return result


def quality_identity(checkpoint, split, samples, seed=PROMPT_SEED, baseline=None):
    """What ``quality_report`` with these arguments would record under "evaluation"."""
    config = Config.from_dict(json.loads((Path(checkpoint).resolve().parent / "config.json").read_text()))
    settings = {"samples_per_cell": samples, "prompt_seed": seed, "grades": list(GRADES), "angles": list(ANGLES),
                "prompt_mode": PROMPT_MODE, "temperature": TEMPERATURE, "top_k": TOP_K,
                "likelihood_modes": list(EVAL_MODES),
                "baseline_checkpoint_sha256": checkpoint_sha256(baseline) if baseline else None}
    sources = [("kilter", ("quality.py",))] + ([("kilter.baseline", ())] if baseline else [])
    return evaluation_identity("quality", QUALITY_EVALUATOR_VERSION, checkpoint, split,
                               data_specification(config.data)[1], settings, sources)


def quality_report(checkpoint, baseline=None, split="validation", samples=20, output=None, seed=PROMPT_SEED):
    torch.set_num_threads(2)
    identity = quality_identity(checkpoint, split, samples, seed, baseline)
    model, board, config, metadata = load_run(checkpoint)
    arrays = load_dataset(config, Path(checkpoint).parent)[2]
    split_id = SPLIT_IDS[split]
    y_of = {h["id"]: h["y"] for h in metadata["board"]}
    references = reference_sets(arrays)
    offsets = arrays["offsets"]
    held = np.flatnonzero(arrays["splits"] == split_id)
    rows = [(arrays["tokens"][offsets[i]:offsets[i + 1]], arrays["conditions"][i]) for i in held]
    base = None
    if baseline:
        from .baseline.experiment import load_run as load_baseline
        base = load_baseline(baseline)[0]
    report = {"evaluation": identity, "checkpoint": str(checkpoint), "baseline": str(baseline) if baseline else None,
              "split": split, "samples_per_cell": samples, "grades": GRADES, "angles": ANGLES}
    report["likelihood"] = {"v2": evaluate_grades(model, arrays, split_id, "cpu", config.train.p_null_grade)}
    if base is not None:
        report["likelihood"]["baseline_scratch_nll_per_climb"] = baseline_nll(base, rows)

    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    methods, prompts = defaultdict(list), {}
    # A model that always trained on null grades has never seen a real one.
    null_only = config.train.p_null_grade == 1
    max_holds = config.data.max_holds

    def add(method, grade, angle, tokens, given=(), ended=True):
        methods[method].append({"grade": grade, "angle": angle, "tokens": [int(t) for t in tokens],
                                "given": list(given), "ended": bool(ended)})

    def add_sample(method, grade, angle, result, given=()):
        add(method, grade, angle, board.encode(result["holds"]), given, result["ended_with_eos"])

    for grade in GRADES:
        for angle in ANGLES:
            cell = [i for i, (_, c) in enumerate(rows) if c[0] == grade and round(c[1] * 90) == angle]
            if not cell:
                continue
            chosen = rng.choice(cell, size=min(samples, len(cell)), replace=False)
            prompts[f"V{grade}@{angle}"] = [str(arrays["row_ids"][held[i]]) for i in chosen]
            for index in chosen:
                tokens = rows[index][0]
                add("real", grade, angle, tokens)
                mask = given_mask(tokens, PROMPT_MODE)
                given = [(board.holds[board.split(t)[0]], board.split(t)[1]) for t in tokens[mask]]
                completed = generate(model, board, None if null_only else grade, angle, given,
                                     temperature=TEMPERATURE, top_k=TOP_K, generator=generator, max_holds=max_holds)
                add_sample("v2_given_starts_feet", grade, angle, completed, given)
                if base is not None:
                    holds, ended = baseline_sample(base, board, grade, angle, [int(t) for t in tokens[mask]], generator)
                    add("baseline_barrier_given_starts_feet", grade, angle, holds, given, ended)
            for _ in range(len(chosen)):
                for name, constrained in (("v2_scratch", True), ("v2_scratch_unconstrained", False)):
                    result = generate(model, board, None if null_only else grade, angle, temperature=TEMPERATURE,
                                      top_k=TOP_K, generator=generator, constrained=constrained, max_holds=max_holds)
                    add_sample(name, grade, angle, result)
                if 0 < config.train.p_null_grade < 1:
                    result = generate(model, board, None, angle, temperature=TEMPERATURE, top_k=TOP_K,
                                      generator=generator, max_holds=max_holds)
                    add_sample("v2_scratch_null_grade", grade, angle, result)
                if base is not None:
                    holds, ended = baseline_sample(base, board, grade, angle, [], generator, max_holds=128)
                    add("baseline_scratch", grade, angle, holds, (), ended)
    report["prompts"] = {"cells": prompts, "sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()}
    report["samples"] = {name: summarize(climbs, references, board, y_of, max_holds) for name, climbs in methods.items()}
    # Given holds always come from the real climb, so novelty there measures the completion.
    output = Path(output) if output else Path(checkpoint).parent / f"quality_{split}.json"
    write_json(output, report)
    print(json.dumps({name: {k: v for k, v in s.items() if k not in {"by_grade", "by_cell"}}
                      for name, s in report["samples"].items()}, indent=2))
    print(json.dumps(report["likelihood"], indent=2))
    return report


__all__ = ["quality_report", "quality_identity", "climb_checks", "baseline_nll", "baseline_sample", "START"]
