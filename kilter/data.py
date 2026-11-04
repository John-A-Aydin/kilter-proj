"""Board geometry, canonical climb order, filtering, and given/rest training examples.

A climb is serialized as ``BOS <given holds> SEP <remaining holds> EOS``. Both
blocks list hold/role tokens in canonical order: start, hand, finish, foot,
and within each role bottom to top, then left to right. Loss covers only the
remaining holds and EOS, so the model learns p(rest | given, grade, angle).
Given sets are resampled every epoch, covering generation from nothing, from
start holds plus some feet, and from arbitrary picks.
"""
from collections import Counter
import csv
import gzip
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import write_json

PAD, BOS, EOS, SEP = range(4)
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "SEP"]
HAND, FOOT, START, FINISH = range(4)
ROLE_NAMES = ["hand", "foot", "start", "finish"]
# Generation order: anchor the start, climb the hands to the finish, then place feet.
ROLE_ORDER = {START: 0, HAND: 1, FINISH: 2, FOOT: 3}
MAX_STARTS = MAX_FINISHES = 2
CACHE_VERSION = 1
EVAL_MODES = ("scratch", "starts", "starts_feet")


def open_lines(path):
    """Open a JSONL source as text, decompressing it if it ends in .gz."""
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Board:
    """Hold geometry and the canonical token order shared by training, sampling, and the web app."""

    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["id"])
        self.holds = [r["id"] for r in rows]
        if len(set(self.holds)) != len(self.holds):
            raise ValueError("Duplicate physical holds in vocabulary")
        self.dense = {hold: i for i, hold in enumerate(self.holds)}
        self.x = np.array([r["x"] for r in rows], dtype=np.float32)
        self.y = np.array([r["y"] for r in rows], dtype=np.float32)
        self.kickboard = np.array([r["kickboard"] for r in rows], dtype=np.float32)
        self.screw_on = np.array([r["set"] == 20 for r in rows], dtype=np.float32)
        self.vocab_size = 4 + 4 * len(rows)
        order = sorted(((ROLE_ORDER[role], float(self.y[i]), float(self.x[i]), i, role)
                        for i in range(len(rows)) for role in range(4)))
        self.rank = np.full(self.vocab_size, -1, dtype=np.int64)
        for position, (*_, i, role) in enumerate(order):
            self.rank[self.token(i, role)] = position

    @classmethod
    def from_csv(cls, path):
        with Path(path).open(newline="") as source:
            return cls([{"id": int(r["hold_token"]), "x": float(r["hold_x"]), "y": float(r["hold_y"]),
                         "kickboard": int(r["on_kickboard"]), "set": int(r.get("set_id") or 0)}
                        for r in csv.DictReader(source)])

    @staticmethod
    def token(dense, role):
        return 4 + 4 * dense + role

    @staticmethod
    def split(token):
        return divmod(int(token) - 4, 4)

    def encode(self, holds):
        """Hold dicts or (hold_id, role) pairs to canonically ordered tokens."""
        pairs = [(h["hold_token"], h["hold_use_type"]) if isinstance(h, dict) else tuple(h) for h in holds]
        if any(role not in range(4) for _, role in pairs):
            raise ValueError("Unknown hold role")
        return sorted((self.token(self.dense[hold], role) for hold, role in pairs), key=lambda t: self.rank[t])

    def decode(self, tokens):
        holds = []
        for token in tokens:
            if int(token) >= 4:
                dense, role = self.split(token)
                holds.append({"hold_token": self.holds[dense], "hold_use_type": role})
        return holds

    def features(self, frequencies):
        """Per-hold geometric features: raw position flags plus Fourier features of x and y."""
        columns = [self.x, self.y, self.kickboard, self.screw_on]
        for k in range(frequencies):
            for value in (self.x, self.y):
                columns += [np.sin(2**k * np.pi * value), np.cos(2**k * np.pi * value)]
        return np.stack(columns, axis=1).astype(np.float32)

    def to_json(self):
        return [{"id": h, "x": float(self.x[i]), "y": float(self.y[i]), "kickboard": bool(self.kickboard[i]),
                 "set": 20 if self.screw_on[i] else 1} for i, h in enumerate(self.holds)]

    @classmethod
    def from_json(cls, holds):
        return cls([{"id": h["id"], "x": h["x"], "y": h["y"], "kickboard": int(h["kickboard"]), "set": h["set"]}
                    for h in holds])


def v_grade(label):
    if label is None or label == "":
        return None
    if not re.fullmatch(r"V\d+", label):
        raise ValueError(f"Unsupported V grade: {label!r}")
    return float(label[1:])


def split_for_uuid(uuid, config):
    # Identical to kilter.data.split_for_uuid so held-out rows match the baseline.
    digest = hashlib.sha256(f"{config.split_seed}:{uuid}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return 0 if value < config.train_fraction else (1 if value < config.train_fraction + config.validation_fraction else 2)


def climb_problem(frames, max_holds):
    """Return why a climb is outside the generated climb family, or None."""
    if len(frames) != 1:
        return "multi_frame"
    holds = frames[0]
    ids = [h["hold_token"] for h in holds]
    roles = Counter(h["hold_use_type"] for h in holds)
    if len(set(ids)) != len(ids):
        return "repeated_hold"
    if not 1 <= roles[START] <= MAX_STARTS:
        return "start_count"
    if not 1 <= roles[FINISH] <= MAX_FINISHES:
        return "finish_count"
    if len(holds) > max_holds:
        return "too_many_holds"
    return None


def prepare(config):
    """Encode and cache the dataset, keyed by the contents of every input and this file."""
    board = Board.from_csv(config.vocabulary)
    specification = {
        "cache_version": CACHE_VERSION,
        "source_sha256": sha256(config.source),
        "vocabulary_sha256": sha256(config.vocabulary),
        "preprocessing_sha256": sha256(__file__),
        "config": {k: v for k, v in asdict(config).items() if k not in {"source", "vocabulary", "cache_dir"}},
    }
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    directory = Path(config.cache_dir) / fingerprint[:20]
    metadata_path = directory / "metadata.json"
    if metadata_path.exists() and (directory / "dataset.npz").exists():
        metadata = json.loads(metadata_path.read_text())
        if sha256(directory / "dataset.npz") != metadata["encoded_sha256"]:
            raise ValueError(f"Corrupt data cache: {directory}")
        return directory, metadata
    directory.mkdir(parents=True, exist_ok=True)
    tokens, offsets, conditions, ascents, uuids, row_ids, splits, signatures = [], [0], [], [], [], [], [], []
    excluded = Counter()
    with open_lines(config.source) as source:
        for line in source:
            row = json.loads(line)
            grade = v_grade(row["grade_hueco"])
            if grade is None:
                excluded["missing_grade"] += 1
                continue
            if not config.include_computed_grades and row["grade_status"] == "computed":
                excluded["computed_grade"] += 1
                continue
            problem = climb_problem(row["frames"], config.max_holds)
            if problem:
                excluded[problem] += 1
                continue
            angle = float(row["angle"])
            if not 0 <= angle <= 90:
                raise ValueError(f"Angle outside [0,90]: {angle}")
            sequence = board.encode(row["frames"][0])
            tokens.extend(sequence)
            offsets.append(len(tokens))
            conditions.append((grade, angle / 90.0))
            ascents.append(max(1, int(row.get("ascensionist_count") or 1)))
            uuids.append(row["climb_uuid"])
            row_ids.append(row["climb_angle_id"])
            splits.append(split_for_uuid(row["climb_uuid"], config))
            signatures.append(hashlib.sha1(np.asarray(sorted(sequence), dtype=np.int32).tobytes()).hexdigest()[:16])
    splits = np.asarray(splits, dtype=np.uint8)
    signatures = np.asarray(signatures)
    leaked = 0
    if config.remove_split_leakage:
        held_out = set(signatures[splits != 0])
        mask = (splits == 0) & np.isin(signatures, list(held_out))
        leaked = int(mask.sum())
        splits[mask] = 255
    if not all((splits == i).any() for i in range(3)):
        raise ValueError("Dataset must contain examples in each split")
    arrays = {
        "tokens": np.asarray(tokens, dtype=np.int32), "offsets": np.asarray(offsets, dtype=np.int64),
        "conditions": np.asarray(conditions, dtype=np.float32), "ascents": np.asarray(ascents, dtype=np.int32),
        "uuids": np.asarray(uuids), "row_ids": np.asarray(row_ids), "splits": splits, "signatures": signatures,
    }
    temporary = directory / "dataset.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(directory / "dataset.npz")
    metadata = {
        **specification, "fingerprint": fingerprint, "board": board.to_json(),
        "special_tokens": SPECIAL_TOKENS, "vocab_size": board.vocab_size,
        "condition_order": ["v_grade", "angle_divided_by_90"], "role_order": ROLE_ORDER,
        "excluded": dict(excluded), "training_rows_removed_for_leakage": leaked,
        "max_holds": int(np.diff(arrays["offsets"]).max()),
        "encoded_sha256": sha256(directory / "dataset.npz"),
        "splits": {name: {"rows": int((splits == i).sum()), "unique_climbs": len(set(arrays["uuids"][splits == i]))}
                   for i, name in enumerate(("train", "validation", "test"))},
    }
    write_json(metadata_path, metadata)
    return directory, metadata


def load_arrays(directory):
    with np.load(Path(directory) / "dataset.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


class ClimbDataset(Dataset):
    def __init__(self, arrays, split):
        self.arrays = arrays
        self.indices = np.flatnonzero(arrays["splits"] == split)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        offsets = self.arrays["offsets"]
        return self.arrays["tokens"][offsets[i]:offsets[i + 1]], self.arrays["conditions"][i], self.arrays["ascents"][i]


def roles_of(tokens):
    return (np.asarray(tokens, dtype=np.int64) - 4) % 4


def given_mask(tokens, mode, rng=None):
    """Which holds of a canonically ordered climb are given. Random modes need ``rng``."""
    roles = roles_of(tokens)
    if mode == "scratch":
        return np.zeros(len(tokens), dtype=bool)
    if mode == "starts":
        return roles == START
    if mode == "starts_feet":
        # Deterministic evaluation variant: starts plus every other foot.
        feet = np.flatnonzero(roles == FOOT)
        mask = roles == START
        mask[feet[::2]] = True
        return mask
    if mode == "starts_random_feet":
        return (roles == START) | ((roles == FOOT) & (rng.random(len(tokens)) < rng.random()))
    if mode == "random":
        return rng.random(len(tokens)) < 0.8 * rng.random()
    raise ValueError(f"Unknown given mode: {mode}")


def build_example(tokens, mask):
    """Input/target pair for BOS given SEP rest EOS; targets before SEP are PAD (not predicted)."""
    given = [int(t) for t, m in zip(tokens, mask) if m]
    rest = [int(t) for t, m in zip(tokens, mask) if not m]
    sequence = [BOS, *given, SEP, *rest, EOS]
    targets = sequence[1:]
    targets[:len(given) + 1] = [PAD] * (len(given) + 1)
    return sequence[:-1], targets


class Collator:
    """Pads examples; picks each row's given holds by ``mode`` or by the training mixture."""

    def __init__(self, mode=None, probabilities=(0.3, 0.45, 0.25), weighting="none", seed=0):
        self.mode = mode
        self.probabilities = np.asarray(probabilities, dtype=np.float64)
        self.weighting = weighting
        self.rng = np.random.default_rng(seed)

    def row_weight(self, ascents):
        return float(min(np.log2(1 + ascents), 6.0)) if self.weighting == "log" else 1.0

    def __call__(self, examples):
        pairs, conditions, weights = [], [], []
        for tokens, condition, ascents in examples:
            mode = self.mode or ("scratch", "starts_random_feet", "random")[
                self.rng.choice(3, p=self.probabilities)]
            pairs.append(build_example(tokens, given_mask(tokens, mode, self.rng)))
            conditions.append(condition)
            weights.append(self.row_weight(ascents))
        length = max(len(x) for x, _ in pairs)
        inputs = torch.full((len(pairs), length), PAD, dtype=torch.long)
        targets = torch.full_like(inputs, PAD)
        for i, (x, y) in enumerate(pairs):
            inputs[i, :len(x)] = torch.tensor(x)
            targets[i, :len(y)] = torch.tensor(y)
        return inputs, targets, torch.as_tensor(np.stack(conditions)), torch.tensor(weights)
