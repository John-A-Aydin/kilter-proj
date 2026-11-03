"""Stable tokenization and UUID-grouped splits, cached as compact ragged arrays."""
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import write_json

PAD, BOS, EOS, FRAME = range(4)
SPECIAL_TOKENS = ["PAD", "BOS", "EOS", "FRAME"]
CACHE_VERSION = 1


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class Tokenizer:
    def __init__(self, holds):
        self.holds = sorted(int(x) for x in holds)
        if len(set(self.holds)) != len(self.holds):
            raise ValueError("Duplicate physical holds in vocabulary")
        self.dense = {token: i for i, token in enumerate(self.holds)}
        self.vocab_size = 4 + 4 * len(self.holds)

    @classmethod
    def from_csv(cls, path):
        with Path(path).open(newline="") as source:
            return cls([row["hold_token"] for row in csv.DictReader(source)])

    def encode(self, frames):
        if not frames:
            raise ValueError("Climb has no frames")
        result = [BOS]
        for i, frame in enumerate(frames):
            if i:
                result.append(FRAME)
            for hold in sorted(frame, key=lambda h: (h["hold_token"], h["hold_use_type"])):
                role = hold["hold_use_type"]
                if role not in range(4):
                    raise ValueError(f"Unknown hold role {role}")
                result.append(4 + 4 * self.dense[hold["hold_token"]] + role)
        return result + [EOS]

    def decode(self, tokens):
        frames = [[]]
        for token in tokens:
            token = int(token)
            if token == EOS:
                break
            if token == FRAME:
                frames.append([])
            elif token >= 4:
                index, role = divmod(token - 4, 4)
                frames[-1].append({"hold_token": self.holds[index], "hold_use_type": role})
        return frames


def v_grade(label):
    if label is None or label == "":
        return None
    if not re.fullmatch(r"V\d+", label):
        raise ValueError(f"Unsupported V grade: {label!r}")
    return float(label[1:])


def split_for_uuid(uuid, config):
    digest = hashlib.sha256(f"{config.split_seed}:{uuid}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return 0 if value < config.train_fraction else (1 if value < config.train_fraction + config.validation_fraction else 2)


def prepare(config):
    """Verify source contents on every call; reuse content-addressed encoded data."""
    tokenizer = Tokenizer.from_csv(config.vocabulary)
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
    tokens, offsets, conditions, uuids, splits, row_ids = [], [0], [], [], [], []
    excluded = Counter()
    grade_histogram, angle_histogram = Counter(), Counter()
    with Path(config.source).open() as source:
        for line in source:
            row = json.loads(line)
            grade = v_grade(row["grade_hueco"])
            if grade is None:
                excluded["missing_grade"] += 1
                continue
            if not config.include_computed_grades and row["grade_status"] == "computed":
                excluded["computed_grade"] += 1
                continue
            sequence = tokenizer.encode(row["frames"])
            # The context length is the number of input/target pairs, including EOS.
            if len(sequence) - 1 > config.max_seq_len:
                excluded["over_context_length"] += 1
                continue
            angle = float(row["angle"])
            if not 0 <= angle <= 90:
                raise ValueError(f"Angle outside [0,90]: {angle}")
            tokens.extend(sequence)
            offsets.append(len(tokens))
            conditions.append((grade, angle / 90.0))
            uuids.append(row["climb_uuid"])
            row_ids.append(row["climb_angle_id"])
            splits.append(split_for_uuid(row["climb_uuid"], config))
            grade_histogram[str(int(grade))] += 1
            angle_histogram[str(int(angle))] += 1
    if set(splits) != {0, 1, 2}:
        raise ValueError("Dataset must contain examples in each split")
    arrays = {
        "tokens": np.asarray(tokens, dtype=np.int32),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "conditions": np.asarray(conditions, dtype=np.float32),
        "uuids": np.asarray(uuids), "row_ids": np.asarray(row_ids),
        "splits": np.asarray(splits, dtype=np.uint8),
    }
    temporary = directory / "dataset.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(directory / "dataset.npz")
    metadata = {
        **specification, "fingerprint": fingerprint, "holds": tokenizer.holds,
        "special_tokens": SPECIAL_TOKENS, "vocab_size": tokenizer.vocab_size,
        "condition_order": ["v_grade", "angle_divided_by_90"],
        "excluded": dict(excluded), "grade_histogram": dict(grade_histogram),
        "angle_histogram": dict(angle_histogram),
        "max_sequence_length": int(np.diff(arrays["offsets"]).max() - 1),
        "encoded_sha256": sha256(directory / "dataset.npz"),
        "splits": {
            name: {"rows": splits.count(i), "unique_climbs": len({u for u, s in zip(uuids, splits) if s == i})}
            for i, name in enumerate(("train", "validation", "test"))
        },
    }
    write_json(metadata_path, metadata)
    return directory, metadata


class ClimbDataset(Dataset):
    def __init__(self, arrays, split):
        self.arrays = arrays
        self.indices = np.flatnonzero(arrays["splits"] == split)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        offsets = self.arrays["offsets"]
        return (self.arrays["tokens"][offsets[i]:offsets[i + 1]], self.arrays["conditions"][i])


def collate(examples):
    length = max(len(tokens) - 1 for tokens, _ in examples)
    inputs = torch.full((len(examples), length), PAD, dtype=torch.long)
    targets = torch.full_like(inputs, PAD)
    for i, (tokens, _) in enumerate(examples):
        n = len(tokens) - 1
        inputs[i, :n] = torch.as_tensor(tokens[:-1].astype(np.int64))
        targets[i, :n] = torch.as_tensor(tokens[1:].astype(np.int64))
    return inputs, targets, torch.as_tensor(np.stack([c for _, c in examples]))


def load_arrays(directory):
    with np.load(Path(directory) / "dataset.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
