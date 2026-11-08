"""Hand labels from data/hold_labels.csv as extra per-hold features.

Only three columns are read: ``type``, ``grip`` and ``on_kickboard``. Size, notes and
the CSV's coordinates are ignored. ``on_kickboard`` is already a geometry feature
taken from the vocabulary, so it is checked against the vocabulary, not added twice.

A hold gets a one-hot type (7 columns) and a multi-hot grip (16 compass directions,
22.5 degrees apart, clockwise from the top of the hold). A blank type or grip is all zeros.
The labels a run trained with are saved in its directory, so later edits to the CSV
do not change a trained model.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np

TYPES = ("jug", "edge", "sloper", "pinch", "pocket", "foot", "unsure")
DIRECTION_STEP = 22.5
DIRECTIONS = tuple(i * DIRECTION_STEP for i in range(int(360 / DIRECTION_STEP)))
FEATURE_NAMES = [f"type_{name}" for name in TYPES] + [f"grip_{direction:g}" for direction in DIRECTIONS]


def parse_grip(value):
    if not value.strip():
        return []
    directions = sorted({float(part) for part in value.split(";")})
    bad = [d for d in directions if d not in DIRECTIONS]
    if bad:
        raise ValueError(f"{bad[0]:g} is not a grip direction")
    return directions


def read_hold_labels(path, board):
    """Read the CSV for every hold on ``board``; the result is what a run saves and reloads."""
    path = Path(path)
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    holds = {}
    for row in rows:
        hold = int(row["hold_token"])
        if hold not in board.dense:
            raise ValueError(f"Hold {hold} in {path} is not on the board")
        if hold in holds:
            raise ValueError(f"Hold {hold} appears twice in {path}")
        if int(row["on_kickboard"]) != int(board.kickboard[board.dense[hold]]):
            raise ValueError(f"Hold {hold}: on_kickboard in {path} disagrees with the vocabulary")
        if row["type"] and row["type"] not in TYPES:
            raise ValueError(f"Hold {hold}: unknown type {row['type']!r}")
        holds[str(hold)] = {"type": row["type"], "grip": parse_grip(row["grip"])}
    missing = [hold for hold in board.holds if str(hold) not in holds]
    if missing:
        raise ValueError(f"{len(missing)} holds have no row in {path}, e.g. {missing[0]}")
    return {
        "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "columns_used": ["type", "grip", "on_kickboard"], "feature_names": FEATURE_NAMES,
        "typed": sum(bool(h["type"]) for h in holds.values()),
        "with_grip": sum(bool(h["grip"]) for h in holds.values()),
        "holds": holds,
    }


def label_features(labels, board):
    """(holds, 23) array in the board's dense order: type one-hot, then grip multi-hot."""
    features = np.zeros((len(board.holds), len(FEATURE_NAMES)), dtype=np.float32)
    for i, hold in enumerate(board.holds):
        label = labels["holds"][str(hold)]
        if label["type"]:
            features[i, TYPES.index(label["type"])] = 1.0
        for direction in label["grip"]:
            features[i, len(TYPES) + DIRECTIONS.index(direction)] = 1.0
    return features
