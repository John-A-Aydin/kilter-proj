"""Export a v2 checkpoint for browser inference as ONNX.

Also exports every real climb in the dataset, so the page can show the closest one to a generated climb.

The manifest's ``condition_schema`` says how to build the model's condition row: its columns in
order (grade, angle / 90, then one 0/1 column per style tag for a tag model), the grade range,
and whether and how a grade can be null. The browser builds its input from it (web/src/decoding.mjs
conditionVector), and the parity fixtures cover null grades and tags whenever the model has them.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import write_json
from .data import BOS, EOS, MAX_FINISHES, MAX_STARTS, PAD, ROLE_ORDER, SEP, load_arrays, open_lines, prepare, sha256
from .experiment import load_run
from .model import NULL_GRADE, parameter_count
from .sampling import allowed_tokens, condition_tags

GRADE_RANGE = (0, 16)


class NextToken(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, conditions):
        return self.model(tokens, conditions)[:, -1, :]


def condition_row(model, grade, angle, tags=()):
    """The model's condition row, exactly as kilter.sampling.generate builds it. ``grade=None`` is null."""
    if grade is None and not model.config.null_grade:
        raise ValueError("This model was trained without null grades")
    return [float(NULL_GRADE if grade is None else grade), angle / 90.0, *condition_tags(model, tags)]


def condition_schema(model, run_dir):
    """How to build a condition row for ``model``, for the manifest."""
    tags = list(model.config.climb_tags and json.loads((Path(run_dir) / "climb_tags.json").read_text())["tags"] or [])
    schema = {"columns": ["v_grade", "angle_divided_by_90", *(f"tag:{tag}" for tag in tags)],
              "grade": {"min": GRADE_RANGE[0], "max": GRADE_RANGE[1], "nullable": bool(model.config.null_grade),
                        "null_value": NULL_GRADE},
              "angle": {"min": 0, "max": 90, "divide_by": 90}, "tags": tags}
    if tags:
        from .climb_tags import TAGS
        if tags != TAGS:
            raise ValueError(f"{run_dir} was trained with tags {tags}, but kilter/climb_tags.py now has {TAGS}")
        record = json.loads((Path(run_dir) / "climb_tags.json").read_text())
        # Unset tags are "not mentioned", not "absent": all zeros means a typical untagged climb.
        schema["tag_training_rows"] = record["rows_per_tag"]
        schema["tags_sha256"] = record["sha256"]
    return schema


def condition_cases(model, schema):
    """(grade, angle, tags) for the parity grid: numeric extremes, plus null grade and tags when supported."""
    cases = [(0, 0, ()), (5, 40, ()), (GRADE_RANGE[1], 90, ())]
    if schema["grade"]["nullable"]:
        cases.append((None, 40, ()))
    tags = schema["tags"]
    if tags:
        cases += [(5, 40, (tags[0],)), (None if schema["grade"]["nullable"] else 8, 50, tuple(tags[1:3])),
                  (11, 30, tuple(tags))]
    return cases


def fixture_tokens(rng, board, given, rest):
    holds = rng.choice(len(board.holds), size=given + rest, replace=False)
    tokens = [board.token(int(h), int(rng.integers(4))) for h in holds]
    return np.array([[BOS, *tokens[:given], SEP, *tokens[given:]]], dtype=np.int64)


def decoding_fixtures(board, max_holds, rng, walks=12):
    """Random valid generation walks, recording each state and the tokens Python's rules allow."""
    fixtures = []
    for walk in range(walks):
        used, last_rank, starts, finishes = set(), -1, 0, 0
        count = int(rng.integers(0, 3)) if walk % 3 else 0
        for dense in rng.choice(len(board.holds), size=count, replace=False):
            used.add(int(dense))
        if walk % 4 == 1:
            starts = min(count, 2)
        for _ in range(max_holds + 1):
            state = {"used": sorted(used), "last_rank": last_rank, "starts": starts, "finishes": finishes,
                     "count": count, "max_holds": max_holds}
            allowed = np.flatnonzero(allowed_tokens(board, **{**state, "used": used}))
            fixtures.append({**state, "allowed": allowed.tolist()})
            # Prefer holds over EOS so walks reach late-climb states.
            choices = allowed[allowed != EOS] if (allowed != EOS).any() and rng.random() < 0.85 else allowed
            token = int(rng.choice(choices))
            if token == EOS:
                break
            dense, role = board.split(token)
            used.add(dense)
            starts += role == 2
            finishes += role == 3
            last_rank = int(board.rank[token])
            count += 1
    return fixtures


def climb_catalogue(arrays, board, names):
    """One entry per climb UUID across every split, most total ascents first.

    Returns JSON metadata and the typed arrays that web/src/similar.mjs reads from climbs.bin, in order.
    Holds are ``hold_id * 4 + role``; each angle row has its angle, V grade, and ascensionist count.
    """
    rows = defaultdict(list)
    for i, uuid in enumerate(arrays["uuids"]):
        rows[str(uuid)].append(i)
    offsets, tokens = arrays["offsets"], arrays["tokens"]
    ascents = arrays["ascents"].astype(np.int64)
    order = sorted(rows, key=lambda uuid: -ascents[rows[uuid]].sum())
    holds, hold_counts, angle_counts, angles, grades, row_ascents = [], [], [], [], [], []
    for uuid in order:
        first = rows[uuid][0]
        climb = tokens[offsets[first]:offsets[first + 1]]
        if any(not np.array_equal(tokens[offsets[i]:offsets[i + 1]], climb) for i in rows[uuid]):
            raise ValueError(f"Climb {uuid} has different holds at different angles")
        for token in climb:
            dense, role = board.split(token)
            holds.append(board.holds[dense] * 4 + role)
        hold_counts.append(len(climb))
        angle_counts.append(len(rows[uuid]))
        for i in rows[uuid]:
            grade, angle = arrays["conditions"][i]
            angles.append(round(float(angle) * 90))
            grades.append(round(float(grade)))
            row_ascents.append(ascents[i])
    # Widest types first, so every typed array starts on an aligned byte offset.
    columns = [("ascents", np.uint32, row_ascents), ("holds", np.uint16, holds), ("hold_counts", np.uint8, hold_counts),
               ("angle_counts", np.uint8, angle_counts), ("angles", np.uint8, angles), ("grades", np.uint8, grades)]
    packed = []
    for name, dtype, values in columns:
        values = np.asarray(values)
        if values.min() < 0 or values.max() > np.iinfo(dtype).max:
            raise ValueError(f"{name} does not fit in {np.dtype(dtype).name}")
        packed.append((name, values.astype(np.dtype(dtype).newbyteorder("<"))))
    metadata = {"format": "kilter-climbs-v1", "climbs": len(order), "rows": len(row_ascents),
                "names": [names.get(uuid, "") for uuid in order],
                "arrays": [{"name": name, "type": np.dtype(values.dtype).name, "length": len(values)}
                           for name, values in packed]}
    return metadata, [values for _, values in packed]


def export_climbs(config, board, output):
    with open_lines(config.data.source) as source:
        names = {row["climb_uuid"]: row["name"] for row in map(json.loads, source)}
    directory, metadata = prepare(config.data)
    catalogue, columns = climb_catalogue(load_arrays(directory), board, names)
    destination = output / "climbs.bin"
    with destination.open("wb") as target:
        for values in columns:
            target.write(values.tobytes())
    write_json(output / "climbs.json", {**catalogue, "data_fingerprint": metadata["fingerprint"],
                                        "bin_sha256": sha256(destination)})
    return catalogue["climbs"], destination.stat().st_size


def export(checkpoint, output, vocabulary):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(1)
    model, board, config, metadata = load_run(checkpoint)
    schema = condition_schema(model, Path(checkpoint).resolve().parent)
    if [c for c in schema["columns"][:2]] != metadata["condition_order"]:
        raise ValueError("The checkpoint's dataset orders its conditions differently from the schema")
    wrapper = NextToken(model).eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "model.onnx"
    example = (torch.tensor([[BOS, 5, SEP, 9]], dtype=torch.long), torch.tensor([condition_row(model, 5, 40)]))
    torch.onnx.export(wrapper, example, str(destination), dynamo=False,
                      input_names=["tokens", "conditions"], output_names=["logits"],
                      dynamic_axes={"tokens": {1: "sequence"}}, opset_version=17, external_data=False)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    fixtures, max_error = [], 0.0
    # Empty context, starts only, starts + feet mid-generation, and the full context.
    for given, rest in ((0, 0), (2, 0), (4, 9), (0, config.data.max_holds), (config.data.max_holds // 2,
                                                                                config.data.max_holds // 2)):
        for grade, angle, tags in condition_cases(model, schema):
            ids = fixture_tokens(rng, board, given, rest)
            numeric = np.array([condition_row(model, grade, angle, tags)], dtype=np.float32)
            with torch.no_grad():
                expected = wrapper(torch.from_numpy(ids), torch.from_numpy(numeric)).numpy()
            actual = session.run(None, {"tokens": ids, "conditions": numeric})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            max_error = max(max_error, float(np.abs(actual - expected).max()))
            # Browser parity fixtures: the everyday request, a null grade, and a tag combination.
            if (grade, angle, tags) in ((5, 40, ()), (None, 40, ())) or len(tags) > 1:
                fixtures.append({"tokens": ids[0].tolist(), "conditions": numeric[0].tolist(), "grade": grade,
                                 "angle": angle, "tags": list(tags), "logits": [round(v, 6) for v in expected[0].tolist()]})
    with Path(vocabulary).open(newline="") as source:
        rows = {int(r["hold_token"]): r for r in csv.DictReader(source)}
    if set(rows) != set(board.holds):
        raise ValueError("Display vocabulary does not match the checkpoint")
    holds = [{"id": token, "x": float(rows[token]["hold_x"]), "y": float(rows[token]["hold_y"]),
              "name": rows[token]["hole_name"], "kickboard": bool(int(rows[token]["on_kickboard"])),
              "set": int(rows[token]["set_id"])} for token in board.holds]
    manifest = {
        "format": "kilter-v2", "name": config.name, "parameters": parameter_count(model),
        "vocab_size": board.vocab_size, "max_seq_len": model.max_len, "max_holds": config.data.max_holds,
        "special_tokens": {"PAD": PAD, "BOS": BOS, "EOS": EOS, "SEP": SEP},
        "condition_order": schema["columns"], "condition_schema": schema, "holds": holds,
        # Canonical generation order of every token (-1 for specials); decoding must follow it.
        "rank": board.rank.tolist(), "role_order": {str(k): v for k, v in ROLE_ORDER.items()},
        "max_starts": MAX_STARTS, "max_finishes": MAX_FINISHES,
        "checkpoint_sha256": sha256(checkpoint), "model_sha256": sha256(destination),
        "data_fingerprint": metadata["fingerprint"], "onnx_max_absolute_error": max_error,
        "roles": {"hand": 0, "foot": 1, "start": 2, "finish": 3},
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "parity-fixtures.json", fixtures)
    write_json(output / "decoding-fixtures.json", decoding_fixtures(board, config.data.max_holds, rng))
    climbs, climb_bytes = export_climbs(config, board, output)
    print(json.dumps({"output": str(output), "parameters": manifest["parameters"],
                      "onnx_bytes": destination.stat().st_size, "max_absolute_error": max_error,
                      "real_climbs": climbs, "climb_bytes": climb_bytes}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("web/public/model"))
    parser.add_argument("--vocabulary", type=Path, default=Path("data/training/hold_vocabulary.csv"))
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.vocabulary)


if __name__ == "__main__":
    main()
