"""Export a v2 checkpoint for browser inference (same ONNX interface as kilter/export_web.py)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import write_json
from .data import BOS, EOS, MAX_FINISHES, MAX_STARTS, PAD, ROLE_ORDER, SEP, sha256
from .experiment import load_run
from .model import parameter_count
from .sampling import allowed_tokens


class NextToken(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, conditions):
        return self.model(tokens, conditions)[:, -1, :]


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


def export(checkpoint, output, vocabulary):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(1)
    model, board, config, metadata = load_run(checkpoint)
    wrapper = NextToken(model).eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "model.onnx"
    example = (torch.tensor([[BOS, 5, SEP, 9]], dtype=torch.long), torch.tensor([[5., 40 / 90]]))
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
        for grade, angle in ((0, 0), (5, 40), (16, 90)):
            ids = fixture_tokens(rng, board, given, rest)
            numeric = np.array([[grade, angle / 90]], dtype=np.float32)
            with torch.no_grad():
                expected = wrapper(torch.from_numpy(ids), torch.from_numpy(numeric)).numpy()
            actual = session.run(None, {"tokens": ids, "conditions": numeric})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            max_error = max(max_error, float(np.abs(actual - expected).max()))
            if grade == 5:
                fixtures.append({"tokens": ids[0].tolist(), "conditions": numeric[0].tolist(),
                                 "logits": expected[0].tolist()})
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
        "condition_order": metadata["condition_order"], "holds": holds,
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
    print(json.dumps({"output": str(output), "parameters": manifest["parameters"],
                      "onnx_bytes": destination.stat().st_size, "max_absolute_error": max_error}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("web/public/model"))
    parser.add_argument("--vocabulary", type=Path, default=Path("data/processed/hold_vocabulary.csv"))
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.vocabulary)


if __name__ == "__main__":
    main()
