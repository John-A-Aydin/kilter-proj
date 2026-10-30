"""Export an experiment checkpoint for local or hosted browser inference."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .config import write_json
from .data import sha256
from .experiment import load_run
from .model import parameter_count


class NextToken(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, conditions):
        return self.model(tokens, conditions)[:, -1, :]


def export(checkpoint, output, vocabulary):
    import onnx
    import onnxruntime as ort

    torch.set_num_threads(1)
    model, config, metadata = load_run(checkpoint)
    wrapper = NextToken(model).eval()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "model.onnx"
    tokens = torch.tensor([[1, 5, 10]], dtype=torch.long)
    conditions = torch.tensor([[5., 40 / 90]], dtype=torch.float32)
    torch.onnx.export(wrapper, (tokens, conditions), str(destination), dynamo=False,
                      input_names=["tokens", "conditions"], output_names=["logits"],
                      dynamic_axes={"tokens": {1: "sequence"}}, opset_version=17,
                      external_data=False)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    fixtures = []
    max_error = 0.0
    for length in (1, 2, 17, config.data.max_seq_len):
        for grade, angle in ((0, 0), (5, 40), (16, 90)):
            ids = rng.integers(4, metadata["vocab_size"], size=(1, length), dtype=np.int64)
            ids[0, 0] = 1
            numeric = np.array([[grade, angle / 90]], dtype=np.float32)
            with torch.no_grad():
                expected = wrapper(torch.from_numpy(ids), torch.from_numpy(numeric)).numpy()
            actual = session.run(None, {"tokens": ids, "conditions": numeric})[0]
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            max_error = max(max_error, float(np.abs(actual - expected).max()))
            if length in (1, 17, config.data.max_seq_len):
                fixtures.append({"tokens": ids[0].tolist(), "conditions": numeric[0].tolist(),
                                 "logits": expected[0].tolist()})
    with Path(vocabulary).open(newline="") as source:
        rows = {int(r["hold_token"]): r for r in csv.DictReader(source)}
    if set(rows) != set(metadata["holds"]):
        raise ValueError("Display vocabulary does not match the checkpoint")
    holds = [{"id": token, "x": float(rows[token]["hold_x"]), "y": float(rows[token]["hold_y"]),
              "name": rows[token]["hole_name"], "kickboard": bool(int(rows[token]["on_kickboard"])),
              "set": int(rows[token]["set_id"])} for token in metadata["holds"]]
    manifest = {
        "name": config.name, "parameters": parameter_count(model), "vocab_size": metadata["vocab_size"],
        "max_seq_len": config.data.max_seq_len, "condition_order": metadata["condition_order"],
        "holds": holds, "checkpoint_sha256": sha256(checkpoint), "model_sha256": sha256(destination),
        "data_fingerprint": metadata["fingerprint"], "onnx_max_absolute_error": max_error,
        "roles": {"hand": 0, "foot": 1, "start": 2, "finish": 3},
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "parity-fixtures.json", fixtures)
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
