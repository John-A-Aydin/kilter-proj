import argparse
import csv
import json
from pathlib import Path
import sys

from .config import Config, load_config
from .data import prepare
from .experiment import evaluate_run, sample, train


def main():
    parser = argparse.ArgumentParser(description="Grade- and angle-conditioned climb experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/ar_transformer.toml")
        sub.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        if command == "train":
            sub.add_argument("--resume", type=Path, help="Resume an interrupted run from last.pt using its saved config")
    sub = commands.add_parser("evaluate")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--split", choices=("validation", "test"), default="test")
    sub.add_argument("--device", default="cpu")
    sub = commands.add_parser("sample")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--grade", type=float, required=True, help="Numeric V scale, e.g. 5 for V5")
    sub.add_argument("--angle", type=float, required=True, help="Degrees in [0,90]; normalized internally")
    sub.add_argument("--count", type=int, default=1)
    sub.add_argument("--temperature", type=float, default=1.0)
    sub.add_argument("--top-k", type=int, default=0)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--device", default="cpu")
    sub = commands.add_parser("compare", help="Write completed-run summaries as CSV")
    sub.add_argument("directory", nargs="?", default="runs", type=Path)
    args = parser.parse_args()
    if args.command in {"prepare", "train"}:
        if args.command == "train" and args.resume:
            if args.set:
                parser.error("--resume uses the saved config; --set is not allowed")
            config = Config.from_dict(json.loads((args.resume.parent / "config.json").read_text()))
        else:
            config = load_config(args.config, args.set)
        if args.command == "prepare":
            directory, metadata = prepare(config.data)
            print(json.dumps({"cache": str(directory), **metadata}, indent=2))
        else:
            train(config, args.resume)
    elif args.command == "evaluate":
        evaluate_run(args.checkpoint, args.split, args.device)
    elif args.command == "sample":
        print(json.dumps(sample(args.checkpoint, args.grade, args.angle, args.count, args.temperature,
                                args.top_k, args.seed, args.device), indent=2))
    elif args.command == "compare":
        summaries = []
        for path in sorted(args.directory.glob("*/summary.json")):
            summary = json.loads(path.read_text())
            config = json.loads((path.parent / "config.json").read_text())
            summary["positional_embeddings"] = config["model"]["positional_embeddings"]
            summary["seed"] = config["train"]["seed"]
            summaries.append(summary)
        if summaries:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


if __name__ == "__main__":
    main()
