import argparse
import csv
import json
from pathlib import Path
import sys

from .config import Config, load_config

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "default.toml"


def parse_given(values):
    """--given 1234:start --given 1301:foot ..."""
    names = {"hand": 0, "foot": 1, "start": 2, "finish": 3}
    given = []
    for value in values:
        hold, _, role = value.partition(":")
        if role not in names:
            raise argparse.ArgumentTypeError(f"--given expects HOLD:ROLE with role in {sorted(names)}")
        given.append((int(hold), names[role]))
    return given


def main():
    parser = argparse.ArgumentParser(prog="python -m kilter_v2", description="Kilter climb generation, v2")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default=str(DEFAULT_CONFIG))
        sub.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        if command == "train":
            sub.add_argument("--resume", type=Path, help="Resume an interrupted run from last.pt")
    sub = commands.add_parser("evaluate", help="Per-climb likelihood for each given-hold mode")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--split", choices=("validation", "test"), default="test")
    sub.add_argument("--device", default="cpu")
    sub = commands.add_parser("sample", help="Generate climbs from nothing or from given holds")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--grade", type=float, help="Numeric V scale, e.g. 5 for V5; omit for a null grade")
    sub.add_argument("--angle", type=float, required=True, help="Degrees in [0,90]")
    sub.add_argument("--given", action="append", default=[], metavar="HOLD:ROLE",
                     help="Given hold, e.g. 1176:start or 1091:foot; repeat for more")
    sub.add_argument("--count", type=int, default=1)
    sub.add_argument("--temperature", type=float, default=1.0)
    sub.add_argument("--top-k", type=int, default=0)
    sub.add_argument("--seed", type=int, default=42)
    sub.add_argument("--unconstrained", action="store_true", help="Disable structural decoding rules")
    sub = commands.add_parser("quality", help="Sample-quality report, optionally against the kilter/ baseline")
    sub.add_argument("checkpoint", type=Path)
    sub.add_argument("--baseline", type=Path, help="kilter/ checkpoint to compare")
    sub.add_argument("--split", choices=("validation", "test"), default="validation")
    sub.add_argument("--samples", type=int, default=20, help="Samples per grade/angle cell and mode")
    sub.add_argument("--output", type=Path)
    sub = commands.add_parser("compare", help="Write completed-run summaries as CSV")
    sub.add_argument("directory", nargs="?", default="runs", type=Path)
    args = parser.parse_args()
    if args.command in {"prepare", "train"}:
        from .data import prepare
        from .experiment import train
        if args.command == "train" and args.resume:
            if args.set:
                parser.error("--resume uses the saved config; --set is not allowed")
            config = Config.from_dict(json.loads((args.resume.parent / "config.json").read_text()))
        else:
            config = load_config(args.config, args.set)
        if args.command == "prepare":
            directory, metadata = prepare(config.data)
            print(json.dumps({"cache": str(directory), **{k: v for k, v in metadata.items() if k != "board"}}, indent=2))
        else:
            train(config, args.resume)
    elif args.command == "evaluate":
        from .experiment import evaluate_run
        evaluate_run(args.checkpoint, args.split, args.device)
    elif args.command == "sample":
        import torch
        from .experiment import load_run
        from .sampling import generate
        model, board, config, _ = load_run(args.checkpoint)
        generator = torch.Generator().manual_seed(args.seed)
        given = parse_given(args.given)
        print(json.dumps([generate(model, board, args.grade, args.angle, given, args.temperature, args.top_k,
                                   generator, not args.unconstrained, config.data.max_holds)
                          for _ in range(args.count)], indent=2))
    elif args.command == "quality":
        from .quality import quality_report
        quality_report(args.checkpoint, args.baseline, args.split, args.samples, args.output)
    elif args.command == "compare":
        rows = [json.loads(p.read_text()) for p in sorted(args.directory.glob("*/summary.json"))]
        rows = [r for r in rows if "best_validation_selection_nll" in r]
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
