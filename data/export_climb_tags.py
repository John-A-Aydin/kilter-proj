"""Write every training climb's style tags, read from its name and setter description.

Tags and their rules are in kilter/climb_tags.py. Reads the climb text from the raw catalogue
snapshot (data/raw/kilter_original.snapshot.sqlite3, see data/README.md) for every climb in
data/training/climbs.jsonl.gz, and writes:

- data/training/climb_tags.csv.gz: climb_uuid and one 0/1 column per tag, sorted by uuid.
- results/climb_tags/evidence.csv.gz: the text that set each tag, for checking by eye.
- results/climb_tags/counts.csv: how many climbs carry each tag, and how many from the
  description or only from the name.

    .venv/bin/python data/export_climb_tags.py
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kilter.climb_tags import TAGS, extract  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/raw/kilter_original.snapshot.sqlite3")
    parser.add_argument("--climbs", type=Path, default=ROOT / "data/training/climbs.jsonl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/training/climb_tags.csv.gz")
    parser.add_argument("--results", type=Path, default=ROOT / "results/climb_tags")
    args = parser.parse_args()
    with gzip.open(args.climbs, "rt") as source:
        uuids = sorted({json.loads(line)["climb_uuid"] for line in source})
    wanted = set(uuids)
    text = {}
    with sqlite3.connect(args.snapshot) as database:
        for uuid, name, description in database.execute("select uuid, name, description from board_climbs"):
            if uuid in wanted:
                text[uuid] = (name, description)
    missing = wanted - set(text)
    if missing:
        raise SystemExit(f"{len(missing)} training climbs are not in {args.snapshot}")
    args.results.mkdir(parents=True, exist_ok=True)
    counts = {tag: {"climbs": 0, "from_description": 0, "name_only": 0} for tag in TAGS}
    # gzip with a fixed mtime so the same input gives the same bytes.
    with gzip.GzipFile(args.output, "wb", mtime=0) as raw, gzip.GzipFile(args.results / "evidence.csv.gz", "wb", mtime=0) as raw_evidence:
        out = csv.writer(_text(raw))
        evidence = csv.writer(_text(raw_evidence))
        out.writerow(["climb_uuid", *TAGS])
        evidence.writerow(["climb_uuid", "tag", "evidence"])
        for uuid in uuids:
            found = extract(*text[uuid])
            out.writerow([uuid, *(int(tag in found) for tag in TAGS)])
            for tag in TAGS:
                if tag in found:
                    evidence.writerow([uuid, tag, found[tag]])
                    counts[tag]["climbs"] += 1
                    counts[tag]["from_description" if found[tag].startswith("description") else "name_only"] += 1
    with (args.results / "counts.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["tag", "climbs", "from_description", "name_only"])
        for tag in TAGS:
            writer.writerow([tag, *counts[tag].values()])
    print(json.dumps({tag: c["climbs"] for tag, c in counts.items()}, indent=1))


def _text(raw):
    import io
    return io.TextIOWrapper(raw, encoding="utf-8", newline="", write_through=True)


if __name__ == "__main__":
    main()
