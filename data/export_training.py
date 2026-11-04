"""Write the compact, committed training data from the full processed catalogue.

The full export under data/processed/ is about 1.5 GB and stays out of Git. Training,
evaluation, and the website's similar-climb lookup only need a few fields per row, so
this writes those to data/training/, which is committed:

- climbs.jsonl.gz: one JSON object per climb-angle row, in the same order as
  processed/climbs.jsonl, with only the fields listed in FIELDS. Each hold keeps
  hold_use_type and hold_token; coordinates come from the vocabulary.
- hold_vocabulary.csv: an unchanged copy of processed/hold_vocabulary.csv.

Output is byte-identical for the same input (gzip header has no name or timestamp).
"""

import argparse
import gzip
import json
from pathlib import Path
import shutil

from download import ROOT

FIELDS = ["climb_angle_id", "climb_uuid", "angle", "grade_hueco", "grade_status",
          "ascensionist_count", "name", "setter_username"]
HOLD_FIELDS = ["hold_use_type", "hold_token"]


def slim(climb):
    row = {key: climb[key] for key in FIELDS}
    row["frames"] = [[{key: hold[key] for key in HOLD_FIELDS} for hold in frame] for frame in climb["frames"]]
    return row


def export(processed_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "climbs.jsonl.gz"
    temporary = destination.with_suffix(".part")
    rows = 0
    with (processed_dir / "climbs.jsonl").open(encoding="utf-8") as source, temporary.open("wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as target:
        for line in source:
            target.write((json.dumps(slim(json.loads(line)), ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False) + "\n").encode("utf-8"))
            rows += 1
    temporary.replace(destination)
    shutil.copyfile(processed_dir / "hold_vocabulary.csv", output_dir / "hold_vocabulary.csv")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training")
    args = parser.parse_args()
    rows = export(args.processed_dir, args.output_dir)
    print(f"Wrote {rows:,} climb-angle rows to {args.output_dir}")


if __name__ == "__main__":
    main()
