"""Validate source coverage, output checksums, and every converted hold offline."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from boardlib.db.aurora import get_difficulty_mapping
import sqlite3

from download import ROOT, check_database, sha256
from prepare import HOLD_FIELDS, QUERY, angle_metadata, convert_frame, decode_frames, read_geometry


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(raw_dir, output_dir, progress=False):
    manifest = json.loads((output_dir / "manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        require(sha256(output_dir / name) == digest, f"Checksum failed: {name}")
    geometry = raw_dir / "kilter.sqlite3"
    snapshot = raw_dir / "kilter_original.snapshot.sqlite3"
    grades = raw_dir / "kilter_original.grades.sqlite3"
    for path, key in [(geometry, "geometry_sha256"), (snapshot, "snapshot_sha256"), (grades, "grades_sha256")]:
        check_database(path)
        require(sha256(path) == manifest["sources"][key], f"Source checksum failed: {path}")
    layout, size, roles, holds = read_geometry(geometry)
    with sqlite3.connect(f"{snapshot.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("ATTACH DATABASE ? AS geometry", (f"{geometry.resolve().as_uri()}?mode=ro",))
        db.execute("ATTACH DATABASE ? AS grades", (f"{grades.resolve().as_uri()}?mode=ro",))
        require(db.execute("SELECT COUNT(*) FROM grades.board_climb_grades").fetchone()[0]
                == manifest["sources"]["snapshot"]["grades"]["tables"]["board_climb_grades"]["rowCount"],
                "Grade artifact row count mismatch")
        for table, info in manifest["sources"]["snapshot"]["tables"].items():
            require(table in {"board_climbs", "board_climb_stats"}, "Unexpected source table")
            require(db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == info["rowCount"],
                    f"Source row count mismatch: {table}")
        require(not db.execute("""
            SELECT 1 FROM board_climbs c WHERE c.layout_id = :layout_id
              AND EXISTS (SELECT 1 FROM json_each(c.compatible_size_ids) WHERE value IN (8, 14))
              AND NOT EXISTS (SELECT 1 FROM json_each(c.compatible_size_ids) WHERE value = :size_id)
              AND c.is_listed = 1 AND COALESCE(c.is_draft, 0) = 0 AND COALESCE(c.is_hidden, 0) = 0
        """, {"layout_id": layout["id"], "size_id": size["id"]}).fetchone(),
                "An 8x12 or 7x10 climb is not compatible with the exported board")
        source_rows = db.execute(QUERY, {"layout_id": layout["id"], "size_id": size["id"]})
        grade_mapping = get_difficulty_mapping(geometry)
        require(manifest["schema_version"] == 2 and manifest["row_key"] == ["climb_uuid", "angle"],
                "Expected schema 2 with a climb-angle row key")
        counts = Counter()
        angle_counts = Counter()
        used_tokens = set()
        previous_key = None
        with (output_dir / "climbs.jsonl").open(encoding="utf-8") as source, \
                (output_dir / "climbs.csv").open(encoding="utf-8", newline="") as metadata_file, \
                (output_dir / "holds.csv").open(encoding="utf-8", newline="") as flat_file:
            flat = csv.DictReader(flat_file)
            metadata_rows = csv.DictReader(metadata_file)
            for line in source:
                climb = json.loads(line)
                uuid = climb["climb_uuid"]
                key = (uuid, climb["angle"])
                row = next(source_rows, None)
                require(row is not None and key == (row["uuid"], row["stats_angle"]),
                        f"Missing, duplicate, unexpected, or misordered climb-angle pair: {key}")
                require(previous_key is None or previous_key < key, f"Duplicate row key: {key}")
                require(climb["ascensionist_count"] > 0, f"Wrong ascent filter: {key}")
                for field, value in angle_metadata(row, grade_mapping).items():
                    require(climb[field] == value, f"Wrong angle-specific {field}: {key}")
                for field in ["name", "setter_username", "description", "created_at", "frames_pace"]:
                    require(climb[field] == row[field], f"Source metadata mismatch: {field} on {key}")
                require(climb["setter_angle"] == row["angle"], f"Wrong setter angle: {key}")
                require(climb["raw_frames"] == row["frames"], f"Source frames changed: {key}")
                if previous_key is None or previous_key[0] != uuid:
                    expected = [convert_frame(frame, roles, holds) for frame in decode_frames(row["frames"])]
                    counts["unique_climbs"] += 1
                    counts["multi_frame_climbs"] += int(len(expected) > 1)
                require(climb["frames"] == expected, f"Hold conversion mismatch: {key}")
                require(len(expected) == climb["frames_count"] and climb["source_frames_count"] == row["frames_count"],
                        f"Wrong frame count: {key}")
                require(climb["frame_count_matches_source"] == (len(expected) == row["frames_count"]),
                        f"Unflagged source frame-count mismatch: {key}")
                metadata = next(metadata_rows, None)
                expected_metadata = {k: "" if v is None else str(v) for k, v in climb.items()
                                     if k not in {"frames", "raw_frames"}}
                require(metadata == expected_metadata, f"Climb CSV disagrees with JSON: {key}")
                for frame_index, frame in enumerate(climb["frames"]):
                    for hold_index, hold in enumerate(frame):
                        require(set(hold) == set(HOLD_FIELDS), f"Unexpected hold schema: {key}")
                        require(type(hold["hold_use_type"]) is int and hold["hold_use_type"] in range(4), "Invalid use type")
                        require(0 <= hold["hold_x"] <= 1 and 0 <= hold["hold_y"] <= 1, "Invalid coordinates")
                        require(type(hold["on_kickboard"]) is int and hold["on_kickboard"] in (0, 1), "Invalid kickboard flag")
                        require(type(hold["hold_token"]) is int and hold["hold_token"] in holds, "Unknown token")
                        actual = next(flat, None)
                        require(actual is not None, "Hold CSV truncated")
                        require(actual["climb_uuid"] == uuid and actual["climb_angle_id"] == climb["climb_angle_id"]
                                and int(actual["angle"]) == climb["angle"]
                                and int(actual["frame_index"]) == frame_index
                                and int(actual["hold_index"]) == hold_index, "CSV grouping mismatch")
                        for field in HOLD_FIELDS:
                            require(float(actual[field]) == hold[field], f"CSV value mismatch: {field}")
                        used_tokens.add(hold["hold_token"])
                        counts["hold_entries"] += 1
                        counts["kickboard_entries"] += hold["on_kickboard"]
                        counts[f"use_type_{hold['hold_use_type']}"] += 1
                counts["climb_angle_rows"] += 1
                counts["frames"] += len(expected)
                counts["multi_frame_rows"] += int(len(expected) > 1)
                counts["frame_count_mismatch_rows"] += int(len(expected) != row["frames_count"])
                counts[climb["ascent_evidence_source"]] += 1
                counts[f"grade_{climb['grade_status']}"] += 1
                angle_counts[climb["angle"]] += 1
                previous_key = key
                if progress and counts["climb_angle_rows"] % 25000 == 0:
                    print(f"Validated {counts['climb_angle_rows']:,} climb-angle rows", flush=True)
            require(next(source_rows, None) is None, "Export is missing qualifying climb-angle pairs")
            require(next(flat, None) is None, "Hold CSV contains extra rows")
            require(next(metadata_rows, None) is None, "Climb CSV contains extra rows")
    require(dict(counts) == manifest["counts"], "Manifest counts do not match data")
    require(sorted(angle_counts) == manifest["angles"], "Manifest angle coverage disagrees")
    require({str(k): v for k, v in angle_counts.items()} == manifest["rows_by_angle"], "Per-angle counts disagree")
    require(len(used_tokens) == manifest["used_hold_count"], "Used token count disagrees")
    with (output_dir / "hold_vocabulary.csv").open(encoding="utf-8", newline="") as source:
        vocabulary = list(csv.DictReader(source))
    require(len(vocabulary) == len(holds), "Vocabulary size mismatch")
    require(len({r["hold_token"] for r in vocabulary}) == len(vocabulary), "Duplicate vocabulary tokens")
    for row in vocabulary:
        expected = holds[int(row["hold_token"])]
        require(all(row[k] == str(v) for k, v in expected.items()), "Vocabulary disagrees with geometry")
    print(f"Validated {counts['climb_angle_rows']:,} climb-angle rows from {counts['unique_climbs']:,} climbs, "
          f"{len(angle_counts)} angles, {counts['hold_entries']:,} hold entries, "
          f"{len(holds)} unique physical holds; grades, full source coverage, and checksums match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "processed")
    args = parser.parse_args()
    validate(args.raw_dir, args.output_dir, progress=True)
