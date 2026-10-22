"""Export public 12x12 Original climbs as one row per recorded ascent angle."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import sqlite3
import tempfile

from boardlib.api.aurora import difficulty_to_grade
from boardlib.db.aurora import get_difficulty_mapping, get_shared_syncs
from download import ROOT, sha256

USE_TYPES = {"middle": 0, "hand": 0, "foot": 1, "start": 2, "finish": 3}
HOLD_FIELDS = ["hold_use_type", "hold_x", "hold_y", "on_kickboard", "hold_token"]
TOKEN = re.compile(r"p([0-9]+)r([0-9]+)|x([0-9]+)")
# BKB Foot Only appears in a few Original climbs, but is absent from the APK's
# placement_roles table. Its meaning is explicit in Boardsesh's Kilter decoder:
# https://github.com/boardsesh/boardsesh/blob/main/packages/board-constants/src/hold-states.ts
LEGACY_ROLE_NAMES = {31: "foot"}

QUERY = """
WITH angles AS (
    SELECT climb_uuid, angle FROM board_climb_stats
    WHERE board_type = 'kilter' AND ascensionist_count > 0
    UNION
    SELECT climb_uuid, angle FROM geometry.climb_stats WHERE ascensionist_count > 0
)
SELECT c.*, a.angle AS stats_angle,
       s.ascensionist_count AS current_ascensionist_count,
       l.ascensionist_count AS legacy_ascensionist_count,
       s.display_difficulty AS current_display_difficulty,
       s.difficulty_average AS current_difficulty_average,
       s.benchmark_difficulty AS current_benchmark_difficulty,
       s.quality_average AS current_quality_average,
       s.fa_username AS current_fa_username, s.fa_at AS current_fa_at,
       l.display_difficulty AS legacy_display_difficulty,
       l.difficulty_average AS legacy_difficulty_average,
       l.benchmark_difficulty AS legacy_benchmark_difficulty,
       l.quality_average AS legacy_quality_average,
       l.fa_username AS legacy_fa_username, l.fa_at AS legacy_fa_at,
       g.local_grade AS computed_local_grade,
       g.universal_grade AS computed_universal_grade,
       g.confidence AS computed_grade_confidence
FROM angles a
JOIN board_climbs c ON c.uuid = a.climb_uuid
LEFT JOIN board_climb_stats s
  ON s.board_type = c.board_type AND s.climb_uuid = c.uuid AND s.angle = a.angle
LEFT JOIN geometry.climb_stats l ON l.climb_uuid = c.uuid AND l.angle = a.angle
LEFT JOIN grades.board_climb_grades g
  ON g.board_type = c.board_type AND g.climb_uuid = c.uuid AND g.angle = a.angle
WHERE c.board_type = 'kilter' AND c.layout_id = :layout_id
  AND c.is_listed = 1 AND COALESCE(c.is_draft, 0) = 0
  AND COALESCE(c.is_hidden, 0) = 0
  AND (COALESCE(s.ascensionist_count, 0) > 0 OR COALESCE(l.ascensionist_count, 0) > 0)
  AND EXISTS (SELECT 1 FROM json_each(c.compatible_size_ids) WHERE value = :size_id)
ORDER BY c.uuid, a.angle
"""


def angle_metadata(row, grade_mapping):
    """Keep angle-specific statistics together and resolve a readable boulder grade."""
    current = row["current_ascensionist_count"] or 0
    legacy = row["legacy_ascensionist_count"] or 0
    prefix = "current" if current > 0 else "legacy"
    evidence = "current_snapshot" if current > 0 else "boardlib_snapshot"
    if not (current > 0 or legacy > 0):
        raise ValueError("Climb-angle pair has no recorded ascents")
    stats = {key: row[f"{prefix}_{key}"] for key in [
        "display_difficulty", "difficulty_average", "benchmark_difficulty",
        "quality_average", "fa_username", "fa_at",
    ]}
    difficulty = stats["display_difficulty"]
    grade_source = evidence
    if difficulty is None:
        difficulty = stats["benchmark_difficulty"]
    if difficulty is None:
        difficulty = stats["difficulty_average"]
    if difficulty is None and row["legacy_display_difficulty"] is not None:
        difficulty = row["legacy_display_difficulty"]
        grade_source = "boardlib_snapshot"
    if difficulty is None and row["computed_local_grade"] is not None:
        difficulty = row["computed_local_grade"]
        grade_source = "boardsesh_computed"
    grade = difficulty_to_grade(grade_mapping, difficulty)
    font, separator, hueco = grade.partition("/") if grade else (None, None, None)
    if grade is not None and not separator:
        raise ValueError(f"Expected Font/Hueco boulder grade, got {grade!r}")
    grade_status = ("computed" if grade_source == "boardsesh_computed" else "reported") if grade else "unavailable"
    return {
        "climb_angle_id": f"{row['uuid']}:{row['stats_angle']}",
        "angle": row["stats_angle"],
        "grade": grade, "grade_font": font, "grade_hueco": hueco,
        "grade_index": int(round(difficulty)) if difficulty is not None else None,
        "grade_difficulty": difficulty, "grade_source": grade_source if difficulty is not None else None,
        "grade_status": grade_status,
        **{key: row[key] for key in ["computed_local_grade", "computed_universal_grade", "computed_grade_confidence"]},
        "ascensionist_count": current if current > 0 else legacy,
        "ascent_evidence_source": evidence,
        "current_ascensionist_count": current,
        "boardlib_ascensionist_count": legacy,
        **stats,
    }


def decode_frames(frames):
    """Decode absolute and quoted delta frames, retaining quoted hold frames.

    A comma separates frames; a leading quote means update the previous state.
    An unquoted frame replaces that state. x<ID> turns a hold off. Source order
    is drawing/encoding order, never a claimed climbing movement sequence.
    """
    if not frames:
        raise ValueError("Empty frames string")
    state = {}
    result = []
    for index, segment in enumerate(frames.split(",")):
        delta = index > 0 and segment.startswith('"')
        body = segment[1:] if delta else segment
        if not delta and not body:
            continue
        if not delta:
            state = {}
        position = 0
        for match in TOKEN.finditer(body):
            if match.start() != position:
                raise ValueError(f"Malformed frame near {body[position:position + 30]!r}")
            if match[3] is not None:
                state.pop(int(match[3]), None)
            else:
                state[int(match[1])] = int(match[2])
            position = match.end()
        if position != len(body):
            raise ValueError(f"Malformed frame suffix {body[position:]!r}")
        result.append(dict(state))
    if not result:
        raise ValueError("No decoded frames")
    return result


def read_geometry(path):
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        layout, = db.execute(
            "SELECT * FROM layouts WHERE name = 'Kilter Board Original'"
        ).fetchall()
        size, = db.execute(
            "SELECT * FROM product_sizes WHERE product_id = ? AND name = '12 x 12 with kickboard'",
            (layout["product_id"],),
        ).fetchall()
        roles = {r["id"]: USE_TYPES[r["name"]] for r in db.execute("SELECT * FROM placement_roles")
                 if r["name"] in USE_TYPES}
        for role, name in LEGACY_ROLE_NAMES.items():
            roles.setdefault(role, USE_TYPES[name])
        holds = {}
        for row in db.execute("""
            SELECT p.id, p.set_id, h.name, h.x, h.y FROM placements p
            JOIN holes h ON h.id = p.hole_id
            JOIN product_sizes_layouts_sets psls
              ON psls.layout_id = p.layout_id AND psls.set_id = p.set_id
            WHERE p.layout_id = ? AND psls.product_size_id = ? AND psls.is_listed = 1
              AND h.x > ? AND h.x < ? AND h.y > ? AND h.y < ?
            ORDER BY p.id
        """, (layout["id"], size["id"], size["edge_left"], size["edge_right"],
               size["edge_bottom"], size["edge_top"])):
            holds[row["id"]] = {
                "hold_x": (row["x"] - size["edge_left"]) / (size["edge_right"] - size["edge_left"]),
                "hold_y": (row["y"] - size["edge_bottom"]) / (size["edge_top"] - size["edge_bottom"]),
                "on_kickboard": int(row["name"].split(",")[-1].startswith("KB")),
                "hold_token": row["id"],
                "hole_name": row["name"], "raw_x": row["x"], "raw_y": row["y"],
                "set_id": row["set_id"],
            }
        if not holds:
            raise ValueError("Board geometry has no holds")
        return dict(layout), dict(size), roles, holds


def convert_frame(frame, roles, holds):
    result = []
    for placement, role in sorted(frame.items()):
        if placement not in holds:
            raise ValueError(f"Placement {placement} is outside the 12x12 board or unknown")
        if role not in roles:
            raise ValueError(f"Unknown role {role} on placement {placement}")
        hold = {"hold_use_type": roles[role]}
        hold.update({key: holds[placement][key] for key in HOLD_FIELDS[1:]})
        result.append(hold)
    return result


def export(raw_dir, output_dir, progress=False):
    geometry = raw_dir / "kilter.sqlite3"
    snapshot = raw_dir / "kilter_original.snapshot.sqlite3"
    grades = raw_dir / "kilter_original.grades.sqlite3"
    sources = json.loads((raw_dir / "sources.json").read_text())
    for path, key in [(geometry, "geometry_sha256"), (snapshot, "snapshot_sha256"), (grades, "grades_sha256")]:
        if sha256(path) != sources[key]:
            raise ValueError(f"Source checksum changed: {path}")
    layout, size, roles, holds = read_geometry(geometry)
    grade_mapping = get_difficulty_mapping(geometry)
    counts = Counter()
    angle_counts = Counter()
    used_tokens = set()
    # Stage all files together: a bad source row must never publish a partial export.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".kilter-export-") as temporary:
        staging = Path(temporary)
        with sqlite3.connect(f"{snapshot.resolve().as_uri()}?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            db.execute("ATTACH DATABASE ? AS geometry", (f"{geometry.resolve().as_uri()}?mode=ro",))
            db.execute("ATTACH DATABASE ? AS grades", (f"{grades.resolve().as_uri()}?mode=ro",))
            expected = db.execute("SELECT COUNT(*) FROM (" + QUERY + ")",
                                  {"layout_id": layout["id"], "size_id": size["id"]}).fetchone()[0]
            with (staging / "climbs.jsonl").open("w", encoding="utf-8", newline="\n") as climbs_file, \
                    (staging / "climbs.csv").open("w", encoding="utf-8", newline="") as metadata_file, \
                    (staging / "holds.csv").open("w", encoding="utf-8", newline="") as holds_file:
                writer = csv.DictWriter(holds_file, fieldnames=["climb_angle_id", "climb_uuid", "angle", "frame_index", "hold_index", *HOLD_FIELDS])
                writer.writeheader()
                metadata_writer = None
                previous_uuid = None
                for row in db.execute(QUERY, {"layout_id": layout["id"], "size_id": size["id"]}):
                    try:
                        if row["uuid"] != previous_uuid:
                            raw_frames = decode_frames(row["frames"])
                            frames = [convert_frame(frame, roles, holds) for frame in raw_frames]
                            counts["unique_climbs"] += 1
                            counts["multi_frame_climbs"] += int(len(frames) > 1)
                            previous_uuid = row["uuid"]
                        stats = angle_metadata(row, grade_mapping)
                    except ValueError as error:
                        raise ValueError(f"Climb {row['uuid']} at {row['stats_angle']}° ({row['name']}): {error}") from error
                    climb = {
                        "climb_uuid": row["uuid"], "name": row["name"],
                        "setter_username": row["setter_username"], "description": row["description"],
                        **stats, "setter_angle": row["angle"],
                        "layout_id": layout["id"], "product_size_id": size["id"],
                        "created_at": row["created_at"],
                        "frames_count": len(frames), "frames_pace": row["frames_pace"],
                        "source_frames_count": row["frames_count"],
                        "frame_count_matches_source": len(frames) == row["frames_count"],
                        "raw_frames": row["frames"], "frames": frames,
                    }
                    climbs_file.write(json.dumps(climb, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    metadata = {k: v for k, v in climb.items() if k not in {"frames", "raw_frames"}}
                    if metadata_writer is None:
                        metadata_writer = csv.DictWriter(metadata_file, fieldnames=list(metadata))
                        metadata_writer.writeheader()
                    metadata_writer.writerow(metadata)
                    for frame_index, frame in enumerate(frames):
                        for hold_index, hold in enumerate(frame):
                            writer.writerow({"climb_angle_id": stats["climb_angle_id"], "climb_uuid": row["uuid"],
                                             "angle": stats["angle"], "frame_index": frame_index,
                                             "hold_index": hold_index, **hold})
                            used_tokens.add(hold["hold_token"])
                            counts["hold_entries"] += 1
                            counts[f"use_type_{hold['hold_use_type']}"] += 1
                            counts["kickboard_entries"] += hold["on_kickboard"]
                    counts["climb_angle_rows"] += 1
                    counts["frames"] += len(frames)
                    counts["multi_frame_rows"] += int(len(frames) > 1)
                    counts["frame_count_mismatch_rows"] += int(len(frames) != row["frames_count"])
                    counts[stats["ascent_evidence_source"]] += 1
                    counts[f"grade_{stats['grade_status']}"] += 1
                    angle_counts[stats["angle"]] += 1
                    if progress and counts["climb_angle_rows"] % 25000 == 0:
                        print(f"Exported {counts['climb_angle_rows']:,}/{expected:,} climb-angle rows", flush=True)
        if counts["climb_angle_rows"] != expected or not expected:
            raise ValueError("Export count does not match the full source query or catalogue is empty")
        with (staging / "hold_vocabulary.csv").open("w", encoding="utf-8", newline="") as vocabulary:
            writer = csv.DictWriter(vocabulary, fieldnames=list(next(iter(holds.values()))))
            writer.writeheader()
            writer.writerows(holds.values())
        manifest = {
            "schema_version": 2, "board": layout["name"], "size": size["name"],
            "row_key": ["climb_uuid", "angle"], "angles": sorted(angle_counts),
            "rows_by_angle": dict(sorted(angle_counts.items())),
            "counts": dict(counts), "board_hold_count": len(holds), "used_hold_count": len(used_tokens),
            "selection": "Current public/listed/non-draft/non-hidden climbs compatible with size 10; one row per angle with positive ascent evidence in either snapshot.",
            "grade_mapping": grade_mapping,
            "grade_rounding": "boardlib.api.aurora.difficulty_to_grade: int(round(difficulty)), Python nearest-even rounding",
            "role_mapping": USE_TYPES, "source_role_mapping": roles,
            "coordinate_system": {"x": "left to right", "y": "bottom to top; includes kickboard",
                                  "bounds": {k: size[k] for k in ["edge_left", "edge_right", "edge_bottom", "edge_top"]}},
            "hold_token": "Stable Original placement ID, independent of role and catalogue subset; not contiguous.",
            "sources": sources, "geometry_shared_syncs": get_shared_syncs(geometry),
            "outputs": {path.name: sha256(path) for path in sorted(staging.iterdir())},
            "scripts": {name: sha256(ROOT / name) for name in ["download.py", "prepare.py"]},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in staging.iterdir():
            path.replace(output_dir / path.name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "processed")
    args = parser.parse_args()
    manifest = export(args.raw_dir, args.output_dir, progress=True)
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Exported and checksummed in {args.output_dir}")


if __name__ == "__main__":
    main()
