"""Download boardlib's geometry database and a complete public Original snapshot."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import gzip
import importlib.metadata
import json
from pathlib import Path
import sqlite3
import shutil

from boardlib.db.aurora import download_database, get_shared_syncs
import requests

ROOT = Path(__file__).resolve().parent
MANIFEST_URL = (
    "https://snapshots.boardsesh.com/board-snapshots/v1-gzip/manifest.json"
)


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_database(path):
    with sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError(f"SQLite integrity check failed: {path}")


def download_snapshot(entry, destination, workers=12):
    """Use checked byte ranges; retain completed chunks to resume interrupted runs."""
    if entry["contentEncoding"] not in {"identity", "gzip"}:
        raise ValueError("Unsupported snapshot encoding")
    chunk_size = 2 * 1024 * 1024
    source_id = hashlib.sha256(entry["url"].encode()).hexdigest()[:12]
    parts = destination.with_name(destination.name + f".{source_id}.parts")
    parts.mkdir(exist_ok=True)
    # Bind cached chunks to the immutable source URL and size.
    identity = json.dumps({"url": entry["url"], "bytes": entry["bytes"]}, sort_keys=True)
    marker = parts / "source.json"
    if marker.exists() and marker.read_text() != identity:
        raise ValueError(f"Different snapshot cached in {parts}; use a fresh --raw-dir")
    marker.write_text(identity)

    def fetch(start):
        end = min(start + chunk_size, entry["bytes"]) - 1
        path = parts / f"{start:012d}"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(3):
            try:
                with requests.get(
                    entry["url"], headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": entry["contentEncoding"]},
                    timeout=(30, 120), stream=True,
                ) as response:
                    response.raise_for_status()
                    # Range offsets refer to the compressed object. Do not let
                    # requests decompress fragments of a gzip stream.
                    content = response.raw.read(decode_content=False)
                    if (response.status_code != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{entry['bytes']}"
                            or len(content) != end - start + 1):
                        raise ValueError("Server returned an invalid byte range")
                temporary = path.with_suffix(".part")
                temporary.write_bytes(content)
                temporary.replace(path)
                return path
            except requests.RequestException:
                if attempt == 2:
                    raise

    starts = list(range(0, entry["bytes"], chunk_size))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, start) for start in starts]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 10 == 0 or count == len(starts):
                print(f"Snapshot chunks: {count}/{len(starts)}", flush=True)
    compressed = parts / "assembled"
    with compressed.open("wb") as target:
        for start in starts:
            target.write((parts / f"{start:012d}").read_bytes())
    temporary = destination.with_suffix(".part")
    if entry["contentEncoding"] == "gzip":
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
    else:
        shutil.copyfile(compressed, temporary)
    if temporary.stat().st_size != entry.get("uncompressedBytes", entry["bytes"]):
        raise ValueError("Decoded snapshot size does not match manifest")
    check_database(temporary)
    temporary.replace(destination)
    for start in starts:
        (parts / f"{start:012d}").unlink()
    marker.unlink()
    compressed.unlink()
    parts.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--manifest", type=Path, help="Saved manifest to pin an exact snapshot")
    selection.add_argument("--latest", action="store_true", help="Resolve the current published manifest")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    geometry = args.raw_dir / "kilter.sqlite3"
    if not geometry.exists():
        print("Downloading boardlib Kilter database (geometry and role metadata)", flush=True)
        temporary = geometry.with_suffix(".part")
        download_database("kilter", temporary)
        check_database(temporary)
        temporary.replace(geometry)
    check_database(geometry)

    manifest_path = args.raw_dir / "snapshot_manifest.json"
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    elif not args.latest and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif not args.latest and (ROOT / "snapshot-lock.json").exists():
        manifest = json.loads((ROOT / "snapshot-lock.json").read_text())
    else:
        response = requests.get(MANIFEST_URL, timeout=30)
        response.raise_for_status()
        manifest = response.json()
    entry, = [e for e in manifest["entries"] if e["boardType"] == "kilter" and e["layoutId"] == 1]
    if manifest.get("geometrySha256") and sha256(geometry) != manifest["geometrySha256"]:
        raise ValueError("Geometry differs from snapshot lock; exact reproduction requires the retained raw database")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    snapshot = args.raw_dir / "kilter_original.snapshot.sqlite3"
    provenance_path = args.raw_dir / "sources.json"
    if snapshot.exists():
        if not provenance_path.exists():
            raise ValueError("Existing snapshot lacks sources.json; use a fresh --raw-dir")
        provenance = json.loads(provenance_path.read_text())
        if provenance["snapshot"]["url"] != entry["url"] or provenance["snapshot_sha256"] != sha256(snapshot):
            raise ValueError("Existing snapshot does not match pinned provenance")
    else:
        print(f"Downloading full public Kilter Original catalogue built {entry['builtAt']}", flush=True)
        download_snapshot(entry, snapshot, args.workers)
    check_database(snapshot)
    if manifest.get("snapshotSha256") and sha256(snapshot) != manifest["snapshotSha256"]:
        raise ValueError("Snapshot checksum differs from snapshot lock")
    with sqlite3.connect(f"{snapshot.resolve().as_uri()}?mode=ro", uri=True) as db:
        for table in ["board_climbs", "board_climb_stats"]:
            if db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] != entry["tables"][table]["rowCount"]:
                raise ValueError(f"Downloaded row count disagrees with manifest: {table}")
    grades = args.raw_dir / "kilter_original.grades.sqlite3"
    if not grades.exists():
        print("Downloading the matching angle-specific grade table", flush=True)
        download_snapshot(entry["grades"], grades, args.workers)
    check_database(grades)
    grades_digest = sha256(grades)
    if manifest.get("gradesSha256") and grades_digest != manifest["gradesSha256"]:
        raise ValueError("Grade table checksum differs from snapshot lock")
    with sqlite3.connect(f"{grades.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("SELECT COUNT(*) FROM board_climb_grades").fetchone()[0] != entry["grades"]["tables"]["board_climb_grades"]["rowCount"]:
            raise ValueError("Grade table count disagrees with manifest")
    provenance = {
        "boardlib_version": importlib.metadata.version("boardlib"),
        "geometry_source": "boardlib.db.aurora.download_database('kilter', ...)",
        "geometry_sha256": sha256(geometry),
        "geometry_shared_syncs": get_shared_syncs(geometry),
        "manifest_url": MANIFEST_URL,
        "snapshot": entry,
        "snapshot_sha256": sha256(snapshot),
        "grades_sha256": grades_digest,
        "freshness_note": "Complete published snapshot, not a live sync. Boardlib provides geometry and historical ascent evidence.",
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Sources validated and recorded in {provenance_path}")


if __name__ == "__main__":
    main()
