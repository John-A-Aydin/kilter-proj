"""Versioned split manifests: which partition each climb is in, how that was decided, and checks.

The historical split (``HISTORICAL_SPLIT``) hashes each climb UUID (kilter.data.split_for_uuid),
then drops training rows whose exact hold-and-role set also appears in validation or test. That
keeps every angle of a climb together, but related climbs still straddle partitions: about 4% of
historical test rows share their hand holds with a training climb.

A family split assigns whole families instead. A family joins climbs that share

* a UUID (every angle of a climb),
* an exact hold-and-role set, or
* a hand set: the holds not used as feet, ignoring start/hand/finish differences.

Families are the connected components of those links. That is safe here because the largest
family is a few dozen climbs; ``build_split_manifest`` refuses anything above
``MAX_FAMILY_CLIMBS``. Looser links are not safe: chaining "one hand hold changed" joins thousands
of climbs into one component (the manifest records how many). So near duplicates are measured
rather than grouped. Every held-out climb records the Jaccard similarity of its hand set to the
nearest training hand set, and evaluations report results per band (``SIMILARITY_BANDS``). Climbs
at or above ``NEAR_DUPLICATE_JACCARD`` are near duplicates of training climbs; the others form the
stricter robustness partition.

A manifest is a JSON file (rules, inputs, statistics, and a checksum of its membership) next to a
gzipped CSV with one row per climb UUID. Both are committed. ``python -m kilter split`` builds one.
"""
from collections import Counter, defaultdict
import csv
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

HISTORICAL_SPLIT = "historical_uuid"
MANIFEST_SCHEMA = 1
PARTITIONS = ("train", "validation", "test")
NEAR_DUPLICATE_JACCARD = 0.75
# (name, lowest similarity) from least to most similar; each band runs up to the next one's start.
SIMILARITY_BANDS = (("distant", 0.0), ("related", 0.5), ("near_duplicate", NEAR_DUPLICATE_JACCARD))
MAX_FAMILY_CLIMBS = 200
MEMBERSHIP_COLUMNS = ("climb_uuid", "family", "partition", "nearest_train_hand_jaccard")


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def band_masks(similarity):
    """Boolean row mask per similarity band; NaN (training rows) is in none of them."""
    similarity = np.asarray(similarity, dtype=np.float64)
    masks = {}
    for i, (name, low) in enumerate(SIMILARITY_BANDS):
        high = SIMILARITY_BANDS[i + 1][1] if i + 1 < len(SIMILARITY_BANDS) else np.inf
        masks[name] = (similarity >= low) & (similarity < high)
    return masks


def hand_sets(arrays):
    """Per row, the sorted dense indices of holds not used as feet."""
    from .data import FOOT
    tokens, offsets = arrays["tokens"].astype(np.int64), arrays["offsets"]
    holds, roles = (tokens - 4) // 4, (tokens - 4) % 4
    return [tuple(sorted(holds[a:b][roles[a:b] != FOOT].tolist())) for a, b in zip(offsets[:-1], offsets[1:])]


def _components(count, groups, parent=None):
    """Union-find over ``count`` items, joining each group of item indices; returns the parent array."""
    parent = np.arange(count) if parent is None else parent.copy()

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for members in groups:
        first = find(members[0])
        for member in members[1:]:
            other = find(member)
            if other != first:
                first, other = min(first, other), max(first, other)
                parent[other] = first
    return np.array([find(i) for i in range(count)])


def _grouped(keys, items):
    groups = defaultdict(list)
    for key, item in zip(keys, items):
        groups[key].append(item)
    return [sorted(set(members)) for members in groups.values() if len(set(members)) > 1]


def partition_for(key, seed, train_fraction, validation_fraction):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return 0 if value < train_fraction else (1 if value < train_fraction + validation_fraction else 2)


def nearest_jaccard(queries, references, n_holds, chunk=256):
    """For each query hold set, the largest Jaccard similarity to any reference hold set."""
    def matrix(sets):
        m = np.zeros((len(sets), n_holds), dtype=np.float32)
        for i, holds in enumerate(sets):
            m[i, list(holds)] = 1
        return m

    reference = matrix(references)
    reference_sizes = reference.sum(1)
    best = np.zeros(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk):
        query = matrix(queries[start:start + chunk])
        shared = query @ reference.T
        union = query.sum(1)[:, None] + reference_sizes[None, :] - shared
        best[start:start + chunk] = (shared / np.maximum(union, 1)).max(1)
    return best


def _one_edit_chain_largest(climb_hands, parent):
    """Largest family if climbs whose hand sets differ by one added, removed, or swapped hold were linked too."""
    keys = defaultdict(list)
    for climb, holds in enumerate(climb_hands):
        keys[holds].append(climb)
        for i in range(len(holds)):
            keys[holds[:i] + holds[i + 1:]].append(climb)
    roots = _components(len(climb_hands), [m for m in keys.values() if len(m) > 1], parent)
    return int(Counter(roots.tolist()).most_common(1)[0][1])


def build_split_manifest(data_config, name, seed, output_dir):
    """Write ``<name>.json`` and ``<name>.membership.csv.gz`` under ``output_dir``; return the JSON path.

    Rows come from the historical encoding of ``data_config`` (same filters), so the manifest lists
    exactly the climbs a config with ``split_manifest`` set will encode.
    """
    from .data import EXCLUDED, load_arrays, prepare, sha256 as file_sha256
    historical = replace(data_config, split_manifest="")
    directory, metadata = prepare(historical)
    arrays = load_arrays(directory)
    uuids, signatures, old_splits = arrays["uuids"], arrays["signatures"], arrays["splits"]
    row_hands = hand_sets(arrays)
    climbs, row_climb = np.unique(uuids, return_inverse=True)
    climb_hands = [()] * len(climbs)
    for climb, holds in zip(row_climb, row_hands):
        climb_hands[climb] = holds

    # Families: connected components of shared UUID (already one item), hold-and-role set, or hand set.
    parent = _components(len(climbs), _grouped(signatures, row_climb) + _grouped(row_hands, row_climb))
    roots, family = np.unique(parent, return_inverse=True)  # roots are each family's smallest UUID index.
    sizes = np.bincount(family)
    if sizes.max() > MAX_FAMILY_CLIMBS:
        raise ValueError(f"A family has {sizes.max()} climbs, more than {MAX_FAMILY_CLIMBS}; the links are too loose")
    family_partition = np.array([partition_for(climbs[root], seed, data_config.train_fraction,
                                               data_config.validation_fraction) for root in roots], dtype=np.uint8)
    climb_partition = family_partition[family]
    row_partition = climb_partition[row_climb]

    # Nothing may cross: these hold by construction, and are checked anyway.
    crossing = {}
    for label, keys in (("climb_uuid", row_climb), ("hold_and_role_set", signatures), ("hand_set", row_hands)):
        seen = defaultdict(set)
        for key, part in zip(keys, row_partition):
            seen[key].add(int(part))
        crossing[label] = sum(len(parts) > 1 for parts in seen.values())
    if any(crossing.values()):
        raise AssertionError(f"Families cross partitions: {crossing}")

    train_hands = sorted({holds for holds, part in zip(climb_hands, climb_partition) if part == 0})
    held_climbs = np.flatnonzero(climb_partition != 0)
    similarity = np.full(len(climbs), np.nan, dtype=np.float32)
    similarity[held_climbs] = nearest_jaccard([climb_hands[i] for i in held_climbs], train_hands,
                                              len(metadata["board"]))
    # Rounded once, so the statistics describe exactly the values in the membership file.
    similarity = np.round(similarity, 4)
    row_similarity = similarity[row_climb]

    lines = io.StringIO(newline="")
    writer = csv.writer(lines, lineterminator="\n")
    writer.writerow(MEMBERSHIP_COLUMNS)
    for i, uuid in enumerate(climbs):
        value = "" if np.isnan(similarity[i]) else f"{similarity[i]:.4f}"
        writer.writerow([uuid, int(family[i]), PARTITIONS[climb_partition[i]], value])
    membership = lines.getvalue().encode()

    grades = np.round(arrays["conditions"][:, 0]).astype(int)
    angles = np.round(arrays["conditions"][:, 1] * 90).astype(int)

    def counts(values):
        return {str(k): int(v) for k, v in sorted(Counter(values.tolist()).items())}

    partitions = {}
    for index, label in enumerate(PARTITIONS):
        rows = row_partition == index
        partitions[label] = {"rows": int(rows.sum()), "climbs": int((climb_partition == index).sum()),
                             "families": int((family_partition == index).sum()),
                             "rows_by_grade": counts(grades[rows]), "rows_by_angle": counts(angles[rows])}
        if index:
            bands = band_masks(row_similarity)
            climb_bands = band_masks(similarity)
            partitions[label]["nearest_train_hand_jaccard"] = {
                "rows_by_band": {band: int((mask & rows).sum()) for band, mask in bands.items()},
                "climbs_by_band": {band: int((mask & (climb_partition == index)).sum())
                                   for band, mask in climb_bands.items()},
                "row_quantiles": {str(q): round(float(np.quantile(row_similarity[rows], q)), 4) if rows.any() else None
                                  for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}

    historical_train_hands = {h for h, s in zip(row_hands, old_splits) if s == 0}
    historical = {}
    for index, label in ((1, "validation"), (2, "test")):
        old, new = old_splits == index, row_partition == index
        shared = np.array([h in historical_train_hands for h in row_hands]) & old
        historical[label] = {"rows_also_in_historical": int((old & new).sum()),
                             "historical_rows_sharing_a_train_hand_set": int(shared.sum()),
                             "historical_rows": int(old.sum())}

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "name": name,
        "description": "Family-separated split: climbs that share a UUID, an exact hold-and-role set, or a hand "
                       "set are always in the same partition. Held-out climbs record how similar the nearest "
                       "training hand set is.",
        "rules": {
            "family_links": ["same climb_uuid", "same hold-and-role set",
                             "same hand set (holds not used as feet; start, hand and finish roles ignored)"],
            "family_closure": "connected components",
            "family_key": "smallest climb_uuid in the family",
            "assignment": "sha256('<seed>:<family_key>'), first 8 bytes as a fraction of 2**64: train below "
                          "train_fraction, validation below train_fraction + validation_fraction, test above",
            "seed": seed, "train_fraction": data_config.train_fraction,
            "validation_fraction": data_config.validation_fraction,
            "max_family_climbs": MAX_FAMILY_CLIMBS,
            "similarity": "Jaccard similarity of a held-out climb's hand set to the nearest training hand set",
            "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
            "similarity_bands": {band: low for band, low in SIMILARITY_BANDS},
        },
        "built_from": {
            "source_sha256": metadata["source_sha256"], "vocabulary_sha256": metadata["vocabulary_sha256"],
            "encoded_sha256": metadata["encoded_sha256"], "data_fingerprint": metadata["fingerprint"],
            "filters": {k: v for k, v in metadata["config"].items()
                        if k in {"max_holds", "include_computed_grades"}},
            "builder_sha256": file_sha256(__file__),
        },
        "membership": {"file": f"{name}.membership.csv.gz", "sha256": _sha256(membership),
                       "columns": list(MEMBERSHIP_COLUMNS), "climbs": len(climbs)},
        "statistics": {
            "rows": len(uuids), "climbs": len(climbs), "families": len(roots),
            "rows_excluded_by_historical_split": int((old_splits == EXCLUDED).sum()),
            "family_size": {"largest_climbs": int(sizes.max()),
                            "largest_rows": int(np.bincount(family[row_climb]).max()),
                            "families_with_several_climbs": int((sizes > 1).sum()),
                            "climbs_in_shared_families": int(sizes[sizes > 1].sum())},
            "crossing_partitions": crossing,
            "one_hold_edit_chaining_largest_family_climbs": _one_edit_chain_largest(climb_hands, parent),
            "partitions": partitions,
            "compared_with_historical_split": historical,
        },
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / manifest["membership"]["file"]
    temporary = membership_path.with_suffix(".part")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as target:
        target.write(membership)
    temporary.replace(membership_path)
    path = output_dir / f"{name}.json"
    from .config import write_json
    write_json(path, manifest)
    read_split_manifest(path)  # Reads back, membership checksum included.
    return path, manifest


def read_split_manifest(path, rows=True):
    """A manifest's name, checksum, and (with ``rows``) verified membership by climb UUID."""
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a split manifest with schema version {MANIFEST_SCHEMA}")
    result = {"name": manifest["name"], "sha256": _sha256(raw), "manifest": manifest}
    if not rows:
        return result
    data = gzip.decompress((path.parent / manifest["membership"]["file"]).read_bytes())
    if _sha256(data) != manifest["membership"]["sha256"]:
        raise ValueError(f"Membership of split manifest {manifest['name']} does not match its checksum")
    partition, family, similarity = {}, {}, {}
    reader = csv.DictReader(io.StringIO(data.decode()))
    if tuple(reader.fieldnames) != MEMBERSHIP_COLUMNS:
        raise ValueError(f"Split manifest membership must have columns {MEMBERSHIP_COLUMNS}")
    for row in reader:
        uuid = row["climb_uuid"]
        partition[uuid] = PARTITIONS.index(row["partition"])
        family[uuid] = int(row["family"])
        if row["nearest_train_hand_jaccard"]:
            similarity[uuid] = float(row["nearest_train_hand_jaccard"])
    return {**result, "partition": partition, "family": family, "nearest_train_hand_jaccard": similarity}
