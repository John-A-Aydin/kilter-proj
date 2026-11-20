"""Identities of training runs and evaluations, so finished work is reused only for the same request.

A run's identity is everything that decides what training produces: the resolved config (except
the output folder), the training source code, the dataset fingerprint (input contents,
preprocessing code, filters, and split), the hold-label and style-tag files, and the device type.
``train`` writes it to ``identity.json`` in the run folder and refuses to resume a run under a
different identity. Runs trained before identities existed have none, so they are never reused.

An evaluation's identity is its checkpoint, its evaluator (a version that changes whenever a
metric's meaning changes, plus the source code it runs), the split and dataset, and every setting
that changes the numbers. Each evaluation output records it under ``"evaluation"``.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

IDENTITY_VERSION = 1
PACKAGE = Path(__file__).resolve().parent
# Files training never runs. Changing them leaves what a run produces unchanged.
NOT_TRAINING = {"kilter": {"__main__.py", "export_web.py", "identity.py", "quality.py", "baseline"},
                "kilter.baseline": {"__main__.py"}}


def file_sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


@lru_cache(maxsize=None)
def _cached_sha256(path, modified, size):
    return file_sha256(path)


def checkpoint_sha256(path):
    """sha256 of a (large) file, computed once per version of it in this process."""
    path = Path(path).resolve()
    status = path.stat()
    return _cached_sha256(str(path), status.st_mtime_ns, status.st_size)


def canonical(value):
    """``value`` as it reads back from JSON, so in-memory and saved identities compare equal."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_files(package, extra=()):
    """sha256 of each Python file ``package`` trains with, by path within the package, plus ``extra`` files."""
    root = PACKAGE if package == "kilter" else PACKAGE / "baseline"
    files = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in relative or (relative.split("/")[0] in NOT_TRAINING[package] and relative not in extra):
            continue
        files[relative] = file_sha256(path)
    return files


def training_identity(package, config):
    """The identity of training ``config`` (a resolved Config of ``package``) with the current code and data."""
    from .config import resolve_device
    values = config.to_dict()
    values.pop("output_dir")
    if package == "kilter":
        from .data import data_specification
        inputs = {f"{key}_sha256": file_sha256(path) if path else None
                  for key, path in (("hold_labels", config.model.hold_labels), ("climb_tags", config.model.climb_tags))}
    elif package == "kilter.baseline":
        from .baseline.data import data_specification
        inputs = {}
    else:
        raise ValueError(f"Unknown package {package}")
    identity = canonical({"identity_version": IDENTITY_VERSION, "package": package, "config": values,
                          "device_type": resolve_device(config.train.device), "source": source_files(package),
                          "data_fingerprint": data_specification(config.data)[1], "inputs": inputs})
    return {"run_id": digest(identity)[:20], **identity}


def read_run_identity(run_dir):
    path = Path(run_dir) / "identity.json"
    return json.loads(path.read_text()) if path.exists() else None


def evaluation_identity(kind, version, checkpoint, split, data_fingerprint, settings, sources):
    """``sources`` lists (package, extra files) whose code the evaluation runs."""
    checkpoint = Path(checkpoint)
    run = read_run_identity(checkpoint.resolve().parent)
    identity = canonical({
        "identity_version": IDENTITY_VERSION, "kind": kind, "evaluator_version": version,
        "evaluator_source": {package: source_files(package, extra) for package, extra in sources},
        "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "run_id": run["run_id"] if run else None, "split": split, "data_fingerprint": data_fingerprint,
        "settings": settings,
    })
    return {"evaluation_id": digest(identity)[:20], **identity}


def differences(requested, recorded, prefix=""):
    """Dotted keys whose values differ between two identities, for explaining a mismatch."""
    if not isinstance(requested, dict) or not isinstance(recorded, dict):
        return [] if requested == recorded else [prefix or "(value)"]
    keys = sorted(set(requested) | set(recorded))
    found = []
    for key in keys:
        if key in {"run_id", "evaluation_id"}:
            continue
        name = f"{prefix}.{key}" if prefix else key
        if key not in requested or key not in recorded:
            found.append(name)
        else:
            found.extend(differences(requested[key], recorded[key], name))
    return found
