"""Command line: `python -m wireframe extract|synth --help`."""
import argparse
import json
from pathlib import Path

from .extract import PoseFile, evaluate, extract
from .geometry import find_climb, find_climb_by_uuid


def climb_from(args):
    if args.uuid:
        return find_climb_by_uuid(args.uuid, args.angle)
    return find_climb(args.climb, args.angle)


def main():
    parser = argparse.ArgumentParser(prog="python -m wireframe", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("extract", help="Wireframes from a video, in board coordinates")
    run.add_argument("video", type=Path)
    run.add_argument("--climb", help="Climb name, as in the dataset")
    run.add_argument("--uuid", help="Climb UUID, when the name is ambiguous")
    run.add_argument("--angle", type=int, required=True, help="Wall angle in degrees")
    run.add_argument("--interval", type=float, default=0.25, help="Seconds between wireframes")
    run.add_argument("--output", type=Path, required=True, help="Output JSON")
    run.add_argument("--pose-file", type=Path, help="Pre-computed landmarks instead of MediaPipe")
    run.add_argument("--points", type=Path, help='Manual board points: JSON {"placement": [u, v], ...}')
    run.add_argument("--focal-px", type=float, help="Camera focal length in pixels, if known")
    run.add_argument("--camera", choices=["auto", "static", "moving"], default="auto")
    run.add_argument("--overlay-dir", type=Path, help="Write each sampled frame with the fit drawn on it")
    run.add_argument("--ground-truth", type=Path, help="Synthetic truth.json, to report errors")

    synth = commands.add_parser("synth", help="Make a synthetic test video with ground truth")
    synth.add_argument("--climb", help="Climb name, as in the dataset")
    synth.add_argument("--uuid")
    synth.add_argument("--angle", type=int, required=True)
    synth.add_argument("--output-dir", type=Path, required=True)
    synth.add_argument("--seed", type=int, default=0)

    tables = commands.add_parser("tables", help="Flat JSON tables from a result, for dashboards")
    tables.add_argument("result", type=Path, help="Output of extract")
    tables.add_argument("--ground-truth", type=Path, help="Synthetic truth.json, to add true joints")
    tables.add_argument("--output-dir", type=Path, required=True)
    tables.add_argument("--images", action="store_true", help="Also write each sampled frame as a small JPEG")

    args = parser.parse_args()
    if args.command == "tables":
        from .tables import write
        print(json.dumps(write(args.result, args.output_dir, args.ground_truth, args.images)))
        return
    if not (args.climb or args.uuid):
        parser.error("pass --climb or --uuid")
    climb = climb_from(args)
    if args.command == "synth":
        from .synthetic import make
        print(json.dumps(make(climb, args.output_dir, seed=args.seed)))
        return

    model = None
    if args.pose_file:
        pose = PoseFile(args.pose_file)
        source = "pose-file"
    else:
        from .pose import MediaPipePose
        model = MediaPipePose()
        pose = lambda frame, time_ms, index: model(frame, time_ms)  # noqa: E731
        source = f"mediapipe pose_landmarker_heavy sha256 {model.model_sha256}"
    points = json.loads(args.points.read_text()) if args.points else None
    try:
        result = extract(args.video, climb, args.interval, pose, points, args.focal_px, args.camera, args.overlay_dir)
    finally:
        if model is not None:
            model.close()
    result["pose_source"] = source
    if args.ground_truth:
        result["evaluation"] = evaluate(result, json.loads(args.ground_truth.read_text()))
        print(json.dumps(result["evaluation"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    posed = sum(f["pose"] for f in result["frames"])
    print(f"{len(result['frames'])} samples, {result['registration']['registered']} with the board found, "
          f"{posed} with a wireframe -> {args.output}")


if __name__ == "__main__":
    main()
